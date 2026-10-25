"""Correlated and multinomial varying effects, categorical marginal means and
group-level summaries for sampled models.

There is no exact R oracle for MCMC: these tests check the model's log density
against independent SciPy formulas, recover simulated parameters, recompute
summaries from the draws, and compare loosely with brms posteriors fitted with
the same priors (tests/r_brms_cases.R).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import stats
from scipy.special import betaln, expit, softmax

from stats3 import (bayes_emmeans, bayes_glm, bayes_ranef, pairs, posterior_summary, predict, prior, tidy)
from oracle import BACKENDS, as_input, as_polars

pytestmark = pytest.mark.skipif(importlib.util.find_spec('pymc') is None, reason='requires stats3[bayes]')

DATA = Path(__file__).resolve().parent / 'data'
PRIORS = {'beta': prior('normal', mu=0, sigma=2.5), '(Intercept)': prior('normal', mu=0, sigma=5),
          'sigma': prior('exponential', lam=1), 'group_sd': prior('half_normal', sigma=1),
          'group_cor': prior('lkj', eta=2)}
# Multinomial coefficients are named by category, as brms's dpars are.
MULTINOMIAL_PRIORS = {'beta': PRIORS['beta'], 'b:(Intercept)': PRIORS['(Intercept)'],
                      'c:(Intercept)': PRIORS['(Intercept)'], 'group_sd': PRIORS['group_sd']}


def groups():
    """Ten groups of 16: y has correlated intercepts/slopes (sd .8, .5, cor .6,
    sigma .5); choice is a three-category outcome with random intercepts."""
    return pl.read_csv(DATA / 'bayes_groups.csv')


def _codes(frame):
    levels = sorted(set(frame['g']))
    return np.array([levels.index(v) for v in frame['g']])


def _values(model, point):
    outs = model.unobserved_value_vars
    return dict(zip([v.name for v in outs], model.compile_fn(outs, inputs=model.value_vars, on_unused_input='ignore')(point)))


def _points(model, count, seed):
    rng = np.random.default_rng(seed)
    start = model.initial_point()
    return [{k: v + rng.normal(scale=.4, size=np.shape(v)) for k, v in start.items()} for _ in range(count)]


def _lkj2(r, eta):
    """LKJ(eta) density of a 2 x 2 correlation: (r+1)/2 ~ Beta(eta, eta)."""
    return stats.beta.logpdf((r+1)/2, eta, eta) - np.log(2)


def test_correlated_log_density_matches_independent_formula():
    source = groups()
    priors = {**PRIORS, 'beta': prior('normal', mu=0, sigma=2), '(Intercept)': prior('normal', mu=0, sigma=3),
              'group_sd': prior('half_normal', sigma=1.5), 'group_cor': prior('lkj', eta=3)}
    fit = bayes_glm(source, 'y~x+(1+x|g)', priors=priors, draws=5, tune=5, chains=1, seed=1)
    model = fit.pymc_model
    x, y, codes = source['x'].to_numpy(), source['y'].to_numpy(), _codes(source)
    logp = model.compile_logp(jacobian=False)
    observed, expected = [], []
    for point in _points(model, 4, 2):
        v = _values(model, point)
        sd, r, z = v['group_0_sd'], v['group_0_cor'][0], v['z_group_0']
        chol = np.array([[sd[0], 0], [sd[1]*r, sd[1]*np.sqrt(1-r*r)]])
        np.testing.assert_allclose(v['group_0'], z @ chol.T, atol=1e-12)
        b = z @ chol.T
        # Slopes are sampled for the centred predictor; the varying slope is not centred.
        eta = v['coefficient_0'] + v['coefficient_1']*(x-x.mean()) + b[codes, 0] + b[codes, 1]*x
        # The sampled value is the packed Cholesky factor L of the covariance;
        # |d(L11, L21, L22)/d(sd1, sd2, r)| = sd2 / sqrt(1 - r^2).
        expected.append(stats.norm.logpdf(v['coefficient_0'], 0, 3) + stats.norm.logpdf(v['coefficient_1'], 0, 2)
            + stats.expon.logpdf(v['sigma']) + stats.norm.logpdf(z).sum()
            + stats.halfnorm.logpdf(sd, scale=1.5).sum() + _lkj2(r, 3) + np.log(np.sqrt(1-r*r)/sd[1])
            + stats.norm.logpdf(y, eta, v['sigma']).sum())
        observed.append(logp(point))
    # PyMC 6.3's LKJ normalizing constant has the wrong sign (it adds
    # log B(1/2, eta) for n = 2), a constant that does not change the
    # posterior; compare log densities up to that constant.
    np.testing.assert_allclose(np.diff(observed), np.diff(expected), rtol=1e-10, atol=1e-8)
    assert observed[0] - expected[0] == pytest.approx(2*betaln(.5, 3), abs=1e-6)


def test_multinomial_log_density_and_pointwise_likelihood():
    source = groups()
    fit = bayes_glm(source, 'choice~x+(1|g)', family='multinomial', priors=MULTINOMIAL_PRIORS,
                    draws=20, tune=30, chains=1, seed=3)
    assert [e['category'] for e in fit.random] == [0, 1]
    model = fit.pymc_model
    x, codes = source['x'].to_numpy(), _codes(source)
    y = np.array([['a', 'b', 'c'].index(v) for v in source['choice']])
    logp = model.compile_logp(jacobian=False)
    for point in _points(model, 2, 4):
        v = _values(model, point)
        theta = np.array([v[f'coefficient_{j}'] for j in range(4)]).reshape(2, 2)
        eta = theta[:, 0] + np.outer(x-x.mean(), theta[:, 1])
        eta += np.column_stack([v['group_0'][codes, 0], v['group_1'][codes, 0]])
        np.testing.assert_allclose(v['group_0'][:, 0], v['z_group_0'][:, 0]*v['group_0_sd'][0])
        probability = softmax(np.column_stack([np.zeros(len(x)), eta]), axis=1)
        expected = (stats.norm.logpdf(theta[:, 0], 0, 5).sum() + stats.norm.logpdf(theta[:, 1], 0, 2.5).sum()
            + sum(stats.norm.logpdf(v[f'z_group_{j}']).sum() + stats.halfnorm.logpdf(v[f'group_{j}_sd']).sum() for j in (0, 1))
            + np.log(probability[np.arange(len(x)), y]).sum())
        # PyTensor rewrites the categorical log density (log-softmax); the two
        # agree to about 1e-9 relative.
        assert logp(point) == pytest.approx(expected, rel=1e-8)
    # The stored pointwise likelihood uses the per-category group effects.
    beta = fit.beta_draws.reshape(1, 20, 2, 2)
    eta = beta[..., 0, None] + beta[..., 1, None]*x
    eta = eta + np.stack([fit.extra_draws[f'group_{j}'][:, :, codes, 0] for j in (0, 1)], axis=2)
    logits = np.concatenate([np.zeros((1, 20, 1, len(x))), eta], axis=2)
    expected = np.take_along_axis(np.log(softmax(logits, axis=2)), y[None, None, None, :], axis=2)[:, :, 0]
    np.testing.assert_allclose(fit.idata.log_likelihood['observed'].values, expected, atol=1e-10)
    assert tidy(fit)['term'].to_list()[-2:] == ['sd_g__mub_Intercept', 'sd_g__muc_Intercept']
    conditional = predict(fit, source.head(2))
    assert conditional.height == 6
    assert not np.allclose(conditional['fit'], predict(fit, source.head(2), re_formula='population')['fit'])


def test_three_correlated_terms_follow_brms_order_and_double_bar_stays_independent():
    source = groups().with_columns(w=(pl.col('x')*3).sin())
    fit = bayes_glm(source, 'y~x+(1+x+w|g)', draws=20, tune=30, chains=1, seed=5)
    terms = tidy(fit, conf_int=True)['term'].to_list()
    assert terms[2:] == ['sd_g__Intercept', 'sd_g__x', 'sd_g__w', 'cor_g__Intercept__x',
                         'cor_g__Intercept__w', 'cor_g__x__w']
    # The correlations are stored in that order: rebuilding the Cholesky
    # factor from them reproduces the sampled effects.
    sd, cor = fit.extra_draws['group_0_sd'], fit.extra_draws['group_0_cor']
    matrix = np.broadcast_to(np.eye(3), (*cor.shape[:2], 3, 3)).copy()
    for value, (i, j) in zip(np.moveaxis(cor, -1, 0), [(0, 1), (0, 2), (1, 2)]):
        matrix[..., i, j] = matrix[..., j, i] = value
    chol = sd[..., :, None]*np.linalg.cholesky(matrix)
    z = fit.idata.posterior['z_group_0'].values
    np.testing.assert_allclose(fit.extra_draws['group_0'], np.einsum('cdlk,cdjk->cdlj', z, chol), atol=1e-10)
    assert not np.allclose(cor, 0)
    independent = bayes_glm(source, 'y~x+(1+x+w||g)', draws=20, tune=30, chains=1, seed=5)
    assert tidy(independent)['term'].to_list()[2:] == ['sd_g__Intercept', 'sd_g__x', 'sd_g__w']
    assert 'group_0_cor' not in independent.extra_draws
    assert not any('cholesky' in v.name for v in independent.pymc_model.free_RVs)
    # Exponentiation leaves the SDs and correlations on their own scale.
    raw, exponentiated = tidy(fit), tidy(fit, exponentiate=True)
    np.testing.assert_allclose(raw['estimate'][2:], exponentiated['estimate'][2:])
    np.testing.assert_allclose(exponentiated['estimate'][:2], np.exp(fit.beta_draws).mean(axis=(0, 1)))


def test_new_groups_draw_correlated_effects():
    fit = bayes_glm(groups(), 'y~x+(1+x|g)', draws=10, tune=10, chains=1, seed=6)
    shape = (2, 20000)
    samples = {'group_0_sd': np.broadcast_to([1., 2.], (*shape, 2)), 'group_0_cor': np.full((*shape, 1), .7)}
    b = fit._new_effect(samples, fit.random[0], np.random.default_rng(1)).reshape(-1, 2)
    np.testing.assert_allclose(b.std(axis=0), [1, 2], rtol=.02)
    assert np.corrcoef(b.T)[0, 1] == pytest.approx(.7, abs=.01)
    new = pl.DataFrame({'x': [.5, .5], 'g': ['new', 'new']})
    with pytest.raises(ValueError, match='new level'):
        predict(fit, new)
    assert predict(fit, new, allow_new_levels=True, seed=2)['fit'].is_finite().all()


def test_priors_for_correlations():
    source = groups()
    fit = bayes_glm(source, 'y~x+(1+x|g)', priors={'group_cor': prior('lkj', eta=4)}, draws=5, tune=5, chains=1, seed=7)
    chol = fit.pymc_model['z_group_0_cholesky']
    assert chol.owner.inputs[-2].eval() == pytest.approx(4)
    default = bayes_glm(source, 'y~x+(1+x|g)', draws=5, tune=5, chains=1, seed=7)
    assert default.pymc_model['z_group_0_cholesky'].owner.inputs[-2].eval() == pytest.approx(2)
    with pytest.raises(ValueError, match='positive scalar eta'):
        prior('lkj', eta=-1)
    with pytest.raises(ValueError, match='positive scalar eta'):
        prior('lkj', shape=2)
    with pytest.raises(ValueError, match="group_cor needs prior"):
        bayes_glm(source, 'y~x+(1+x|g)', priors={'group_cor': prior('normal')}, draws=5)
    with pytest.raises(ValueError, match="only for group_cor"):
        bayes_glm(source, 'y~x', priors={'sigma': prior('lkj', eta=2)}, draws=5)
    with pytest.raises(ValueError, match='unused prior'):
        bayes_glm(source, 'y~x+(1+x||g)', priors={'group_cor': prior('lkj', eta=2)}, draws=5)
    with pytest.raises(ValueError, match='unused prior'):
        bayes_glm(source, 'y~x+(1|g)', priors={'group_cor': prior('lkj', eta=2)}, draws=5)


def test_simulated_correlated_effects_are_recovered():
    rng = np.random.default_rng(8)
    groups_, size = 40, 12
    truth = {'sd': np.array([1., .6]), 'cor': .6, 'sigma': .4, 'beta': np.array([.5, -1.])}
    chol = np.linalg.cholesky([[1, truth['cor']], [truth['cor'], 1]])*truth['sd'][:, None]
    b = rng.normal(size=(groups_, 2)) @ chol.T
    codes = np.repeat(np.arange(groups_), size)
    x = rng.uniform(-1, 1, groups_*size)
    y = truth['beta'][0] + truth['beta'][1]*x + b[codes, 0] + b[codes, 1]*x + rng.normal(0, truth['sigma'], len(x))
    source = pl.DataFrame({'y': y, 'x': x, 'g': [f'g{c:02d}' for c in codes]})
    fit = bayes_glm(source, 'y~x+(1+x|g)', draws=300, tune=400, chains=2, seed=9)
    table = tidy(fit, conf_int=True)
    values = [*truth['beta'], *truth['sd'], truth['cor']]
    for term, value in zip(['(Intercept)', 'x', 'sd_g__Intercept', 'sd_g__x', 'cor_g__Intercept__x'], values):
        row = table.filter(pl.col('term') == term)
        assert row['conf_low'][0] < value < row['conf_high'][0], term
    assert posterior_summary(fit).filter(pl.col('term') == 'sigma')['estimate'][0] == pytest.approx(truth['sigma'], rel=.15)
    effects = bayes_ranef(fit)
    inside = (effects['conf_low'].to_numpy() < b.reshape(-1)) & (b.reshape(-1) < effects['conf_high'].to_numpy())
    assert inside.mean() > .85
    assert np.corrcoef(effects['estimate'].to_numpy(), b.reshape(-1))[0, 1] > .9
    assert fit.glance()['divergences'][0] < 10


@pytest.mark.parametrize('backend', BACKENDS)
def test_group_level_summaries_on_every_backend(backend):
    source = groups()
    data, _ = as_input(source, backend, [])
    fit = bayes_glm(data, 'y~x+(1+x|g)', draws=20, tune=30, chains=1, seed=10)
    effects = as_polars(bayes_ranef(fit, level=.9))
    assert effects.columns == ['group', 'level', 'term', 'estimate', 'std_error', 'conf_low', 'conf_high', 'prob_positive']
    assert effects['level'].to_list() == [f's{i:02d}' for i in range(10) for _ in (0, 1)]
    assert effects['term'].to_list() == ['(Intercept)', 'x']*10
    draws = fit.extra_draws['group_0'].reshape(20, 20)
    np.testing.assert_allclose(effects['estimate'], draws.mean(axis=0))
    np.testing.assert_allclose(effects['std_error'], draws.std(axis=0, ddof=1))
    np.testing.assert_allclose(effects['conf_low'], np.quantile(draws, .05, axis=0))
    expected = {'polars': 'DataFrame', 'lazy': 'LazyFrame', 'pandas': 'DataFrame', 'tidy3': 'TidyFrame'}[backend]
    assert type(bayes_ranef(fit)).__name__ == expected
    multinomial = bayes_glm(data, 'choice~x+(1|g)', family='multinomial', draws=20, tune=30, chains=1, seed=10)
    effects = as_polars(bayes_ranef(multinomial))
    assert effects['term'].to_list()[:4] == ['b:(Intercept)', 'c:(Intercept)']*2
    assert effects['level'].to_list()[:4] == ['s00', 's00', 's01', 's01']
    np.testing.assert_allclose(effects['estimate'][1], multinomial.extra_draws['group_1'][0, :, 0, 0].mean())
    with pytest.raises(ValueError, match='no varying effects'):
        bayes_ranef(bayes_glm(data, 'y~x', draws=5, tune=5, chains=1, seed=1))


def test_grouped_fits_stack_group_level_summaries():
    source = groups().with_columns(half=pl.when(pl.col('x') < 0).then(pl.lit('left')).otherwise(pl.lit('right')))
    fits = bayes_glm(source, 'y~x+(1|g)', by='half', draws=10, tune=10, chains=1, seed=11)
    effects = bayes_ranef(fits)
    assert effects['half'].unique().sort().to_list() == ['left', 'right']
    assert effects.height == 20
    assert tidy(fits).filter(pl.col('term') == 'sd_g__Intercept').height == 2


def _ordinal_cells(fit, x):
    """Cell probabilities for f = u, v with x at its mean (the reference
    grid), recomputed from the draws: (chain, draw, cell, category)."""
    out = []
    for v in (0, 1):
        eta = fit.beta_draws[..., 0]*x.mean() + fit.beta_draws[..., 1]*v
        cumulative = expit(fit.extra_draws['cutpoints'] - eta[..., None])
        out.append(np.diff(np.concatenate([np.zeros((*eta.shape, 1)), cumulative, np.ones((*eta.shape, 1))], axis=-1), axis=-1))
    return np.stack(out, axis=2)


@pytest.mark.parametrize('backend', BACKENDS)
def test_categorical_marginal_means_are_probabilities_per_cell(backend):
    source = groups().with_columns(f=pl.Series(['u', 'v']*80), ordered=pl.Series(
        np.digitize(pl.read_csv(DATA/'bayes_groups.csv')['y'].to_numpy(), [.3, 1.2])).replace_strict(
        {0: 'low', 1: 'mid', 2: 'high'}).cast(pl.Enum(['low', 'mid', 'high'])))
    data, _ = as_input(source, backend, [])
    fit = bayes_glm(data, 'ordered~x+f', family='ordinal', draws=40, tune=60, chains=2, seed=12)
    probabilities = _ordinal_cells(fit, source['x'].to_numpy())
    table = as_polars(tidy(bayes_emmeans(fit, 'f', level=.9)))
    assert table['category'].cast(pl.String).to_list() == ['low', 'low', 'mid', 'mid', 'high', 'high']
    assert table['f'].to_list() == ['u', 'v']*3
    expected = probabilities.transpose(3, 2, 0, 1).reshape(6, -1)
    np.testing.assert_allclose(table['estimate'], expected.mean(axis=1), rtol=1e-12)
    np.testing.assert_allclose(table['conf_high'], np.quantile(expected, .95, axis=1), rtol=1e-12)
    np.testing.assert_allclose(table.group_by('f').agg(pl.col('estimate').sum())['estimate'], 1, rtol=1e-12)
    contrasts = as_polars(tidy(pairs(bayes_emmeans(fit, 'f'))))
    assert contrasts['contrast'].to_list() == ['u - v']*3
    np.testing.assert_allclose(contrasts['estimate'], (expected[0::2]-expected[1::2]).mean(axis=1), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(contrasts['estimate'].sum(), 0, atol=1e-12)
    latent = as_polars(tidy(bayes_emmeans(fit, 'f', type='link')))
    eta = np.stack([fit.beta_draws[..., 0]*source['x'].mean() + fit.beta_draws[..., 1]*v for v in (0, 1)], axis=-1)
    np.testing.assert_allclose(latent['estimate'], (eta - fit.extra_draws['cutpoints'].mean(axis=-1)[..., None]).mean(axis=(0, 1)))
    assert 'category' not in latent.columns

    multinomial = bayes_glm(data, 'choice~f+(1|g)', family='multinomial', draws=40, tune=60, chains=2, seed=12)
    beta = multinomial.beta_draws.reshape(2, 40, 2, 2)
    logits = np.stack([np.concatenate([np.zeros((2, 40, 1)), beta[..., 0] + beta[..., 1]*v], axis=-1) for v in (0, 1)], axis=2)
    table = as_polars(tidy(bayes_emmeans(multinomial, 'f')))
    np.testing.assert_allclose(table['estimate'], softmax(logits, axis=-1).mean(axis=(0, 1)).T.reshape(-1), rtol=1e-12)
    latent = as_polars(tidy(bayes_emmeans(multinomial, 'f', type='link')))
    centred = logits - logits.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(latent['estimate'], centred.mean(axis=(0, 1)).T.reshape(-1), rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match='nonempty/distinct'):
        bayes_emmeans(multinomial, 'category')


def _brms():
    reference = json.loads((DATA/'brms_groups.json').read_text())
    return {name: {t: (m, s) for t, m, s in zip(part['term'], part['mean'], part['sd'])} for name, part in reference.items()}


def _close(ours, theirs, label):
    # Same priors and likelihood, different samplers: posterior means agree
    # within a quarter of a posterior SD (Monte Carlo error with 2000 draws
    # is about 0.03 SD per side), and posterior SDs within 20%; a
    # distributional comparison, not an exact one.
    (mean, sd), (reference_mean, reference_sd) = ours, theirs
    assert abs(mean-reference_mean) < .25*reference_sd, label
    assert sd == pytest.approx(reference_sd, rel=.2), label


def test_gaussian_correlated_posterior_agrees_with_brms():
    reference = _brms()['gaussian']
    fit = bayes_glm(groups(), 'y~x+(1+x|g)', priors=PRIORS, draws=1000, tune=1000, chains=2, seed=13)
    table = tidy(fit)
    names = {'(Intercept)': 'b_Intercept', 'x': 'b_x', 'sd_g__Intercept': 'sd_g__Intercept',
             'sd_g__x': 'sd_g__x', 'cor_g__Intercept__x': 'cor_g__Intercept__x'}
    for row in table.iter_rows(named=True):
        _close((row['estimate'], row['std_error']), reference[names[row['term']]], row['term'])
    sigma = fit.extra_draws['sigma']
    _close((sigma.mean(), sigma.std(ddof=1)), reference['sigma'], 'sigma')
    for row in bayes_ranef(fit).iter_rows(named=True):
        term = 'Intercept' if row['term'] == '(Intercept)' else row['term']
        _close((row['estimate'], row['std_error']), reference[f"r_g[{row['level']},{term}]"], row['level'])


def test_multinomial_varying_intercepts_agree_with_brms():
    reference = _brms()['categorical']
    priors = MULTINOMIAL_PRIORS
    fit = bayes_glm(groups(), 'choice~x+(1|g)', family='multinomial', priors=priors, draws=1000, tune=1000, chains=2, seed=14)
    names = {'b:(Intercept)': 'b_mub_Intercept', 'c:(Intercept)': 'b_muc_Intercept', 'b:x': 'b_mub_x', 'c:x': 'b_muc_x',
             'sd_g__mub_Intercept': 'sd_g__mub_Intercept', 'sd_g__muc_Intercept': 'sd_g__muc_Intercept'}
    for row in tidy(fit).iter_rows(named=True):
        _close((row['estimate'], row['std_error']), reference[names[row['term']]], row['term'])
    for row in bayes_ranef(fit).iter_rows(named=True):
        category, _ = row['term'].split(':')
        _close((row['estimate'], row['std_error']), reference[f"r_g__mu{category}[{row['level']},Intercept]"], row['term'])
