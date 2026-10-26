"""Multinomial logit against nnet::multinom and broom, across the four data backends."""

import warnings

import numpy as np
import polars as pl
import pytest
from scipy.optimize import approx_fprime

from stats3 import anova, augment, emmeans, glance, lm, modelsummary, multinom, predict, tidy, vcov
from stats3.multinomial import _deviance, _information, _log_probabilities, _predictor, _score
from oracle import BACKENDS, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r, penguins


SPECIES = ['Adelie', 'Chinstrap', 'Gentoo']


def data():
    return penguins().with_row_index('i', offset=1).with_columns(
        pl.col('species').cast(pl.Enum(SPECIES)), pl.col('sex').cast(pl.Enum(['female', 'male'])),
        mass=pl.col('body_mass_g')/1000, flipper=pl.col('flipper_length_mm')/10,
        mass2=pl.col('body_mass_g')/1000,
        w=pl.when(pl.col('i').is_in([2, 30, 160, 250])).then(0).otherwise(1 + pl.col('i') % 3),
        off=(pl.col('i') % 5)/10, oa=pl.lit(0.0), ob=(pl.col('i') % 5)/10, oc=(pl.col('i') % 5)/-20)


CASES = dict(main=('species ~ mass + sex', {}), flipper=('species ~ flipper', {}),
    intercept=('species ~ 1', {}), no_intercept=('species ~ 0 + mass + sex', {}),
    weighted=('species ~ mass + sex', {'weights': 'w'}),
    sum_contrasts=('species ~ mass + sex + factor(year)', {'contrasts': 'sum'}),
    aliased=('species ~ mass + mass2 + sex', {}), reference=('species ~ mass + sex', {}),
    offset_matrix=('species ~ mass', {'offset': ['oa', 'ob', 'oc']}),
    binary=('sex ~ mass + species', {}), binary_offset=('sex ~ mass + species + offset(off)', {}),
    decay=('species ~ mass + sex', {'decay': 0.5}), decay_aliased=('species ~ mass + mass2 + sex', {'decay': 0.5}),
    counts=('cbind(Adelie, Chinstrap, Gentoo) ~ factor(year) + sex', {}),
    counts_weighted=('cbind(Adelie, Chinstrap, Gentoo) ~ factor(year) + sex', {'weights': 'cw'}))


def counts():
    """Counts of each species by year and sex, in R's aggregate() order (sex varying slowest)."""
    return (data().drop_nulls('sex')
            .group_by('year', 'sex').agg(**{s: (pl.col('species') == s).sum() for s in SPECIES})
            .sort('sex', 'year')
            .with_row_index('j', offset=1).with_columns(cw=1 + pl.col('j') % 2))


@pytest.fixture(scope='module', params=BACKENDS)
def fits(request):
    backend = request.param
    source, _ = as_input(data(), backend, [])
    gentoo, _ = as_input(data().with_columns(pl.col('species').cast(pl.String).cast(
        pl.Enum(['Gentoo', 'Adelie', 'Chinstrap']))), backend, [])
    tallies, _ = as_input(counts(), backend, [])
    sources = {'reference': gentoo, 'counts': tallies, 'counts_weighted': tallies}
    result = {name: multinom(sources.get(name, source), formula, **kwargs)
              for name, (formula, kwargs) in CASES.items()}
    return backend, result


# nnet's BFGS (vmmin) stops on a relative change in the objective; even with
# the oracle's maxit=10000 and reltol=1e-15 it reaches the MLE only to about
# 5e-6 relative in the coefficients (the weighted and Gentoo-reference fits are
# worst; stats3's deviance is lower than nnet's in every case). stats3 runs
# Newton to full precision, so the coefficients, their Wald tables and the
# fitted probabilities are compared at nnet's accuracy: 1e-5 relative (2e-5 for
# covariances, 1e-4 for p-values, whose relative error grows with |z|; 1e-6
# absolute for probabilities). Deviance and AIC are flat at the optimum and match
# at the default 1e-9. Standard errors come from the exact Hessian
# (multinomHess) at each optimum, so they share the coefficients' accuracy.
LOOSE = {c: (1e-5, 1e-8) for c in ('estimate', 'std_error', 'statistic', 'conf_low', 'conf_high')}
LOOSE.update(p_value=(1e-4, 1e-15), **{c: (1e-5, 1e-6) for c in (*SPECIES, 'female', 'male')})

# Where nnet stops depends on the platform's rounding: on x86-64 it stopped
# 4e-8 above the minimum deviance for 'species ~ flipper' (1.3e-10 relative),
# which moves the coefficients by up to 7e-4 relative, and restarting it from
# its own weights stops at once. stats3's deviance must never be above
# nnet's. Where nnet's is measurably higher (relative gap over STOPPED_SHORT),
# its tables are compared at the accuracy that gap allows.
STOPPED_SHORT = 1e-11
SHORT_LOOSE = (1e-3, 1e-6)


def _r_deviance(expected):
    glance_table = expected['glance']
    return glance_table['data'][glance_table['columns'].index('deviance')][0]


@pytest.mark.parametrize('table', ['tidy', 'vcov', 'probs', 'class', 'new_probs', 'new_class', 'glance'])
def test_tables_match_r(fits, table):
    backend, models = fits
    new, _ = as_input(data()[[0, 1, 2, 4, 5, 199, 299]], backend, [])
    for name, fit in models.items():
        actual = {'tidy': lambda: tidy(fit, conf_int=True), 'vcov': lambda: vcov(fit),
                  'probs': lambda: predict(fit, type='probs'), 'class': lambda: predict(fit),
                  'new_probs': lambda: predict(fit, new, type='probs'), 'new_class': lambda: predict(fit, new),
                  'glance': lambda: glance(fit)}[table]()
        expected = _oracle('multinom')[name]
        assert expected['converged']
        r_deviance = _r_deviance(expected)
        gap = (r_deviance - fit.deviance) / fit.deviance
        assert gap > -1e-12, f"{name}: stats3's deviance {fit.deviance} is above nnet's {r_deviance}"
        loose = dict(LOOSE)
        if table == 'vcov':
            loose.update({n: (2e-5, 1e-9) for n in fit.parameter_names})
        if table == 'tidy':
            # nnet's accuracy is relative to the coefficient vector's size, not
            # each element's: a coefficient near zero is only that close.
            columns = dict(zip(expected[table]['columns'], expected[table]['data']))
            for c in ('estimate', 'statistic', 'conf_low', 'conf_high'):
                largest = max(abs(v) for v in columns[c] if v is not None)
                loose[c] = (LOOSE[c][0], max(LOOSE[c][1], LOOSE[c][0] * largest))
        if gap > STOPPED_SHORT and table not in ('glance', 'class', 'new_class'):
            loose = {c: SHORT_LOOSE for c in expected[table]['columns']}
        assert_matches_r(actual, expected[table], loose=loose)


@pytest.mark.parametrize('backend', BACKENDS)
def test_groups_comparison_and_output_types(backend):
    source, by = as_input(data(), backend, ['year'])
    grouped = multinom(source, 'species ~ mass', by=by)
    assert len(grouped) == 3
    # On the smaller per-year fits nnet stops further from the MLE: 6e-5 relative
    # in the 2009 coefficients, so they compare at 1e-4.
    grouped_loose = {**LOOSE, **{c: (1e-4, 1e-8) for c in ('estimate', 'statistic', 'conf_low', 'conf_high')}}
    for table, fn in [('tidy', lambda f: tidy(f, conf_int=True)), ('glance', glance)]:
        assert_matches_r(fn(grouped), _oracle('multinom')['grouped'][table], loose=grouped_loose)
    complete, _ = as_input(data().drop_nulls(['mass', 'sex']), backend, [])
    small = multinom(complete, 'species ~ 1')
    middle = multinom(complete, 'species ~ mass')
    big = complete >> multinom('species ~ mass + sex')
    assert_matches_r(anova(big, small, middle), _oracle('multinom')['compare'])
    assert 'p_value' not in as_polars(anova(small, big, test='none')).columns
    expected = {'polars': ('polars', 'DataFrame'), 'lazy': ('polars', 'LazyFrame'),
                'pandas': ('pandas', 'DataFrame'), 'tidy3': ('tidy3', 'TidyFrame')}[backend]
    for output in [tidy(big), glance(big), predict(big), predict(big, type='probs'), vcov(big), augment(big)]:
        assert (type(output).__module__.split('.')[0], type(output).__name__) == expected
    assert as_polars(big >> tidy()).height == 6
    summary = as_polars(modelsummary(big, small, statistic='conf_int'))
    assert 'Gentoo:sexmale' in summary['term'].to_list()


def test_score_and_information_match_numerical_derivatives():
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(6), rng.normal(size=6)])
    y = np.array([0, 1, 2, 1, 0, 2]); weights = np.array([1., 2., 0., 3., .5, 1.])
    offset = rng.normal(size=(6, 3))
    coef = np.array([[.3, -.2], [-.5, .7]])
    loglik = lambda b: -_deviance(x, b.reshape(2, 2), y, weights, offset) / 2
    score = lambda b: _score(x, b.reshape(2, 2), y, weights, offset)
    np.testing.assert_allclose(score(coef.ravel()), approx_fprime(coef.ravel(), loglik, 1e-7), atol=1e-5)
    probabilities = np.exp(_log_probabilities(_predictor(x, coef, offset)))
    numerical = np.column_stack([approx_fprime(coef.ravel(), lambda b: score(b)[j], 1e-7) for j in range(4)])
    np.testing.assert_allclose(_information(x, probabilities, weights), -numerical, atol=1e-5)


def test_missing_rows_exclude_and_offsets():
    source = data()
    assert source.select(pl.any_horizontal(pl.col('mass', 'sex').is_null()).sum()).item() == 11
    fit = multinom(source, 'species ~ mass + sex', na_action='exclude')
    assert fit.n == 333
    assert predict(fit).height == 344 and predict(fit)['fit'].null_count() == 11
    assert predict(fit, type='probs')['Gentoo'].null_count() == 11
    out = augment(fit)
    assert out['.fitted'].null_count() == 11 and '.prob_Adelie' in out.columns
    np.testing.assert_allclose(out.drop_nulls('.fitted').select(pl.col('^\\.prob_.*$')).sum_horizontal(), 1)
    omitted = multinom(source, 'species ~ mass + sex')
    assert augment(omitted).height == 333
    np.testing.assert_allclose(fit.coef, omitted.coef)
    # the binary offset as an argument equals the offset term
    term = multinom(source, 'sex ~ mass + species + offset(off)')
    argument = multinom(source, 'sex ~ mass + species', offset='off')
    np.testing.assert_allclose(term.coef, argument.coef)
    # a missing offset value drops its row, as model.frame does
    gap = source.with_columns(ob=pl.when(pl.col('i') == 1).then(None).otherwise(pl.col('ob')))
    assert multinom(gap, 'species ~ mass', offset=['oa', 'ob', 'oc']).n == 341
    with pytest.raises(ValueError, match='one column per'):
        multinom(source, 'species ~ mass + offset(off)')
    with pytest.raises(ValueError, match='one column per response level'):
        multinom(source, 'species ~ mass', offset=['oa', 'ob'])
    with pytest.raises(ValueError, match='missing'):
        multinom(source, 'species ~ mass', na_action='fail')


def test_response_levels_and_empty_groups():
    numeric = data().with_columns(code=pl.col('species').cast(pl.String).replace_strict(
        {'Adelie': 3, 'Chinstrap': 1, 'Gentoo': 2}))
    fit = multinom(numeric, 'code ~ mass')
    assert fit.response_levels == ['1', '2', '3']
    assert tidy(fit)['y_level'].unique(maintain_order=True).to_list() == ['2', '3']
    factor = multinom(numeric, 'factor(code) ~ mass')
    np.testing.assert_allclose(factor.coef, fit.coef)
    strings = multinom(data().with_columns(pl.col('species').cast(pl.String)), 'species ~ mass')
    assert strings.response_levels == SPECIES
    dream = data().filter(pl.col('island') == 'Dream')
    with pytest.warns(UserWarning, match="group 'Gentoo' is empty"):
        two = multinom(dream, 'species ~ mass')
    assert two.response_levels == ['Adelie', 'Chinstrap'] and two.edf == 2
    with pytest.raises(ValueError, match='two or more classes'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            multinom(data().filter(pl.col('island') == 'Torgersen'), 'species ~ mass')


def test_exponentiate_aliases_and_convergence():
    fit = multinom(data(), 'species ~ mass + sex')
    plain, exp = tidy(fit, conf_int=True), tidy(fit, conf_int=True, exponentiate=True)
    for column in ('estimate', 'conf_low', 'conf_high'):
        np.testing.assert_allclose(exp[column], np.exp(plain[column]))
    np.testing.assert_allclose(exp['std_error'], plain['std_error'])
    aliased = multinom(data(), 'species ~ mass + mass2 + sex')
    assert aliased.edf == fit.edf == 6 and aliased.deviance == pytest.approx(fit.deviance, rel=1e-12)
    np.testing.assert_allclose(predict(aliased, type='probs'), predict(fit, type='probs'), rtol=1e-10)
    with pytest.warns(UserWarning, match='did not converge'):
        separated = multinom(data(), 'species ~ mass + flipper + sex', maxit=5)
    assert not separated.converged


@pytest.mark.parametrize('kwargs', [{'maxit': 0}, {'maxit': True}, {'na_action': 'drop'},
                                    {'contrasts': 'helmert'}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        multinom(data(), 'species ~ mass', **kwargs)


def test_unsupported_inference():
    fit = multinom(data(), 'species ~ mass')
    with pytest.raises(ValueError, match='vcov'):
        tidy(fit, vcov='HC3')
    with pytest.raises(TypeError, match='single multinom'):
        anova(fit)
    with pytest.raises(ValueError, match="'class' or 'probs'"):
        predict(fit, type='link')
    with pytest.raises(TypeError, match='multinom fits'):
        anova(fit, lm(data(), 'mass ~ flipper'))
    with pytest.raises(ValueError, match='same size'):
        anova(fit, multinom(data(), 'species ~ mass + sex'))


def test_counts_match_the_same_fit_on_rows():
    """A cbind() count response is the row-level fit with each row counted:
    the same coefficients, deviance and standard errors."""
    tallies = counts()
    rows = data().drop_nulls('sex')
    by_counts = multinom(tallies, 'cbind(Adelie, Chinstrap, Gentoo) ~ factor(year) + sex')
    by_rows = multinom(rows, 'species ~ factor(year) + sex')
    assert by_counts.response_levels == SPECIES
    assert by_counts.coef == pytest.approx(by_rows.coef, rel=1e-9, abs=1e-12)
    assert by_counts.deviance == pytest.approx(by_rows.deviance, rel=1e-12)
    assert tidy(by_counts)['std_error'].to_list() == pytest.approx(tidy(by_rows)['std_error'].to_list(), rel=1e-9)
    assert glance(by_counts)['nobs'][0] == tallies.height


def test_decay_penalizes_every_coefficient():
    fit = multinom(data(), 'species ~ mass + sex', decay=0.5)
    plain = multinom(data(), 'species ~ mass + sex')
    used = data().drop_nulls(['mass', 'sex'])
    x = np.column_stack([np.ones(used.height), used['mass'].to_numpy(), (used['sex'] == 'male').to_numpy()])
    y = used['species'].to_physical().to_numpy()
    unpenalized = _deviance(x, fit.coef, y, np.ones(used.height), np.zeros((used.height, 3)))
    # nnet reports twice the penalized criterion, intercepts included in the penalty.
    assert fit.deviance == pytest.approx(unpenalized + 2 * 0.5 * float(np.sum(fit.coef**2)), rel=1e-12)
    assert np.sum(fit.coef**2) < np.sum(plain.coef**2)
    for bad in (-1, True, 'a'):
        with pytest.raises(ValueError, match='decay must be a nonnegative number'):
            multinom(data(), 'species ~ mass', decay=bad)
    with pytest.raises(ValueError, match='nonnegative'):
        multinom(counts().with_columns(Adelie=-pl.col('Adelie').cast(pl.Int64)), 'cbind(Adelie, Chinstrap, Gentoo) ~ sex')
