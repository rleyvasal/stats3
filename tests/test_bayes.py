"""Exact Bayesian inference, posterior workflows and backend integration."""

import subprocess
import sys

import numpy as np
import polars as pl
import pytest
from scipy.integrate import quad
from scipy.stats import beta, gamma, t

from stats3 import (anova, augment, bayes_binom_test, bayes_compare, bayes_diagnostics,
    bayes_emmeans, bayes_glm, bayes_lm, bayes_poisson_test, contrast, emmeans, glance,
    hypothesis, modelsummary, pairs, posterior_draws, posterior_epred, posterior_predict,
    posterior_summary, pp_check, predict, prior, prior_predict, tidy, vcov)
from oracle import BACKENDS, DATA, ROOT, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r


def cars():
    return pl.read_csv(DATA / "mtcars.csv").with_row_index("i", offset=1).with_columns(w=1+pl.col("i")%3)


CASES = {"basic": ("mpg~wt+hp", {}), "factors": ("mpg~wt+factor(cyl)", {}),
    "weighted": ("mpg~wt+hp", {"weights": "w"}), "offset": ("mpg~wt+hp+offset(hp)", {}),
    "no_intercept": ("mpg~0+wt+hp", {}), "intercept": ("mpg~1", {})}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES)
def test_exact_matches_r(backend, case):
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    data, _ = as_input(cars(), backend, [])
    formula, kwargs = CASES[case]
    fit = bayes_lm(data, formula, prior_mean=.3, prior_scale=3, prior_shape=2.5,
                   prior_rate=1.7, draws=20, chains=2, seed=1, **kwargs)
    reference = _oracle("bayes")[case]
    for table, actual in (("tidy", tidy(fit, conf_int=True)), ("vcov", vcov(fit)),
                          ("predict", predict(fit)), ("predictive", predict(fit, interval="prediction"))):
        assert_matches_r(actual, reference[table])
    assert_matches_r(as_polars(glance(fit)).select("log_marginal_likelihood"), reference["evidence"])


def data():
    return pl.DataFrame({"y": [1., 2., 2., 4., 4., 6., 5., 7.], "x": [-2., -1., 0., 1., 2., 3., 4., 5.],
                         "g": ["a", "b"]*4, "h": ["u"]*4+["v"]*4, "off": [.25]*8, "w": [2.]*8})


def test_hand_computed_posterior_and_predictive():
    f = bayes_lm({"y": [1., 2., 3.]}, "y~1", prior_mean=0, prior_scale=1,
                 prior_shape=2, prior_rate=1, draws=3000, chains=2, seed=71)
    assert f.coef[0] == 1.5
    assert f.posterior_shape == 3.5
    assert f.posterior_rate == 3.5
    assert f.posterior_scale[0, 0] == .25
    assert f.posterior_covariance[0, 0] == .35
    row = tidy(f, conf_int=True).row(0, named=True)
    np.testing.assert_allclose([row['conf_low'], row['conf_high']], t.ppf([.025, .975], 7, loc=1.5, scale=.5))
    np.testing.assert_allclose(f.beta_draws.mean(), 1.5, atol=.04)
    np.testing.assert_allclose(np.var(f.beta_draws), .35, rtol=.1)
    predicted = predict(f, {"unused": [0.]}, interval="prediction").row(0, named=True)
    assert predicted["upr"] == pytest.approx(1.5+t.ppf(.975,7)*np.sqrt(1.25))
    # Direct integration of the Student-t marginal likelihood for n=1.
    one = bayes_lm({"y": [1.2]}, "y~1", prior_scale=1, prior_shape=2, prior_rate=1, draws=5, seed=1)
    assert one.log_evidence == pytest.approx(t.logpdf(1.2, df=4, scale=1), abs=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_grouping_pipes_and_model_tables(backend):
    frame, by = as_input(data(), backend, ["h"])
    fits = bayes_lm(frame, "y~x+g", by=by, seed=5, draws=30, chains=2)
    assert len(fits) == 2
    for fn in (tidy, glance, vcov, posterior_summary, posterior_draws, bayes_diagnostics):
        out = as_polars(fn(fits))
        assert out["h"].unique().sort().to_list() == ["u", "v"]
    assert not np.array_equal(fits[0].beta_draws, fits[1].beta_draws)
    frame, _ = as_input(data(), backend, [])
    fit = frame >> bayes_lm("y~x+g", seed=1, draws=20)
    assert as_polars(fit >> tidy()).height == 3
    assert as_polars(augment(fit)).height == 8
    assert as_polars(modelsummary(fit, statistic="conf_int")).height > 0
    assert as_polars(predict(fit, frame)).height == 8


def test_missing_rows_offsets_factor_order_and_new_levels():
    d = data().with_columns(pl.when(pl.col("x") == 0).then(None).otherwise(pl.col("x")).alias("x"),
                            pl.col("g").cast(pl.Enum(["b", "a"])))
    f = bayes_lm(d, "y~x+g", offset="off", seed=1, na_action="exclude", draws=30)
    assert f.parameter_names == ["(Intercept)", "x", "ga"]
    assert predict(f)["fit"].null_count() == 1
    assert augment(f)[".resid"].null_count() == 1
    assert posterior_epred(f)[".row"].unique().sort().to_list() == [1,2,4,5,6,7,8]
    new = data().head(2)
    x, off, _, _ = f._new_design(new)
    np.testing.assert_allclose(predict(f, new)["fit"], x@f.coef+off)
    with pytest.raises(ValueError, match="new levels"):
        predict(f, new.with_columns(pl.lit("new").alias("g")))
    with pytest.raises(ValueError, match="missing"):
        bayes_lm(d, "y~x", na_action="fail")


def test_rank_deficient_likelihood_is_identified_by_prior():
    d = data().with_columns(z=pl.col("x"))
    f = bayes_lm(d, "y~x+z", draws=10, seed=2)
    assert f.coef[1] == pytest.approx(f.coef[2])
    assert np.linalg.eigvalsh(f.posterior_covariance).min() > 0
    assert predict(f, d)["fit"].is_finite().all()


def test_reproducibility_draws_and_predictive_checks():
    a = bayes_lm(data(), "y~x", seed=12, draws=50, chains=2)
    b = bayes_lm(data(), "y~x", seed=12, draws=50, chains=2)
    assert posterior_draws(a).equals(posterior_draws(b))
    # A seeded step piped twice repeats its draws, as a data-first call does.
    reused = bayes_lm("y~x", seed=12, draws=50, chains=2)
    assert posterior_draws(data() >> reused).equals(posterior_draws(a))
    assert posterior_draws(data() >> reused).equals(posterior_draws(a))
    assert posterior_predict(a, seed=4).equals(posterior_predict(a, seed=4))
    assert prior_predict(a, draws=30, seed=4).equals(prior_predict(a, draws=30, seed=4))
    epred = posterior_epred(a)
    assert epred.height == 2*50*8
    np.testing.assert_allclose(epred[".prediction"].to_numpy().reshape(2, 50, 8),
                               np.einsum("cdp,np->cdn", a.beta_draws, a.x))
    check = pp_check(a, seed=4)
    assert check["observed"][0] == data()["y"].mean()
    assert 0 <= check["prob_greater"][0] <= 1
    diagnostic = bayes_diagnostics(a)
    assert diagnostic["r_hat"].null_count() == 2
    assert diagnostic["ess_bulk"].to_list() == [100., 100.]
    assert tidy(a)["p_value"].null_count() == 2


def test_weighted_predictive_draws_observation_precision():
    f = bayes_lm(data(), "y~x", weights="w", draws=2000, chains=2, seed=1)
    predictive = posterior_predict(f, seed=6)[".prediction"].to_numpy().reshape(2,2000,8)
    eta = np.einsum("cdp,np->cdn", f.beta_draws, f.x)
    standardized = (predictive-eta)/f.extra_draws["sigma"][...,None]*np.sqrt(2)
    assert standardized.std() == pytest.approx(1, rel=.02)


def test_posterior_contrasts_marginal_means_and_intervals():
    f = bayes_lm(data(), "y~x+g+h", seed=1, draws=100, chains=2)
    h = hypothesis(f, {"gb": 1, "x": -1}, value=.5, rope=(-.1,.1))
    values = f.beta_draws[:,:,2]-f.beta_draws[:,:,1]-.5
    assert h["estimate"][0] == pytest.approx(values.mean())
    assert h["rope_probability"][0] == np.mean(np.abs(values) <= .1)
    e = bayes_emmeans(f, "g", by="h", at={"x": [1,2]})
    means = tidy(e)
    assert means.height == 4
    differences = tidy(pairs(e))
    assert differences.height == 2
    np.testing.assert_allclose(differences["estimate"], -f.beta_draws[:,:,2].mean())
    custom = tidy(contrast(e, {"b-a": [-1,1]}))
    np.testing.assert_allclose(custom["estimate"], -differences["estimate"].to_numpy())
    with pytest.raises(ValueError, match="adjust"):
        pairs(e, adjust="tukey")
    with pytest.raises(TypeError, match="bayes_emmeans"):
        emmeans(f, "g")
    with pytest.raises(TypeError, match="bayes_compare"):
        anova(f)


def test_marginal_offsets_use_the_same_cell_weights():
    d = pl.DataFrame({'y':[1.,2.,3.,4.,5.,6.], 'g':['a','a','a','b','b','b'], 'h':['u','u','v','u','v','v'], 'off':[0.,0.,3.,0.,3.,3.]})
    f=bayes_lm(d,'y~g+h+offset(off)',seed=1,draws=50,chains=2)
    e=bayes_emmeans(f,'g',weights='cells',at={'off':[0.,3.]})
    assert tidy(e).height == 2
    # Cell frequencies are (2/3,1/3) for a and (1/3,2/3) for b.
    beta=f.beta_draws
    expected=np.stack([beta[:,:,0]+beta[:,:,2]/3+1,
                       beta[:,:,0]+beta[:,:,1]+2*beta[:,:,2]/3+2],axis=-1)
    for labels, values, _ in e.parts:
        np.testing.assert_allclose(values,expected)
    g=bayes_lm(d,'y~g+h',offset='off',seed=1,draws=50,chains=2)
    means=tidy(bayes_emmeans(g,'g'))
    np.testing.assert_allclose(means['estimate'].to_numpy(),
        [np.mean(g.beta_draws[:,:,0]+g.beta_draws[:,:,2]/2)+1.5,
         np.mean(g.beta_draws[:,:,0]+g.beta_draws[:,:,1]+g.beta_draws[:,:,2]/2)+1.5])


def test_evidence_comparison_and_reject_different_responses():
    a=bayes_lm(data(),'y~1',draws=10,seed=1)
    b=bayes_lm(data(),'y~x',draws=10,seed=1)
    comparison=bayes_compare({'null':a,'slope':b},method='evidence')
    assert comparison['log_bayes_factor'][1] == b.log_evidence-a.log_evidence
    assert comparison['weight'].sum() == pytest.approx(1)
    with pytest.raises(ValueError,match='identical'):
        bayes_compare({'a':a,'b':bayes_lm(data().with_columns(y=pl.col('y')+1),'y~x')},method='evidence')


def test_scalar_conjugate_inference_and_bayes_factors_by_quadrature():
    b=bayes_binom_test(7,10,alpha=2,beta=3,null=.4)
    assert b['estimate'][0] == pytest.approx(9/15)
    np.testing.assert_allclose([b['conf_low'][0],b['conf_high'][0]],beta(9,6).ppf([.025,.975]))
    integrated=quad(lambda p: p**7*(1-p)**3*beta.pdf(p,2,3),0,1)[0]
    assert b['log_bayes_factor'][0] == pytest.approx(np.log(integrated)-7*np.log(.4)-3*np.log(.6))
    p=bayes_poisson_test(5,2,shape=2,rate=3,null=1.2)
    assert p['estimate'][0] == pytest.approx(7/5)
    integrated=quad(lambda rate: rate**5*np.exp(-2*rate)*gamma.pdf(rate,2,scale=1/3),0,np.inf)[0]
    assert p['log_bayes_factor'][0] == pytest.approx(np.log(integrated)-5*np.log(1.2)+2*1.2)
    assert bayes_binom_test(0,0)['log_bayes_factor'][0] == pytest.approx(0)
    assert np.isinf(bayes_binom_test(1,1,null=0)['log_bayes_factor'][0])


@pytest.mark.parametrize('kwargs',[{'prior_scale':0},{'prior_shape':1},{'prior_rate':0},{'draws':0},
    {'chains':0},{'draws':True},{'prior_mean':{'missing':1}},{'prior_mean':np.nan}])
def test_invalid_gaussian_arguments(kwargs):
    with pytest.raises(ValueError):
        bayes_lm(data(),'y~x',**kwargs)


def test_invalid_interfaces():
    f=bayes_lm(data(),'y~x',draws=10)
    with pytest.raises(ValueError):
        tidy(f,vcov='HC3')
    with pytest.raises(ValueError):
        hypothesis(f,{'missing':1})
    with pytest.raises(ValueError):
        hypothesis(f,{'x':1},rope=(1,-1))
    with pytest.raises(ValueError):
        predict(f,interval='confidence')
    with pytest.raises(ValueError,match='finite mean'):
        tidy(f,exponentiate=True)
    with pytest.raises(ValueError):
        prior('unsupported')
    with pytest.raises(ValueError):
        posterior_predict(bayes_lm(data(),'y~x',by='g',draws=10))
    with pytest.raises(ValueError):
        bayes_glm(data(),'y~x',family='quasipoisson')


def test_exact_import_does_not_load_optional_bayesian_dependencies():
    script="import sys; sys.path.insert(0,sys.argv[1]); from stats3 import bayes_lm; bayes_lm({'y':[1.,2.]},'y~1',draws=5); assert 'pymc' not in sys.modules; assert 'arviz' not in sys.modules"
    subprocess.run([sys.executable,'-c',script,str(ROOT/'src')],check=True)


def test_auxiliary_draw_names_do_not_overwrite_coefficients():
    f=bayes_lm(data().rename({'x':'sigma'}),'y~sigma',draws=20,seed=1)
    out=posterior_draws(f)
    assert 'sigma' in out.columns and 'aux:sigma' in out.columns
    np.testing.assert_allclose(out['sigma'].to_numpy(),f.beta_draws[:,:,1].reshape(-1))


def test_response_marginal_means_transform_before_averaging():
    # A fixed joint posterior with a nonlinear response isolates the averaging
    # operation from Monte Carlo error or the inference engine.
    f=bayes_lm(data(),'y~g+h',draws=20,chains=2,seed=1)
    f._mean=lambda samples,eta: np.exp(eta)
    response=tidy(bayes_emmeans(f,'g',type='response'))
    from stats3.emm import _reference_grid
    grid,_=_reference_grid(f,{})
    x,off,rows,_=f._new_design(grid)
    predictions=np.exp(f._eta(f._posterior_samples(),x,off,rows))
    for name,expected in zip(response['g'],response['estimate']):
        mask=grid['g'].to_numpy()==name
        assert expected==pytest.approx(predictions[:,:,mask].mean())


@pytest.mark.parametrize('dependency',['pymc','arviz'])
def test_missing_extra_is_actionable(dependency):
    # A fresh interpreter exercises the real import path whether or not the
    # optional dependency is installed in the test runner's environment.
    script='''
import importlib.abc
import sys
sys.path.insert(0,sys.argv[1])
dependency=sys.argv[2]
class MissingDependency(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.partition('.')[0]==dependency:
            raise ModuleNotFoundError('optional dependency deliberately unavailable',name=dependency)
sys.meta_path.insert(0,MissingDependency())
from stats3 import bayes_compare,bayes_glm,bayes_lm
try:
    if dependency=='pymc':
        bayes_glm({'y':[1.,2.,3.]},'y~1',draws=5,tune=5)
    else:
        a=bayes_lm({'y':[1.,2.,3.]},'y~1',draws=5,seed=1)
        b=bayes_lm({'y':[1.,2.,3.]},'y~1',prior_scale=2,draws=5,seed=1)
        bayes_compare({'a':a,'b':b},method='loo')
except ImportError as error:
    assert 'stats3[bayes]' in str(error)
    assert error.__cause__.name==dependency
else:
    raise AssertionError('missing optional dependency did not raise ImportError')
'''
    subprocess.run([sys.executable,'-c',script,str(ROOT/'src'),dependency],check=True)
