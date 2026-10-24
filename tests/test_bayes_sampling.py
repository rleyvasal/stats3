"""Real NUTS fits and independent posterior checks for the optional extra."""

import importlib.util

import numpy as np
import polars as pl
import pytest
from scipy.integrate import quad
from scipy.special import expit, log_expit

from stats3 import (augment, bayes_compare, bayes_diagnostics, bayes_emmeans, bayes_glm,
    binomial, negative_binomial,
    posterior_draws, posterior_epred, posterior_predict, posterior_summary, pp_check,
    predict, prior, prior_predict, tidy, vcov)
from oracle import BACKENDS, as_input, as_polars

pytestmark = pytest.mark.skipif(importlib.util.find_spec('pymc') is None, reason='requires stats3[bayes]')


def synthetic():
    rng=np.random.default_rng(19)
    x=np.linspace(-1.5,1.5,30)
    eta=.2+.5*x
    ordinal=np.digitize(eta+rng.logistic(size=30),[-.5,.7])
    return pl.DataFrame({'x':x,'g':['a','b','c']*10,'off':np.full(30,.1),
        'gaussian':eta+rng.normal(size=30),'student_t':eta+rng.standard_t(5,size=30),
        'binomial':rng.binomial(1,expit(eta)),'poisson':rng.poisson(np.exp(eta)),
        'negative_binomial':rng.negative_binomial(3,3/(3+np.exp(eta))),
        'gamma':rng.gamma(3,np.exp(eta)/3),'lognormal':rng.lognormal(eta,.4),
        'exponential':rng.exponential(np.exp(eta)),
        'ordinal':pl.Series([['low','middle','high'][i] for i in ordinal],dtype=pl.Enum(['low','middle','high'])),
        'multinomial':['red','blue','green']*10})


@pytest.fixture(scope='module',params=['gaussian','student_t','binomial','poisson','negative_binomial',
                                     'gamma','lognormal','exponential','ordinal','multinomial'])
def sampled(request):
    family=request.param
    fit=bayes_glm(synthetic(),f'{family}~x+offset(off)',family=family,
        draws=60,tune=100,chains=2,seed=14)
    return family,fit


def test_all_likelihoods_have_usable_posterior_workflows(sampled):
    family,fit=sampled
    assert tidy(fit)['estimate'].is_finite().all()
    assert np.isfinite(vcov(fit).drop('term').to_numpy()).all()
    assert posterior_draws(fit).height == 120
    assert posterior_summary(fit).height >= len(fit.parameter_names)
    diagnostics=bayes_diagnostics(fit)
    assert 'r_hat' in diagnostics.columns
    prediction=predict(fit,synthetic().head(3))
    draws=posterior_predict(fit,synthetic().head(3),seed=4)
    assert draws.height == 360
    assert draws.equals(posterior_predict(fit,synthetic().head(3),seed=4))
    prior_draws=prior_predict(fit,synthetic().head(3),draws=12,seed=4)
    assert prior_draws.height == 36
    assert prior_draws.equals(prior_predict(fit,synthetic().head(3),draws=12,seed=4))
    if family in ('ordinal','multinomial'):
        probs=posterior_epred(fit,synthetic().head(3))
        sums=probs.group_by('.chain','.draw','.row').agg(pl.col('.prediction').sum())
        np.testing.assert_allclose(sums['.prediction'],1,atol=1e-14)
        assert probs['.prediction'].min() >= 0
        assert draws['.prediction'].min() >= 0 and draws['.prediction'].max() <= 2
        assert prediction.height == 9
    else:
        assert prediction.height == 3
        assert augment(fit).height == 30
        assert pp_check(fit,seed=1).height == 1
        noisy=predict(fit,synthetic().head(3),interval='prediction',seed=2)
        np.testing.assert_allclose(noisy['fit'].to_numpy(),prediction['fit'].to_numpy())
    if family=='multinomial':
        link=predict(fit,synthetic().head(3),type='link')
        assert link.height==9
        assert link.filter(pl.col('category')==fit.response_levels[0])['fit'].to_list()==[0.]*3
    if family in ('ordinal','multinomial'):
        with pytest.raises(ValueError,match='posterior_predict'):
            predict(fit,interval='prediction')


@pytest.mark.parametrize('family',['binomial','poisson'])
def test_sampled_posterior_matches_independent_quadrature(family):
    if family=='binomial':
        data=pl.DataFrame({'y':[.7], 'trials':[20]})
        kwargs={'weights':'trials'}
        logp=lambda b: 14*log_expit(b)+6*log_expit(-b)-b*b/2
    else:
        data=pl.DataFrame({'y':[2,3,1]})
        kwargs={}
        logp=lambda b: 6*b-3*np.exp(b)-b*b/2
    z=quad(lambda b: np.exp(logp(b)), -20,10,epsabs=1e-13)[0]
    mean=quad(lambda b: b*np.exp(logp(b)),-20,10,epsabs=1e-13)[0]/z
    variance=quad(lambda b: (b-mean)**2*np.exp(logp(b)),-20,10,epsabs=1e-13)[0]/z
    fit=bayes_glm(data,'y~1',family=family,priors={'(Intercept)':prior('normal',mu=0,sigma=1)},
                  draws=1000,tune=500,chains=2,seed=71,**kwargs)
    diag=bayes_diagnostics(fit)
    assert abs(fit.coef[0]-mean) < max(6*diag['mcse_mean'][0],.01)
    assert np.var(fit.beta_draws) == pytest.approx(variance,rel=.15)
    assert diag['r_hat'][0] < 1.05
    assert diag['divergences'][0] == 0


@pytest.mark.parametrize('backend',BACKENDS)
def test_sampled_backend_grouping_and_column_metadata(backend):
    source=synthetic().with_columns(part=pl.when(pl.col('x')<0).then(pl.lit('left')).otherwise(pl.lit('right')))
    data,by=as_input(source,backend,['part'])
    fits=bayes_glm(data,'binomial~x',family='binomial',by=by,draws=15,tune=20,chains=1,seed=5)
    assert len(fits)==2
    assert as_polars(tidy(fits))['part'].unique().sort().to_list() == ['left','right']
    assert as_polars(posterior_draws(fits)).height==30
    new,_=as_input(source.head(2),backend,[])
    assert as_polars(predict(fits[0],new)).height==2


def test_noncentered_random_intercepts_slopes_and_new_groups():
    source=synthetic()
    fit=bayes_glm(source,'gaussian~x+(1+x|g)',draws=100,tune=150,chains=2,seed=31)
    assert fit.random[0]['slopes']==['x']
    assert posterior_summary(fit).filter(pl.col('term').str.starts_with('group_0_sd')).height==2
    known=predict(fit,source.head(3))
    population=predict(fit,source.head(3),re_formula='population')
    assert not np.allclose(known['fit'],population['fit'])
    new=source.head(3).with_columns(pl.lit('new').alias('g'))
    with pytest.raises(ValueError,match='new level'):
        predict(fit,new)
    a=posterior_predict(fit,new,allow_new_levels=True,seed=10)
    assert a.equals(posterior_predict(fit,new,allow_new_levels=True,seed=10))
    assert a['.prediction'].is_finite().all()
    assert predict(fit,new.drop('g'),re_formula='population').height==3
    augmented=augment(fit,re_formula='population')
    np.testing.assert_allclose(augmented['.resid'].to_numpy(),source['gaussian'].to_numpy()-augmented['.fitted'].to_numpy())
    # Random effects are deliberately zero for marginal means.
    assert tidy(bayes_emmeans(fit,'x',at={'x':[-1,1]})).height==2


@pytest.mark.parametrize('link',['probit','cloglog'])
def test_binomial_links(link):
    fit=bayes_glm(synthetic(),'binomial~x',family='binomial',link=link,draws=20,tune=30,chains=1,seed=14)
    assert predict(fit)['fit'].is_between(0,1).all()


def test_ordinal_intercept_only_and_missing_row_alignment():
    source=synthetic().with_columns(pl.when(pl.col('x')>1.4).then(None).otherwise(pl.col('ordinal')).alias('ordinal'))
    fit=bayes_glm(source,'ordinal~1',family='ordinal',link='probit',draws=25,tune=40,chains=1,seed=3,na_action='exclude')
    assert fit.parameter_names==[]
    assert posterior_summary(fit).height==2
    assert predict(fit).height==90
    assert predict(fit)['fit'].null_count()==3
    assert bayes_diagnostics(fit).height>=2


def test_binomial_counts_predictions_and_validation():
    source=pl.DataFrame({'yes':[1,4,7], 'no':[9,6,3], 'x':[-1.,0.,1.]})
    fit=bayes_glm(source,'cbind(yes,no)~x',family='binomial',draws=40,tune=60,chains=1,seed=12)
    prediction=posterior_predict(fit,seed=1)
    assert prediction['.prediction'].is_between(0,10).all()
    assert posterior_predict(fit,source.head(1),trials=5,seed=1)['.prediction'].max()<=5
    with pytest.raises(ValueError,match='integers'):
        bayes_glm({'y':[1.5]},'y~1',family='poisson',draws=10)
    with pytest.raises(ValueError,match='unused prior'):
        bayes_glm(source,'yes~x',family='poisson',priors={'typo':prior()},draws=10)
    with pytest.raises(ValueError,match='positive'):
        bayes_glm(source,'yes~x',priors={'sigma':prior('normal',mu=1,sigma=1)},draws=10)
    with pytest.raises(ValueError,match='dispersion'):
        bayes_glm(source,'yes~x',family=negative_binomial(2),draws=10)


def test_boolean_all_successes_keep_the_success_category():
    fit=bayes_glm({'y':[True]*10},'y~1',family=binomial,draws=100,tune=100,chains=2,seed=2)
    assert fit.y.tolist()==[1]*10
    assert predict(fit)['fit'].mean()>.8


def test_loo_waic_and_evidence_guard():
    source=synthetic()
    a=bayes_glm(source,'binomial~1',family='binomial',draws=100,tune=100,chains=2,seed=42)
    b=bayes_glm(source,'binomial~x',family='binomial',draws=100,tune=100,chains=2,seed=43)
    for method in ('loo','waic'):
        out=bayes_compare({'intercept':a,'slope':b},method=method)
        assert out.height==2
        assert out['weight'].sum()==pytest.approx(1)
    with pytest.raises(ValueError,match='only for exact'):
        bayes_compare({'a':a,'b':b},method='evidence')
