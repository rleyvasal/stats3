"""Real NUTS fits and independent posterior checks for the optional extra."""

import importlib.util
import itertools

import numpy as np
import polars as pl
import pytest
from scipy.integrate import quad
from scipy.special import expit, log_expit, log_softmax, logsumexp
from scipy import stats

from stats3 import (augment, bayes_compare, bayes_diagnostics, bayes_emmeans, bayes_glm,
    bayes_lm, binomial, negative_binomial,
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


FAMILIES=['gaussian','student_t','binomial','poisson','negative_binomial',
          'gamma','lognormal','exponential','ordinal','multinomial']


@pytest.fixture(scope='module',params=list(itertools.product(FAMILIES,BACKENDS)),
                ids=lambda item:'-'.join(item))
def sampled(request):
    family,backend=request.param
    source,_=as_input(synthetic(),backend,[])
    fit=bayes_glm(source,f'{family}~x+offset(off)',family=family,
        draws=60,tune=100,chains=2,seed=14)
    return family,backend,fit


def test_all_likelihoods_have_usable_posterior_workflows(sampled):
    family,backend,fit=sampled
    assert as_polars(tidy(fit))['estimate'].is_finite().all()
    assert np.isfinite(as_polars(vcov(fit)).drop('term').to_numpy()).all()
    assert as_polars(posterior_draws(fit)).height == 120
    assert as_polars(posterior_summary(fit)).height >= len(fit.parameter_names)
    diagnostics=as_polars(bayes_diagnostics(fit))
    assert 'r_hat' in diagnostics.columns
    new,_=as_input(synthetic().head(3),backend,[])
    prediction=as_polars(predict(fit,new))
    draws=as_polars(posterior_predict(fit,new,seed=4))
    assert draws.height == 360
    assert draws.equals(as_polars(posterior_predict(fit,new,seed=4)))
    prior_draws=as_polars(prior_predict(fit,new,draws=12,seed=4))
    assert prior_draws.height == 36
    assert prior_draws.equals(as_polars(prior_predict(fit,new,draws=12,seed=4)))
    expected_type={'polars':('polars','DataFrame'),'lazy':('polars','LazyFrame'),
                   'pandas':('pandas','DataFrame'),'tidy3':('tidy3','TidyFrame')}[backend]
    for output in (tidy(fit),posterior_draws(fit),predict(fit,new)):
        assert (type(output).__module__.split('.')[0],type(output).__name__)==expected_type
    if family in ('ordinal','multinomial'):
        probs=as_polars(posterior_epred(fit,new))
        sums=probs.group_by('.chain','.draw','.row').agg(pl.col('.prediction').sum())
        np.testing.assert_allclose(sums['.prediction'],1,atol=1e-14)
        assert probs['.prediction'].min() >= 0
        assert draws['.prediction'].min() >= 0 and draws['.prediction'].max() <= 2
        assert prediction.height == 9
    else:
        assert prediction.height == 3
        assert as_polars(augment(fit)).height == 30
        assert as_polars(pp_check(fit,seed=1)).height == 1
        noisy=as_polars(predict(fit,new,interval='prediction',seed=2))
        np.testing.assert_allclose(noisy['fit'].to_numpy(),prediction['fit'].to_numpy())
    if family=='multinomial':
        link=as_polars(predict(fit,new,type='link'))
        assert link.height==9
        assert link.filter(pl.col('category')==fit.response_levels[0])['fit'].to_list()==[0.]*3
    if family in ('ordinal','multinomial'):
        with pytest.raises(ValueError,match='posterior_predict'):
            predict(fit,interval='prediction')


def _reference_distribution(family,fit):
    """Independent SciPy distributions, evaluated at the retained draws."""
    beta=fit.beta_draws
    x=synthetic()['x'].to_numpy()
    if family=='multinomial':
        assert fit.response_levels==['blue','green','red']
        beta=beta.reshape(2,60,2,2)
        eta=beta[:,:,:,0,None]+beta[:,:,:,1,None]*x+.1
        logits=np.concatenate([np.zeros((2,60,1,30)),eta],axis=2).transpose(0,1,3,2)
        return np.exp(log_softmax(logits,axis=-1))
    eta=beta[:,:,0,None]+beta[:,:,1,None]*x+.1 if family!='ordinal' else beta[:,:,0,None]*x+.1
    aux=fit.extra_draws
    if family=='gaussian':
        return stats.norm(eta,aux['sigma'][...,None])
    if family=='student_t':
        return stats.t(aux['nu'][...,None],loc=eta,scale=aux['sigma'][...,None])
    if family=='binomial':
        return stats.binom(1,expit(eta))
    if family=='poisson':
        return stats.poisson(np.exp(eta))
    if family=='negative_binomial':
        alpha=aux['alpha'][...,None]
        return stats.nbinom(alpha,alpha/(alpha+np.exp(eta)))
    if family=='gamma':
        alpha=aux['alpha'][...,None]
        return stats.gamma(alpha,scale=np.exp(eta)/alpha)
    if family=='lognormal':
        return stats.lognorm(aux['sigma'][...,None],scale=np.exp(eta))
    if family=='exponential':
        return stats.expon(scale=np.exp(eta))
    cumulative=stats.logistic.cdf(aux['cutpoints'][:,:,None,:]-eta[...,None])
    return np.diff(np.concatenate([np.zeros((2,60,30,1)),cumulative,np.ones((2,60,30,1))],axis=-1),axis=-1)


def test_pointwise_likelihood_matches_independent_scipy_distribution(sampled):
    family,_,fit=sampled
    reference=_reference_distribution(family,fit)
    if family in ('ordinal','multinomial'):
        expected=np.log(np.take_along_axis(reference,np.broadcast_to(fit.y,(2,60,30))[...,None],axis=-1)[...,0])
    else:
        expected=reference.logpmf(fit.y) if family in ('binomial','poisson','negative_binomial') else reference.logpdf(fit.y)
    np.testing.assert_allclose(fit.idata.log_likelihood['observed'].values,expected,rtol=1e-9,atol=1e-9)
    means=as_polars(posterior_epred(fit))['.prediction'].to_numpy()
    expected=reference if family in ('ordinal','multinomial') else reference.mean()
    np.testing.assert_allclose(means.reshape(expected.shape),expected,rtol=1e-12,atol=1e-12)


def test_predictive_draws_follow_the_declared_distribution(sampled):
    family,_,fit=sampled
    reference=_reference_distribution(family,fit)
    values=as_polars(posterior_predict(fit,seed=39))['.prediction'].to_numpy().reshape(2,60,30)
    if family in ('ordinal','multinomial'):
        cumulative=reference.cumsum(axis=-1)
        right=np.take_along_axis(cumulative,values[...,None],axis=-1)[...,0]
        mass=np.take_along_axis(reference,values[...,None],axis=-1)[...,0]
        pit=right-mass+np.random.default_rng(40).uniform(size=values.shape)*mass
    elif family in ('binomial','poisson','negative_binomial'):
        pit=reference.cdf(values-1)+np.random.default_rng(40).uniform(size=values.shape)*reference.pmf(values)
    else:
        pit=reference.cdf(values)
    # DKW bounds the chance of exceeding this distance by 1e-5 per case.
    distance=stats.kstest(pit.reshape(-1),'uniform').statistic
    assert distance < np.sqrt(np.log(2/1e-5)/(2*pit.size))


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


def test_weighted_gaussian_posterior_matches_independent_quadrature():
    source=pl.DataFrame({'y':[.5,1.,2.,2.5],'off':[.1,.2,.3,.4],'w':[1.,2.,.5,3.]})
    y=(source['y']-source['off']).to_numpy()
    w=source['w'].to_numpy()
    def conditional(sd):
        covariance=np.diag(sd**2/w)+.8**2*np.ones((4,4))
        density=np.exp(stats.multivariate_normal.logpdf(y,mean=np.full(4,.2),cov=covariance)
                       +stats.halfnorm.logpdf(sd,scale=1.3))
        variance=1/(1/.8**2+w.sum()/sd**2)
        mean=variance*(.2/.8**2+w@y/sd**2)
        return density,mean,variance
    z=quad(lambda sd:conditional(sd)[0],0,np.inf,epsabs=1e-11)[0]
    mean=quad(lambda sd:np.prod(conditional(sd)[:2]),0,np.inf,epsabs=1e-11)[0]/z
    variance=quad(lambda sd:conditional(sd)[0]*(conditional(sd)[2]+(conditional(sd)[1]-mean)**2),
                  0,np.inf,epsabs=1e-11)[0]/z
    fit=bayes_glm(source,'y~1+offset(off)',weights='w',
        priors={'(Intercept)':prior('normal',mu=.2,sigma=.8),'sigma':prior('half_normal',sigma=1.3)},
        draws=1000,tune=500,chains=2,seed=71)
    diagnostics=bayes_diagnostics(fit).filter(pl.col('term')=='(Intercept)')
    assert abs(fit.coef[0]-mean)<max(6*diagnostics['mcse_mean'][0],.01)
    assert np.var(fit.beta_draws)==pytest.approx(variance,rel=.15)
    assert diagnostics['r_hat'][0]<1.05
    eta=fit.beta_draws[:,:,0,None]+source['off'].to_numpy()
    expected=stats.norm.logpdf(source['y'].to_numpy(),eta,fit.extra_draws['sigma'][...,None]/np.sqrt(w))
    np.testing.assert_allclose(fit.idata.log_likelihood['observed'].values,expected,atol=1e-10)


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
    repeated=pl.DataFrame({'x':[.5,.5,.5],'g':['new','new','another']})
    epred=posterior_epred(fit,repeated,allow_new_levels=True,seed=10)
    assert epred.equals(posterior_epred(fit,repeated,allow_new_levels=True,seed=10))
    values=epred['.prediction'].to_numpy().reshape(2,100,3)
    np.testing.assert_array_equal(values[:,:,0],values[:,:,1])
    assert not np.allclose(values[:,:,0],values[:,:,2])
    augmented=augment(fit,re_formula='population')
    np.testing.assert_allclose(augmented['.resid'].to_numpy(),source['gaussian'].to_numpy()-augmented['.fitted'].to_numpy())
    # Random effects are deliberately zero for marginal means.
    assert tidy(bayes_emmeans(fit,'x',at={'x':[-1,1]})).height==2


@pytest.mark.parametrize('link',['probit','cloglog'])
def test_binomial_links(link):
    fit=bayes_glm(synthetic(),'binomial~x',family='binomial',link=link,draws=20,tune=30,chains=1,seed=14)
    assert predict(fit)['fit'].is_between(0,1).all()
    eta=fit.beta_draws[:,:,0,None]+fit.beta_draws[:,:,1,None]*synthetic()['x'].to_numpy()
    probability=stats.norm.cdf(eta) if link=='probit' else stats.gumbel_l.cdf(eta)
    np.testing.assert_allclose(posterior_epred(fit)['.prediction'].to_numpy().reshape(probability.shape),probability)
    np.testing.assert_allclose(fit.idata.log_likelihood['observed'].values,
        stats.binom.logpmf(fit.y,1,probability),atol=1e-10)


def test_ordinal_intercept_only_and_missing_row_alignment():
    source=synthetic().with_columns(pl.when(pl.col('x')>1.4).then(None).otherwise(pl.col('ordinal')).alias('ordinal'))
    fit=bayes_glm(source,'ordinal~1',family='ordinal',link='probit',draws=25,tune=40,chains=1,seed=3,na_action='exclude')
    assert fit.parameter_names==[]
    assert posterior_summary(fit).height==2
    assert predict(fit).height==90
    assert predict(fit)['fit'].null_count()==3
    assert bayes_diagnostics(fit).height>=2
    cumulative=stats.norm.cdf(fit.extra_draws['cutpoints'])
    probabilities=np.diff(np.concatenate([np.zeros((1,25,1)),cumulative,np.ones((1,25,1))],axis=-1),axis=-1)
    expected=np.broadcast_to(probabilities[:,:,None,:],(1,25,fit.n,3))
    np.testing.assert_allclose(posterior_epred(fit)['.prediction'].to_numpy().reshape(expected.shape),expected)
    loglik=np.log(np.take_along_axis(expected,np.broadcast_to(fit.y,(1,25,fit.n))[...,None],axis=-1)[...,0])
    np.testing.assert_allclose(fit.idata.log_likelihood['observed'].values,loglik,atol=1e-10)
    assert predict(fit)['category'].head(3).to_list()==['low','middle','high']


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


def test_exact_models_support_predictive_comparison_and_waic_definition():
    source=synthetic()
    fits={'intercept':bayes_lm(source,'gaussian~1',draws=500,chains=2,seed=41),
          'slope':bayes_lm(source,'gaussian~x',draws=500,chains=2,seed=42)}
    for method in ('loo','waic'):
        comparison=bayes_compare(fits,method=method)
        assert comparison.height==2
        assert comparison['weight'].sum()==pytest.approx(1)
        assert comparison[f'elpd_{method}'].is_finite().all()
        if method=='waic':
            for name,fit in fits.items():
                eta=np.einsum('cdp,np->cdn',fit.beta_draws,fit.x)
                loglik=stats.norm.logpdf(fit.y,eta,fit.extra_draws['sigma'][...,None]).reshape(1000,30)
                expected=np.sum(logsumexp(loglik,axis=0)-np.log(1000)-loglik.var(axis=0))
                assert comparison.filter(pl.col('model')==name)['elpd_waic'][0]==pytest.approx(expected,abs=1e-10)
