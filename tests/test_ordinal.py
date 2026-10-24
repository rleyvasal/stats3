"""Ordinal regression against MASS::polr, across the four data backends."""

import warnings

import numpy as np
import polars as pl
import pytest
from scipy.optimize import check_grad

from stats3 import anova, augment, emmeans, glance, lm, modelsummary, polr, predict, tidy, vcov
from stats3.ordinal import _distribution, _objective, _score
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r


def housing():
    return pl.read_csv(DATA/'housing.csv').with_row_index('i',offset=1).with_columns(
        pl.col('Sat').cast(pl.Enum(['Low','Medium','High'])),
        pl.col('Infl').cast(pl.Enum(['Low','Medium','High'])),
        pl.col('Type').cast(pl.Enum(['Tower','Apartment','Atrium','Terrace'])),
        pl.col('Cont').cast(pl.Enum(['Low','High'])),
        off=pl.Series(np.linspace(-.3,.3,72)),x=(pl.col('i')%7)/4,
        w0=pl.when(pl.col('i').is_in([2,11,28])).then(0).otherwise(pl.col('Freq')))


CASES={name:('Sat ~ Infl + Type + Cont',{'method':name}) for name in ('logistic','probit','loglog','cloglog')}
CASES.update(intercept=('Sat ~ 1',{}),numeric=('Sat ~ x + Cont',{}),no_intercept=('Sat ~ 0 + x',{}),
    offset=('Sat ~ Infl + Type + offset(off)',{}),offset_arg=('Sat ~ Infl + Type',{'offset':'off'}),
    zero_weights=('Sat ~ Infl + Type + Cont',{'weights':'w0'}),sum_contrasts=('Sat ~ Infl + Type + Cont',{'contrasts':'sum'}))


@pytest.fixture(scope='module',params=BACKENDS)
def fits(request):
    backend=request.param
    source,_=as_input(housing(),backend,[])
    result={}
    for name,(formula,kwargs) in CASES.items():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='polr.*intercept is needed')
            result[name]=polr(source,formula,**{'weights':'Freq',**kwargs})
    return backend,result


@pytest.mark.parametrize('table',['tidy','vcov','probs','class','new_probs','new_class','augment','glance'])
def test_tables_match_r(fits,table):
    if R_COMMAND is None:
        pytest.skip('R oracle unavailable')
    backend,models=fits
    new,_=as_input(housing().head(5),backend,[])
    for name,fit in models.items():
        actual={'tidy':lambda:tidy(fit,conf_int=True),'vcov':lambda:vcov(fit),
            'probs':lambda:predict(fit,type='probs'),'class':lambda:predict(fit),
            'new_probs':lambda:predict(fit,new,type='probs'),'new_class':lambda:predict(fit,new),
            'augment':lambda:as_polars(augment(fit)).select('.fitted','.linear_predictor'),
            'glance':lambda:glance(fit)}[table]()
        expected=_oracle('ordinal')[name][table]
        # R optim stops on objective change; BFGS here stops on the gradient.
        # The oracle tightens R's tolerance; compare at optimizer accuracy.
        loose={c:(2e-6,2e-8) for c in ('estimate','std_error','statistic','conf_low','conf_high',
                                     'Low','Medium','High')}
        # Near-zero Wald endpoints retain the absolute optimizer error after subtraction;
        # near-zero linear predictors sum the absolute errors of several slopes (up to 7e-8).
        loose.update({c:(2e-6,1e-7) for c in ('conf_low','conf_high','.linear_predictor')})
        if table=='vcov':
            loose.update({n:(2e-6,2e-8) for n in fit.parameter_names})
        assert_matches_r(actual,expected,loose=loose)


@pytest.mark.parametrize('backend',BACKENDS)
def test_groups_comparison_and_output_types(backend):
    source,by=as_input(housing(),backend,['Cont'])
    grouped=polr(source,'Sat ~ Infl + Type',weights='Freq',by=by)
    assert len(grouped)==2
    if R_COMMAND is not None:
        for table,fn in [('tidy',lambda f:tidy(f,conf_int=True)),('glance',glance)]:
            assert_matches_r(fn(grouped),_oracle('ordinal')['grouped'][table],
                loose={c:(2e-6,1e-7) for c in ('estimate','std_error','statistic','conf_low','conf_high')})
    source,_=as_input(housing(),backend,[])
    small=polr(source,'Sat ~ 1',weights='Freq')
    big=source >> polr('Sat ~ Infl + Type + Cont',weights='Freq')
    comparison=anova(big,small)
    if R_COMMAND is not None:
        assert_matches_r(comparison,_oracle('ordinal')['compare'])
    assert 'p_value' not in as_polars(anova(small,big,test='none')).columns
    expected={'polars':('polars','DataFrame'),'lazy':('polars','LazyFrame'),
              'pandas':('pandas','DataFrame'),'tidy3':('tidy3','TidyFrame')}[backend]
    for output in [tidy(big),glance(big),predict(big),vcov(big),augment(big)]:
        assert (type(output).__module__.split('.')[0],type(output).__name__)==expected
    assert as_polars(big >> tidy()).height==8
    assert as_polars(modelsummary(big,statistic='conf_int')).height>0


@pytest.mark.parametrize('method',['logistic','probit','loglog','cloglog'])
def test_analytic_score_matches_numerical_derivative(method):
    x=np.array([[-1.],[0.],[1.],[2.]])
    y=np.array([0,1,2,1]); weights=np.array([1.,2.,0.,3.]); offset=np.array([.1,0.,-.1,.2])
    parameters=np.array([.3,-.8,np.log(1.7)])
    distribution=_distribution(method)
    objective=lambda p:_objective(p,x,y,weights,offset,distribution.cdf)
    score=lambda p:_score(p,x,y,weights,offset,distribution.cdf,distribution.pdf)
    assert check_grad(objective,score,parameters)<2e-6


def test_missing_rows_offsets_and_factor_response():
    source=housing().with_columns(pl.when(pl.col('i')==4).then(None).otherwise(pl.col('Infl')).alias('Infl'))
    fit=polr(source,'Sat ~ Infl + Type',weights='Freq',offset='off',na_action='exclude')
    assert predict(fit)['fit'].null_count()==1
    assert predict(fit,type='probs')['Low'].null_count()==1
    assert augment(fit)['.fitted'].null_count()==1
    new=housing().head(5)
    x,offset,_,_=fit._new_design(new)
    np.testing.assert_allclose(predict(fit,new,type='link')['fit'].to_numpy(),
                               x[:,fit.slope_indices]@fit.coef+offset)
    numeric=housing().with_columns(rating=pl.col('Sat').cast(pl.String).replace_strict({'Low':1,'Medium':2,'High':3}))
    ordered=polr(numeric,'factor(rating) ~ Infl + Type',weights='Freq')
    assert list(ordered.zeta)==['1|2','2|3']
    with pytest.raises(ValueError,match='missing'):
        polr(source,'Sat ~ Infl',na_action='fail')


def test_aliases_probabilities_start_and_convergence():
    source=housing().with_columns(z=pl.col('x'))
    with pytest.warns(UserWarning,match='rank-deficient'):
        fit=polr(source,'Sat ~ x + z',weights='Freq')
    assert list(fit.coefficients)==['x']
    with pytest.warns(UserWarning,match='rank-deficient'):
        probabilities=predict(fit,source,type='probs')
    np.testing.assert_allclose(probabilities.sum_horizontal(),1)
    start=np.r_[fit.coef,fit.thresholds]
    refit=polr(source,'Sat ~ x',weights='Freq',start=start)
    np.testing.assert_allclose(refit.coef,fit.coef,atol=1e-7)
    with pytest.warns(UserWarning,match='did not converge'):
        incomplete=polr(source,'Sat ~ Infl + Type + Cont',weights='Freq',maxit=1)
    assert not incomplete.converged


@pytest.mark.parametrize('kwargs',[{'method':'cauchit'},{'maxit':0},{'epsilon':0},{'epsilon':np.nan},
    {'start':[0.,0.]},{'start':[0.,1.,0.]}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        polr(housing(),'Sat ~ x',weights='Freq',**kwargs)


def test_unsupported_inference_and_invalid_responses():
    fit=polr(housing(),'Sat ~ Infl',weights='Freq')
    assert tidy(fit)['p_value'].null_count()==4
    with pytest.raises(ValueError,match='vcov'):
        tidy(fit,vcov='HC3')
    with pytest.raises(TypeError,match='single polr'):
        anova(fit)
    with pytest.raises(TypeError,match='ordinal marginal'):
        emmeans(fit,'Infl')
    with pytest.raises(ValueError,match='categorical'):
        polr(housing(),'Freq ~ x')
    with pytest.raises(ValueError,match='three levels'):
        polr(housing(),'Cont ~ x')
    with pytest.raises(ValueError,match='positive total weight'):
        polr(housing().with_columns(w=pl.when(pl.col('Sat')=='High').then(0).otherwise(1)),'Sat ~ x',weights='w')
    with pytest.raises(ValueError,match='new levels'):
        predict(fit,pl.DataFrame({'Infl':['new']}))
    with pytest.raises(TypeError,match='polr fits'):
        anova(fit,lm(housing(),'Freq ~ x'))
