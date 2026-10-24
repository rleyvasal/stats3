# Bayesian models

This first pass provides exact Gaussian regression, exact binomial/Poisson
scalar inference and optional sampled generalized/hierarchical regression.
It is not an implementation of every Bayesian method.

## Exact linear regression

```python
from stats3 import *

fit = bayes_lm(data, "y ~ x + group", prior_mean=0, prior_scale=10,
               prior_shape=2, prior_rate=1, draws=1000, chains=4, seed=42)
tidy(fit, conf_int=True)
predict(fit, newdata, interval="credible")
predict(fit, newdata, interval="prediction")
```

The model is `y ~ Normal(X beta + offset, sigma² / weights)`, with
`beta | sigma² ~ Normal(prior_mean, sigma² diag(prior_scale²))` and
`sigma² ~ InverseGamma(prior_shape, prior_rate)`. The inverse-gamma uses
shape/rate, with density proportional to
`variance**(-shape-1) * exp(-rate/variance)`. Shape must exceed 1, so the
reported posterior covariance is finite. These are proper priors; predictor
units matter. The defaults are not universal noninformative priors.

`prior_mean` and `prior_scale` accept scalars, coefficient-order vectors,
or dictionaries such as `{"x": 2}`. Unspecified dictionary entries use 0
and 10, respectively. Scales must be positive. Known precision weights
must be strictly positive and are not frequency weights. New-row predictive
noise has unit precision; training predictions honor the original weights.
Finite proper priors identify collinear designs without dropping coefficients.

Coefficient means, SDs, credible intervals, predictions and normalized
marginal likelihoods are exact. Stored posterior draws are IID samples,
not MCMC chains. Marginal means, contrasts, auxiliary parameters and ROPE
probabilities are summarized from those draws. `exponentiate=True` is
rejected for exact fits: exponentiated Student-t coefficient posteriors
do not have finite means. Draws remain available for other transformations.

## Sampled models

Install `pip install "stats3[bayes]"`. The supported dependency series are
PyMC 5.28–5.x and ArviZ 0.21–0.22, with their existing InferenceData API.
Importing stats3 and using exact methods never imports these optional packages.

```python
fit = bayes_glm(data, "success ~ x", family="binomial",
                priors={"beta": prior("normal", mu=0, sigma=1.5)},
                chains=4, draws=1000, tune=1000, target_accept=.9, seed=42)
```

| Family | Linear predictor / parameters |
|---|---|
| `gaussian` | identity mean; residual `sigma` |
| `student_t` | identity mean; scale `sigma`; `nu=2+nu_minus_two` |
| `binomial` | logit probability; optional probit or cloglog |
| `poisson` | log mean |
| `negative_binomial` | log mean; variance `mu + mu²/alpha` |
| `gamma` | log mean; shape `alpha`, rate `alpha/mu` |
| `lognormal` | eta is mean log outcome; response mean `exp(eta+sigma²/2)` |
| `exponential` | log mean |
| `ordinal` | cumulative logit or probit; `P(Y<=k)=F(cutpoint[k]-eta)` |
| `multinomial` | softmax; the first factor level is the zero-logit reference |

Ordinal and multinomial responses use categorical columns. Set the intended
order with Polars `Enum`, pandas `category`, or tidy3 factor levels; plain
strings use sorted levels. Ordinal regression removes the fixed intercept
because cutpoints provide it. Ordinal needs at least three categories.

Binomial inputs can be 0/1, a two-level categorical response,
`cbind(successes, failures)`, or proportions with integer `weights=` trials.
Counts and trials must be integral; rows need at least one trial.
Gaussian and Student-t weights are positive observation precisions, with
residual scale divided by their square root. Other likelihoods do not
accept weights. Offsets enter the linear predictor on both fitted and new rows.

### Priors

Defaults on the raw predictor scale:

- coefficients: `Normal(0, 2.5)`;
- Gaussian/Student-t intercept: `Normal(0, 10)`;
- `sigma`: `HalfNormal(5)`;
- varying-effect SDs: `HalfNormal(2.5)`;
- `alpha`: `Exponential(1)`;
- `nu_minus_two`: `Exponential(1/30)`;
- ordered cutpoints: iid `Normal(0, 5)` restricted to increasing order.

Choose units/center/scale your predictors before fitting and examine prior
predictions. Broad log-link priors can imply extremely large counts.
Sampled Gaussian priors are independent of sigma and differ from the exact
engine's conjugate conditional prior; the two engines do not fit the same
default model.

Override `beta`, an individual coefficient name (including `(Intercept)`),
`sigma`, `alpha`, `group_sd`, `nu_minus_two`, or `cutpoints` in `priors`.
Unused names raise an error. `prior()` accepts `normal`, `student_t`,
`half_normal`, `half_student_t`, `exponential`, `gamma`, `inverse_gamma`,
and `beta`, using PyMC parameter names. Choose distributions whose support
matches the parameter. Cutpoint priors must have scalar, exchangeable
parameters; prior predictive cutpoints are sorted iid prior draws.

### Hierarchical models

```python
fit = bayes_glm(data, "y ~ x + (1 + x | subject)", seed=42)
predict(fit, newdata)                              # conditional on subject effects
predict(fit, newdata, re_formula="population")     # subject effects set to zero
posterior_predict(fit, newdata, allow_new_levels=True, seed=42)
```

Varying coefficients are independent and noncentered; the `|` syntax here
does not estimate correlations. Slopes must name numeric columns. Multiple
grouping columns are supported as separate terms. Unknown groups raise by
default; `allow_new_levels=True` samples one new effect per distinct new
group and posterior draw, preserving within-group dependence. The seed
controls these predictions. Setting effects to zero is not the same as
integrating a nonlinear response over their distribution.

`by=` / tidy3 `group_by()` fits independent models with independently seeded
draws; it does not pool their parameters. Select `models[0]` for predictive
draws. Multinomial varying effects are deferred.

## Posterior workflow

```python
posterior_summary(fit)              # includes auxiliary parameters
posterior_draws(fit)                # .chain, .draw, coefficient/auxiliary columns
posterior_epred(fit, newdata)       # conditional response mean/probability draws
posterior_predict(fit, newdata, seed=42)
prior_predict(fit, newdata, draws=1000, seed=42)
pp_check(fit, statistic="zero_fraction", seed=42)
hypothesis(fit, {"x": 1, "z": -1}, value=0, rope=(-.1, .1))
```

`tidy()` and `posterior_summary()` report posterior means, posterior SDs
(`std_error`), equal-tail credible bounds, and `prob_positive`.
`tidy()` leaves `statistic`/`p_value` missing; these are not frequentist
coefficient tests. `vcov()` is posterior covariance. `modelsummary()` can
display these fits too. The robust `vcov=` options and `anova()` do not
apply to Bayesian fits.

Predictive draw tables contain `.chain`, `.draw`, `.row`, `.prediction`.
Training row IDs retain their original input positions after missing rows
are omitted. `na_action="exclude"` pads ordinary prediction/augmentation
tables; draw tables include only usable rows. New-row IDs start at 1.
Binomial posterior predictive draws are success counts: training draws use
the original trial totals; new rows default to one trial, overridable via
`trials=`. Ordinal/multinomial outcome draws are zero-based category codes;
`fit.response_levels` supplies their labels. Expected prediction draws for
these models contain one row per category with a `category` label.
`predict()` reports category probabilities and credible intervals; use
`posterior_predict()` for categorical outcome draws, and `posterior_epred()`
to retain joint uncertainty. Categorical augmentation is deferred.

`pp_check()` compares mean, SD, maximum or zero fraction with replicated
data. Its `prob_greater` is a posterior predictive tail area, not a Bayes
factor; categorical predictive checks use raw draws instead.

### Marginal means and contrasts

```python
emm = bayes_emmeans(fit, "treatment", by="site", weights="cells")
tidy(emm)
tidy(pairs(emm))
tidy(contrast(emm, {"custom": [-1, .5, .5]}))
```

The reference grid and equal/cell/proportional weights follow `emmeans`.
`type="response"` transforms each posterior draw before averaging.
`type="link"` averages linear predictors. Offsets use the same cell weights;
an `offset=` column contributes its mean over fitted rows. Varying effects
are zero for these marginal means. Multinomial/ordinal marginal means are
deferred. Pairwise/named contrasts retain joint posterior draws and report
credible intervals and sign probabilities; frequentist multiplicity
adjustments are not applied. `emmeans()` directs Bayesian fits to this API.

### Diagnostics and comparison

```python
bayes_diagnostics(fit)
bayes_compare({"small": small, "large": large}, method="loo")
bayes_compare({"small": small, "large": large}, method="waic")
bayes_compare({"small": exact_small, "large": exact_large}, method="evidence")
```

Sampled diagnostics include rank-normalized R-hat, bulk/tail ESS, MCSE,
divergences, maximum tree-depth hits and minimum chain BFMI. Exact draws
are IID: R-hat is not assigned, ESS equals the independent sample count and
coefficient MCSE follows from the exact covariance. Inspect all parameters,
including auxiliary scales and varying effects, before interpreting MCMC.
The short-chain workflow tests do not establish convergence for user data;
separate weighted Gaussian, logistic and Poisson tests check Monte Carlo
accuracy against independent quadrature. All ten likelihood families run
through all four data backends and are checked against independent SciPy
pointwise likelihoods, response means and predictive distributions.

LOO and WAIC use ArviZ and return its diagnostics/warnings and stacking
weights. Both require the extra, including for exact models. Compared fits
must have identical response values, category order/trials, used row positions
and weights. Exact Gaussian `evidence` comparison uses proper prior
normalizations, returns log Bayes factors relative to the first model,
and weights under equal prior model odds. Evidence is sensitive to priors.
General sampled-model marginal likelihoods and point-null coefficient Bayes
factors are not approximated with posterior sign probabilities.

## Exact scalar conjugate inference

```python
bayes_binom_test(7, 10, alpha=1, beta=1, null=.5)
bayes_poisson_test(5, exposure=2, shape=1, rate=1, null=1)
```

These scalar functions return Polars tables with posterior moments,
equal-tail credible intervals, `prob_greater` than the null, posterior
shape parameters and log Bayes factors for the specified proper alternative
versus the point null. Gamma uses shape/rate. They do not yet accept grouped
data-column inputs. The normal regression interface covers posterior
comparisons of numeric group means via factor coefficients/contrasts.

## Remaining scope

Correlated varying effects, multinomial varying effects, categorical marginal
means/augmentation, general model evidence, custom likelihoods, censoring,
survival, time-series and nonparametric Bayesian models are not implemented.
Bayesian tests are not claimed to reproduce unspecified defaults from R's
BayesFactor, brms or rstanarm. Exact results are checked against independent
base-R formulas; MCMC results are checked with quadrature and actual PyMC
fits across the supported workflows. The optional dependencies have their
own compiler/runtime requirements; local tests can use PyTensor's
`cxx=` execution mode where native compilation is unavailable.
