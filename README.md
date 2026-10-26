# stats3

**R's statistical tests on Polars, for Polars, pandas and tidy3 data.**
Write a test the way R users do, run it once per group, and get R's numbers
back as a table in the same kind of frame you passed in.

```bash
pip install stats3              # Polars data
pip install "stats3[pandas]"    # pandas data too
pip install "stats3[fast]"      # Numba: compiled k-means, hclust, dist and concordance, at R's speed on large data
```

stats3 needs Python 3.12+, Polars and SciPy 1.18+ (the first SciPy whose
distribution functions match R to full precision).

Every test takes R's arguments and defaults, drops missing values as R does,
raises R's errors and warnings, and is checked against R 4.6.

## One test per group

```python
import polars as pl
from stats3 import *

penguins = pl.read_csv(
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv",
    null_values="NA",
)

t_test(penguins, "body_mass_g ~ sex", by="species")
```

```text
┌───────────┬─────────────┬────────────┬────────────┬────────────┬─────────────┬─────────────┐
│ species   ┆ estimate    ┆ statistic  ┆ df         ┆ p_value    ┆ conf_low    ┆ conf_high   │
│ ---       ┆ ---         ┆ ---        ┆ ---        ┆ ---        ┆ ---         ┆ ---         │
│ str       ┆ f64         ┆ f64        ┆ f64        ┆ f64        ┆ f64         ┆ f64         │
╞═══════════╪═════════════╪════════════╪════════════╪════════════╪═════════════╪═════════════╡
│ Adelie    ┆ -674.657534 ┆ -13.126286 ┆ 135.693579 ┆ 6.4023e-26 ┆ -776.301162 ┆ -573.013906 │
│ Chinstrap ┆ -411.764706 ┆ -5.207734  ┆ 62.575105  ┆ 0.000002   ┆ -569.790344 ┆ -253.739068 │
│ Gentoo    ┆ -805.094686 ┆ -14.761368 ┆ 116.643504 ┆ 1.8678e-28 ┆ -913.113043 ┆ -697.07633  │
└───────────┴─────────────┴────────────┴────────────┴────────────┴─────────────┴─────────────┘
```

(The table also has `estimate1`, `estimate2`, `method` and `alternative`.)
`by=` is dplyr's `.by`: one test per group, one row per test, groups in order.

## Any frame in, the same frame out

| You pass | Groups | You get back |
|---|---|---|
| Polars `DataFrame` / `LazyFrame` | `by=` | Polars `DataFrame` / `LazyFrame` |
| pandas `DataFrame` | `by=` | pandas `DataFrame` |
| tidy3 frame | `group_by()` | tidy3 frame |
| dict of columns | `by=` | Polars `DataFrame` |

`as_="polars"`, `"pandas"` or `"tidy3"` asks for another type. Factor levels
come from the data: pandas `category` and Polars `Enum` columns order the
levels of `y ~ g` (the first level is `estimate1`) and of the groups.

Each test can be called with the data first, or piped:

```python
t_test(penguins, "body_mass_g ~ sex", by="species")       # data first
penguins >> t_test("body_mass_g ~ sex", by="species")     # pipe
penguins_pd.pipe(t_test("body_mass_g ~ sex", by="species"))  # pandas .pipe
```

With [tidy3](https://github.com/rleyvasal/tidy3) the tests are pipe steps
like any verb, and its groups and factor levels carry through:

```python
from tidy3 import *

tidy(penguins)
>> group_by(species, island)
>> wilcox_test(body_mass_g ~ sex)
>> group_by(species)
>> mutate(p_adj = p_adjust(p_value, "BH"))
```

## In a notebook: R's shorthand

In Jupyter or IPython, after `from stats3 import *` (or `%load_ext
stats3.jupyter`), from the next cell on formulas need no quotes and columns
are bare names, inside stats3's calls:

```python
penguins >> t_test(body_mass_g ~ sex, by=species)
penguins >> binom_test(sex == "male")
cor_test(penguins, bill_length_mm, bill_depth_mm, method="kendall")
```

A name the notebook defines keeps its value; any other bare name in a stats3
call is a column. With tidy3 loaded the two packages share the shorthand. In
a `.py` script write `"body_mass_g ~ sex"` and `pl.col("sex") == "male"`.

## Models

`lm()` and `glm()` take R's formulas and print a summary you can read at a
glance: the coefficients with their 95% intervals, then the fit.

```python
fit = lm(penguins, "body_mass_g ~ flipper_length_mm + species")
fit
```

```text
lm(body_mass_g ~ flipper_length_mm + species)   n = 342
┌───────────────────┬──────────────┬──────────────┬──────────────┬───────────┬────────────┐
│ term              ┆ estimate     ┆ conf_low     ┆ conf_high    ┆ std_error ┆ p_value    │
╞═══════════════════╪══════════════╪══════════════╪══════════════╪═══════════╪════════════╡
│ (Intercept)       ┆ -4031.476891 ┆ -5180.506847 ┆ -2882.446935 ┆ 584.15134 ┆ 2.5515e-11 │
│ flipper_length_mm ┆ 40.705401    ┆ 34.664683    ┆ 46.746119    ┆ 3.07102   ┆ 1.4016e-32 │
│ speciesChinstrap  ┆ -206.51012   ┆ -320.066724  ┆ -92.953517   ┆ 57.730646 ┆ 0.000398   │
│ speciesGentoo     ┆ 266.809603   ┆ 79.425129    ┆ 454.194077   ┆ 95.263741 ┆ 0.005392   │
└───────────────────┴──────────────┴──────────────┴──────────────┴───────────┴────────────┘
residual SE 375.5 on 338 df · R² 0.7826 · adjusted R² 0.7807
F(3, 338) = 405.7, p < 2.2e-16
```

broom's verbs give tables (in the data's frame type), and `by=` fits one
model per group:

```python
tidy(fit, conf_int=True)    # term, estimate, std_error, statistic, p_value, conf_low, conf_high
glance(fit)                 # r_squared, adj_r_squared, sigma, statistic, p_value, AIC, BIC, ...
augment(fit)                # the rows with .fitted, .resid, .hat, .sigma, .cooksd, .std_resid
anova(fit)                  # sequential sums of squares; anova(fit1, fit2) compares models
anova(fit, type=2)          # car's type II tests; type=3 with contrasts="sum"
predict(fit, new_rows, interval="prediction")

glance(lm(penguins, "body_mass_g ~ flipper_length_mm", by="species"))   # one row per species

glm(penguins, "sex ~ body_mass_g + species", family=binomial())
glm(mtcars, "carb ~ hp + offset(log(gear))", family=poisson())
```

Robust standard errors work as lmtest's `coeftest` with sandwich's
estimators, and `modelsummary` puts models side by side:

```python
tidy(fit, vcov="HC3")                  # heteroskedasticity-consistent (HC0-HC5)
tidy(fit, vcov=vcov_cl("island"))      # clustered; vcov_cl(["island", "year"]) for two-way
tidy(fit, vcov=vcov_nw())              # Newey-West
glance(fit, vcov="HC1")                # robust Wald F test of the slopes

modelsummary({"Base": fit0, "Species": fit}, stars=True, output="markdown")
```

Estimated marginal means and contrasts follow R's emmeans:

```python
emm = emmeans(fit, "species")
tidy(emm)                            # adjusted means, standard errors, intervals
tidy(pairs(emm))                     # pairwise differences, Tukey adjustment
tidy(contrast(emm, "trt.vs.ctrl"))   # comparisons with the first level
```

`by=` compares levels within each subgroup, `at=` sets covariate values,
and `weights=` controls averaging over other factors. `vcov="HC3"` uses
robust standard errors; `type="response"` back-transforms GLM estimates.
Offsets use the same averaging weights as the means. With `weights="cells"`,
R's emmeans 2.0.4 uses equal weights for offsets; matching stats3 in that
case requires supplying the cell-weighted offsets explicitly in R.

For overdispersed counts, `glm_nb()` estimates the negative binomial size
`theta` alongside the coefficients, as MASS's `glm.nb`:

```python
fit = glm_nb(counts, "events ~ treatment + offset(log(exposure))")
fit.theta, fit.se_theta              # size estimate and its standard error
tidy(fit, conf_int=True)             # profile intervals, with theta fixed
anova(smaller_fit, fit)              # likelihood ratio, theta fitted for each model
```

`glm(..., family=negative_binomial(2))` fixes theta at 2, as MASS's
`negative.binomial`. Both forms support log, identity and sqrt links,
grouped fits, weights, offsets, predictions, robust covariance and marginal
means. Single-model `anova(fit)` holds theta fixed and warns, as MASS does.

Robust M-estimation and conditional quantiles use the same formulas and
frame types:

```python
robust_fit = rlm(penguins, "body_mass_g ~ flipper_length_mm + species")
quantile_fit = rq(penguins, "body_mass_g ~ flipper_length_mm + species", tau=0.75)
tidy(robust_fit, conf_int=True)
predict(quantile_fit, new_rows, interval="confidence")
```

`rlm()` follows MASS's M-estimation with Huber, bisquare or Hampel
scores, MAD or Huber proposal-2 scale, inverse-variance or case weights,
and `XtX` or `XtWX` coefficient covariance. `method="MM"` gives MASS's
MM-estimate: an S-estimate from `lqs`, then bisquare M-steps at its fixed
scale. `init="lts"` starts the M iterations from `lqs`'s least trimmed
squares. Like MASS, `lqs` tries every subset when there are fewer than
5000, and otherwise random ones (`lqs_control={"nsamp": ...}`); those come
from R's generator, so `seed=` gives the same fit as `set.seed()` in R.
`rlm()` reports R's t statistics and normal confidence intervals; p-values
remain missing because `summary.rlm` supplies none.

`rq()` minimizes weighted pinball loss with a sparse linear program. It
fits one `tau` at a time with inference as in `summary.rq`. By default,
fits with fewer than 1001 rows get rank-inversion confidence intervals
(`se="rank"`, with `iid=False` for quantreg's nid rank variant) and larger
fits get `nid` standard errors. `se="iid"` gives iid standard errors, and
`se="boot"` the xy-pair bootstrap with `r` replications (200 by default),
drawn from R's generator so that `seed=` matches `set.seed()` in R. Rank
fits have intervals but no standard errors or tests, as in R; their
`vcov()`, prediction intervals and marginal means use `nid` covariance, as
quantreg's do. `anova(full_fit, reduced_fit)` gives Wald tests for nested
models with the same tau. Offsets are subtracted from the response
(quantreg 6.1 silently ignores `offset()` terms). For nonunique optima, the
LP solver may choose different coefficients from R, and rank intervals may
follow a different vertex path from R's; R warns about these fits.
Both models support groups, row alignment, `vcov()`, `modelsummary()` and
estimated marginal means. Quantile fit statistics follow `logLik.rq`'s
asymmetric-Laplace convention; R leaves BIC missing. Robust fits report their
scale and convergence, without Gaussian likelihood statistics.

Ordered responses get MASS's proportional-odds model, `polr`:

```python
housing = housing.with_columns(pl.col("Sat").cast(pl.Enum(["Low", "Medium", "High"])))
fit = polr(housing, "Sat ~ Infl + Type + Cont", weights="Freq")
tidy(fit, conf_int=True)             # slopes, then the thresholds Low|Medium and Medium|High
predict(fit, new_rows, type="probs") # one column per level; "class" (default) or "link"
anova(smaller_fit, fit)              # likelihood ratio test of nested fits
tidy(fit, vcov=vcov_cl("Type"))      # clustered standard errors, as sandwich's vcovCL
emmeans(fit, "Infl", by="Sat", mode="prob")  # class probabilities; "latent" (default), "mean.class"
```

As in MASS, P(Y <= k) = F(zeta_k - x'beta - offset), with the logistic,
probit, loglog, cloglog or cauchit link (`method=`). The response needs three or more
levels, ordered by a Polars `Enum`, a pandas category or `factor()`. Weights,
offsets, groups and `na_action` work as for `glm`. Standard errors come from
the Hessian, as `summary.polr`; p-values stay missing, as R reports none.
`conf_int=True` gives `confint.polr`'s profile likelihood intervals for the
slopes (MASS profiles no thresholds); `conf_method="wald"` gives Wald
intervals for every row. Robust covariance is sandwich's `vcovCL` (HC0 or
HC1), tested as `coeftest` with t on the residual df; as in R, its bread
scales by the total weight, so frequency-weighted fits get inflated robust
errors, and `vcov_hc`, `vcov_nw`, clustered HC2/HC3 and loglog fits raise
R's errors. `emmeans` follows emmeans' polr modes: `"latent"` (the linear
predictor minus the mean threshold), `"prob"` (the response becomes a grid
factor to put in specs or `by=`) and `"mean.class"`, with `pairs` and
`contrast`. For loglog fits, stats3 gives the correct class probabilities,
where emmeans 2.0.4 back-transforms with the wrong tail.

Unordered categorical responses get nnet's baseline-category logit, `multinom`:

```python
penguins = penguins.with_columns(mass=pl.col("body_mass_g") / 1000)
fit = multinom(penguins, "species ~ mass + sex")
tidy(fit, conf_int=True)             # one row per (y_level, term): Chinstrap and Gentoo against Adelie
predict(fit, new_rows, type="probs") # one column per level; "class" (default)
emmeans(fit, "species", by="sex")    # class probabilities averaged over the grid
anova(smaller_fit, fit)              # likelihood ratio test of nested fits, as anova.multinom
```

The first response level is the reference (a Polars `Enum`, a pandas category
or `factor()` sets the order; other responses become factors with sorted
levels, as in R). Weights, sum contrasts, groups and `na_action` work as for
`glm`; an offset is one column for a two-level response, or one column per
level (`offset=["o1", "o2", "o3"]`, R's `offset(cbind(...))`). stats3 finds the
maximum likelihood by Newton's method rather than nnet's BFGS, so it agrees
with nnet to the precision nnet reaches (about 1e-5); standard errors come
from the exact Hessian, as `summary.multinom`, and `tidy` gives broom's Wald z
tests (`exponentiate=True` for odds ratios). Aliased columns share nnet's
minimum-norm solution. `glance` gives `edf`, `deviance`, `AIC` and `nobs`.
`decay=` is nnet's weight decay (the reported deviance, as nnet's, includes the
penalty), a `cbind(n1, n2, ...)` response gives counts per level, and
`emmeans(fit, "species", by="sex")` gives emmeans' class probabilities
(`mode="latent"` for its centred log-odds), with pairs and contrasts.

Several responses in `cbind()` make a multivariate linear model, R's `mlm`,
and `manova` gives `summary.manova`'s table:

```python
fit = lm(iris, "cbind(sepal_length, sepal_width, petal_length, petal_width) ~ species")
tidy(fit, conf_int=True)              # one coefficient table per response, with a response column
anova(fit, test="Wilks")              # anova.mlm: Pillai (default), Wilks, Hotelling-Lawley, Roy
manova(iris, "cbind(sepal_length, petal_length) ~ species", test="Roy")
anova(fit, test="Spherical", X="~ 1") # univariate F with Greenhouse-Geisser and Huynh-Feldt p-values
mauchly_test(fit, X="~ 1")            # Mauchly's test of sphericity
```

The tests use R's sequential sums of squares and cross products and its F
approximations. `X=`, `M=` (formulas on `idata=`, one row per response, or
matrices) or `T=` test contrasts of the responses, as in repeated-measures
designs; `anova(fit1, fit2)` compares nested multivariate fits, as
`anova.mlmlist`. `predict` and `vcov` follow `predict.mlm` and `vcov.mlm`.
As in R, `anova` and `mauchly_test` refuse weighted fits (`manova` takes
weights). `glance`, `augment`, `emmeans` and `modelsummary` are not
available for multivariate fits.

Nonlinear least squares follows R's `nls`. The model is an arithmetic
expression of columns and parameters (`+ - * / ^` and R's math functions,
such as `exp`, `log`, `sqrt`, `pnorm` or `lgamma`); the names in `start`
are the parameters:

```python
fit = nls(puromycin, "rate ~ Vm * conc / (K + conc)", start={"Vm": 200, "K": 0.1})
tidy(fit, conf_int=True)             # t tests; confint.nls's profile intervals
nls(dnase, "density ~ SSlogis(log(conc), Asym, xmid, scal)", by="Run")  # self-starting
nls(puromycin, "rate ~ conc / (K + conc)", start={"K": 0.1}, algorithm="plinear")  # .lin is Vm
nls(puromycin, "rate ~ Vm * conc / (K + conc)", start={"Vm": 200, "K": 0.1},
    algorithm="port", lower=[0, 0.07])
anova(smaller_fit, fit)              # F test of each model against the one before
```

The default algorithm is R's Gauss-Newton iteration, with R's
relative-offset convergence criterion, step halving, `control=` (`maxiter`,
`tol`, `min_factor`, `warn_only`, `scale_offset`) and error messages
("singular gradient", "step factor ... reduced below 'minFactor'").
Gradients come from symbolic differentiation of the expression, where R
uses forward differences. `SSlogis`, `SSasymp`, `SSmicmen`, `SSgompertz`,
`SSfpl` and `SSweibull` find their own starting values with R's
initial-value methods. `algorithm="plinear"` is Golub-Pereyra, with the
conditionally linear terms as `cbind(...)` columns. `algorithm="port"`
takes `lower`/`upper` bounds and reaches R's bounded optimum with a
trust-region solver rather than the PORT library, so `glance()`'s `fin_tol`
is missing for it. Profile intervals solve for the end points that
`confint.nls` interpolates from a spline, so they can differ from R's in the
fourth significant digit. `weights`, `na_action`, `by=`, `glance`,
`augment`, `predict` and `vcov` work as for `lm`. Indexed parameters
(`a[group]`), `subset` and `trace` are not included.

Scatterplot smoothers are base R's, number for number:

```python
fit = loess(cars, "dist ~ speed")                  # span=0.75, degree=2, family="gaussian"
glance(fit)                                        # enp, sigma, trace_hat, one_delta, two_delta
predict(fit, new_rows, se=True)                    # fit, se_fit, residual_scale, df
anova(loess(cars, "dist ~ speed", span=0.4), fit)  # anova.loess's F test
lowess(cars, "speed", "dist", f=2/3)               # sorted x and the smooth y
spline = smooth_spline(cars, "speed", "dist")      # GCV; cv=True, df=, spar= or lambda_=
predict(spline, [10, 20], deriv=1)                 # the spline or its derivatives
ksmooth(cars, "speed", "dist", kernel="normal", bandwidth=2)
```

`loess` ports the netlib dloess code R uses. The default
`surface="interpolate"` builds R's k-d tree, fits at its vertices and blends
cubic Hermite interpolants between them, so fitted values, the equivalent
number of parameters, the residual standard error and the trace of the hat
matrix are R's, not those of an exact local fit at every point
(`surface="direct"` gives that). `family="symmetric"` adds R's bisquare
robustness iterations, `normalize`, `statistics="exact"`, `trace_hat`,
`cell` and `iterations` are `loess.control`'s, and one to four numeric
predictors are allowed. Conditionally parametric terms and `drop.square` are
not included. `smooth_spline` follows `smooth.spline`: `.nknots.smspl`
knots, x values merged within `tol`, GCV, leave-one-out CV or df matching,
the `spar`-to-lambda scale, and linear extrapolation in `predict`. Where it
searches for `spar` it agrees with R to about 1e-9, as Brent's parabolic
steps amplify last-bit differences in the criterion; the fit at a given
`spar`, `lambda_` or `df` matches to full precision. Each smoother takes
`by=`, and drops rows missing x or y.

Penalized regression follows glmnet: lasso, ridge and elastic-net paths for
the gaussian, binomial, poisson, multinomial and Cox families, and
`cv_glmnet` to choose lambda.

```python
fit = glmnet(mtcars, "mpg ~ .", alpha=0.5)        # a formula, as glmnetUtils; or glmnet(x=X, y=y)
tidy(fit)                                       # term, step, estimate, lambda, dev_ratio (broom)
coef(fit, s=0.1)                                # interpolated along the path, as coef.glmnet
predict(fit, new_rows, s=[1, 0.1], type="response")
cv = cv_glmnet(mtcars, "am ~ wt + hp + qsec", family="binomial", foldid="fold", type_measure="auc")
glance(cv)                                      # lambda_min, lambda_1se
fit = glmnet(lung, "Surv(time, status) ~ age + sex + ph_ecog", family="cox", cox_ties="efron")
```

Model-matrix columns come from the formula code without the intercept column
(factors get treatment contrasts, as glmnetUtils' `use.model.frame=TRUE`).
The lambda sequence (`nlambda`, `lambda_min_ratio`, or `lambda_=`),
standardization, `intercept`, `penalty_factor`, `exclude`, lower/upper limits,
weights, offsets, `dfmax` and glmnet's early stopping rules follow glmnet 5.1.
Where glmnet stops coordinate descent at `thresh`, stats3 solves each lambda
exactly (Newton steps on the exact likelihood, with coordinate descent to find
the active set and a linear solve on it), so `thresh` is accepted but not
needed; R's coefficients agree to its own convergence (about 1e-8 relative to
the largest coefficient for gaussian fits at a tight `thresh`). `glance` gives
`nulldev`, `npasses` (stats3's own passes, not glmnet's) and `nobs`.
`cv_glmnet` matches R exactly when given the same `foldid` (a column name or
one fold per row); without it, folds are drawn with NumPy (`seed=`). Grouped
multinomial (`type.multinomial="grouped"`), `mgaussian`, start-stop and
stratified Cox responses, relaxed fits and `exact=TRUE` predictions are not
included.

Linear mixed models follow lme4's `lmer`, fitted by REML (or `reml=False`
for maximum likelihood) over the profiled deviance, as lme4 does:

```python
sleep = pl.read_csv("tests/data/sleepstudy.csv")
fit = lmer(sleep, "Reaction ~ Days + (Days || Subject)")
tidy(fit)                            # fixed effects, then sd__ rows per variance component
ranef(fit)                           # conditional modes with their conditional SDs
predict(fit, new_rows)               # with the random effects; re_form="population" without
predict(fit, new_rows, allow_new_levels=True)
anova(lmer(sleep, "Reaction ~ Days + (1 | Subject)"), fit)   # likelihood ratio, refitted with ML
```

Random effects are `(1 | g)`, `(0 + x | g)`, `(1 + x | g)` for a correlated
intercept and slope (any number of columns, factors included), `(x || g)`
for an independent intercept and slope, crossed `(1 | g1) + (1 | g2)`,
`(1 | g1:g2)` and nested `(1 | g1/g2)`. `tidy(fit)` is broom.mixed's table: fixed
terms (`effect="fixed"`, t statistics without p-values or degrees of freedom,
as `summary.merMod`) and the standard deviations and correlations as
`effect="ran_pars"` rows (`sd__Days`, `cor__(Intercept).Days`);
`effects="ran_vals"` gives the conditional modes. `glance` has `logLik`,
`AIC`, `BIC`, `REMLcrit`, `deviance` and `sigma`; `augment` adds `.fitted`,
`.resid`, `.hat`, `.cooksd` and `.fixed`; `vcov` is the fixed effects'
covariance, and `anova(fit)` lme4's F table without p-values. Weights,
offsets, grouped fits, `na_action` and `modelsummary` work as for `lm`.
The optimizer finds the optimum of lme4's criterion more precisely than
lme4's default stopping rule does (see the tests for the tolerances), and
the q x q system is factored densely up to 1,500 random-effect columns and
by sparse LU above that, so a few thousand group levels fit in under a second.

`na_action="exclude"` (R's `na.exclude`) keeps every row in `augment` and
`predict`, with missing values where a row was not used, so the results line
up with the data.

Formulas follow R: `a * b`, `a:b`, `(a + b)^2`, `0 +` / `- 1`, `.`, `log(x)`,
`I(x^2)`, `factor(cyl)`, `offset(...)`, and `cbind(successes, failures)` for
binomial models. Factors use treatment contrasts with the first level as the
reference (a pandas category or Polars `Enum` sets the order), with R's
column names (`contrasts="sum"` gives R's `contr.sum`, as type III tests
want), and redundant columns are reported as aliased, as in R. For a `glm`,
`tidy(conf_int=True)` gives R's profile-likelihood intervals (`confint`);
`conf_method="wald"` gives `confint.default`.
Families: `gaussian`, `binomial`, `poisson`, `Gamma`, `inverse_gaussian`,
`quasibinomial`, `quasipoisson`, with R's links. `fit >> tidy()` works too.

tidy3 has a `tidy()` of its own, for data. stats3's `tidy` hands data to
it, and from tidy3 0.8.3 tidy3's `tidy` hands models to stats3 (0.1.1 or
later), so either import order works. With tidy3 0.8.2 or earlier, import stats3 after tidy3
(or call `stats3.tidy(fit)`).

The engine is stats3's own: weighted least squares by QR for `lm`, and a
port of R's `glm.fit` (same starting values, convergence rule and step
halving) for `glm`. On 1,000,000 rows with 13 coefficients `lm` takes about
0.16 s and a Poisson `glm` about 0.6 s, against about 1.1 s and 1.9 s for
statsmodels' formula interface.

## Survival analysis

```python
from stats3 import coxph, cox_zph, survdiff, survfit, survreg, tidy, glance, predict

km = survfit(lung, "Surv(time, status) ~ sex")     # Kaplan-Meier, log intervals
tidy(km)                                            # one row per time per stratum
glance(km)                                          # medians and their intervals
glance(survdiff(lung, "Surv(time, status) ~ sex"))  # log-rank; rho=1 for Peto-Peto

fit = coxph(lung, "Surv(time, status) ~ age + sex + strata(inst)")
tidy(fit, conf_int=True, exponentiate=True)         # hazard ratios
glance(fit)                                         # LR, Wald, score tests, concordance
cox_zph(fit)                                        # proportional-hazards test
survfit(fit, newdata=new_rows)                      # predicted survival curves

survreg(lung, "Surv(time, status) ~ age + sex", dist="weibull")
survreg(lung, 'Surv(lower, upper, type="interval2") ~ age')  # interval-censored
survfit(lung, 'Surv(lower, upper, type="interval2") ~ sex')  # Turnbull's estimator
```

These follow R's survival package. The response is `Surv(time, status)`
(status 0/1, 1/2 or logical, or a comparison such as `status == 2`) or
`Surv(start, stop, status)` for counting-process data. `coxph` uses Efron
ties by default (`ties="breslow"` or `"exact"`), takes `strata()`,
weights and offsets, and gives robust standard errors with `cluster(id)`,
`robust=True` or non-integer weights, as R does. Its Newton steps,
step halving and stopping rule are R's, so coefficients agree to rounding.
`predict` gives `type="lp"`, `"risk"` or `"expected"` (with R's
stratum-centred linear predictors), and `anova` gives sequential tables and
nested comparisons. `survreg` fits Weibull, exponential, lognormal,
loglogistic, gaussian, logistic and extreme-value models; `predict` gives
the response, linear predictor and quantiles. `by=` fits one model per
group, and `na_action="exclude"` lines predictions up with the data.

Left- and interval-censored data take R's `Surv` forms:
`Surv(time, status, type="left")`, `Surv(lower, upper, type="interval2")`
(a missing bound is unbounded; both missing is a missing row) and
`Surv(time, time2, event, type="interval")` (event 0 right-censored, 1
exact, 2 left-censored, 3 interval-censored). `survreg` fits them by R's
likelihood, starting values and Newton steps, so its coefficients and
iteration counts are R's; `survfit` gives Turnbull's estimator by R's
self-consistency iteration (with its Aitken acceleration and stopping rule,
so the curve is R's, short of the limit as R's is). `coxph` and `survdiff`
refuse them, as R does. Kaplan-Meier curves with non-integer weights use
R's robust (infinitesimal jackknife) variance, with R's `std.err` on the
survival scale.

Not included yet: multi-state models, `tt()` terms, frailty, `survreg`
strata and robust variances, and `survfit` on a stratified Cox model with
new data. Times are compared exactly (R's `timefix` rounding is not applied).

### Generalized additive models

`gam()` is mgcv's `gam`: penalized regression smooths with their smoothing
parameters chosen by GCV (UBRE for binomial and Poisson), the default, or
`method="REML"`.

```python
fit = gam(penguins, "body_mass_g ~ s(flipper_length_mm) + s(bill_depth_mm, bs='cr') + species",
          method="REML")
tidy(fit)                    # parametric terms, as summary.gam
tidy(fit, parametric=False)  # smooths: edf, ref_df, F (or Chi.sq) and p-value
```

```text
┌──────────────────────┬──────────┬──────────┬───────────┬─────────┐
│ term                 ┆ edf      ┆ ref_df   ┆ statistic ┆ p_value │
╞══════════════════════╪══════════╪══════════╪═══════════╪═════════╡
│ s(flipper_length_mm) ┆ 1.366226 ┆ 1.640748 ┆ 39.397472 ┆ 0.0     │
│ s(bill_depth_mm)     ┆ 2.588033 ┆ 3.266838 ┆ 32.547571 ┆ 0.0     │
└──────────────────────┴──────────┴──────────┴───────────┴─────────┘
```

Smooths are `s()` with `bs="tp"` (thin plate regression splines, the
default, with mgcv's eigen-truncation and, past 2000 unique values, its
knot sample from R's random number generator), `"cr"`, `"cs"`, `"ps"` or
`"re"`, with `k=`, `m=`, `fx=` and `by=` (a factor gives one smooth per
level; a number multiplies the smooth); `te()` builds tensor products of
those marginals. The bases, penalty scaling and centring constraints are
ports of mgcv's, and the families are `glm`'s, with weights and offsets.
Every trial set of smoothing parameters is fitted to convergence (mgcv's
outer iteration, with full Newton weights for non-canonical links).
`glance` gives the adjusted R², deviance explained, GCV/UBRE/REML score,
scale, AIC and BIC (with mgcv's smoothing-parameter-corrected degrees of
freedom under REML); `augment` and `predict(type="link"/"response"/
"terms", se_fit=True)` give fits and standard errors; `anova(fit)` is
`anova.gam`'s Wald table and `anova(fit1, fit2)` its deviance comparison;
`vcov(fit)` is the Bayesian covariance `Vp` (`"Ve"` frequentist, `"Vc"`
with smoothing parameter uncertainty). broom's `tidy.gam` defaults to the
smooth table; stats3's `tidy` keeps to its other models and gives the
parametric table unless `parametric=False`.

mgcv stops its optimizers early; stats3 optimizes the same criteria to full
precision, and agrees with mgcv run with tight tolerances to about 1e-7.
Thin plate basis columns match mgcv's up to sign (LAPACK's eigenvector
signs), so compare their coefficients up to sign; fitted values, degrees of
freedom, smoothing parameters and tests do not depend on it. Two mgcv
behaviours rest on rounding: `cs`'s shrinkage of the cubic spline
penalty's null space depends on which null eigenvectors LAPACK returns
first, so mgcv's own `cs` fits differ between LAPACK builds (stats3 and R
agree on macOS; on Linux the tests check only the parts that do not depend
on the choice), and Gaussian GCV fits end with a probe towards infinite smoothing
parameters in the direction of the gradient's sign, which at a converged
optimum is noise. Usually both agree with mgcv; when they don't, the fits
differ. `ti()`, `t2()`, `id=`, `gam.side` constraints for nested smooths,
`select=`, other bases and `method="ML"`/`"P-REML"` are not included.

## Multivariate exploration

```python
cols = ["Murder", "Assault", "UrbanPop", "Rape"]
pca = prcomp(usarrests, "~ Murder + Assault + UrbanPop + Rape", scale_=True, labels="state")
tidy(pca, matrix="pcs")         # broom: "x" scores, "rotation" loadings, "pcs" variance explained
fa = factanal(mtcars, ["mpg", "disp", "hp", "drat", "wt", "qsec"], 2, rotation="varimax")
glance(fa)                      # the chi-squared test that two factors suffice
tree = usarrests >> dist(cols, labels="state") >> hclust("average")
cutree(tree, k=4)               # item, cluster
km = kmeans(usarrests, cols, usarrests.head(4))  # given centres; tidy, glance, augment as broom
```

Base R's `prcomp`, `princomp`, `factanal` (with `varimax` and `promax`),
`dist`, `hclust` with `cutree`, `cmdscale` and `kmeans`, each with `by=`.
`dist` returns broom's distance table (`item1`, `item2`, `distance`), which
`hclust` and `cmdscale` take, grouping columns and all. The algorithms are
ports of R's own code (hclust.f, kmns.f, kmeans.c, distance.c), so the merge
matrix, heights and leaf order of `hclust`, ties included, and k-means from
given centres are R's exactly. Sums of squares round as R's build for the
machine computes them (fused on 64-bit ARM, not on x86-64), since the last
bit decides which tied merge comes first. With `stats3[fast]` these run
compiled (k-means on 100k rows of 50 columns in under a second, as R);
without it, k-means compares rows on bounded fast distances and settles
only ties with R's exact sum, and `dist` emulates the fused rounding in
NumPy. Random k-means starts use NumPy's generator,
so with `centers=k` only the objective is comparable with R. `factanal` is
R's maximum-likelihood fit over the uniquenesses, run to full precision.
Component signs come from the LAPACK routines R calls (dgesdd for `prcomp`,
dsyevr for `cmdscale`); they match R's here, but another LAPACK may flip a
component, scores and loadings together. `princomp` and `factanal` fix
their signs as R does.
| `nls`, `confint.nls` | `nls("y ~ a * exp(-b * x)", start=...)`, `SSlogis` and other self-starting models, `algorithm="plinear"` / `"port"` |
| `loess`, `lowess`, `smooth.spline`, `ksmooth` | `loess("y ~ x1 + x2", span=0.75)`, `lowess(x, y)`, `smooth_spline(x, y, df=)`, `ksmooth(x, y, "normal")` |
| `acf`, `pacf`, `ccf` | `acf("x", lag_max=, type=)`, `pacf("x")`, `ccf("x", "y")`: one row per lag, with the white-noise bounds |
| `Box.test` | `box_test("x", lag=10, type="Ljung-Box", fitdf=)`, or `box_test(fit, ...)` on an arima fit's residuals |
| `arima`, `predict.Arima` | `arima("x", order=(p, d, q), seasonal=(P, D, Q), period=12, xreg=)`, `predict(fit, n_ahead=12)` |

## Time series

A series is a column, in row order or ordered by `order_by=`; `by=` gives
one series per group, so one call fits every series in a long frame.

```python
acf(sales, "units", order_by="month", by="store")      # lag, acf, conf_low, conf_high
box_test(sales, "units", lag=12, type="Ljung-Box", by="store")
fit = arima(sales, "units", order=(0, 1, 1), seasonal=(0, 1, 1), period=12,
            order_by="month", by="store")
tidy(fit)                  # coefficients with R's standard errors
glance(fit)                # sigma2, logLik, AIC, AICc, BIC, nobs
predict(fit, n_ahead=12)   # horizon, fit, se_fit, for every store
box_test(fit, lag=24, type="Ljung-Box", fitdf=2)   # residual check, as tsdiag
```

`arima` ports R's: the exact likelihood by Kalman filter on the state-space
form of `makeARIMA`, R's diffuse start for differencing, its CSS start
(`method="CSS-ML"`), its AR parameter transformation, `fixed=`, `xreg=`,
and missing values inside the series. R's `optim` stops early (its gradient
is a finite difference); stats3 finds the optimum to full precision, so its
log likelihood is never below R's, and estimates agree with R's to about
1e-4. Lags count observations (R's lag times the frequency), and missing
values follow `na_action` as in R: `"fail"` by default, or `"pass"`,
`"omit"` or `"contiguous"`.
| survival: `survfit`, `survdiff` | `survfit("Surv(time, status) ~ g")` (Kaplan-Meier), `survdiff("Surv(time, status) ~ g", rho=0)` |
| survival: `coxph`, `cox.zph` | `coxph("Surv(time, status) ~ x + strata(s) + cluster(id)")`, `cox_zph(fit)`, `survfit(fit, newdata)` |
| survival: `survreg` | `survreg("Surv(time, status) ~ x", dist="weibull")` |
| glmnet | `glmnet("y ~ .", family=, alpha=)`, `cv_glmnet(..., foldid=)`, `coef(fit, s=)` |
| mgcv::gam | `gam("y ~ s(x) + te(x, z) + f", method="REML")` |

## What's in it

| R | stats3 |
|---|---|
| `t.test` | `t_test("y ~ g")`, `t_test(x, mu=)`, `t_test(x, y, paired=True)` |
| `wilcox.test` | `wilcox_test(...)`, same forms; R 4.6's exact tests with ties |
| `kruskal.test`, `friedman.test`, `quade.test` | `kruskal_test("y ~ g")`, `friedman_test("y ~ g \| block")`, `quade_test("y ~ g \| block")` |
| `ansari.test`, `mood.test` | `ansari_test("y ~ g")`, `ansari_test(x, y, conf_int=True)`: exact or normal, with the ratio of scales; `mood_test("y ~ g")`, with ties as in R 4.x |
| `aov`, `TukeyHSD` | `aov("y ~ a * b")`, `aov("y ~ x + a")`: `summary(aov())`'s sequential table; `tukey_hsd("y ~ a * b", which=, ordered=)`: per-term comparisons, interactions included (no `Error()` strata) |
| `oneway.test` | `oneway_test("y ~ g")`: Welch's ANOVA, or `var_equal=True` |
| `pairwise.t.test`, `pairwise.wilcox.test`, `pairwise.prop.test` | `pairwise_t_test("y ~ g")`, `pairwise_wilcox_test("y ~ g")`, `pairwise_prop_test(x, n)`: one row per pair, `p_adjust_method=` |
| `cor.test` | `cor_test(x, y, method="pearson" / "spearman" / "kendall")` |
| `chisq.test` | `chisq_test(x, y)`, `chisq_test(x, p=[...])` |
| `fisher.test` | `fisher_test(x, y)`: odds ratio for 2 x 2, exact p-value for larger tables |
| `mcnemar.test`, `mantelhaen.test` | `mcnemar_test(x, y)`, `mantelhaen_test(x, y, strata)`, `exact=True` for 2 x 2 x K |
| `prop.test`, `prop.trend.test`, `binom.test` | `prop_test(x, n)`, `prop_trend_test(x, n, score=)`, `binom_test(sex == "male")` |
| `poisson.test` | `poisson_test(events, time)`: one rate, or the ratio of two |
| `shapiro.test`, `ks.test` | `shapiro_test(x)`, `ks_test(x, "pnorm", mean=, sd=)`, `ks_test(x, y)` |
| `bartlett.test`, `fligner.test`, `var.test`, `car::leveneTest` | `bartlett_test`, `fligner_test`, `var_test`, `levene_test` |
| `p.adjust` | `p_adjust(p, "BH")`: on a vector or Series, or a column in `with_columns` / `mutate` |
| effectsize | `cohens_d`, `hedges_g`, `eta_squared` |
| `power.t.test`, `power.prop.test`, `power.anova.test` | `power_t_test`, `power_prop_test`, `power_anova_test` |
| `lm`, `glm` | `lm`, `glm`, with broom's `tidy`, `glance`, `augment`, and `anova`, `predict` |
| `lm` with `cbind()`, `manova`, `anova.mlm`, `mauchly.test` | `lm("cbind(y1, y2) ~ x")`, `manova(...)`, `anova(fit, test="Spherical", X="~ 1")`, `mauchly_test(fit, X="~ 1")` |
| `car::Anova` | `anova(fit, type=2)`, `anova(fit, type=3)` |
| sandwich, `lmtest::coeftest` | `tidy(fit, vcov="HC3")`, `vcov_cl(...)`, `vcov_nw()`, `vcov(fit)` |
| modelsummary | `modelsummary(fit1, fit2)`: a frame, Markdown or HTML |
| emmeans | `emmeans(fit, "species")`, `pairs(emm)`, `contrast(emm, ...)` |
| MASS | `glm_nb("y ~ x")`, `glm("y ~ x", family=negative_binomial(theta))` |
| MASS::polr | `polr("y ~ x", method="logistic")` |
| nnet::multinom | `multinom("y ~ x")` |
| lme4::lmer | `lmer("y ~ x + (x \| g)")`, `(x \|\| g)`, `ranef(fit)`, `predict(fit, re_form=...)` |
| MASS::rlm | `rlm("y ~ x", psi="huber")` |
| quantreg::rq | `rq("y ~ x", tau=0.5, se="rank")` |
| `prcomp`, `princomp`, `factanal` | `prcomp("~ a + b", scale_=True)`, `princomp(cols, cor=True)`, `factanal(cols, 2, rotation="promax")`; `varimax`, `promax` |
| `dist`, `hclust`, `cutree`, `cmdscale`, `kmeans` | `dist(cols) >> hclust("average")`, `cutree(fit, k=3)`, `dist(cols) >> cmdscale(2)`, `kmeans(cols, centers)` |

## Bayesian statistics

```python
fit = bayes_lm(penguins, "body_mass_g ~ flipper_length_mm + species", seed=42)
tidy(fit, conf_int=True)              # posterior means, SDs and credible intervals
predict(fit, interval="prediction")  # includes uncertainty in future observations
hypothesis(fit, {"flipper_length_mm": 1}, rope=(-1, 1))
tidy(pairs(bayes_emmeans(fit, "species")))
```

`bayes_lm` uses a proper normal-inverse-gamma prior and exact inference;
it needs only the base dependencies. `prior_scale` is conditional on the
residual SD. Defaults and parameterizations are explicit in the
[Bayesian guide](docs/bayesian.md).

Install `pip install "stats3[bayes]"` for sampled models:

```python
fit = bayes_glm(
    penguins, "body_mass_g ~ flipper_length_mm + species + (1 | island)",
    priors={"flipper_length_mm": prior("normal", mu=0, sigma=50)},
    draws=1000, tune=1000, chains=4, seed=42,
)
posterior_summary(fit)
bayes_diagnostics(fit)                # R-hat, ESS, MCSE, divergences and BFMI
posterior_predict(fit, seed=42)
pp_check(fit, statistic="sd", seed=42)
```

The sampled engine supports Gaussian, Student-t, binomial, Poisson,
negative binomial, gamma, lognormal, exponential, ordinal and multinomial
models, plus varying intercepts/slopes: `(1 + x | g)` correlated under an
LKJ prior (named as brms names them in `tidy()`), `(1 + x || g)`
independent, multinomial ones per category, with `bayes_ranef` group-level
summaries. It uses optional PyMC 6
and ArviZ 1 dependencies. All four data backends, formulas, offsets, grouped
fits and missing-row alignment carry through. `posterior_draws`,
`posterior_epred`, `prior_predict`, `bayes_emmeans` (category probabilities
for ordinal and multinomial fits), contrasts and ROPE
probabilities expose posterior uncertainty; `bayes_compare` offers LOO,
WAIC and exact Gaussian model evidence. `bayes_binom_test` and
`bayes_poisson_test` give exact conjugate scalar inference and log Bayes factors.

This is a first Bayesian implementation. Custom likelihoods,
survival/time-series models and general model-evidence estimation remain
outside its scope.

## How it works, and how it is checked

Most tests need only a few numbers per group: counts, means, variances,
rank sums, cross-tabulated counts. Polars computes them for every group in
one lazy query, and SciPy's distribution functions (Polars has none) turn
them into p-values for all groups at once. Over 1,000 groups of 1,000 rows
most tests run 4-30 times faster than calling SciPy once per group. Tests
that need every value (Shapiro-Wilk, Kendall's tau, the Ansari-Bradley
interval) run per group. stats3
needs only Polars and SciPy.

`tests/r_oracle_cases.R` runs the same tests in R; `pytest` compares every
column, to 1e-9 relative, on Polars, lazy Polars, pandas and tidy3 input.
Marginal means and contrasts are also checked against R's emmeans, and mixed
models against lme4 (to 1e-7 against lme4's own deviance function minimized
to 1e-9 in theta, and to the accuracy of lme4's default optimizer, about
5e-4 on weakly identified variance components, against `lmer()` as called).
Where R's own numerics are approximate, stats3 either follows R's
algorithm (Tukey's distribution is a port of `ptukey` and `qtukey`, and
`fisher.test`'s odds ratio and interval come from the same `uniroot`
searches at R's tolerance, so both match R closely) or computes to full
precision and the test compares at R's accuracy (Dunnett's quantile
approximation, the Ansari-Bradley interval, the noncentral t above ncp
37.62, the asymptotic Kolmogorov series, the exact Mantel-Haenszel odds
ratio), with the reason written next to the tolerance. Without R, the
SciPy comparisons still run.

The algorithms that walk the rows one by one (k-means transfers, the
nearest-neighbour chain of `hclust`, distances, the concordance sweep) run
in NumPy, and compiled when `stats3[fast]` (Numba) is installed: the
kernels do the same operations in the same order, with the fused
multiply-add R's compiled code uses on ARM, so the results are the same
bits either way and only the speed differs. `pytest` checks the two paths
against each other.

```bash
pip install -e ".[dev]"
pytest
```
