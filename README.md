# stats3

**R's statistical tests on Polars, for Polars, pandas and tidy3 data.**
Write a test the way R users do, run it once per group, and get R's numbers
back as a table in the same kind of frame you passed in.

```bash
pip install stats3              # Polars data
pip install "stats3[pandas]"    # pandas data too
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

`rlm()` follows MASS's default M-estimation with Huber, bisquare or Hampel
scores, MAD or Huber proposal-2 scale, inverse-variance or case weights,
and `XtX` or `XtWX` coefficient covariance. It reports R's t statistics
and normal confidence intervals; p-values remain missing because
`summary.rlm` supplies none. MM/LTS initialization is not included.

`rq()` minimizes weighted pinball loss with a sparse linear program. It
fits one `tau` at a time and uses quantreg's `nid` inference by default, or
`se="iid"`, with t tests and t intervals. This differs from R's small-sample
default of rank intervals; rank/bootstrap inference is not included.
`anova(full_fit, reduced_fit)` gives Wald tests for nested models with the
same tau. Offsets are rejected; fit an explicitly adjusted response instead.
For nonunique optima, the LP solver may choose different coefficients from R.
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
```

As in MASS, P(Y <= k) = F(zeta_k - x'beta - offset), with the logistic,
probit, loglog or cloglog link (`method=`). The response needs three or more
levels, ordered by a Polars `Enum`, a pandas category or `factor()`. Weights,
offsets, groups and `na_action` work as for `glm`. Standard errors come from
the Hessian, as `summary.polr`; p-values stay missing, as R reports none, and
`conf_int=True` gives Wald intervals rather than `confint.polr`'s profile
likelihood. The cauchit link, robust covariance and marginal means for
ordinal fits are not included yet.

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

Not included yet: left- and interval-censored data, multi-state models,
`tt()` terms, frailty, `survreg` strata and robust variances, `survfit`
on a stratified Cox model with new data, and Kaplan-Meier curves with
non-integer weights (R's robust variance). Times are compared exactly
(R's `timefix` rounding is not applied).

## What's in it

| R | stats3 |
|---|---|
| `t.test` | `t_test("y ~ g")`, `t_test(x, mu=)`, `t_test(x, y, paired=True)` |
| `wilcox.test` | `wilcox_test(...)`, same forms; R 4.6's exact tests with ties |
| `kruskal.test`, `friedman.test` | `kruskal_test("y ~ g")`, `friedman_test("y ~ g \| block")` |
| `aov`, `TukeyHSD` | `aov("y ~ g")`, `tukey_hsd("y ~ g")` (one grouping column) |
| `oneway.test` | `oneway_test("y ~ g")`: Welch's ANOVA, or `var_equal=True` |
| `pairwise.t.test`, `pairwise.wilcox.test`, `pairwise.prop.test` | `pairwise_t_test("y ~ g")`, `pairwise_wilcox_test("y ~ g")`, `pairwise_prop_test(x, n)`: one row per pair, `p_adjust_method=` |
| `cor.test` | `cor_test(x, y, method="pearson" / "spearman" / "kendall")` |
| `chisq.test` | `chisq_test(x, y)`, `chisq_test(x, p=[...])` |
| `fisher.test` | `fisher_test(x, y)`: odds ratio for 2 x 2, exact p-value for larger tables |
| `mcnemar.test`, `mantelhaen.test` | `mcnemar_test(x, y)`, `mantelhaen_test(x, y, strata)`, `exact=True` for 2 x 2 x K |
| `prop.test`, `binom.test` | `prop_test(x, n)`, `binom_test(sex == "male")` |
| `poisson.test` | `poisson_test(events, time)`: one rate, or the ratio of two |
| `shapiro.test`, `ks.test` | `shapiro_test(x)`, `ks_test(x, "pnorm", mean=, sd=)`, `ks_test(x, y)` |
| `bartlett.test`, `fligner.test`, `var.test`, `car::leveneTest` | `bartlett_test`, `fligner_test`, `var_test`, `levene_test` |
| `p.adjust` | `p_adjust(p, "BH")`: on a vector or Series, or a column in `with_columns` / `mutate` |
| effectsize | `cohens_d`, `hedges_g`, `eta_squared` |
| `power.t.test`, `power.prop.test`, `power.anova.test` | `power_t_test`, `power_prop_test`, `power_anova_test` |
| `lm`, `glm` | `lm`, `glm`, with broom's `tidy`, `glance`, `augment`, and `anova`, `predict` |
| `car::Anova` | `anova(fit, type=2)`, `anova(fit, type=3)` |
| sandwich, `lmtest::coeftest` | `tidy(fit, vcov="HC3")`, `vcov_cl(...)`, `vcov_nw()`, `vcov(fit)` |
| modelsummary | `modelsummary(fit1, fit2)`: a frame, Markdown or HTML |
| emmeans | `emmeans(fit, "species")`, `pairs(emm)`, `contrast(emm, ...)` |
| MASS | `glm_nb("y ~ x")`, `glm("y ~ x", family=negative_binomial(theta))` |
| MASS::polr | `polr("y ~ x", method="logistic")` |
| MASS::rlm | `rlm("y ~ x", psi="huber")` |
| quantreg::rq | `rq("y ~ x", tau=0.5, se="nid")` |
| survival: `survfit`, `survdiff` | `survfit("Surv(time, status) ~ g")` (Kaplan-Meier), `survdiff("Surv(time, status) ~ g", rho=0)` |
| survival: `coxph`, `cox.zph` | `coxph("Surv(time, status) ~ x + strata(s) + cluster(id)")`, `cox_zph(fit)`, `survfit(fit, newdata)` |
| survival: `survreg` | `survreg("Surv(time, status) ~ x", dist="weibull")` |

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
models, plus independent varying intercepts/slopes. It uses optional PyMC 6
and ArviZ 1 dependencies. All four data backends, formulas, offsets, grouped
fits and missing-row alignment carry through. `posterior_draws`,
`posterior_epred`, `prior_predict`, `bayes_emmeans`, contrasts and ROPE
probabilities expose posterior uncertainty; `bayes_compare` offers LOO,
WAIC and exact Gaussian model evidence. `bayes_binom_test` and
`bayes_poisson_test` give exact conjugate scalar inference and log Bayes factors.

This is a first Bayesian implementation. Correlated varying effects,
multinomial varying effects, categorical marginal means, custom likelihoods,
survival/time-series models and general model-evidence estimation remain
outside its scope. Frequentist multinomial and mixed models are still
pending; Bayesian work now takes priority over those remaining tier 2 items.

## How it works, and how it is checked

Most tests need only a few numbers per group: counts, means, variances,
rank sums, cross-tabulated counts. Polars computes them for every group in
one lazy query, and SciPy's distribution functions (Polars has none) turn
them into p-values for all groups at once. Over 1,000 groups of 1,000 rows
most tests run 4-30 times faster than calling SciPy once per group. Tests
that need every value (Shapiro-Wilk, Kendall's tau) run per group. stats3
needs only Polars and SciPy.

`tests/r_oracle_cases.R` runs the same tests in R; `pytest` compares every
column, to 1e-9 relative, on Polars, lazy Polars, pandas and tidy3 input.
Marginal means and contrasts are also checked against R's emmeans.
Where R's own numerics are approximate (Tukey's distribution, Dunnett's
quantile approximation, intervals
found with `uniroot`, the noncentral t above ncp 37.62, the asymptotic
Kolmogorov series, the exact Mantel-Haenszel odds ratio), stats3 computes to full precision and the test compares
at R's accuracy, with the reason written next to the tolerance. Without R,
the SciPy comparisons still run.

```bash
pip install -e ".[dev]"
pytest
```
