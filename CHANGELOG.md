# Changelog

All notable changes to stats3. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may change behaviour, and those changes are listed
under **Changed**.

## Unreleased

### Added

- `aov` and `tukey_hsd` take several terms: `aov("y ~ a + b")`, `aov("y ~ a * b")` and covariates (`aov("y ~ x + a")`) give `summary(aov())`'s sequential (type I) table, fitted with the `lm` engine; `tukey_hsd` gives `TukeyHSD`'s table for every factor term, interactions included, with R's `which=` and `ordered=`. Unbalanced designs use R's `model.tables` means (later terms adjusted for earlier ones) and Tukey-Kramer intervals, and empty interaction cells give missing comparisons, as in R. `Error()` strata raise a clear error. The one-factor forms are unchanged.
- `rlm(..., method="MM")`: MASS's MM-estimation, starting from `lqs`'s S-estimate, and `rlm(..., init="lts")` (or given coefficients) to start M-estimation from least trimmed squares. `lqs`'s subset search is MASS's own: exhaustive below 5000 subsets, otherwise random subsets from R's Mersenne-Twister stream, so `seed=` reproduces `set.seed()` in R. Checked against MASS on all four backends, including weights, case weights, custom bisquare tuning and offsets.
- `rq(..., se="rank")`: quantreg's rank-inversion confidence intervals, with its iid and nid (`iid=False`) variants; unbounded sides are infinite. `rq(..., se="boot")`: quantreg's xy-pair bootstrap with `r` replications and `seed=`, drawing the same resamples as `set.seed(seed)` in R. Both checked against quantreg on all four backends.
- `rq()` accepts offsets, in the formula or `offset=`, and subtracts them from the response. quantreg 6.1 ignores `offset()` terms; stats3 is checked against R's fit of the adjusted response.
- `bayes_glm` varying effects are complete. `(1 + x | g)` now estimates correlated intercepts and slopes, with an LKJ prior on their correlation (`LKJ(2)` by default; `priors={"group_cor": prior("lkj", eta=...)}`) and a noncentred Cholesky parameterization; `(1 + x || g)` keeps independent effects. Multinomial fits accept varying effects, one set per non-reference category, as brms's `categorical()` does. `tidy()` adds the group SDs and correlations under brms's names (`sd_g__Intercept`, `cor_g__Intercept__x`), and the new `bayes_ranef()` summarizes group-level deviations with credible intervals. `bayes_emmeans` gives ordinal and multinomial category probabilities per reference-grid cell, computed draw by draw, with contrasts within each category and emmeans's latent scale for `type="link"`. Checked against independent log-density formulas, simulated-parameter recovery, and brms fits with the same priors.
- `ansari_test("y ~ g")` and `ansari_test(x, y)`: R's `ansari.test` of equal scale, with its exact distribution for samples under 50 without ties (or `exact=`), the normal approximation with R's tie correction otherwise, and `conf_int=True` for R's estimate and interval of the ratio of scales, exact or asymptotic. As in R, ties with small samples give the normal p-value with a warning, and an exact interval with ties is an error (`exact=False` gives the asymptotic one). The asymptotic estimate and bounds pick the root R's `uniroot` picks, then locate that jump of the step function to 1e-12 rather than R's 1e-4.
- `mood_test("y ~ g")` and `mood_test(x, y)`: R's `mood.test` of scale on the finite values, with R 4.x's tie handling.
- `quade_test("y ~ g | b")`: R's `quade.test` for unreplicated complete blocks, with the design checks and missing values of `friedman_test`, and R's degenerate p-value `k!^(1-b)` when F is undefined.
- `prop_trend_test(x, n, score=)`: R's `prop.trend.test` chi-squared test for trend in proportions; `score` is a column or numbers, 1, 2, 3, ... by default.
- All four run once per `by=` group on Polars, lazy Polars, pandas and tidy3 data, and take bare names in notebooks. Checked against R 4.6.1 on all four backends.
- `polr()` gaps closed: `tidy(fit, conf_int=True)` now gives MASS's profile likelihood intervals (`confint.polr`, slopes only), with `conf_method="wald"` for the previous Wald intervals; `method="cauchit"` (with MASS's ±100 clamp of the latent bounds, which moves the cauchit optimum); `tidy(fit, vcov=vcov_cl(...))` and `vcov(fit, vcov_cl(...))` reproduce sandwich's `vcovCL` (HC0/HC1, multi-way, `cadjust`), tested as `coeftest`, while `vcovHC`, `NeweyWest`, clustered HC2/HC3 and loglog fits raise R's errors; `emmeans(fit, "x", mode=...)` gives emmeans' `"latent"`, `"prob"` and `"mean.class"` scales, with `pairs` and `contrast`, cell weights from the prior weights, offsets and robust covariance. Checked against MASS, sandwich and emmeans on all four backends; loglog probabilities are validated through the equivalent cloglog fit of the reversed response, since emmeans 2.0.4 mishandles loglog.

### Changed

- `rq()`'s `se=` now defaults to quantreg's rule: rank-inversion intervals below 1001 rows, `nid` otherwise. Small fits' coefficient tables therefore have intervals but no standard errors or p-values, as in `summary.rq`; pass `se="nid"` for the previous behaviour. `vcov()`, prediction intervals and marginal means of rank fits still use `nid` covariance, as in quantreg.
- The `bayes` extra now needs PyMC 6 (`pymc>=6.3,<7`, with PyTensor 3) and ArviZ 1 (`arviz>=1.1,<2`). PyTensor 2, which PyMC 5 requires, cannot compile its C extensions with Apple clang 21 (Xcode 27): it passes the `-ld64` linker flag that this clang removed, so every sampled `bayes_glm` fit failed on current macOS toolchains without a local compiler wrapper. PyTensor 3 chooses the flag by the compiler's version instead, and runs the model graph through Numba by default, so the C++ compiler only builds PyTensor's small bootstrap module. `bayes_compare(method="loo")` still uses ArviZ's PSIS-LOO; ArviZ 1 dropped WAIC, so `method="waic"` is now computed by stats3 from the pointwise log likelihoods with the same definition, and both methods return the same table as before (rank, `elpd_*`, `p_*`, `elpd_diff`, stacking `weight`, `se`, `dse`, `warning`). As in ArviZ 1, `se` and `dse` now use the sample variance of the pointwise values (ArviZ 0.x used the population variance, a factor of `sqrt(n/(n-1))` smaller). A sampled fit's `idata` is now an xarray `DataTree`, as PyMC 6 returns.

## 0.2.0 — 2026-10-24

Models beyond lm and glm: estimated marginal means, robust standard errors, side-by-side model tables, negative binomial, robust and quantile regression, ordinal regression, and a first Bayesian layer with exact and sampled models.

### Added

- Expanded Bayesian validation: all ten sampled likelihoods now run through all four data backends, with independent SciPy checks of pointwise log likelihoods, response means and predictive distributions. Added weighted Gaussian posterior quadrature, exact-model WAIC calculations, shared new-group effects and explicit marginal-offset checks. Missing PyMC/ArviZ error paths run in isolated interpreters even when the optional dependencies are installed.
- Bayesian first pass: `bayes_lm` with proper normal-inverse-gamma priors, exact moments/credible and predictive intervals, independent posterior draws and normalized model evidence, checked with independent R calculations on all four backends. `bayes_glm` adds optional PyMC NUTS inference for ten likelihood families and noncentered independent varying intercepts/slopes via `stats3[bayes]`. Includes prior specification, posterior/auxiliary summaries, draws, prior/posterior prediction, predictive checks, R-hat/ESS/MCSE/divergence/BFMI diagnostics, posterior hypotheses and ROPE probabilities, marginal means/contrasts evaluated draw by draw, and LOO/WAIC/evidence comparisons. `bayes_binom_test` and `bayes_poisson_test` provide scalar conjugate posteriors and point-null log Bayes factors. Coefficient `std_error` is posterior SD and p-values stay missing. Correlated effects, multinomial varying effects, categorical marginal means and general marginal likelihoods are deferred. Bayesian work now precedes the remaining frequentist tier 2 models.
- `rlm("y ~ x")`: MASS's M-estimation with Huber, bisquare or Hampel scores, custom tuning, MAD/Huber proposal-2 scale, inverse-variance/case weights and XtX/XtWX covariance. Coefficient tables keep R's missing p-values; confidence intervals are normal. Includes grouped fits, offsets, row alignment, predictions, covariance, model tables and marginal means. MM/LTS initialization is not included.
- `rq("y ~ x", tau=0.5)`: quantile regression by sparse linear programming, with quantreg's `nid` or `iid` covariance, t tests/intervals and nested-model Wald comparisons at one tau. Includes weights, grouped fits, row alignment, predictions, covariance, model tables and marginal means. Defaults to `nid`, rather than R's small-sample rank intervals; rank/bootstrap inference and offsets are not included. R comparisons cover coefficients, covariance, fit statistics, predictions and grouping on all four backends.
- `polr("y ~ x")`: MASS's proportional-odds cumulative-link model, with logistic, probit, loglog and cloglog links. Slopes and ordered thresholds (`fit.coefficients`, `fit.zeta`) with Hessian standard errors and Wald intervals; p-values stay missing, as in `summary.polr`. `predict()` gives classes, probabilities or the linear predictor, and `anova(fit1, fit2)` gives likelihood ratio tests of nested fits. Includes weights (zero weights drop rows), offsets, sum contrasts, aliased columns, grouped fits, missing-row alignment and model tables. Checked against MASS on all four backends. Profile intervals, the cauchit link, robust covariance and ordinal marginal means are not included.
- `glm_nb("y ~ x")` and `negative_binomial(theta)`: MASS's `glm.nb` (alternating coefficient and size estimation, with `theta` and `se_theta`) and its fixed-theta GLM family. Log, identity and sqrt links; grouped fits, weights, offsets, missing-row alignment, profile/Wald intervals, predictions, robust standard errors and marginal means. `anova(fit1, fit2)` compares fitted likelihoods with theta re-estimated; a single-model ANOVA holds theta fixed and warns. Checked against MASS on all four backends, including theta, information SEs, AIC/BIC and zero-weight rows.
- `emmeans(fit, "species")`, `pairs(emm)` and `contrast(emm, ...)`: estimated marginal means and contrasts, checked against R's emmeans on all four data backends. Includes reference grids, binary covariates, weighting, grouped fits, robust covariance, GLM response scales, offsets and multiplicity adjustments. `tidy(emm)` and `emm >> tidy()` return tables. Offsets use the same averaging weights as the means, including cell weights (validated against explicit offsets in R, whose emmeans 2.0.4 defaults to equal offset weights in that case).
- `tidy(fit, vcov=...)` tests the coefficients with a robust covariance, as lmtest's `coeftest` and `coefci`: `vcov="HC3"` (or `"HC0"`-`"HC5"`, `"const"`; sandwich's `vcovHC`), `vcov=vcov_cl("island")` for clustered standard errors (one or more columns, types HC0-HC3; `vcovCL`), and `vcov=vcov_nw()` for Newey-West (`NeweyWest`, with its lag selection and prewhitening). `lm` fits keep t tests on the residual degrees of freedom; `glm` fits get z tests and Wald intervals.
- `glance(fit, vcov=...)` for an `lm`: the robust Wald F test of all slopes, as `waldtest(fit, vcov=)`.
- `vcov(fit)` and `vcov(fit, "HC3")`: the coefficient covariance matrix as a table.
- `lm(..., na_action="exclude")` and `glm(...)`: as R's `na.exclude`, `augment()` and `predict()` keep every row, with missing values for rows the model didn't use, so results line up with the data. `na_action="fail"` raises on missing values; `"omit"` stays the default.
- `modelsummary(fit1, fit2)` (or a dict of named fits, or a grouped fit): models side by side with standard errors, intervals or stars and fit statistics, as a frame or as Markdown or HTML text.

### Changed

- `bayes_glm`'s default priors are autoscaled to the data, as in rstanarm: coefficients `Normal(0, 2.5 s_y / s_x)`, the intercept `Normal(mean response, 2.5 s_y)`, `sigma` `Exponential(1 / s_y)` and group SDs in the response's units. As in rstanarm, NUTS samples with the predictors centred at their means, so intercept and cutpoint priors apply at those means; reported coefficients are unchanged. On penguin body mass this halves sampling time and brings the mean tree depth from about 10 to 5. The fixed defaults (`sigma` `HalfNormal(5)`, Gaussian intercept `Normal(0, 10)`, coefficients `Normal(0, 2.5)`) dominated the likelihood for responses in large units: on penguin body mass in grams they halved the posterior sigma and the flipper-length slope.

### Fixed

- `predict(fit, interval="prediction")` for a binomial `bayes_glm` gave the interval in success counts but `fit` as a probability; the interval is now in proportions too, as in `augment()`. `posterior_predict()` still draws counts.
- `tidy(fit, exponentiate=True)` on a sampled Bayesian fit reported `prob_positive` as 1 for every coefficient (the probability that exp(beta) is positive); it is now P(beta > 0), that is, P(exp(beta) > 1).
- A `bayes_lm` or `bayes_glm` step with `seed=`, piped into data more than once, gave different draws on each use. Each application now starts from the seed.
- `bayes_glm(..., family=Gamma())` failed on R's default inverse link; it now warns and uses the log link.
- A `factor()` response (`factor(y) ~ x`) is now read as categorical. Before, `factor()` was ignored on the response: `lm("factor(cyl) ~ hp")` fitted `cyl` as a number, where it now raises that the response must be numeric. A binomial `glm` with `factor(am)` gives the same fit as before.

## 0.1.3 — 2026-10-24

More of base R's tests: Welch's ANOVA, pairwise comparisons, Friedman, McNemar, Cochran-Mantel-Haenszel, exact Poisson and Fligner-Killeen tests, and ANOVA power.

### Added

- `oneway_test("y ~ g")`: R's `oneway.test`, Welch's one-way ANOVA by default, the classic F test with `var_equal=True`.
- `pairwise_t_test("y ~ g")`, `pairwise_wilcox_test("y ~ g")` and `pairwise_prop_test(x, n)`: R's `pairwise.*.test`, with `p_adjust_method` (Holm by default), pooled or Welch SDs and paired tests. One row per pair (`group1`, `group2`, `p_value`), as broom's `tidy` gives, and each group adjusted on its own.
- `friedman_test("y ~ g | b")`: R's `friedman.test` for unreplicated complete block designs, with its tie correction.
- `mcnemar_test(x, y)`: R's `mcnemar.test`, with the continuity correction for 2 x 2 tables.
- `mantelhaen_test(x, y, z)`: R's `mantelhaen.test`: the Mantel-Haenszel test and common odds ratio for 2 x 2 x K tables (`exact=True` for the exact conditional test), and the generalized Cochran-Mantel-Haenszel test for larger tables.
- `poisson_test(x, T)`: R's `poisson.test`, the exact test of one rate or of the ratio of two.
- `fligner_test("y ~ g")`: R's `fligner.test` of equal variances.
- `power_anova_test(...)`: R's `power.anova.test` for a balanced one-way ANOVA.

## 0.1.2 — 2026-10-23

`predict()` keeps `offset=` on new rows and gives intervals on the fitted rows.

### Fixed

- `predict(fit, newdata)` left out an offset given as `offset=` (an `offset()` term in the formula was kept), so predictions were too low by the offset. The `offset=` column or expression is now evaluated on the new rows, for `lm` and `glm`, as R's `predict` does.
- `predict(fit)` without `newdata` ignored `interval=` and `se_fit=` (and `type=` for a `glm` was all it honoured). Predictions on the fitted rows now give intervals and standard errors too. As in R, prediction intervals on those rows warn that they are for future responses, and use the fitting weights of a weighted `lm`; on new rows a weighted fit warns that the prediction variance is taken as constant.

## 0.1.1 — 2026-10-23

`tidy(fit)` works whichever of stats3 and tidy3 is imported last (with tidy3 0.8.3 or later).

### Fixed

- `tidy(fit)` failed when tidy3 was imported after stats3, because tidy3's `tidy()` replaced stats3's and treated the model as data. Fitted models (and grouped models) now have a `__tidy__` method that tidy3's `tidy()` calls from tidy3 0.8.3 on, so `tidy(fit)` and `fit >> tidy()` work whichever package is imported last.

## 0.1.0 — 2026-10-23

The first version: R's common statistical tests for Polars, pandas and tidy3 data.

### Added

- Works on Polars DataFrames and LazyFrames, pandas DataFrames, tidy3 frames and dicts of columns, and returns the same kind of frame (or `as_=`). `by=` groups Polars and pandas data; tidy3's `group_by` groups tidy3 frames. pandas `category` and Polars `Enum` columns, and tidy3's factor levels, order the levels. Needs only Polars and SciPy; `stats3[pandas]` adds pandas and pyarrow, and tidy3 is not a dependency.
- Each test is called with the data first (`t_test(df, "y ~ g")`) or piped (`df >> t_test("y ~ g")`, `df.pipe(...)`).
- In Jupyter and IPython, formulas need no quotes and bare names are columns inside stats3's calls (`t_test(body_mass_g ~ sex, by=species)`, `binom_test(sex == "male")`), sharing tidy3's shorthand when both are loaded.

- Tests on means: `t_test` (Welch, Student, paired, one-sample), `aov` (one grouping column) and `tukey_hsd`.
- Rank tests and correlations: `wilcox_test` (R 4.6's exact tests with ties and zeros, and its Edgeworth corrections), `kruskal_test`, `cor_test` (Pearson, Spearman with AS 89, Kendall).
- Tests on counts: `chisq_test` (independence and goodness of fit), `fisher_test` (2 x 2), `prop_test`, `binom_test`.
- Checks of distribution and spread: `shapiro_test`, `ks_test` (R 4.6's exact two-sample distribution with ties), `bartlett_test`, `levene_test`, `var_test`.
- `p_adjust` on vectors and Series, or as a column expression for `with_columns` and tidy3's `mutate` (by group), `cohens_d`, `hedges_g`, `eta_squared`, `power_t_test`, `power_prop_test`.
- Models: `lm()` and `glm()` with R's formulas (`*`, `:`, `^`, `.`, `0`/`-1`, `log()`, `I()`, `factor()`, `offset()`, `cbind()` responses), treatment contrasts and R's column names, aliased columns, weights and offsets; families `gaussian`, `binomial`, `poisson`, `Gamma`, `inverse_gaussian`, `quasibinomial`, `quasipoisson` with R's links. `glm` is a port of R's `glm.fit`. A printed fit reads as a summary; `tidy`, `glance` and `augment` give broom's tables, `anova` gives sequential tables and model comparisons, and `predict` / `augment(newdata=)` predict new rows. `by=` (or tidy3's `group_by`) fits one model per group. `tidy()` hands data that is not a model to tidy3's `tidy()`.
- `tidy(conf_int=True)` on a `glm` gives R's profile-likelihood intervals (`confint`: `profile.glm`'s grid, spline and inversion); `conf_method="wald"` gives `confint.default`.
- `anova(fit, type=2)` and `anova(fit, type=3)`: car's type II and III tests (F for `lm`; likelihood ratio for binomial and Poisson `glm`, F on the deviance for the others). `lm` and `glm` take `contrasts="sum"` (R's `contr.sum`), per factor or for all.
- `fisher_test` gives an exact p-value for tables larger than 2 x 2, by a network algorithm as R's FEXACT, without R's workspace limit.
- Every test runs once per group and returns one row per group (one table per group for `aov` and `tukey_hsd`).
- Tests compare results with R 4.6 through an R oracle, on all four kinds of input, and with SciPy: the models' coefficients, intervals, fit statistics, influence measures, anova tables and predictions for 23 models, on R's penguins and mtcars. Where R's own numerics are approximate (Tukey's distribution, uniroot intervals, noncentral t above ncp 37.62, the asymptotic Kolmogorov series), stats3 computes to full precision and the tests compare at R's accuracy, with the reason next to each tolerance.
