# Changelog

All notable changes to stats3. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may change behaviour, and those changes are listed
under **Changed**.

## Unreleased

Robust standard errors, rows that line up with the data, and side-by-side model tables.

### Added

- `tidy(fit, vcov=...)` tests the coefficients with a robust covariance, as lmtest's `coeftest` and `coefci`: `vcov="HC3"` (or `"HC0"`-`"HC5"`, `"const"`; sandwich's `vcovHC`), `vcov=vcov_cl("island")` for clustered standard errors (one or more columns, types HC0-HC3; `vcovCL`), and `vcov=vcov_nw()` for Newey-West (`NeweyWest`, with its lag selection and prewhitening). `lm` fits keep t tests on the residual degrees of freedom; `glm` fits get z tests and Wald intervals.
- `glance(fit, vcov=...)` for an `lm`: the robust Wald F test of all slopes, as `waldtest(fit, vcov=)`.
- `vcov(fit)` and `vcov(fit, "HC3")`: the coefficient covariance matrix as a table.
- `lm(..., na_action="exclude")` and `glm(...)`: as R's `na.exclude`, `augment()` and `predict()` keep every row, with missing values for rows the model didn't use, so results line up with the data. `na_action="fail"` raises on missing values; `"omit"` stays the default.
- `modelsummary(fit1, fit2)` (or a dict of named fits, or a grouped fit): models side by side with standard errors, intervals or stars and fit statistics, as a frame or as Markdown or HTML text.

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
