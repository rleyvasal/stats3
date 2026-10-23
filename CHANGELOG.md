# Changelog

All notable changes to stats3. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may change behaviour, and those changes are listed
under **Changed**.

## 0.1.0 — unreleased

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
- Every test runs once per group and returns one row per group (one table per group for `aov` and `tukey_hsd`).
- Tests compare results with R 4.6 through an R oracle, on all four kinds of input, and with SciPy: the models' coefficients, intervals, fit statistics, influence measures, anova tables and predictions for 23 models, on R's penguins and mtcars. Where R's own numerics are approximate (Tukey's distribution, uniroot intervals, noncentral t above ncp 37.62, the asymptotic Kolmogorov series), stats3 computes to full precision and the tests compare at R's accuracy, with the reason next to each tolerance.
