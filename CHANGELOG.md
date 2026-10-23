# Changelog

All notable changes to stats3. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may change behaviour, and those changes are listed
under **Changed**.

## 0.1.0 — unreleased

The first version: R's common statistical tests as tidy3 pipe steps.

### Added

- Tests on means: `t_test` (Welch, Student, paired, one-sample), `aov` (one grouping column) and `tukey_hsd`.
- Rank tests and correlations: `wilcox_test` (R 4.6's exact tests with ties and zeros, and its Edgeworth corrections), `kruskal_test`, `cor_test` (Pearson, Spearman with AS 89, Kendall).
- Tests on counts: `chisq_test` (independence and goodness of fit), `fisher_test` (2 x 2), `prop_test`, `binom_test`.
- Checks of distribution and spread: `shapiro_test`, `ks_test` (R 4.6's exact two-sample distribution with ties), `bartlett_test`, `levene_test`, `var_test`.
- `p_adjust` for `mutate` (by group), `cohens_d`, `hedges_g`, `eta_squared`, `power_t_test`, `power_prop_test`.
- Every test runs once per group after `group_by` and returns one row per group (one table per group for `aov` and `tukey_hsd`).
- Tests compare results with R 4.6 through an R oracle and with scipy. Where R's own numerics are approximate (Tukey's distribution, uniroot intervals, noncentral t above ncp 37.62, the asymptotic Kolmogorov series), stats3 computes to full precision and the tests compare at R's accuracy, with the reason next to each tolerance.
