# stats3

**R's statistical tests on Polars, as [tidy3](https://github.com/rleyvasal/tidy3) pipe steps.**
Write a test the way R users do, run it once per group, and get R's numbers
back as a table you can keep piping.

```bash
pip install stats3
```

stats3 is the statistics half of tidy3: tidy3 wrangles the data, stats3 tests
it. Every test takes R's arguments and defaults, drops missing values as R
does, raises R's errors and warnings, and is checked against R 4.6.

## One test per group

```python
from tidy3 import *
from stats3 import *

penguins = scan_csv(
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv",
    null_values="NA",
) >> compute()

penguins >> group_by(species) >> t_test("body_mass_g ~ sex")
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
The result is an ordinary tidy3 frame, so the next step can adjust the
p-values within each group:

```python
penguins
>> group_by(species, island)
>> wilcox_test("body_mass_g ~ sex")
>> mutate(p_adj = p_adjust(p_value, "BH"))
```

ANOVA and Tukey's comparisons read the same way:

```python
penguins >> aov("body_mass_g ~ species")
penguins >> tukey_hsd("body_mass_g ~ species")
```

```text
┌─────────┬──────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ term    ┆ contrast         ┆ estimate    ┆ conf_low    ┆ conf_high   ┆ adj_p_value │
╞═════════╪══════════════════╪═════════════╪═════════════╪═════════════╪═════════════╡
│ species ┆ Chinstrap-Adelie ┆ 32.425984   ┆ -126.500196 ┆ 191.352163  ┆ 0.880667    │
│ species ┆ Gentoo-Adelie    ┆ 1375.354009 ┆ 1243.178628 ┆ 1507.529389 ┆ 1.4433e-14  │
│ species ┆ Gentoo-Chinstrap ┆ 1342.928025 ┆ 1178.481013 ┆ 1507.375037 ┆ 1.4433e-14  │
└─────────┴──────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

## What's in it

| R | stats3 |
|---|---|
| `t.test` | `t_test("y ~ g")`, `t_test(x, mu=)`, `t_test(x, y, paired=True)` |
| `wilcox.test` | `wilcox_test(...)`, same forms; R 4.6's exact tests with ties |
| `kruskal.test` | `kruskal_test("y ~ g")` |
| `aov`, `TukeyHSD` | `aov("y ~ g")`, `tukey_hsd("y ~ g")` (one grouping column) |
| `cor.test` | `cor_test(x, y, method="pearson" / "spearman" / "kendall")` |
| `chisq.test` | `chisq_test(x, y)`, `chisq_test(x, p=[...])` |
| `fisher.test` | `fisher_test(x, y)` (2 x 2) |
| `prop.test`, `binom.test` | `prop_test(x, n)`, `binom_test(sex == "male")` |
| `shapiro.test`, `ks.test` | `shapiro_test(x)`, `ks_test(x, "pnorm", mean=, sd=)`, `ks_test(x, y)` |
| `bartlett.test`, `var.test`, `car::leveneTest` | `bartlett_test`, `var_test`, `levene_test` |
| `p.adjust` | `p_adjust(p_value, "BH")` inside `mutate` |
| effectsize | `cohens_d`, `hedges_g`, `eta_squared` |
| `power.t.test`, `power.prop.test` | `power_t_test`, `power_prop_test` |

Models (`lm`, `glm`) are next.

## How it works, and how it is checked

Most tests need only a few numbers per group: counts, means, variances,
rank sums, cross-tabulated counts. Polars computes them for every group in
one lazy query, and scipy's distribution functions turn them into p-values
for all groups at once. Over 1,000 groups of 1,000 rows most tests run 4-30 times
faster than calling scipy once per group. Tests that need every value
(Shapiro-Wilk, Kendall's tau) run per group.

`tests/r_oracle_cases.R` runs the same tests in R; `pytest` compares every
column, to 1e-9 relative. Where R's own numerics are approximate (Tukey's
distribution, intervals found with `uniroot`, the noncentral t above ncp
37.62, the asymptotic Kolmogorov series), stats3 computes to full precision
and the test compares at R's accuracy, with the reason written next to the
tolerance. Without R, the scipy comparisons still run.

```bash
pip install -e ".[dev]"
pytest
```
