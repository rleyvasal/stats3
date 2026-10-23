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
it, and from tidy3 0.8.3 tidy3's `tidy` hands models to stats3, so either
import order works. With tidy3 0.8.2 or earlier, import stats3 after tidy3
(or call `stats3.tidy(fit)`).

The engine is stats3's own: weighted least squares by QR for `lm`, and a
port of R's `glm.fit` (same starting values, convergence rule and step
halving) for `glm`. On 1,000,000 rows with 13 coefficients `lm` takes about
0.16 s and a Poisson `glm` about 0.6 s, against about 1.1 s and 1.9 s for
statsmodels' formula interface.

## What's in it

| R | stats3 |
|---|---|
| `t.test` | `t_test("y ~ g")`, `t_test(x, mu=)`, `t_test(x, y, paired=True)` |
| `wilcox.test` | `wilcox_test(...)`, same forms; R 4.6's exact tests with ties |
| `kruskal.test` | `kruskal_test("y ~ g")` |
| `aov`, `TukeyHSD` | `aov("y ~ g")`, `tukey_hsd("y ~ g")` (one grouping column) |
| `cor.test` | `cor_test(x, y, method="pearson" / "spearman" / "kendall")` |
| `chisq.test` | `chisq_test(x, y)`, `chisq_test(x, p=[...])` |
| `fisher.test` | `fisher_test(x, y)`: odds ratio for 2 x 2, exact p-value for larger tables |
| `prop.test`, `binom.test` | `prop_test(x, n)`, `binom_test(sex == "male")` |
| `shapiro.test`, `ks.test` | `shapiro_test(x)`, `ks_test(x, "pnorm", mean=, sd=)`, `ks_test(x, y)` |
| `bartlett.test`, `var.test`, `car::leveneTest` | `bartlett_test`, `var_test`, `levene_test` |
| `p.adjust` | `p_adjust(p, "BH")`: on a vector or Series, or a column in `with_columns` / `mutate` |
| effectsize | `cohens_d`, `hedges_g`, `eta_squared` |
| `power.t.test`, `power.prop.test` | `power_t_test`, `power_prop_test` |
| `lm`, `glm` | `lm`, `glm`, with broom's `tidy`, `glance`, `augment`, and `anova`, `predict` |
| `car::Anova` | `anova(fit, type=2)`, `anova(fit, type=3)` |

Next: estimated marginal means, negative binomial, robust and quantile
regression, ordinal models, and mixed models.

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
Where R's own numerics are approximate (Tukey's distribution, intervals
found with `uniroot`, the noncentral t above ncp 37.62, the asymptotic
Kolmogorov series), stats3 computes to full precision and the test compares
at R's accuracy, with the reason written next to the tolerance. Without R,
the SciPy comparisons still run.

```bash
pip install -e ".[dev]"
pytest
```
