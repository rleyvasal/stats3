# stats3

**R's statistical tests on Polars, for Polars, pandas and tidy3 data.**
Write a test the way R users do, run it once per group, and get R's numbers
back as a table in the same kind of frame you passed in.

```bash
pip install stats3              # Polars data
pip install "stats3[pandas]"    # pandas data too
```

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
| `p.adjust` | `p_adjust(p, "BH")`: on a vector or Series, or a column in `with_columns` / `mutate` |
| effectsize | `cohens_d`, `hedges_g`, `eta_squared` |
| `power.t.test`, `power.prop.test` | `power_t_test`, `power_prop_test` |

Models (`lm`, `glm`) are next.

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
