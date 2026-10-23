"""Checks of distribution and spread: R's ``shapiro.test``, ``ks.test``,
``bartlett.test``, ``var.test`` and car's ``leveneTest``.

Variances and deviations come from grouped Polars queries. Shapiro-Wilk and
the Kolmogorov-Smirnov statistics need each group's sorted values, so they
run per group on the collected values: Shapiro-Wilk through scipy (Royston's
algorithm, as R uses), and the exact Smirnov distribution, with ties as in
R 4.6, by counting lattice paths here.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import polars as pl

from stats3._data import Step, column_name, expression, step
from stats3.core import (
    _warn,
    _alternative,
    _by_levels,
    _check_conf_level,
    _column,
    _finish,
    _group_label,
    _is_formula,
    _moments,
    _parse_formula,
    _present,
    _sorted_keys,
    _special,
)


def _group_values(
    tf: Any, groups: list[str], columns: dict[str, pl.Expr], keep: pl.Expr
) -> tuple[pl.DataFrame, list[dict[str, np.ndarray]]]:
    """Per group (in dplyr order): each named expression's values as arrays."""
    lf = tf._lf.filter(keep).select(*groups, **columns)
    if groups:
        table = _sorted_keys(tf, lf.group_by(groups).agg(pl.all()).collect(), groups)
    else:
        table = lf.select(pl.all().implode()).collect()
    rows = [
        {name: np.asarray(row[name], dtype=float) for name in columns}
        | {"__label": _group_label(row, groups)}
        for row in table.iter_rows(named=True)
    ]
    return table, rows


# shapiro_test ------------------------------------------------------------------


@step
def shapiro_test(x: Any) -> Step:
    """Shapiro-Wilk normality test, as R's ``shapiro.test``, one row per group.

    Missing values are dropped; each group needs 3 to 5000 values that are not
    all identical. Columns: ``statistic`` (W), ``p_value``, ``method``.
    """

    def _apply(tf: Any) -> Any:
        from scipy.stats import shapiro

        groups = list(tf._groups or [])
        name = _column(tf, x, "x", "shapiro_test")
        table, rows = _group_values(
            tf, groups, {"x": pl.col(name).cast(pl.Float64)}, _present(tf._lf, name)
        )
        statistic, p_value = [], []
        for row in rows:
            values = np.sort(row["x"])
            n = len(values)
            if not 3 <= n <= 5000:
                raise ValueError(f"shapiro_test(): sample size must be between 3 and 5000{row['__label']}")
            if values[-1] - values[0] < 1e-10:
                raise ValueError(f"shapiro_test(): all 'x' values are identical{row['__label']}")
            result = shapiro(values)
            statistic.append(float(result.statistic))
            p_value.append(float(result.pvalue))
        return _finish(tf, table, groups, {
            "statistic": np.array(statistic),
            "p_value": np.array(p_value),
            "method": "Shapiro-Wilk normality test",
        })

    return Step(_apply, "shapiro_test")


# variance tests --------------------------------------------------------------------


@step
def bartlett_test(formula: str) -> Step:
    """Bartlett's test of equal variances, as R's ``bartlett.test(y ~ g)``.

    Columns: ``statistic`` (Bartlett's K-squared), ``df``, ``p_value``,
    ``method``.
    """

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        lf = tf._lf
        response, by = _parse_formula(tf, formula, "bartlett_test", one_sample=False)
        kept = lf.filter(_present(lf, response) & _present(lf, by))
        table = _by_levels(tf, kept, groups, by, _moments(pl.col(response).cast(pl.Float64)), "bartlett_test")
        statistic, df = [], []
        for row in table.iter_rows(named=True):
            n_i = np.array(row["n"], dtype=float)
            if np.any(n_i < 2):
                raise ValueError(
                    "bartlett_test(): there must be at least 2 observations in each group"
                    + _group_label(row, groups)
                )
            v_i = np.array(row["var"], dtype=float)
            k = len(n_i)
            d_i = n_i - 1
            n = d_i.sum()
            pooled = np.sum(d_i * v_i) / n
            k2 = (n * math.log(pooled) - np.sum(d_i * np.log(v_i))) / (
                1 + (np.sum(1 / d_i) - 1 / n) / (3 * (k - 1))
            )
            statistic.append(float(k2))
            df.append(float(k - 1))
        s, d = np.array(statistic), np.array(df)
        return _finish(tf, table, groups, {
            "statistic": s, "df": d, "p_value": special.chdtrc(d, s),
            "method": "Bartlett test of homogeneity of variances",
        })

    return Step(_apply, "bartlett_test")


@step
def levene_test(formula: str, *, center: str = "median") -> Step:
    """Levene's test of equal variances, as car's ``leveneTest(y ~ g)``.

    A one-way ANOVA on absolute deviations from each level's median (the
    Brown-Forsythe form, car's default) or mean (``center="mean"``, Levene's
    original). Columns: ``statistic`` (F), ``df``, ``df_residual``,
    ``p_value``, ``method``.
    """
    if center not in ("median", "mean"):
        raise ValueError(f"levene_test(): center must be 'median' or 'mean', got {center!r}")

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        lf = tf._lf
        response, by = _parse_formula(tf, formula, "levene_test", one_sample=False)
        value = pl.col(response).cast(pl.Float64)
        middle = (value.median() if center == "median" else value.mean()).over([*groups, by])
        kept = lf.filter(_present(lf, response) & _present(lf, by)).with_columns(
            __stats3_dev=(value - middle).abs()
        )
        table = _by_levels(tf, kept, groups, by, _moments(pl.col("__stats3_dev")), "levene_test")
        statistic, df_b, df_w = [], [], []
        for row in table.iter_rows(named=True):
            n = np.array(row["n"], dtype=float)
            m = np.array(row["mean"], dtype=float)
            v = np.array([0.0 if x is None else x for x in row["var"]], dtype=float)
            total = n.sum()
            grand = np.sum(n * m) / total
            between = np.sum(n * (m - grand) ** 2) / (len(n) - 1)
            within = np.sum((n - 1) * v) / (total - len(n))
            statistic.append(float(between / within))
            df_b.append(float(len(n) - 1))
            df_w.append(float(total - len(n)))
        s, d1, d2 = np.array(statistic), np.array(df_b), np.array(df_w)
        return _finish(tf, table, groups, {
            "statistic": s, "df": d1, "df_residual": d2, "p_value": special.fdtrc(d1, d2, s),
            "method": f"Levene's Test for Homogeneity of Variance (center = {center})",
        })

    return Step(_apply, "levene_test")


@step
def var_test(
    x: Any,
    y: Any = None,
    *,
    ratio: float = 1.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> Step:
    """F test comparing two variances, as R's ``var.test``.

    ``var_test("body_mass_g ~ sex")`` or ``var_test(x, y)``. Columns:
    ``estimate`` (ratio of variances), ``num_df``, ``den_df``, ``statistic``,
    ``p_value``, ``conf_low``, ``conf_high``, ``method``, ``alternative``.
    """
    alternative = _alternative(alternative)
    _check_conf_level(conf_level)
    if not (math.isfinite(ratio) and ratio > 0):
        raise ValueError(f"var_test(): ratio must be a single positive number, got {ratio!r}")

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        lf = tf._lf
        if _is_formula(x):
            response, by = _parse_formula(tf, x, "var_test", one_sample=False)
            kept = lf.filter(_present(lf, response) & _present(lf, by))
            stacked = _by_levels(tf, kept, groups, by, _moments(pl.col(response).cast(pl.Float64)),
                                 "var_test", exactly=2)
            table = stacked.select(*groups, *(pl.col(s).list.get(i).alias(f"{s}{i + 1}")
                                              for s in ("n", "var") for i in (0, 1)))
        else:
            if y is None:
                raise ValueError("var_test(): pass a formula, or two columns x and y")
            xc, yc = _column(tf, x, "x", "var_test"), _column(tf, y, "y", "var_test")
            aggs = []
            for side, name in (("1", xc), ("2", yc)):
                value = pl.col(name).filter(_present(lf, name)).cast(pl.Float64)
                aggs += [value.count().alias(f"n{side}"), value.var().alias(f"var{side}")]
            table = lf.group_by(groups).agg(aggs).collect() if groups else lf.select(aggs).collect()
            table = _sorted_keys(tf, table, groups)
        for row in table.iter_rows(named=True):
            label = _group_label(row, groups)
            if row["n1"] < 2:
                raise ValueError(f"var_test(): not enough 'x' observations{label}")
            if row["n2"] < 2:
                raise ValueError(f"var_test(): not enough 'y' observations{label}")
        d1 = table["n1"].cast(pl.Float64).to_numpy() - 1
        d2 = table["n2"].cast(pl.Float64).to_numpy() - 1
        estimate = table["var1"].to_numpy() / table["var2"].to_numpy()
        statistic = estimate / ratio
        lower = special.fdtr(d1, d2, statistic)
        if alternative == "less":
            p = lower
            low = np.zeros_like(estimate)
            high = estimate / special.fdtri(d1, d2, 1 - conf_level)
        elif alternative == "greater":
            p = 1 - lower
            low = estimate / special.fdtri(d1, d2, conf_level)
            high = np.full_like(estimate, np.inf)
        else:
            p = 2 * np.minimum(lower, special.fdtrc(d1, d2, statistic))
            beta = (1 - conf_level) / 2
            low = estimate / special.fdtri(d1, d2, 1 - beta)
            high = estimate / special.fdtri(d1, d2, beta)
        return _finish(tf, table, groups, {
            "estimate": estimate, "num_df": d1, "den_df": d2, "statistic": statistic,
            "p_value": p, "conf_low": low, "conf_high": high,
            "method": "F test to compare two variances", "alternative": alternative,
        })

    return Step(_apply, "var_test")


# ks_test -------------------------------------------------------------------------


def _distribution(name: Any, params: dict[str, float]) -> Callable[[np.ndarray], np.ndarray]:
    """R's distribution functions by name, with R's parameter names."""
    if callable(name):
        return lambda v: np.asarray(name(v), dtype=float)
    from scipy import stats

    def get(key: str, default: float | None = None) -> float:
        if key in params:
            return float(params[key])
        if default is None:
            raise ValueError(f"ks_test(): {name} needs {key}=")
        return default

    known = {
        "pnorm": lambda: stats.norm(loc=get("mean", 0.0), scale=get("sd", 1.0)),
        "punif": lambda: stats.uniform(loc=get("min", 0.0), scale=get("max", 1.0) - get("min", 0.0)),
        "pexp": lambda: stats.expon(scale=1 / get("rate", 1.0)),
        "plnorm": lambda: stats.lognorm(s=get("sdlog", 1.0), scale=math.exp(get("meanlog", 0.0))),
        "pgamma": lambda: stats.gamma(a=get("shape"), scale=1 / get("rate", 1.0)),
        "pt": lambda: stats.t(df=get("df")),
    }
    if name not in known:
        raise ValueError(f"ks_test(): y must be a column, a function, or one of {sorted(known)}; got {name!r}")
    allowed = {"pnorm": {"mean", "sd"}, "punif": {"min", "max"}, "pexp": {"rate"},
               "plnorm": {"meanlog", "sdlog"}, "pgamma": {"shape", "rate"}, "pt": {"df"}}[name]
    extra = set(params) - allowed
    if extra:
        raise ValueError(f"ks_test(): {name} takes {sorted(allowed)}, not {sorted(extra)}")
    return known[name]().cdf


def _smirnov_upper(stat: float, m: int, n: int, ends: np.ndarray, alternative: str) -> float:
    """P(D >= stat) by counting lattice paths, checking only where tie blocks end.

    ``ends[k]`` is True when the k-th sorted pooled value (1-based) ends a run
    of equal values; R 4.6's exact Smirnov distribution with ties.
    """
    s = round(stat * m * n)
    j = np.arange(n + 1)

    def allowed(i: int) -> np.ndarray:
        diff = i * n - j * m
        if alternative == "two.sided":
            diff = np.abs(diff)
        elif alternative == "less":
            diff = -diff
        checked = ends[i + j]
        checked[i + j == 0] = False
        return ~((diff >= s) & checked)

    # row[j]: paths from (0, 0) to (i, j) that stay below the bound.
    row = np.zeros(n + 1)
    for i in range(m + 1):
        good = allowed(i)
        new = np.zeros(n + 1)
        for jj in range(n + 1):
            if not good[jj]:
                continue
            if i == 0 and jj == 0:
                new[jj] = 1.0
            else:
                new[jj] = (row[jj] if i > 0 else 0.0) + (new[jj - 1] if jj > 0 else 0.0)
        row = new
    return float(max(0.0, 1.0 - row[n] / math.comb(m + n, m)))


def _kolmogorov_upper(stat: float, n: int, two_sided: bool, exact: bool) -> float:
    from scipy import stats
    special = _special()

    if exact:
        return float(stats.kstwo.sf(stat, n) if two_sided else special.smirnov(n, stat))
    if two_sided:
        return float(special.kolmogorov(math.sqrt(n) * stat))
    return float(math.exp(-2 * n * stat * stat))


@step
def ks_test(
    x: Any,
    y: Any = "pnorm",
    *,
    alternative: str = "two.sided",
    exact: bool | None = None,
    **params: float,
) -> Step:
    """Kolmogorov-Smirnov test, as R's ``ks.test``.

    - ``ks_test(x, "pnorm", mean=..., sd=...)``: one sample against a
      distribution named as in R (``pnorm``, ``punif``, ``pexp``, ``plnorm``,
      ``pgamma``, ``pt``, with R's parameter names) or a CDF function.
    - ``ks_test(x, y)`` with two columns, or ``ks_test("y ~ g")``: two samples.

    p-values are exact as R decides (one sample: under 100 values without
    ties; two samples: ``n_x * n_y < 10000``, with ties handled exactly as in
    R 4.6). ``alternative`` follows R: ``"less"`` means the CDF of x lies
    below the other. Columns: ``statistic``, ``p_value``, ``method``,
    ``alternative``.
    """
    alternative = _alternative(alternative)

    def _apply(tf: Any) -> Any:
        groups = list(tf._groups or [])
        lf = tf._lf
        two = _is_formula(x) or (isinstance(y, str) and y in tf._lf.collect_schema()) or (
            not isinstance(y, str) and not callable(y)
        )
        if _is_formula(x):
            response, by = _parse_formula(tf, x, "ks_test", one_sample=False)
            kept = lf.filter(_present(lf, response) & _present(lf, by))
            stacked = _by_levels(tf, kept, groups, by, [pl.col(response).cast(pl.Float64).alias("v")],
                                 "ks_test", exactly=2)
            table = stacked
            rows = [{"x": np.asarray(r["v"][0], float), "y": np.asarray(r["v"][1], float),
                     "__label": _group_label(r, groups)} for r in stacked.iter_rows(named=True)]
        elif two:
            xc, yc = _column(tf, x, "x", "ks_test"), _column(tf, y, "y", "ks_test")
            if groups:
                aggs = [pl.col(c).filter(_present(lf, c)).cast(pl.Float64).alias(k) for k, c in (("x", xc), ("y", yc))]
                table = _sorted_keys(tf, lf.group_by(groups).agg(aggs).collect(), groups)
            else:
                table = lf.select(
                    pl.col(xc).filter(_present(lf, xc)).cast(pl.Float64).implode().alias("x"),
                    pl.col(yc).filter(_present(lf, yc)).cast(pl.Float64).implode().alias("y"),
                ).collect()
            rows = [{"x": np.asarray(r["x"], float), "y": np.asarray(r["y"], float),
                     "__label": _group_label(r, groups)} for r in table.iter_rows(named=True)]
        else:
            name = _column(tf, x, "x", "ks_test")
            table, rows = _group_values(tf, groups, {"x": pl.col(name).cast(pl.Float64)}, _present(lf, name))
        cdf = None if (two or _is_formula(x)) else _distribution(y, params)
        if (two or _is_formula(x)) and params:
            _warn(f"Parameter(s) {', '.join(params)} ignored")
        statistic, p_value, method = [], [], []
        for row in rows:
            if cdf is None:
                d, p, m = _two_sample(row["x"], row["y"], alternative, exact, row["__label"])
            else:
                d, p, m = _one_sample(row["x"], cdf, alternative, exact, row["__label"])
            statistic.append(d)
            p_value.append(min(1.0, max(0.0, p)))
            method.append(m)
        return _finish(tf, table, groups, {
            "statistic": np.array(statistic), "p_value": np.array(p_value),
            "method": pl.Series(method, dtype=pl.String), "alternative": alternative,
        })

    return Step(_apply, "ks_test")


def _one_sample(x: np.ndarray, cdf: Any, alternative: str, exact: bool | None, label: str):
    n = len(x)
    if n < 1:
        raise ValueError(f"ks_test(): not enough 'x' data{label}")
    ties = len(np.unique(x)) < n
    if ties:
        _warn("ties should not be present for the one-sample Kolmogorov-Smirnov test")
    use_exact = (n < 100 and not ties) if exact is None else exact
    v = cdf(np.sort(x)) - np.arange(n) / n
    d = {"two.sided": max(np.max(v), np.max(1 / n - v)),
         "greater": np.max(1 / n - v), "less": np.max(v)}[alternative]
    d = float(d)
    if d <= 0:
        p = 1.0
    elif d > 1:
        p = 0.0
    else:
        p = _kolmogorov_upper(d, n, alternative == "two.sided", use_exact)
    kind = "Exact" if use_exact else "Asymptotic"
    return d, p, f"{kind} one-sample Kolmogorov-Smirnov test"


def _two_sample(x: np.ndarray, y: np.ndarray, alternative: str, exact: bool | None, label: str):
    m, n = len(x), len(y)
    if m < 1:
        raise ValueError(f"ks_test(): not enough 'x' data{label}")
    if n < 1:
        raise ValueError(f"ks_test(): not enough 'y' data{label}")
    use_exact = (m * n < 10000) if exact is None else exact
    w = np.concatenate([x, y])
    order = np.argsort(w, kind="stable")
    z = np.cumsum(np.where(order < m, 1 / m, -1 / n))
    sorted_w = w[order]
    ends = np.ones(m + n + 1, dtype=bool)
    ties = len(np.unique(w)) < m + n
    if ties:
        last = np.r_[sorted_w[1:] != sorted_w[:-1], True]
        z = z[last]
        ends[1:] = last
        if not use_exact:
            _warn("p-value will be approximate in the presence of ties")
    d = float({"two.sided": np.max(np.abs(z)), "greater": np.max(z), "less": -np.min(z)}[alternative])
    if d <= 0:
        p = 1.0
    elif d > 1:
        p = 0.0
    elif use_exact:
        p = _smirnov_upper(d, m, n, ends, alternative)
    else:
        size = m * n / (m + n)
        p = float(_special().kolmogorov(math.sqrt(size) * d)) if alternative == "two.sided" else math.exp(-2 * size * d * d)
    kind = "Exact" if use_exact else "Asymptotic"
    return d, p, f"{kind} two-sample Kolmogorov-Smirnov test"
