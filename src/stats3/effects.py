"""Multiple-testing adjustment, effect sizes and power.

``p_adjust`` is R's ``p.adjust``: on a vector it returns a vector, and on a
column name it returns a Polars expression for ``with_columns`` or tidy3's
``mutate`` (by group after ``group_by``). ``cohens_d``, ``hedges_g`` and ``eta_squared``
follow the effectsize package's definitions and its noncentral-t and
noncentral-F intervals. ``power_t_test`` and ``power_prop_test`` are R's
``power.t.test`` and ``power.prop.test``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from stats3._data import OUTPUTS, Data, Step, _is_pandas_series, expression, step
from stats3.core import (
    _by_levels,
    _check_conf_level,
    _check_mu,
    _column,
    _finish,
    _group_label,
    _is_formula,
    _moments,
    _parse_formula,
    _present,
    _sorted_keys,
    _special,
    _summarise,
)

P_ADJUST_METHODS = ("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none")


# p_adjust ------------------------------------------------------------------------


def p_adjust(p: Any, method: str = "holm", n: int | None = None) -> Any:
    """Adjust p-values for multiple comparisons, as R's ``p.adjust``.

    On values (a list, NumPy array, or pandas or Polars Series) it returns
    the adjusted values in the same type. On a column (a name or expression)
    it returns a Polars expression, for ``df.with_columns(p_adj=p_adjust(
    "p_value", "BH"))`` or tidy3's ``mutate``, where each group is adjusted on
    its own after ``group_by``. ``method`` is one of ``"holm"`` (R's default),
    ``"hochberg"``, ``"hommel"``, ``"bonferroni"``, ``"BH"`` (or ``"fdr"``),
    ``"BY"``, ``"none"``. ``n`` defaults to the number of non-missing
    p-values; missing values stay missing.
    """
    if method not in P_ADJUST_METHODS:
        raise ValueError(f"p_adjust(): method must be one of {list(P_ADJUST_METHODS)}, got {method!r}")
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
        raise ValueError(f"p_adjust(): n must be a positive integer, got {n!r}")
    if isinstance(p, pl.Series):
        return _adjust_series(p, method, n)
    if _is_pandas_series(p):
        values = adjust_p(p.to_numpy(dtype=float, na_value=np.nan), method, n)
        return type(p)(values, index=p.index, name=p.name)
    if isinstance(p, (list, tuple, np.ndarray)):
        return adjust_p(np.array([np.nan if v is None else v for v in p], dtype=float), method, n)
    # A Polars expression: tidy3's mutate runs it per group with over().
    return expression(p).map_batches(lambda s: _adjust_series(s, method, n), return_dtype=pl.Float64)


def _adjust_series(series: pl.Series, method: str, n: int | None) -> pl.Series:
    values = series.cast(pl.Float64).to_numpy().astype(float)
    return pl.Series(series.name, adjust_p(values, method, n), dtype=pl.Float64).fill_nan(None)


def adjust_p(p: np.ndarray, method: str, n: int | None = None) -> np.ndarray:
    """R's ``p.adjust`` on a NumPy array (NaN is missing)."""
    if method == "fdr":
        method = "BH"
    out = np.array(p, dtype=float)
    present = ~np.isnan(out)
    q = out[present]
    lp = len(q)
    if n is None:
        n = lp
    if n < lp:
        raise ValueError(f"p_adjust(): n ({n}) is smaller than the number of p-values ({lp})")
    if n <= 1 or lp == 0:
        return out
    if n == 2 and method == "hommel":
        method = "hochberg"
    if method == "bonferroni":
        adjusted = np.minimum(1, n * q)
    elif method == "holm":
        i = np.arange(1, lp + 1)
        o = np.argsort(q, kind="stable")
        adjusted = np.empty(lp)
        adjusted[o] = np.minimum(1, np.maximum.accumulate((n + 1 - i) * q[o]))
    elif method in ("hochberg", "BH", "BY"):
        i = np.arange(lp, 0, -1)
        o = np.argsort(-q, kind="stable")
        if method == "hochberg":
            scaled = (n + 1 - i) * q[o]
        elif method == "BH":
            scaled = n / i * q[o]
        else:
            scaled = np.sum(1 / np.arange(1, n + 1)) * n / i * q[o]
        adjusted = np.empty(lp)
        adjusted[o] = np.minimum(1, np.minimum.accumulate(scaled))
    elif method == "hommel":
        full = np.r_[q, np.ones(n - lp)] if n > lp else q.copy()
        i = np.arange(1, n + 1)
        o = np.argsort(full, kind="stable")
        s = full[o]
        pa = np.full(n, np.min(n * s / i))
        qq = pa.copy()
        for m in range(n - 1, 1, -1):
            i1 = np.arange(n - m + 1)
            i2 = np.arange(n - m + 1, n)
            q1 = np.min(m * s[i2] / np.arange(2, m + 1))
            qq[i1] = np.minimum(m * s[i1], q1)
            qq[i2] = qq[n - m]
            pa = np.maximum(pa, qq)
        result = np.empty(n)
        result[o] = np.maximum(pa, s)
        adjusted = result[:lp]
    else:
        adjusted = q
    out[present] = adjusted
    return out


# effect sizes ----------------------------------------------------------------------


def _ncp_t(t: float, df: float, conf_level: float) -> tuple[float, float]:
    """Noncentrality interval for an observed t (two-sided), as effectsize."""
    from scipy.optimize import brentq
    from scipy.stats import nct

    alpha = 1 - conf_level

    def cdf(ncp: float) -> float:
        value = float(nct.cdf(t, df, ncp))
        if math.isnan(value):  # far in a tail: decreasing in ncp
            return 0.0 if ncp > t else 1.0
        return value

    def solve(prob: float) -> float:
        f = lambda ncp: cdf(ncp) - prob  # noqa: E731
        step = 1.0
        low, high = t - step, t + step
        while f(low) < 0:
            step *= 2
            low = t - step
        step = 1.0
        while f(high) > 0:
            step *= 2
            high = t + step
        return brentq(f, low, high, xtol=1e-12, rtol=1e-14, maxiter=500)

    return solve(1 - alpha / 2), solve(alpha / 2)


def _ncp_lower(cdf: Any, statistic: float, conf_level: float, start: float) -> float:
    """effectsize's lower noncentrality bound for an F or chi-squared
    statistic: the ``ncp`` putting ``statistic`` at the ``conf_level``
    quantile, or 0 when the central distribution already does. (effectsize
    finds it with ``optim``, to about 1e-8; this solves it exactly.)"""
    from scipy.optimize import brentq

    if not math.isfinite(statistic) or cdf(statistic, 0.0) <= conf_level:
        return 0.0
    upper = max(10.0, start)
    while cdf(statistic, upper) > conf_level:
        upper *= 2
    return brentq(lambda ncp: cdf(statistic, ncp) - conf_level, 0.0, upper,
                  xtol=1e-12, rtol=1e-14, maxiter=500)


def _ncp_f_lower(f: float, df1: float, df2: float, conf_level: float) -> float:
    from scipy.stats import f as f_dist, ncf

    def cdf(x: float, ncp: float) -> float:
        return float(f_dist.cdf(x, df1, df2) if ncp == 0 else ncf.cdf(x, df1, df2, ncp))

    return _ncp_lower(cdf, f, conf_level, f * df1 * 4)


def _ncp_chi_lower(chisq: float, df: float, conf_level: float) -> float:
    from scipy.stats import chi2, ncx2

    def cdf(x: float, ncp: float) -> float:
        return float(chi2.cdf(x, df) if ncp == 0 else ncx2.cdf(x, df, ncp))

    return _ncp_lower(cdf, chisq, conf_level, chisq * 4)


def _hedges_j(df: np.ndarray) -> np.ndarray:
    special = _special()
    return np.exp(special.gammaln(df / 2) - np.log(np.sqrt(df / 2)) - special.gammaln((df - 1) / 2))


def d_values(
    table: pl.DataFrame, kind: str, mu: float, pooled_sd: bool, hedges: bool,
    conf_level: float, alternative: str = "two.sided",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cohen's d (or Hedges' g) per row of a moments table, with
    effectsize's noncentral-t interval.

    ``kind`` is ``"one"`` (columns ``n``, ``mean``, ``var``: one sample, or
    paired differences) or ``"two"`` (``n1``, ``mean1``, ``var1``, ...). A
    one-sided ``alternative`` gives effectsize's one-sided interval: the
    two-sided one at ``2 * conf_level - 1``, the other end infinite.
    """
    level = conf_level if alternative == "two.sided" else 2 * conf_level - 1
    estimate, low, high = [], [], []
    for row in table.iter_rows(named=True):
        if kind == "one":
            n = row["n"]
            df = n - 1
            d = (row["mean"] - mu) / math.sqrt(row["var"])
            scale = 1 / math.sqrt(n)
        else:
            n1, n2 = row["n1"], row["n2"]
            v1 = row["var1"] if n1 > 1 else 0.0
            v2 = row["var2"] if n2 > 1 else 0.0
            if pooled_sd:
                df = n1 + n2 - 2
                sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df)
                scale = math.sqrt(1 / n1 + 1 / n2)
            else:
                sd = math.sqrt((v1 + v2) / 2)
                se = math.sqrt(v1 / n1 + v2 / n2)
                df = se**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
                scale = se / sd
            d = (row["mean1"] - row["mean2"] - mu) / sd
        lo, hi = _ncp_t(d / scale, df, level)
        j = float(_hedges_j(np.array([float(df)]))[0]) if hedges else 1.0
        estimate.append(d * j)
        low.append(-math.inf if alternative == "less" else lo * scale * j)
        high.append(math.inf if alternative == "greater" else hi * scale * j)
    return np.array(estimate, dtype=float), np.array(low, dtype=float), np.array(high, dtype=float)


def _d_family(
    x: Any, y: Any, mu: float, paired: bool, pooled_sd: bool, conf_level: float,
    verb: str, hedges: bool,
) -> Step:
    _check_mu(mu)
    _check_conf_level(conf_level)
    formula = _is_formula(x)
    if formula and (paired or y is not None):
        raise ValueError(f"{verb}(): pass a formula, or x and y (paired=True needs x and y)")
    if paired and y is None:
        raise ValueError(f"{verb}(): paired=True needs two columns, x and y")

    def _apply(tf: Any) -> Any:
        groups = list(tf._groups or [])
        lf = tf._lf
        if formula:
            response, by = _parse_formula(tf, x, verb)
            if by is None:
                kind, table = "one", _summarise(
                    lf, groups, _moments(pl.col(response).filter(_present(lf, response)).cast(pl.Float64))
                )
            else:
                kept = lf.filter(_present(lf, response) & _present(lf, by))
                stacked = _by_levels(tf, kept, groups, by, _moments(pl.col(response).cast(pl.Float64)),
                                     verb, exactly=2)
                kind = "two"
                table = stacked.select(*groups, *(pl.col(s).list.get(i).alias(f"{s}{i + 1}")
                                                  for s in ("n", "mean", "var") for i in (0, 1)))
        else:
            xc = _column(tf, x, "x", verb)
            if y is None:
                kind = "one"
                table = _summarise(lf, groups, _moments(pl.col(xc).filter(_present(lf, xc)).cast(pl.Float64)))
            else:
                yc = _column(tf, y, "y", verb)
                if paired:
                    both = _present(lf, xc) & _present(lf, yc)
                    diff = (pl.col(xc).cast(pl.Float64) - pl.col(yc).cast(pl.Float64)).filter(both)
                    kind, table = "one", _summarise(lf, groups, _moments(diff))
                else:
                    kind = "two"
                    aggs = []
                    for side, name in (("1", xc), ("2", yc)):
                        value = pl.col(name).filter(_present(lf, name)).cast(pl.Float64)
                        aggs += [e.name.suffix(side) for e in _moments(value)]
                    table = _summarise(lf, groups, aggs)
        table = _sorted_keys(tf, table, groups)
        for row in table.iter_rows(named=True):
            if (row["n"] if kind == "one" else min(row["n1"], row["n2"])) < 2:
                raise ValueError(f"{verb}(): not enough observations{_group_label(row, groups)}")
        estimate, low, high = d_values(table, kind, mu, pooled_sd, hedges, conf_level)
        return _finish(tf, table, groups, {"estimate": estimate, "conf_low": low, "conf_high": high})

    return Step(_apply, verb)


@step
def cohens_d(
    x: Any, y: Any = None, *, mu: float = 0.0, paired: bool = False,
    pooled_sd: bool = True, conf_level: float = 0.95,
) -> Step:
    """Cohen's d, as effectsize's ``cohens_d``, with its noncentral-t interval.

    ``cohens_d("body_mass_g ~ sex")`` (first level minus second, divided by
    the pooled SD, or by the root mean of the two variances with
    ``pooled_sd=False``), ``cohens_d(x, mu=...)`` for one sample, and
    ``cohens_d(x, y, paired=True)`` for paired differences. Columns:
    ``estimate``, ``conf_low``, ``conf_high``.
    """
    return _d_family(x, y, mu, paired, pooled_sd, conf_level, "cohens_d", hedges=False)


@step
def hedges_g(
    x: Any, y: Any = None, *, mu: float = 0.0, paired: bool = False,
    pooled_sd: bool = True, conf_level: float = 0.95,
) -> Step:
    """Hedges' g: Cohen's d with the exact small-sample correction, as
    effectsize's ``hedges_g``. Arguments and columns as ``cohens_d``."""
    return _d_family(x, y, mu, paired, pooled_sd, conf_level, "hedges_g", hedges=True)


@step
def eta_squared(formula: str, *, conf_level: float = 0.95) -> Step:
    """Eta squared for a one-way ANOVA, as effectsize's ``eta_squared``.

    The share of the response's variance between the levels. The interval is
    effectsize's default one-sided noncentral-F interval, so ``conf_high`` is
    1. Columns: ``term``, ``estimate``, ``conf_low``, ``conf_high``.
    """
    _check_conf_level(conf_level)

    def _apply(tf: Any) -> Any:
        groups = list(tf._groups or [])
        lf = tf._lf
        response, by = _parse_formula(tf, formula, "eta_squared", one_sample=False)
        kept = lf.filter(_present(lf, response) & _present(lf, by))
        table = _by_levels(tf, kept, groups, by, _moments(pl.col(response).cast(pl.Float64)), "eta_squared")
        estimate, low = [], []
        for row in table.iter_rows(named=True):
            n = np.array(row["n"], dtype=float)
            m = np.array(row["mean"], dtype=float)
            v = np.array([0.0 if x is None else x for x in row["var"]], dtype=float)
            total = n.sum()
            grand = np.sum(n * m) / total
            between = float(np.sum(n * (m - grand) ** 2))
            within = float(np.sum((n - 1) * v))
            df1, df2 = len(n) - 1, total - len(n)
            fval = (between / df1) / (within / df2)
            estimate.append(between / (between + within))
            # effectsize 1.0: the bound on eta squared is ncp / (ncp + df_error).
            ncp = _ncp_f_lower(fval, df1, df2, conf_level)
            low.append(ncp / (ncp + df2))
        return _finish(tf, table, groups, {
            "term": by,
            "estimate": np.array(estimate),
            "conf_low": np.array(low),
            "conf_high": np.ones(len(low)),
        })

    return Step(_apply, "eta_squared")


# effect sizes inside the tests -----------------------------------------------------
#
# ``t_test(..., effect_size=True)`` and the others add four columns, before
# ``method``: ``effect`` (the measure's name), ``effect_size``,
# ``effect_low`` and ``effect_high``. Each measure is the one effectsize's
# ``effectsize()`` gives for the R test, with its default interval.

EFFECT_COLUMNS = ("effect", "effect_size", "effect_low", "effect_high")


def effect_choice(value: Any, verb: str, choices: tuple[str, ...]) -> str | None:
    """The measure ``effect_size=`` asks for: None for off, the first of
    ``choices`` (effectsize's default) for True, else one of ``choices``."""
    if value is None or value is False:
        return None
    if value is True:
        return choices[0]
    if isinstance(value, str) and value in choices:
        return value
    allowed = ", ".join(repr(c) for c in choices)
    raise ValueError(f"{verb}(): effect_size must be True, False or one of {allowed}, got {value!r}")


def with_effect(columns: dict[str, Any], name: Any, estimate: Any, low: Any, high: Any) -> dict[str, Any]:
    """``columns`` with the effect size columns added before ``method`` (at
    the end without one). Missing values (NaN) become nulls."""

    def floats(name: str, values: Any) -> pl.Series:
        return pl.Series(name, np.asarray(values, dtype=float), dtype=pl.Float64, nan_to_null=True)

    effect = {
        "effect": name if isinstance(name, str) else pl.Series("effect", name, dtype=pl.String),
        "effect_size": floats("effect_size", estimate),
        "effect_low": floats("effect_low", low),
        "effect_high": floats("effect_high", high),
    }
    out: dict[str, Any] = {}
    for key, value in columns.items():
        if key == "method":
            out.update(effect)
        out[key] = value
    if "method" not in columns:
        out.update(effect)
    return out


def _one_sided_level(conf_level: float, alternative: str) -> float:
    """effectsize's ``.adjust_ci``: a one-sided interval is one end of the
    two-sided interval at ``2 * conf_level - 1``."""
    return conf_level if alternative == "two.sided" else 2 * conf_level - 1


def rank_biserial_values(
    r: np.ndarray, se: np.ndarray, alternative: str, conf_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """effectsize's rank-biserial interval: normal on Fisher's z scale, cut
    at -1 or 1 on the side a one-sided alternative leaves open."""
    special = _special()
    level = _one_sided_level(conf_level, alternative)
    z = float(special.ndtri(1 - (1 - level) / 2))
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        centre = np.arctanh(r)
        low = np.tanh(centre - z * se)
        high = np.tanh(centre + z * se)
    if alternative == "less":
        low = np.full_like(r, -1.0)
    elif alternative == "greater":
        high = np.full_like(r, 1.0)
    return r, low, high


def cramers_v_values(chisq: float, n: float, nrow: int, ncol: int, conf_level: float = 0.95) -> tuple[float, float, float]:
    """Bias-corrected Cramer's V, effectsize's default (``adjust = TRUE``),
    from the uncorrected Pearson statistic, with its one-sided
    noncentral chi-squared interval (so the upper end is 1)."""
    df = (nrow - 1) * (ncol - 1)
    shrink = df / (n - 1)
    k = nrow - (nrow - 1) ** 2 / (n - 1)
    l = ncol - (ncol - 1) ** 2 / (n - 1)
    scale = math.sqrt(min(k - 1, l - 1))

    def v(statistic: float) -> float:
        return math.sqrt(max(0.0, statistic / n - shrink)) / scale

    return v(chisq), v(_ncp_chi_lower(chisq, df, conf_level)), 1.0


def fei_values(chisq: float, n: float, p: np.ndarray, conf_level: float = 0.95) -> tuple[float, float, float]:
    """Fei, effectsize's effect size for a goodness-of-fit test: phi scaled
    by its largest value under the expected probabilities ``p``, with the
    one-sided noncentral chi-squared interval."""
    p = np.asarray(p, dtype=float) / np.sum(p)
    scale = math.sqrt(1 / float(np.min(p)) - 1)
    low = math.sqrt(_ncp_chi_lower(chisq, len(p) - 1, conf_level) / n) / scale
    return math.sqrt(chisq / n) / scale, low, 1.0


def partial_eta_squared(table: pl.DataFrame, conf_level: float = 0.95) -> pl.DataFrame:
    """An ANOVA table (one block per group, each ending in ``Residuals``)
    with effectsize's ``eta_squared`` for each term: partial eta squared,
    which with one term is eta squared, and its one-sided noncentral-F
    interval. ``Residuals`` rows get nulls."""
    terms = table["term"].to_list()
    dfs, ss, fs = table["df"].to_list(), table["sumsq"].to_list(), table["statistic"].to_list()
    name: list[str | None] = [None] * len(terms)
    estimate = np.full(len(terms), np.nan)
    low = np.full(len(terms), np.nan)
    high = np.full(len(terms), np.nan)
    start = 0
    for end, term in enumerate(terms):
        if term != "Residuals":
            continue
        rows = range(start, end)
        label = "eta_squared" if len(rows) == 1 else "eta_squared_partial"
        df_error, ss_error = dfs[end], ss[end]
        for i in rows:
            name[i] = label
            estimate[i] = ss[i] / (ss[i] + ss_error)
            if fs[i] is not None and df_error > 0:
                ncp = _ncp_f_lower(fs[i], dfs[i], df_error, conf_level)
                low[i] = ncp / (ncp + df_error)
                high[i] = 1.0
        start = end + 1
    columns = with_effect({}, name, estimate, low, high)
    return table.with_columns(pl.Series(k, v) if not isinstance(v, pl.Series) else v for k, v in columns.items())


# power -------------------------------------------------------------------------------


def _solve(f: Any, low: float, high: float, extend: str) -> float:
    """Root of an increasing ("up") or decreasing ("down") function, as uniroot."""
    from scipy.optimize import brentq

    for _ in range(200):
        a, b = f(low), f(high)
        if a * b <= 0:
            return brentq(f, low, high, xtol=1e-12, rtol=1e-14, maxiter=1000)
        if extend == "up":
            high *= 2
        elif extend == "down":
            low /= 2
        else:
            low, high = low / 2, high * 2
    raise ValueError("no solution found in the search range")


def _one_row(as_: str, columns: dict[str, list[Any]]) -> Any:
    if as_ not in OUTPUTS:
        raise ValueError(f"as_ must be one of {OUTPUTS}, got {as_!r}")
    return Data(pl.LazyFrame(), [], {}, as_).write(pl.DataFrame(columns))


def _one_null(**values: Any) -> str:
    missing = [k for k, v in values.items() if v is None]
    if len(missing) != 1:
        raise ValueError(f"exactly one of {', '.join(values)} must be None")
    return missing[0]


def power_t_test(
    n: float | None = None,
    delta: float | None = None,
    sd: float | None = 1.0,
    sig_level: float | None = 0.05,
    power: float | None = None,
    *,
    type: str = "two.sample",  # noqa: A002
    alternative: str = "two.sided",
    strict: bool = False,
    as_: str = "polars",
) -> Any:
    """Power of a t-test, or the value of whichever argument is None that
    reaches the given power, as R's ``power.t.test``. Returns a one-row
    frame: Polars, or ``as_="pandas"`` / ``"tidy3"``.

    ``n`` is per group for two samples and the number of pairs for paired.
    """
    from scipy.stats import nct, t as t_dist

    missing = _one_null(n=n, delta=delta, sd=sd, power=power, sig_level=sig_level)
    if type not in ("two.sample", "one.sample", "paired"):
        raise ValueError("type must be 'two.sample', 'one.sample', or 'paired'")
    if alternative not in ("two.sided", "one.sided"):
        raise ValueError("alternative must be 'two.sided' or 'one.sided'")
    tsample = 2 if type == "two.sample" else 1
    tside = 2 if alternative == "two.sided" else 1
    if tside == 2 and delta is not None:
        delta = abs(delta)

    def body(n: float, delta: float, sd: float, sig_level: float) -> float:
        nu = max(1e-7, n - 1) * tsample
        qu = t_dist.isf(sig_level / tside, nu)
        ncp = math.sqrt(n / tsample) * delta / sd
        p = nct.sf(qu, nu, ncp)
        if strict and tside == 2:
            p += nct.cdf(-qu, nu, ncp)
        return float(p)

    values = {"n": n, "delta": delta, "sd": sd, "sig_level": sig_level}
    if missing == "power":
        power = body(**values)
    else:
        def f(v: float) -> float:
            return body(**{**values, missing: v}) - power

        if missing == "n":
            values["n"] = _solve(f, 2.0, 1e7, "up")
        elif missing == "sd":
            values["sd"] = _solve(f, delta * 1e-7, delta * 1e7, "down")
        elif missing == "delta":
            values["delta"] = _solve(f, sd * 1e-7, sd * 1e7, "up")
        else:
            values["sig_level"] = _solve(f, 1e-10, 1 - 1e-10, "both")
    method = {"one.sample": "One-sample", "two.sample": "Two-sample", "paired": "Paired"}[type]
    return _one_row(as_, {
        **{k: [float(v)] for k, v in values.items()},
        "power": [float(power)],
        "alternative": [alternative],
        "method": [f"{method} t test power calculation"],
    })


def power_prop_test(
    n: float | None = None,
    p1: float | None = None,
    p2: float | None = None,
    sig_level: float | None = 0.05,
    power: float | None = None,
    *,
    alternative: str = "two.sided",
    strict: bool = False,
    as_: str = "polars",
) -> Any:
    """Power of a two-sample test of proportions, or the value of whichever
    argument is None that reaches the given power, as R's
    ``power.prop.test``. ``n`` is per group. Returns a one-row frame:
    Polars, or ``as_="pandas"`` / ``"tidy3"``."""

    special = _special()
    missing = _one_null(n=n, p1=p1, p2=p2, power=power, sig_level=sig_level)
    if alternative not in ("two.sided", "one.sided"):
        raise ValueError("alternative must be 'two.sided' or 'one.sided'")
    tside = 2 if alternative == "two.sided" else 1

    def body(n: float, p1: float, p2: float, sig_level: float) -> float:
        qu = float(special.ndtri(1 - sig_level / tside))
        if strict and tside == 2:
            d = abs(p1 - p2)
            pbar = (p1 + p2) / 2
            vbar = pbar * (1 - pbar)
            v = p1 * (1 - p1) + p2 * (1 - p2)
            return float(special.ndtr((math.sqrt(n) * d - qu * math.sqrt(2 * vbar)) / math.sqrt(v))
                         + special.ndtr(-(math.sqrt(n) * d + qu * math.sqrt(2 * vbar)) / math.sqrt(v)))
        return float(special.ndtr(
            (math.sqrt(n) * abs(p1 - p2) - qu * math.sqrt((p1 + p2) * (1 - (p1 + p2) / 2)))
            / math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ))

    values = {"n": n, "p1": p1, "p2": p2, "sig_level": sig_level}
    if missing == "power":
        power = body(**values)
    else:
        def f(v: float) -> float:
            return body(**{**values, missing: v}) - power

        if missing == "n":
            values["n"] = _solve(f, 1.0, 1e7, "up")
        elif missing == "p1":
            values["p1"] = _solve(f, 1e-12, p2, "both")
        elif missing == "p2":
            values["p2"] = _solve(f, p1, 1 - 1e-12, "both")
        else:
            values["sig_level"] = _solve(f, 1e-10, 1 - 1e-10, "up")
    return _one_row(as_, {
        **{k: [float(v)] for k, v in values.items()},
        "power": [float(power)],
        "alternative": [alternative],
        "method": ["Two-sample comparison of proportions power calculation"],
    })


def power_anova_test(
    groups: float | None = None,
    n: float | None = None,
    between_var: float | None = None,
    within_var: float | None = None,
    sig_level: float | None = 0.05,
    power: float | None = None,
    *,
    as_: str = "polars",
) -> Any:
    """Power of a balanced one-way ANOVA, as R's ``power.anova.test``.

    Give all but one of ``groups``, ``n`` (per group), ``between_var``,
    ``within_var``, ``sig_level`` and ``power`` (pass ``sig_level=None`` to
    solve for it); the missing one is solved for. Returns one row: the six
    values and ``method``.
    """
    values = {"groups": groups, "n": n, "between_var": between_var,
              "within_var": within_var, "power": power, "sig_level": sig_level}
    missing = [name for name, value in values.items() if value is None]
    if len(missing) != 1:
        raise ValueError(
            "power_anova_test(): exactly one of 'groups', 'n', 'between_var', "
            "'within_var', 'power', and 'sig_level' must be None"
        )
    if groups is not None and groups < 2:
        raise ValueError("power_anova_test(): number of groups must be at least 2")
    if n is not None and n < 2:
        raise ValueError("power_anova_test(): number of observations in each group must be at least 2")
    for name in ("sig_level", "power"):
        if values[name] is not None and not 0 <= values[name] <= 1:
            raise ValueError(f"power_anova_test(): '{name}' must be numeric in [0, 1]")
    from scipy.stats import f as f_dist, ncf

    def power_of(v: dict[str, float]) -> float:
        g, k = v["groups"], v["n"]
        df1, df2 = g - 1, (k - 1) * g
        ncp = df1 * k * (v["between_var"] / v["within_var"])
        return float(ncf.sf(f_dist.isf(v["sig_level"], df1, df2), df1, df2, ncp))

    target = missing[0]
    if target == "power":
        values["power"] = power_of(values)
    else:
        ranges = {
            "groups": (2, 100), "n": (2, 1e5),
            "within_var": (between_var * 1e-7, between_var * 1e7) if between_var else (0, 0),
            "between_var": (within_var * 1e-7, within_var * 1e7) if within_var else (0, 0),
            "sig_level": (1e-10, 1 - 1e-10),
        }

        def f(value: float) -> float:
            return power_of(values | {target: value}) - power

        from scipy.optimize import brentq

        low, high = ranges[target]
        if f(low) * f(high) > 0:
            raise ValueError(
                f"power_anova_test(): no {target} between {low:g} and {high:g} gives "
                f"power {power}, as R's uniroot searches"
            )
        values[target] = brentq(f, low, high, xtol=1e-12, rtol=1e-14, maxiter=1000)
    return _one_row(as_, {
        **{name: [float(values[name])] for name in ("groups", "n", "between_var", "within_var", "sig_level", "power")},
        "method": ["Balanced one-way analysis of variance power calculation"],
    })
