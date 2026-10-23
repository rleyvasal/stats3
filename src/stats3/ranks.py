"""Rank tests and correlation tests: R's ``wilcox.test``, ``kruskal.test``
and ``cor.test``.

Ranks, rank sums, tie counts and correlations come from one grouped Polars
query. Exact p-values (small samples without ties, as R decides) depend only
on the statistic and the sample sizes, so they come from exact null
distributions computed here and cached by size. Kendall's tau needs every
pair of rows, so it runs per group on the collected values.
"""

from __future__ import annotations

import itertools
import math
import warnings
from functools import lru_cache
from typing import Any

import numpy as np
import polars as pl

from tidy3.verbs import Verb
from stats3.core import (
    _alternative,
    _by_levels,
    _check_conf_level,
    _check_mu,
    _column,
    _finish,
    _group_label,
    _is_formula,
    _keys,
    _p_normal,
    _parse_formula,
    _present,
    _sorted_keys,
    _special,
    _summarise,
)


# exact null distributions -----------------------------------------------


@lru_cache(maxsize=None)
def _wilcox_counts(m: int, n: int) -> np.ndarray:
    """Ways to reach each Mann-Whitney statistic 0..m*n (R's ``dwilcox``)."""
    if m == 0 or n == 0:
        return np.ones(1)
    out = np.zeros(m * n + 1)
    out[n:] += _wilcox_counts(m - 1, n)
    out[: m * (n - 1) + 1] += _wilcox_counts(m, n - 1)
    return out


@lru_cache(maxsize=None)
def _signrank_counts(n: int) -> np.ndarray:
    """Subsets of 1..n reaching each rank sum 0..n(n+1)/2 (R's ``dsignrank``)."""
    out = np.ones(1)
    for k in range(1, n + 1):
        grown = np.zeros(len(out) + k)
        grown[: len(out)] += out
        grown[k:] += out
        out = grown
    return out


@lru_cache(maxsize=None)
def _kendall_counts(n: int) -> np.ndarray:
    """Permutations of n with each number of inversions (R's ``ckendall``)."""
    out = np.ones(1)
    for k in range(2, n + 1):
        out = np.convolve(out, np.ones(k))
    return out


def _cdf(counts: np.ndarray, q: float) -> float:
    """P(X <= q) for a distribution given as counts on 0, 1, 2, ..."""
    q = math.floor(q + 1e-7)
    if q < 0:
        return 0.0
    if q >= len(counts) - 1:
        return 1.0
    return float(counts[: q + 1].sum() / counts.sum())


def _upper(counts: np.ndarray, q: float) -> float:
    """P(X > q)."""
    q = math.floor(q + 1e-7)
    if q < 0:
        return 1.0
    if q >= len(counts) - 1:
        return 0.0
    return float(counts[q + 1 :].sum() / counts.sum())


@lru_cache(maxsize=None)
def _spearman_exact(n: int) -> np.ndarray:
    """Sum of squared rank differences over all permutations of n <= 9."""
    base = np.arange(n)
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int16)
    return np.sort(((perms - base) ** 2).sum(axis=1))


_PRHO = (0.2274, 0.2531, 0.1745, 0.0758, 0.1033, 0.3932,
         0.0879, 0.0151, 0.0072, 0.0831, 0.0131, 4.6e-4)


def _prho(n: int, s: float, lower: bool) -> float:
    """R's ``prho`` (AS 89): P(S >= s), or P(S < s) when ``lower``.

    Exact by enumeration for n <= 9, an Edgeworth series above.
    """
    pv = 0.0 if lower else 1.0
    if s <= 0:
        return pv
    n3 = n * (n * n - 1) / 3
    if s > n3:
        return 1 - pv
    if n <= 9:
        values = _spearman_exact(n)
        at_least = len(values) - np.searchsorted(values, s, side="left")
        return float((len(values) - at_least) / len(values) if lower else at_least / len(values))
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = _PRHO
    b = 1 / n
    x = (6 * (s - 1) * b / (n * n - 1) - 1) * math.sqrt(1 / b - 1)
    y = x * x
    u = x * b * (c1 + b * (c2 + c3 * b) + y * (-c4 + b * (c5 + c6 * b)
        - y * b * (c7 + c8 * b - y * (c9 - c10 * b + y * b * (c11 - c12 * y)))))
    y = u / math.exp(y / 2)
    special = _special()
    pv = (-y + float(special.ndtr(x))) if lower else (y + float(special.ndtr(-x)))
    return min(max(pv, 0.0), 1.0)


def _tie_term(lf: pl.LazyFrame, groups: list[str], value: pl.Expr) -> pl.DataFrame:
    """Per group: sum of t^3 - t over runs of tied values."""
    counts = lf.group_by([*groups, value.alias("__stats3_v")]).agg(t=pl.len())
    term = (pl.col("t").cast(pl.Float64) ** 3 - pl.col("t")).sum().alias("ties")
    if groups:
        return counts.group_by(groups).agg(term).collect()
    return counts.select(term).collect()


def _join_ties(table: pl.DataFrame, ties: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
    if not groups:
        return table.with_columns(ties=ties["ties"])
    return table.join(ties, on=groups, how="left", nulls_equal=True, maintain_order="left")


def _correction(z: np.ndarray, alternative: str, correct: bool) -> np.ndarray:
    if not correct:
        return np.zeros_like(z)
    if alternative == "two.sided":
        return np.sign(z) * 0.5
    return np.full_like(z, 0.5 if alternative == "greater" else -0.5)


def _two_sided(lower_p: float, upper_p: float) -> float:
    return min(2 * min(lower_p, upper_p), 1.0)


# wilcox_test ---------------------------------------------------------------


def _scaled(scores: np.ndarray) -> tuple[np.ndarray, int]:
    """Rank scores as integers: doubled when some are half ranks (ties)."""
    f = 1 if np.all(scores == np.floor(scores)) else 2
    return np.rint(scores * f).astype(np.int64), f


def _subset_sum_counts(scores: np.ndarray) -> np.ndarray:
    """Counts of sums over all subsets of the integer scores."""
    out = np.ones(1)
    for a in scores:
        grown = np.zeros(len(out) + a)
        grown[: len(out)] += out
        grown[a:] += out
        out = grown
    return out


def _choose_sum_counts(scores: np.ndarray, m: int) -> np.ndarray:
    """Counts of sums over all m-element subsets of the integer scores."""
    total = int(scores.sum())
    dp = np.zeros((m + 1, total + 1))
    dp[0, 0] = 1.0
    for a in scores:
        dp[1:, a:] = dp[1:, a:] + dp[:-1, : total + 1 - a].copy()
    return dp[m]


def _tails(counts: np.ndarray, q: float, f: int, shift: float = 0.0) -> tuple[float, float]:
    """P(S <= q) and P(S >= q) for S = (index / f) - shift."""
    k = (q + shift) * f
    total = counts.sum()
    lo = math.floor(k + 1e-7)
    hi = math.ceil(k - 1e-7)
    lower = counts[: max(lo + 1, 0)].sum() / total
    upper = counts[max(hi, 0):].sum() / total
    return float(lower), float(upper)


def _edgeworth(z: float, alternative: str, correct: int, e4: float, e6: float) -> float:
    """R 4.6's normal p-value with ``correct`` 1-3 Edgeworth terms."""
    special = _special()

    def tail(z: float, lower: bool) -> float:
        y = float(special.ndtr(z if lower else -z))
        if correct < 1:
            return y
        z2 = z * z
        e = e4 * z * (z2 - 3)
        if correct > 1:
            e += e6 * z * ((z2 - 10) * z2 + 15)
        if correct > 2:
            e += e4**2 / 2 * z * (((z2 - 21) * z2 + 105) * z2 - 105)
        e *= math.exp(-z2 / 2) / math.sqrt(2 * math.pi)
        return min(1.0, max(y - e if lower else y + e, 0.0))

    if alternative == "less":
        return tail(z, True)
    if alternative == "greater":
        return tail(z, False)
    if correct < 1:
        return 2 * float(special.ndtr(-abs(z)))
    p = tail(z, True)
    return 2 * min(p, 1 - p)


def _correct_level(correct: bool | int) -> int:
    """R 4.6: TRUE is 0 (continuity correction), FALSE is -1, or 0-3."""
    if isinstance(correct, bool):
        return 0 if correct else -1
    if correct not in (0, 1, 2, 3):
        raise ValueError(f"correct must be True, False, or an integer 0-3, got {correct!r}")
    return int(correct)


def _shift(z: float, alternative: str, level: int) -> float:
    if level < 0:
        return 0.0
    if alternative == "two.sided":
        return math.copysign(0.5, z) if z else 0.0
    return 0.5 if alternative == "greater" else -0.5


def wilcox_test(
    x: Any,
    y: Any = None,
    *,
    mu: float = 0.0,
    paired: bool = False,
    exact: bool | None = None,
    correct: bool | int = True,
    alternative: str = "two.sided",
) -> Verb:
    """Wilcoxon rank sum and signed rank tests, as R 4.6's ``wilcox.test``.

    - ``wilcox_test("body_mass_g ~ sex")`` or ``wilcox_test(x, y)``: rank
      sum test (Mann-Whitney); ``statistic`` is R's W.
    - ``wilcox_test(x, mu=...)``: signed rank test; ``statistic`` is V.
    - ``wilcox_test(x, y, paired=True)``: signed rank test on ``x - y``.

    As in R, the p-value is exact when the samples have fewer than 50 values
    (unless ``exact`` says otherwise), using the permutation distribution of
    the ranks, so ties and zero differences are handled exactly. Otherwise
    it is the normal approximation with tie correction; ``correct=True``
    adds the continuity correction, and ``correct=1``-``3`` add R's Edgeworth
    terms when there are no ties. Columns: ``statistic``, ``p_value``,
    ``method``, ``alternative``.
    """
    alternative = _alternative(alternative)
    _check_mu(mu)
    level = _correct_level(correct)
    formula = _is_formula(x)
    if formula and paired:
        raise ValueError(
            "wilcox_test(): paired=True takes two columns, "
            "wilcox_test(x, y, paired=True), not a formula"
        )
    if formula and y is not None:
        raise ValueError("wilcox_test(): pass a formula or x and y, not both")
    if paired and y is None:
        raise ValueError("wilcox_test(): paired=True needs two columns, x and y")

    def _apply(tf: Any) -> Any:
        groups = list(tf._groups or [])
        lf = tf._lf
        if formula:
            response, by = _parse_formula(tf, x, "wilcox_test")
            if by is None:
                value = pl.col(response).cast(pl.Float64)
                return _signed_rank(tf, groups, value, _present(lf, response))
            return _rank_sum_by(tf, groups, response, by)
        xc = _column(tf, x, "x", "wilcox_test")
        if y is None:
            return _signed_rank(tf, groups, pl.col(xc).cast(pl.Float64), _present(lf, xc))
        yc = _column(tf, y, "y", "wilcox_test")
        if paired:
            diff = pl.col(xc).cast(pl.Float64) - pl.col(yc).cast(pl.Float64)
            return _signed_rank(tf, groups, diff, _present(lf, xc) & _present(lf, yc))
        return _rank_sum_columns(tf, groups, xc, yc)

    def _over(expr: pl.Expr, groups: list[str]) -> pl.Expr:
        return expr.over(groups) if groups else expr

    def _signed_rank(tf: Any, groups: list[str], value: pl.Expr, present: pl.Expr) -> Any:
        d = pl.col("__stats3_d")
        lf = tf._lf.filter(present).with_columns(__stats3_d=value - mu)
        # The exact test ranks |d| with zeros included, then leaves the zeros
        # out of the permutation scores; the normal approximation drops zeros
        # before ranking. Both follow R 4.6.
        lf = lf.with_columns(
            __stats3_re=_over(d.abs().rank("average"), groups),
            __stats3_ra=_over(pl.when(d != 0).then(d.abs()).rank("average"), groups),
        )
        table = _summarise(lf, groups, [
            pl.len().alias("n_all"),
            (d != 0).sum().alias("n"),
            (d == 0).sum().alias("zeroes"),
            pl.col("__stats3_re").filter(d > 0).sum().alias("v_exact"),
            pl.col("__stats3_ra").filter(d > 0).sum().alias("v"),
            (pl.col("__stats3_re").n_unique() < pl.len()).alias("ties_exact"),
        ])
        ranked = lf.filter(pl.col("__stats3_ra").is_not_null())
        table = _join_ties(table, _tie_term(ranked, groups, pl.col("__stats3_ra")), groups)
        table = _sorted_keys(tf, table, groups)
        scores = _scores_for(tf, table, groups, lf.filter(d != 0), "__stats3_re",
                             lambda row: (row["n_all"] < 50 if exact is None else exact)
                             and (row["ties_exact"] or row["zeroes"]))
        out: dict[str, list[Any]] = {"statistic": [], "p_value": [], "method": []}
        for i, row in enumerate(table.iter_rows(named=True)):
            n_all = row["n_all"]
            if n_all == 0:
                raise ValueError(
                    f"wilcox_test(): not enough (non-missing) 'x' observations{_group_label(row, groups)}"
                )
            use_exact = n_all < 50 if exact is None else exact
            if use_exact:
                q = float(row["v_exact"] or 0.0)
                if row["ties_exact"] or row["zeroes"]:
                    ints, f = _scaled(scores[i])
                    counts = _subset_sum_counts(ints)
                    middle = float(scores[i].sum()) / 2
                else:
                    counts, f = _signrank_counts(n_all), 1
                    middle = n_all * (n_all + 1) / 4
                lower, upper = _tails(counts, q, f)
                p = {"less": lower, "greater": upper}.get(
                    alternative, min(2 * (upper if q > middle else lower), 1.0)
                )
                method = "Wilcoxon signed rank exact test"
            else:
                n = row["n"]
                q = float(row["v"] or 0.0)
                ties = bool(row["ties"]) or bool(row["zeroes"])
                this = 0 if (level > 0 and ties) else level
                z = q - n * (n + 1) / 4
                sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - (row["ties"] or 0.0) / 48)
                z = (z - _shift(z, alternative, this)) / sigma
                nn2n = n_all * (n_all + 1) * (2 * n_all + 1)
                e4 = -((n_all + 1) * 3 * n_all - 1) / (10 * nn2n)
                e6 = 4 * (((3 * n_all + 6) * n_all**2 - 3) * n_all + 1) / (35 * nn2n**2)
                p = _edgeworth(z, alternative, this, e4, e6)
                method = "Wilcoxon signed rank test" + (" with continuity correction" if level >= 0 else "")
            out["statistic"].append(q)
            out["p_value"].append(p)
            out["method"].append(method)
        return _finish(tf, table, groups, {**out, "alternative": alternative})

    def _rank_sum(tf: Any, groups: list[str], table: pl.DataFrame, ranked: pl.LazyFrame) -> Any:
        """``table`` has n1, n2, r1 (rank sum of the first sample) and ties."""
        def needs_scores(row: dict[str, Any]) -> bool:
            small = (row["n1"] < 50 and row["n2"] < 50) if exact is None else exact
            return bool(small and row["ties"])

        scores = _scores_for(tf, table, groups, ranked, "__stats3_r", needs_scores)
        out: dict[str, list[Any]] = {"statistic": [], "p_value": [], "method": []}
        for i, row in enumerate(table.iter_rows(named=True)):
            n1, n2 = row["n1"], row["n2"]
            label = _group_label(row, groups)
            if not n1:
                raise ValueError(f"wilcox_test(): not enough (non-missing) 'x' observations{label}")
            if not n2:
                raise ValueError(f"wilcox_test(): not enough 'y' observations{label}")
            w = float(row["r1"]) - n1 * (n1 + 1) / 2
            ties = bool(row["ties"])
            use_exact = (n1 < 50 and n2 < 50) if exact is None else exact
            if use_exact:
                if ties:
                    ints, f = _scaled(scores[i])
                    counts = _choose_sum_counts(ints, n1)
                    shift = n1 * (n1 + 1) / 2
                else:
                    counts, f, shift = _wilcox_counts(n1, n2), 1, 0.0
                lower, upper = _tails(counts, w, f, shift)
                p = {"less": lower, "greater": upper}.get(
                    alternative, min(2 * lower, 2 * upper, 1.0)
                )
                method = "Wilcoxon rank sum exact test"
            else:
                total = n1 + n2
                this = 0 if (level > 0 and ties) else level
                z = w - n1 * n2 / 2
                sigma = math.sqrt(
                    (n1 * n2 / 12) * ((total + 1) - row["ties"] / (total * (total - 1)))
                )
                z = (z - _shift(z, alternative, this)) / sigma
                mn, m2, nsq, mpn = n1 * n2, n1**2, n2**2, total
                mpn1 = mpn + 1
                e4 = -((mpn1 * mpn - mn) / (20 * mn * mpn1))
                n5 = (2 * (m2**2 + nsq**2) + 4 * mn * (m2 + nsq) + 6 * m2 * nsq
                      + 4 * (m2 * n1 + nsq * n2) + (7 * mn + mpn - 1) * mpn)
                e6 = n5 / (210 * m2 * nsq * mpn1**2)
                p = _edgeworth(z, alternative, this, e4, e6)
                method = "Wilcoxon rank sum test" + (" with continuity correction" if level >= 0 else "")
            out["statistic"].append(w)
            out["p_value"].append(p)
            out["method"].append(method)
        return _finish(tf, table, groups, {**out, "alternative": alternative})

    def _rank_sum_by(tf: Any, groups: list[str], response: str, by: str) -> Any:
        lf = tf._lf
        kept = lf.filter(_present(lf, response) & _present(lf, by))
        levels = _by_levels(tf, kept, groups, by, [pl.len().alias("n")], "wilcox_test", exactly=2)
        first = levels.select(*groups, pl.col(by).list.first().alias("__stats3_first"))
        if groups:
            kept = kept.join(first.lazy(), on=groups, how="left", nulls_equal=True)
        else:
            kept = kept.with_columns(__stats3_first=pl.lit(first["__stats3_first"][0]))
        # R ranks c(x - mu, y): the first level is shifted by mu.
        is_first = pl.col(by) == pl.col("__stats3_first")
        value = pl.col(response).cast(pl.Float64)
        kept = kept.with_columns(__stats3_v=pl.when(is_first).then(value - mu).otherwise(value))
        return _rank_sum_on(tf, groups, kept.with_columns(__stats3_first=is_first))

    def _rank_sum_columns(tf: Any, groups: list[str], xc: str, yc: str) -> Any:
        lf = tf._lf
        parts = [
            lf.filter(_present(lf, name)).select(
                *groups,
                (pl.col(name).cast(pl.Float64) - shift).alias("__stats3_v"),
                pl.lit(first).alias("__stats3_first"),
            )
            for name, shift, first in ((xc, mu, True), (yc, 0.0, False))
        ]
        return _rank_sum_on(tf, groups, pl.concat(parts))

    def _rank_sum_on(tf: Any, groups: list[str], stacked: pl.LazyFrame) -> Any:
        stacked = stacked.with_columns(
            __stats3_r=_over(pl.col("__stats3_v").rank("average"), groups)
        )
        is_first = pl.col("__stats3_first")
        table = _summarise(stacked, groups, [
            is_first.sum().alias("n1"),
            (~is_first).sum().alias("n2"),
            pl.col("__stats3_r").filter(is_first).sum().alias("r1"),
        ])
        table = _join_ties(table, _tie_term(stacked, groups, pl.col("__stats3_v")), groups)
        return _rank_sum(tf, groups, _sorted_keys(tf, table, groups), stacked)

    return Verb(_apply, "wilcox_test")


def _scores_for(
    tf: Any,
    table: pl.DataFrame,
    groups: list[str],
    ranked: pl.LazyFrame,
    column: str,
    wanted: Any,
) -> list[np.ndarray | None]:
    """Rank scores of the groups whose exact test needs them (small groups)."""
    rows = list(table.iter_rows(named=True))
    need = [i for i, row in enumerate(rows) if wanted(row)]
    out: list[np.ndarray | None] = [None] * len(rows)
    if not need:
        return out
    if not groups:
        out[0] = ranked.select(column).collect()[column].to_numpy()
        return out
    keys = table[need].select(groups)
    values = (
        ranked.join(keys.lazy(), on=groups, how="semi", nulls_equal=True)
        .select(*groups, column).collect()
        .partition_by(groups, as_dict=True)
    )
    for i in need:
        key = tuple(rows[i][g] for g in groups)
        out[i] = values[key][column].to_numpy()
    return out


# kruskal_test ------------------------------------------------------------------


def kruskal_test(formula: str) -> Verb:
    """Kruskal-Wallis rank sum test, as R's ``kruskal.test(y ~ g)``.

    Ranks are computed per group after dropping rows missing the response or
    the grouping column, with R's tie correction. Columns: ``statistic``,
    ``df``, ``p_value``, ``method``.
    """

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        lf = tf._lf
        response, by = _parse_formula(tf, formula, "kruskal_test", one_sample=False)
        kept = lf.filter(_present(lf, response) & _present(lf, by))
        rank = pl.col(response).cast(pl.Float64).rank("average")
        kept = kept.with_columns(__stats3_r=rank.over(groups) if groups else rank)
        table = _by_levels(
            tf, kept, groups, by,
            [pl.len().alias("n"), pl.col("__stats3_r").sum().alias("r")], "kruskal_test",
        )
        table = _join_ties(table, _tie_term(kept, groups, pl.col(response)), groups)
        statistic, df = [], []
        for row in table.iter_rows(named=True):
            n_i = np.array(row["n"], dtype=float)
            r_i = np.array(row["r"], dtype=float)
            n = n_i.sum()
            h = 12 * np.sum(r_i**2 / n_i) / (n * (n + 1)) - 3 * (n + 1)
            h /= 1 - row["ties"] / (n**3 - n)
            statistic.append(float(h))
            df.append(float(len(n_i) - 1))
        statistic_arr, df_arr = np.array(statistic), np.array(df)
        return _finish(tf, table, groups, {
            "statistic": statistic_arr,
            "df": df_arr,
            "p_value": special.chdtrc(df_arr, statistic_arr),
            "method": "Kruskal-Wallis rank sum test",
        })

    return Verb(_apply, "kruskal_test")


# cor_test ----------------------------------------------------------------------

_COR_METHODS = {"pearson", "spearman", "kendall"}


def cor_test(
    x: Any,
    y: Any,
    *,
    method: str = "pearson",
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    exact: bool | None = None,
    continuity: bool = False,
) -> Verb:
    """Correlation test, as R's ``cor.test(x, y)``, on rows where both are present.

    - ``method="pearson"``: ``estimate`` is r; ``statistic`` is t on ``df``
      degrees of freedom; ``conf_low``/``conf_high`` from Fisher's z when there
      are at least 4 pairs.
    - ``method="spearman"``: ``estimate`` is rho; ``statistic`` is S. As in R,
      the p-value is exact (AS 89) for up to 1290 pairs without ties, else from
      the t approximation.
    - ``method="kendall"``: ``estimate`` is tau-b; ``statistic`` is T (exact,
      under 50 pairs without ties) or z (normal approximation).
    """
    alternative = _alternative(alternative)
    _check_conf_level(conf_level)
    if method not in _COR_METHODS:
        raise ValueError(f"method must be 'pearson', 'spearman', or 'kendall', got {method!r}")

    def _apply(tf: Any) -> Any:
        groups = list(tf._groups or [])
        xc = _column(tf, x, "x", "cor_test")
        yc = _column(tf, y, "y", "cor_test")
        lf = tf._lf
        kept = lf.filter(_present(lf, xc) & _present(lf, yc)).with_columns(
            __stats3_x=pl.col(xc).cast(pl.Float64), __stats3_y=pl.col(yc).cast(pl.Float64)
        )
        if method == "pearson":
            return _pearson(tf, groups, kept)
        if method == "spearman":
            return _spearman(tf, groups, kept)
        return _kendall(tf, groups, kept)

    def _pearson(tf: Any, groups: list[str], kept: pl.LazyFrame) -> Any:
        special = _special()
        table = _sorted_keys(tf, _summarise(kept, groups, [
            pl.len().alias("n"), pl.corr("__stats3_x", "__stats3_y").alias("r"),
        ]), groups)
        for row in table.iter_rows(named=True):
            if row["n"] < 3:
                raise ValueError(f"cor_test(): not enough finite observations{_group_label(row, groups)}")
        n = table["n"].cast(pl.Float64).to_numpy()
        r = table["r"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        df = n - 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.sqrt(df) * r / np.sqrt(1 - r**2)
        if alternative == "less":
            p = special.stdtr(df, t)
        elif alternative == "greater":
            p = special.stdtr(df, -t)
        else:
            p = 2 * np.minimum(special.stdtr(df, t), special.stdtr(df, -t))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.arctanh(r)
            sigma = 1 / np.sqrt(n - 3)
        if alternative == "less":
            low = np.full_like(r, -np.inf)
            high = z + special.ndtri(conf_level) * sigma
        elif alternative == "greater":
            low = z - special.ndtri(conf_level) * sigma
            high = np.full_like(r, np.inf)
        else:
            q = special.ndtri((1 + conf_level) / 2)
            low, high = z - q * sigma, z + q * sigma
        has_ci = n > 3
        return _finish(tf, table, groups, {
            "estimate": r,
            "statistic": t,
            "df": df,
            "p_value": p,
            "conf_low": pl.Series(np.where(has_ci, np.tanh(low), np.nan)).fill_nan(None),
            "conf_high": pl.Series(np.where(has_ci, np.tanh(high), np.nan)).fill_nan(None),
            "method": "Pearson's product-moment correlation",
            "alternative": alternative,
        })

    def _spearman(tf: Any, groups: list[str], kept: pl.LazyFrame) -> Any:
        special = _special()
        rx = pl.col("__stats3_x").rank("average")
        ry = pl.col("__stats3_y").rank("average")
        ranked = kept.with_columns(
            __stats3_rx=rx.over(groups) if groups else rx,
            __stats3_ry=ry.over(groups) if groups else ry,
        )
        table = _sorted_keys(tf, _summarise(ranked, groups, [
            pl.len().alias("n"),
            pl.corr("__stats3_rx", "__stats3_ry").alias("r"),
            pl.col("__stats3_x").n_unique().alias("ux"),
            pl.col("__stats3_y").n_unique().alias("uy"),
        ]), groups)
        statistic, p_value = [], []
        use_exact = True if exact is None else exact
        for row in table.iter_rows(named=True):
            n, r = row["n"], row["r"]
            if n < 2:
                raise ValueError(f"cor_test(): not enough finite observations{_group_label(row, groups)}")
            ties = min(row["ux"], row["uy"]) < n
            this_exact = use_exact
            if ties and this_exact:
                this_exact = False
                warnings.warn("cannot compute exact p-value with ties", stacklevel=3)
            q = (n**3 - n) * (1 - r) / 6

            def pspearman(q: float, lower: bool) -> float:
                if n <= 1290 and this_exact:
                    return _prho(n, round(q) + 2 * lower, lower)
                den = n * (n * n - 1) / 6
                rr = 1 - q / den
                t = rr / math.sqrt((1 - rr * rr) / (n - 2))
                return float(special.stdtr(n - 2, -t if lower else t))

            if alternative == "two.sided":
                p = pspearman(q, False) if q > (n**3 - n) / 6 else pspearman(q, True)
                p = min(2 * p, 1.0)
            elif alternative == "greater":
                p = pspearman(q, True)
            else:
                p = pspearman(q, False)
            statistic.append(q)
            p_value.append(p)
        return _finish(tf, table, groups, {
            "estimate": table["r"].cast(pl.Float64),
            "statistic": np.array(statistic, dtype=float),
            "p_value": np.array(p_value, dtype=float),
            "method": "Spearman's rank correlation rho",
            "alternative": alternative,
        })

    def _kendall(tf: Any, groups: list[str], kept: pl.LazyFrame) -> Any:
        from scipy.stats import kendalltau

        values = kept.select(*groups, "__stats3_x", "__stats3_y").collect()
        parts = (
            values.partition_by(groups, maintain_order=True, as_dict=True)
            if groups else {(): values}
        )
        keys = pl.DataFrame([dict(zip(groups, key)) for key in parts], schema={g: values.schema[g] for g in groups}) if groups else pl.DataFrame({"__stats3_k": [0]})
        keys = _sorted_keys(tf, keys, groups) if groups else keys
        estimate, statistic, p_value = [], [], []
        for key in (keys.select(groups).iter_rows() if groups else [()]):
            part = parts[tuple(key)]
            row = dict(zip(groups, key))
            xv, yv = part["__stats3_x"].to_numpy(), part["__stats3_y"].to_numpy()
            n = len(xv)
            if n < 2:
                raise ValueError(f"cor_test(): not enough finite observations{_group_label(row, groups)}")
            tau = float(kendalltau(xv, yv).statistic)
            ties = min(len(np.unique(xv)), len(np.unique(yv))) < n
            use_exact = n < 50 if exact is None else exact
            if use_exact and not ties:
                q = round((tau + 1) * n * (n - 1) / 4)
                counts = _kendall_counts(n)
                lower = _cdf(counts, q)
                upper = _upper(counts, q - 1)
                p = {"less": lower, "greater": upper}.get(
                    alternative, min(2 * (upper if q > n * (n - 1) / 4 else lower), 1.0)
                )
                stat = float(q)
            else:
                tx = np.unique(xv, return_counts=True)[1].astype(float)
                ty = np.unique(yv, return_counts=True)[1].astype(float)
                tx, ty = tx[tx > 1], ty[ty > 1]
                t0 = n * (n - 1) / 2
                t1 = np.sum(tx * (tx - 1)) / 2
                t2 = np.sum(ty * (ty - 1)) / 2
                s = tau * math.sqrt((t0 - t1) * (t0 - t2))
                v0 = n * (n - 1) * (2 * n + 5)
                vt = np.sum(tx * (tx - 1) * (2 * tx + 5))
                vu = np.sum(ty * (ty - 1) * (2 * ty + 5))
                v1 = np.sum(tx * (tx - 1)) * np.sum(ty * (ty - 1))
                v2 = np.sum(tx * (tx - 1) * (tx - 2)) * np.sum(ty * (ty - 1) * (ty - 2))
                var_s = (v0 - vt - vu) / 18 + v1 / (2 * n * (n - 1)) + v2 / (9 * n * (n - 1) * (n - 2))
                if use_exact and ties:
                    warnings.warn("cannot compute exact p-value with ties", stacklevel=3)
                if continuity:
                    s = math.copysign(abs(s) - 1, s)
                stat = s / math.sqrt(var_s)
                p = float(_p_normal(np.array([stat]), alternative)[0])
            estimate.append(tau)
            statistic.append(stat)
            p_value.append(p)
        return _finish(tf, keys, groups, {
            "estimate": np.array(estimate),
            "statistic": np.array(statistic),
            "p_value": np.array(p_value),
            "method": "Kendall's rank correlation tau",
            "alternative": alternative,
        })

    return Verb(_apply, "cor_test")
