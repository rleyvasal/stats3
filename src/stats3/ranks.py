"""Rank tests and correlation tests: R's ``wilcox.test``, ``kruskal.test``,
``friedman.test``, ``quade.test``, ``ansari.test``, ``mood.test`` and
``cor.test``.

Ranks, rank sums, tie counts and correlations come from one grouped Polars
query. Exact p-values (small samples without ties, as R decides) depend only
on the statistic and the sample sizes, so they come from exact null
distributions computed here and cached by size. Kendall's tau needs every
pair of rows, and the Ansari-Bradley interval every ratio x_i / y_j, so they
run per group on the collected values.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Step, column_name, expression, step
from stats3.core import (
    _warn,
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


@step
def wilcox_test(
    x: Any,
    y: Any = None,
    *,
    mu: float = 0.0,
    paired: bool = False,
    exact: bool | None = None,
    correct: bool | int = True,
    alternative: str = "two.sided",
) -> Step:
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

    return Step(_apply, "wilcox_test")


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


@step
def kruskal_test(formula: str) -> Step:
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

    return Step(_apply, "kruskal_test")


def _block_formula(tf: Any, formula: str, verb: str) -> tuple[str, str, str]:
    """``"y ~ g | b"`` gives (y, g, b)."""
    left, bar, right = formula.partition("|")
    if not bar or not right.strip() or "|" in right:
        raise ValueError(f"{verb}(): formula must look like 'y ~ groups | blocks', got {formula!r}")
    response, by = _parse_formula(tf, left, verb, one_sample=False)
    block = _column(tf, right.strip(), "the formula's block", verb)
    if block in (tf._groups or []) or block == by:
        raise ValueError(f"{verb}(): {block!r} can't be the block and a group")
    return response, by, block


def _join_design(
    table: pl.DataFrame, kept: pl.LazyFrame, groups: list[str], by: str, block: str, *extra: pl.Expr
) -> pl.DataFrame:
    """Add each group's design counts (``cells``, ``rows``, ``blocks``) and ``extra``."""
    design = kept.group_by(groups or pl.lit(0)).agg(
        cells=pl.struct(by, block).n_unique(),
        rows=pl.len(),
        blocks=pl.col(block).n_unique(),
        *extra,
    ).collect()
    if groups:
        return table.join(design, on=groups, how="left", nulls_equal=True, maintain_order="left")
    return table.with_columns(design.drop("literal"))


def _check_design(row: dict[str, Any], groups: list[str], k: int, verb: str) -> None:
    if row["rows"] != row["cells"] or row["cells"] != k * row["blocks"]:
        raise ValueError(f"{verb}(): not an unreplicated complete block design{_group_label(row, groups)}")


@step
def friedman_test(formula: str) -> Step:
    """Friedman rank sum test, as R's ``friedman.test(y ~ g | b)``.

    ``g`` is the treatment and ``b`` the block: every block must have exactly
    one value for each treatment (an unreplicated complete block design),
    after dropping rows with a missing value. Values are ranked within each
    block, with R's tie correction. Columns: ``statistic``, ``df``,
    ``p_value``, ``method``.
    """

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        lf = tf._lf
        response, by, block = _block_formula(tf, formula, "friedman_test")
        kept = lf.filter(_present(lf, response) & _present(lf, by) & _present(lf, block))
        value = pl.col(response).cast(pl.Float64)
        kept = kept.with_columns(__stats3_r=value.rank("average").over([*groups, block]))
        table = _by_levels(
            tf, kept, groups, by,
            [pl.len().alias("n"), pl.col("__stats3_r").sum().alias("r")], "friedman_test",
        )
        ties = _tie_term(kept, [*groups, block], value)
        ties = (ties.group_by(groups).agg(pl.col("ties").sum()) if groups else ties.select(pl.col("ties").sum()))
        table = _join_design(_join_ties(table, ties, groups), kept, groups, by, block)
        statistic, df = [], []
        for row in table.iter_rows(named=True):
            r = np.array(row["r"], dtype=float)
            k, n = len(r), row["blocks"]
            _check_design(row, groups, k, "friedman_test")
            stat = 12 * np.sum((r - n * (k + 1) / 2) ** 2) / (n * k * (k + 1) - row["ties"] / (k - 1))
            statistic.append(float(stat))
            df.append(float(k - 1))
        statistic_arr, df_arr = np.array(statistic), np.array(df)
        return _finish(tf, table, groups, {
            "statistic": statistic_arr,
            "df": df_arr,
            "p_value": special.chdtrc(df_arr, statistic_arr),
            "method": "Friedman rank sum test",
        })

    return Step(_apply, "friedman_test")


@step
def quade_test(formula: str) -> Step:
    """Quade test, as R's ``quade.test(y ~ g | b)``.

    The design and missing values are as in ``friedman_test``. Values are
    ranked within each block, and each block is weighted by the rank of its
    range. Columns: ``statistic`` (Quade's F), ``num_df``, ``den_df``,
    ``p_value``, ``method``. As in R, when the treatment rank sums account
    for all the variation (every block ranks the treatments alike, with
    equal ranges), F and its degrees of freedom are missing and the p-value
    is ``k!^(1 - b)`` for ``k`` treatments in ``b`` blocks.
    """

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        lf = tf._lf
        response, by, block = _block_formula(tf, formula, "quade_test")
        kept = lf.filter(_present(lf, response) & _present(lf, by) & _present(lf, block))
        value = pl.col(response).cast(pl.Float64)
        in_block = [*groups, block]
        treatments = pl.col(by).n_unique()
        kept = kept.with_columns(
            __stats3_r=value.rank("average").over(in_block),
            __stats3_k=(treatments.over(groups) if groups else treatments).cast(pl.Float64),
        )
        # Blocks ranked by their range (max - min), within each group.
        spread = kept.group_by(in_block).agg(__stats3_q=value.max() - value.min())
        q = pl.col("__stats3_q").rank("average")
        spread = spread.with_columns(__stats3_q=q.over(groups) if groups else q)
        kept = kept.join(spread, on=in_block, how="left", nulls_equal=True).with_columns(
            __stats3_s=pl.col("__stats3_q") * (pl.col("__stats3_r") - (pl.col("__stats3_k") + 1) / 2)
        )
        table = _by_levels(
            tf, kept, groups, by, [pl.len().alias("n"), pl.col("__stats3_s").sum().alias("s")], "quade_test",
        )
        table = _join_design(table, kept, groups, by, block, (pl.col("__stats3_s") ** 2).sum().alias("a"))
        out: dict[str, list[Any]] = {"statistic": [], "num_df": [], "den_df": [], "p_value": []}
        for row in table.iter_rows(named=True):
            s = np.array(row["s"], dtype=float)
            k, b = len(s), row["blocks"]
            _check_design(row, groups, k, "quade_test")
            a, between = row["a"], float(np.sum(s**2)) / b
            if a == between:
                values = (None, None, None, math.gamma(k + 1) ** (1 - b))
            else:
                stat = (b - 1) * between / (a - between)
                d1, d2 = float(k - 1), float((b - 1) * (k - 1))
                values = (stat, d1, d2, float(special.fdtrc(d1, d2, stat)))
            for name, v in zip(out, values):
                out[name].append(v)
        return _finish(tf, table, groups, {
            **{name: pl.Series(v, dtype=pl.Float64) for name, v in out.items()},
            "method": "Quade test",
        })

    return Step(_apply, "quade_test")


# ansari_test and mood_test -----------------------------------------------------


def _finite(lf: pl.LazyFrame, name: str) -> pl.Expr:
    """Present and finite, as R's ``is.finite``."""
    if lf.collect_schema()[name].is_float():
        return pl.col(name).is_finite()
    return pl.col(name).is_not_null()


def _two_samples(tf: Any, groups: list[str], x: Any, y: Any, verb: str, keep: Any) -> pl.LazyFrame:
    """The two samples stacked: ``__stats3_v`` the values, ``__stats3_first``
    true for x (or the formula's first level). ``keep(lf, name)`` selects the
    values each test uses."""
    lf = tf._lf.with_row_index("__stats3_i")
    if _is_formula(x):
        if y is not None:
            raise ValueError(f"{verb}(): pass a formula or x and y, not both")
        response, by = _parse_formula(tf, x, verb, one_sample=False)
        kept = lf.filter(keep(lf, response) & _present(lf, by))
        levels = _by_levels(tf, kept, groups, by, [pl.len().alias("n")], verb, exactly=2)
        first = levels.select(*groups, pl.col(by).list.first().alias("__stats3_first"))
        if groups:
            kept = kept.join(first.lazy(), on=groups, how="left", nulls_equal=True)
        else:
            kept = kept.with_columns(__stats3_first=pl.lit(first["__stats3_first"][0]))
        return kept.select(
            *groups,
            "__stats3_i",
            __stats3_v=pl.col(response).cast(pl.Float64),
            __stats3_first=pl.col(by) == pl.col("__stats3_first"),
        )
    if y is None:
        raise ValueError(f"{verb}(): pass a formula, or two columns x and y")
    names = ((_column(tf, x, "x", verb), True), (_column(tf, y, "y", verb), False))
    return pl.concat([
        lf.filter(keep(lf, name)).select(
            *groups, "__stats3_i", __stats3_v=pl.col(name).cast(pl.Float64), __stats3_first=pl.lit(first)
        )
        for name, first in names
    ])


def _sample_sizes(table: pl.DataFrame, groups: list[str], verb: str) -> None:
    for row in table.iter_rows(named=True):
        if not row["m"]:
            raise ValueError(f"{verb}(): not enough 'x' observations{_group_label(row, groups)}")
        if not row["n"]:
            raise ValueError(f"{verb}(): not enough 'y' observations{_group_label(row, groups)}")


@lru_cache(maxsize=None)
def _ansari_counts(m: int, n: int) -> np.ndarray:
    """Ways to reach each Ansari-Bradley statistic 0..u (R's ``cansari``),
    zero outside R's support [(m+1)^2 %/% 4, that + m*n %/% 2]."""

    def support(i: int, j: int) -> tuple[int, int]:
        low = (i + 1) * (i + 1) // 4
        return low, low + i * j // 2

    # row[j] holds the counts for (i, j); built up from i = 0.
    row = [np.ones(1) for _ in range(n + 1)]
    for i in range(1, m + 1):
        new: list[np.ndarray] = []
        for j in range(n + 1):
            low, up = support(i, j)
            out = np.zeros(up + 1)
            if j == 0:
                out[low] = 1.0
            else:
                left, below, shift = new[j - 1], row[j], (i + j + 1) // 2
                ks = np.arange(low, up + 1)
                a = np.where(ks < len(left), left[np.minimum(ks, len(left) - 1)], 0.0)
                back = ks - shift
                ok = (back >= 0) & (back < len(below))
                b = np.where(ok, below[np.clip(back, 0, len(below) - 1)], 0.0)
                out[low:] = a + b
            new.append(out)
        row = new
    return row[n]


def _pansari(q: float, m: int, n: int) -> float:
    """P(AB <= q), as R's ``pansari``."""
    counts = _ansari_counts(m, n)
    low = (m + 1) * (m + 1) // 4
    up = low + m * n // 2
    x = math.floor(q + 1e-7)
    if x < low:
        return 0.0
    if x > up:
        return 1.0
    return float(np.cumsum(counts[low:x + 1])[-1] / math.comb(m + n, m))


def _qansari(p: float, m: int, n: int) -> int:
    """The smallest q with P(AB <= q) >= p, as R's ``qansari``."""
    counts = _ansari_counts(m, n)
    low = (m + 1) * (m + 1) // 4
    up = low + m * n // 2
    if p == 0:
        return low
    if p == 1:
        return up
    reached = np.flatnonzero(np.cumsum(counts / math.comb(m + n, m)) >= p)
    return int(reached[0]) if len(reached) else up


def _ansari_scores(values: np.ndarray, m: int) -> tuple[float, bool, np.ndarray]:
    """AB (the x scores' sum), whether there are ties, and every score."""
    from scipy.stats import rankdata

    # R's rank(na.last = TRUE): NaN (from x / -0) ranks last, in order.
    nan = np.isnan(values)
    r = np.empty(len(values))
    r[~nan] = rankdata(values[~nan])
    r[nan] = np.arange((~nan).sum() + 1, len(values) + 1)
    a = np.minimum(r, len(r) - r + 1)
    return float(a[:m].sum()), len(np.unique(r)) < len(r), a


def _r_mean(a: np.ndarray) -> float:
    """R's ``mean``: a sequential sum, then one correction pass, both in C's
    ``long double`` (80-bit on x86-64, the same as double on 64-bit ARM), as
    NumPy's longdouble is. Where the tie-corrected AB z is exactly 0 in exact
    arithmetic, its sign (which R's interval code tests) follows this
    rounding, so the platform's long double decides whether R's estimate
    exists."""
    a = np.asarray(a, dtype=np.longdouble)
    s = np.cumsum(a)[-1] / len(a)
    return float(s + np.cumsum(a - s)[-1] / len(a))


def _zeroin(f: Any, a: float, b: float, fa: float, fb: float, tol: float = 1e-4, maxit: int = 1000):
    """R's ``uniroot`` (``R_zeroin2``): the final b and c, which bracket the root."""
    eps = np.finfo(float).eps
    c, fc = a, fa
    if fa == 0:
        return a, a, 0.0, 0.0
    if fb == 0:
        return b, b, 0.0, 0.0
    for _ in range(maxit + 1):
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * eps * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0:
            return b, c, fb, fc
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p, q = cb * t1, 1.0 - t1
            else:
                q, t1, t2 = fa / fc, fb / fc, fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if p < 0.75 * cb * q - abs(tol_act * q) / 2 and p < abs(prev_step * q / 2):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    return b, c, fb, fc


def _ansari_z(s: float, m: int, n: int, ties: bool, mean_a: float, var_a: float) -> float:
    big = m + n
    if ties:
        return (s - m * mean_a) / math.sqrt(m * n * var_a / big)
    if big % 2 == 0:
        return (s - m * (big + 2) / 4) / math.sqrt(m * n * (big + 2) * (big - 2) / (48 * (big - 1)))
    return (s - m * (big + 1) ** 2 / (4 * big)) / math.sqrt(m * n * (big + 1) * (3 + big**2) / (48 * big**2))


def _ansari_exact_interval(
    x: np.ndarray, y: np.ndarray, alternative: str, conf_level: float
) -> tuple[float, float, float]:
    """R's exact estimate and interval: AB just past each ratio x_i / y_j."""
    m, n = len(x), len(y)
    x, y = np.sort(x), np.sort(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide.outer(x, y)
    sigma = np.sort(ratio[ratio >= 0])
    if len(sigma) < 1:
        _warn("cannot compute confidence set, returning NA")
        raise ValueError("ansari_test(): no nonnegative ratios x / y to estimate the ratio of scales from")
    points = sigma + np.r_[np.diff(sigma) / 2, sigma[-1] * 1.01]
    absigma = np.array([_ansari_scores(np.r_[x / s, y], m)[0] for s in points])

    def nearest(d: np.ndarray, strict: bool, pick: Any) -> float:
        keep = d > 0 if strict else d >= 0
        if not keep.any():
            return math.inf if pick is min else -math.inf
        return float(pick(sigma[keep & (d == d[keep].min())]))

    def cci(alpha: float) -> tuple[float, float]:
        u = absigma - _qansari(alpha / 2, m, n)
        l = absigma - _qansari(1 - alpha / 2, m, n)
        if not (u >= 0).any() or not (l > 0).any():
            _warn("samples differ in location: cannot compute confidence set, returning NA")
            return math.nan, math.nan
        uci, lci = nearest(u, False, min), nearest(l, True, max)
        if uci > lci:
            uci, lci = nearest(l, False, min), nearest(u, True, max)
        return uci, lci

    alpha = 1 - conf_level
    if alternative == "two.sided":
        low, high = cci(alpha)
    elif alternative == "greater":
        low, high = cci(alpha * 2)[0], math.inf
    else:
        low, high = 0.0, cci(alpha * 2)[1]
    u = absigma - _qansari(0.5, m, n)
    below, above = sigma[u <= 0], sigma[u > 0]
    estimate = (below.max() + above.min()) / 2 if len(below) and len(above) else math.nan
    return float(estimate), low, high


def _ansari_asymptotic_interval(
    x: np.ndarray, y: np.ndarray, alternative: str, conf_level: float
) -> tuple[float, float, float]:
    """R's asymptotic estimate and interval: where the normalized AB of
    x / sigma against y crosses each normal quantile.

    R finds these with ``uniroot(tol = 1e-4)``; AB is a step function of
    sigma, so stats3 finds each crossing to full precision (1e-12) instead.
    """
    from scipy.optimize import brentq

    special = _special()
    m, n = len(x), len(y)

    def ab2(sig: float, zq: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            s, ties, a = _ansari_scores(np.r_[x / sig, y], m)
        return _ansari_z(s, m, n, ties, _r_mean(a), float(a.var(ddof=1))) - zq

    ranges = []
    if (x > 0).any() and (y > 0).any():
        ranges += [x[x > 0].min() / y[y > 0].max(), x[x > 0].max() / y[y > 0].min()]
    if (x <= 0).any() and (y < 0).any():
        ranges += [x[x <= 0].min() / y[y < 0].max(), x[x <= 0].max() / y[y < 0].min()]
    if not ranges or not np.all(np.isfinite(ranges)):
        _warn("cannot compute asymptotic confidence set or estimator")
        return math.nan, math.nan, math.nan
    lo, hi = float(min(ranges)), float(max(ranges))

    def root(zq: float) -> float:
        # R's uniroot picks the root (AB can equal its mean on a whole
        # interval); a sign change in its final bracket is then refined.
        b, c, fb, _ = _zeroin(lambda s: ab2(s, zq), lo, hi, ab2(lo, zq), ab2(hi, zq))
        if fb == 0 or b == c:
            return float(b)
        return float(brentq(ab2, min(b, c), max(b, c), args=(zq,), xtol=1e-12, maxiter=1000))

    def ccia(alpha: float) -> tuple[float, float]:
        z_low, z_high = float(special.ndtri(alpha / 2)), float(-special.ndtri(alpha / 2))
        if ab2(lo, z_low) > 0 or ab2(hi, z_high) < 0:
            _warn("samples differ in location: cannot compute confidence set, returning NA")
            return math.nan, math.nan
        u, l = root(z_low), root(z_high)
        return min(u, l), max(u, l)

    alpha = 1 - conf_level
    if alternative == "two.sided":
        low, high = ccia(alpha)
    elif alternative == "greater":
        low, high = ccia(alpha * 2)[0], math.inf
    else:
        low, high = 0.0, ccia(alpha * 2)[1]
    if ab2(lo, 0.0) > 0 or ab2(hi, 0.0) < 0:
        _warn("cannot compute estimate, returning NA")
        estimate = math.nan
    else:
        estimate = root(0.0)
    return estimate, low, high


def _sample_values(
    tf: Any, table: pl.DataFrame, groups: list[str], stacked: pl.LazyFrame
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each group's x and y values, in data order (R's sums run in that order)."""
    frame = stacked.select(*groups, "__stats3_i", "__stats3_v", "__stats3_first").collect().sort("__stats3_i")
    parts = frame.partition_by(groups, as_dict=True, maintain_order=True) if groups else {(): frame}
    out = []
    for row in table.iter_rows(named=True):
        part = parts[tuple(row[g] for g in groups)]
        first = part["__stats3_first"].to_numpy()
        v = part["__stats3_v"].to_numpy()
        out.append((v[first], v[~first]))
    return out


@step
def ansari_test(
    x: Any,
    y: Any = None,
    *,
    alternative: str = "two.sided",
    exact: bool | None = None,
    conf_int: bool = False,
    conf_level: float = 0.95,
) -> Step:
    """Ansari-Bradley test of equal scale, as R's ``ansari.test``.

    ``ansari_test("y ~ g")`` or ``ansari_test(x, y)``. As in R, the p-value is
    exact when both samples have fewer than 50 values and there are no ties
    (``exact`` overrides the size rule), else from the normal approximation,
    with R's tie correction; R warns that it cannot compute an exact
    p-value with ties. ``conf_int=True`` adds R's estimate and interval for
    the ratio of scales of x to y: exact from the permutation distribution,
    or asymptotic when the p-value is (with ties, as in R, only with
    ``exact=False`` or large samples). Columns: ``estimate``, ``statistic``
    (AB), ``p_value``, ``conf_low``, ``conf_high`` (the interval columns only
    with ``conf_int=True``), ``method``, ``alternative``.
    """
    alternative = _alternative(alternative)
    if conf_int:
        _check_conf_level(conf_level)

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        stacked = _two_samples(tf, groups, x, y, "ansari_test", _present)
        v, first = pl.col("__stats3_v"), pl.col("__stats3_first")
        r, big = v.rank("average"), pl.len()
        if groups:
            r, big = r.over(groups), big.over(groups)
        stacked = stacked.with_columns(__stats3_a=pl.min_horizontal(r, big - r + 1))
        a = pl.col("__stats3_a")
        table = _sorted_keys(tf, _summarise(stacked, groups, [
            first.sum().alias("m"),
            (~first).sum().alias("n"),
            a.filter(first).sum().alias("ab"),
            (v.n_unique() < pl.len()).alias("ties"),
            a.mean().alias("mean_a"),
            a.var().alias("var_a"),
        ]), groups)
        _sample_sizes(table, groups, "ansari_test")
        values = _sample_values(tf, table, groups, stacked) if conf_int else [None] * table.height
        out: dict[str, list[Any]] = {"estimate": [], "statistic": [], "p_value": [], "conf_low": [], "conf_high": []}
        for row, samples in zip(table.iter_rows(named=True), values):
            m, n, stat, ties = row["m"], row["n"], float(row["ab"]), row["ties"]
            use_exact = (m < 50 and n < 50) if exact is None else exact
            interval = (math.nan, math.nan, math.nan)
            if use_exact and not ties:
                if alternative == "two.sided":
                    if stat > (m + 1) ** 2 // 4 + (m * n // 2) / 2:
                        p = 1 - _pansari(stat - 1, m, n)
                    else:
                        p = _pansari(stat, m, n)
                    p = min(2 * p, 1.0)
                elif alternative == "less":
                    p = 1 - _pansari(stat - 1, m, n)
                else:
                    p = _pansari(stat, m, n)
                if conf_int:
                    interval = _ansari_exact_interval(*samples, alternative, conf_level)
            else:
                z = _ansari_z(stat, m, n, ties, row["mean_a"], row["var_a"] or 0.0)
                lower = float(special.ndtr(z))
                p = {"two.sided": 2 * min(lower, 1 - lower), "less": 1 - lower, "greater": lower}[alternative]
                if conf_int and not use_exact:
                    interval = _ansari_asymptotic_interval(*samples, alternative, conf_level)
                if use_exact:
                    _warn("cannot compute exact p-value with ties")
                    if conf_int:
                        _warn("cannot compute exact confidence interval with ties")
                        raise ValueError(
                            "ansari_test(): cannot compute an exact confidence interval with ties"
                            f"{_group_label(row, groups)}; pass exact=False for the asymptotic one"
                        )
            for name, value in zip(("estimate", "statistic", "p_value", "conf_low", "conf_high"),
                                   (interval[0], stat, p, interval[1], interval[2])):
                out[name].append(value)
        columns = {name: pl.Series(values_, dtype=pl.Float64).fill_nan(None) for name, values_ in out.items()}
        if not conf_int:
            columns = {name: columns[name] for name in ("statistic", "p_value")}
        else:
            columns = {name: columns[name] for name in ("estimate", "statistic", "p_value", "conf_low", "conf_high")}
        return _finish(tf, table, groups, {**columns, "method": "Ansari-Bradley test", "alternative": alternative})

    return Step(_apply, "ansari_test")


@step
def mood_test(x: Any, y: Any = None, *, alternative: str = "two.sided") -> Step:
    """Mood's two-sample test of scale, as R's ``mood.test``.

    ``mood_test("y ~ g")`` or ``mood_test(x, y)``, on the finite values (R
    drops infinite values too). Ties are handled as in R 4.x: each tied
    value scores the mean of the squared deviations its tied ranks share,
    and the variance is corrected. Columns: ``statistic`` (Z), ``p_value``,
    ``method``, ``alternative``.
    """
    alternative = _alternative(alternative)

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        stacked = _two_samples(tf, groups, x, y, "mood_test", _finite)
        v, first = pl.col("__stats3_v"), pl.col("__stats3_first")
        r, big, t = v.rank("average"), pl.len(), pl.len().over([*groups, "__stats3_v"])
        if groups:
            r, big = r.over(groups), big.over(groups)
        big, t = big.cast(pl.Float64), t.cast(pl.Float64)
        # A tied value's mean squared deviation over its ranks lo..hi:
        # (mean rank - (N + 1) / 2)^2 + (t^2 - 1) / 12.
        stacked = stacked.with_columns(
            __stats3_score=(r - (big + 1) / 2) ** 2 + (t * t - 1) / 12,
            __stats3_ties=(t * t - 1) * (t * t - 4 + 15 * (big + 1 - 2 * r) ** 2),
        )
        table = _sorted_keys(tf, _summarise(stacked, groups, [
            first.sum().alias("m"),
            (~first).sum().alias("n"),
            pl.col("__stats3_score").filter(first).sum().alias("t"),
            pl.col("__stats3_ties").sum().alias("ties"),
        ]), groups)
        statistic = []
        for row in table.iter_rows(named=True):
            m, n = row["m"], row["n"]
            total = m + n
            if total < 3:
                raise ValueError(f"mood_test(): not enough observations{_group_label(row, groups)}")
            expected = m * (total**2 - 1) / 12
            var = m * n * (total + 1) * (total + 2) * (total - 2) / 180
            var -= m * n / (180 * total * (total - 1)) * row["ties"]
            with np.errstate(divide="ignore", invalid="ignore"):
                statistic.append((row["t"] - expected) / np.sqrt(var))
        z = np.array(statistic, dtype=float)
        lower = special.ndtr(z)
        p = {"less": lower, "greater": 1 - lower, "two.sided": 2 * np.minimum(lower, 1 - lower)}[alternative]
        return _finish(tf, table, groups, {
            "statistic": z, "p_value": p, "method": "Mood two-sample test of scale", "alternative": alternative,
        })

    return Step(_apply, "mood_test")


# cor_test ----------------------------------------------------------------------

_COR_METHODS = {"pearson", "spearman", "kendall"}


@step
def cor_test(
    x: Any,
    y: Any,
    *,
    method: str = "pearson",
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    exact: bool | None = None,
    continuity: bool = False,
) -> Step:
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
                _warn("cannot compute exact p-value with ties")
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

        # One query: each group's rows, contiguous in dplyr's group order,
        # with its size and distinct counts; then one SciPy call per slice.
        sizes = [pl.len().alias("__stats3_n"), pl.col("__stats3_x").n_unique().alias("__stats3_ux"),
                 pl.col("__stats3_y").n_unique().alias("__stats3_uy")]
        if groups:
            keys = _sorted_keys(tf, kept.group_by(groups).agg(sizes).collect(), groups)
            order = keys.select(groups).with_row_index("__stats3_order")
            values = (kept.join(order.lazy(), on=groups, how="left", nulls_equal=True)
                      .sort("__stats3_order", maintain_order=True)
                      .select("__stats3_x", "__stats3_y").collect())
        else:
            keys = kept.select(sizes).collect()
            values = kept.select("__stats3_x", "__stats3_y").collect()
        xs, ys = values["__stats3_x"].to_numpy(), values["__stats3_y"].to_numpy()
        ends = np.cumsum(keys["__stats3_n"].to_numpy())
        estimate, statistic, p_value = [], [], []
        for index, row in enumerate(keys.iter_rows(named=True)):
            n = row["__stats3_n"]
            xv, yv = xs[ends[index] - n:ends[index]], ys[ends[index] - n:ends[index]]
            if n < 2:
                raise ValueError(f"cor_test(): not enough finite observations{_group_label(row, groups)}")
            tau = float(kendalltau(xv, yv).statistic)
            ties = min(row["__stats3_ux"], row["__stats3_uy"]) < n
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
                # Tie runs, counted only in a column that has ties.
                tx = np.unique(xv, return_counts=True)[1].astype(float) if row["__stats3_ux"] < n else np.empty(0)
                ty = np.unique(yv, return_counts=True)[1].astype(float) if row["__stats3_uy"] < n else np.empty(0)
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
                    _warn("cannot compute exact p-value with ties")
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

    return Step(_apply, "cor_test")
