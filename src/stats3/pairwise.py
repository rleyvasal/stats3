"""Pairwise comparisons with adjusted p-values: R's ``pairwise.t.test``,
``pairwise.wilcox.test`` and ``pairwise.prop.test``.

Each pair of levels is tested with stats3's own test (all pairs of all
groups in one grouped run), and the p-values of each group are adjusted
together with ``p.adjust``, as R's ``pairwise.table`` does. The result is
broom's long form: one row per pair, ``group1`` the later level and
``group2`` the earlier one, as in R's lower-triangle table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from stats3._data import Step, step
from stats3.checks import _group_values
from stats3.core import (
    _alternative,
    _check_mu,
    _finish,
    _parse_formula,
    _present,
    _special,
)
from stats3.counts import _count_lists, _counts_column, _prop_row
from stats3.effects import P_ADJUST_METHODS, adjust_p


def _check_method(verb: str, method: str) -> None:
    if method not in P_ADJUST_METHODS:
        raise ValueError(
            f"{verb}(): p_adjust_method must be one of {list(P_ADJUST_METHODS)}, got {method!r}"
        )


def _pairs(k: int) -> list[tuple[int, int]]:
    """(i, j) with i > j, by row of R's lower-triangle table."""
    return [(i, j) for i in range(1, k) for j in range(i)]


def _levels(tf: Any, by: str, values: np.ndarray) -> list[Any]:
    """The levels present, in factor order, else sorted: R's ``factor(g)``."""
    present = set(values.tolist())
    levels = tf.levels_of(by)
    if levels is not None:
        return [level for level in levels if level in present]
    return sorted(present)


def _by_pairs(tf: Any, formula: str, verb: str) -> tuple[list[str], str, pl.DataFrame, list[dict[str, Any]]]:
    """Per group: the levels and each level's response values, in row order."""
    groups = list(tf._groups or [])
    response, by = _parse_formula(tf, formula, verb, one_sample=False)
    if by in groups:
        raise ValueError(f"{verb}(): {by!r} is a grouping column; ungroup it first")
    table, rows = _group_values(
        tf, groups, {"y": pl.col(response).cast(pl.Float64), "g": pl.col(by)}, _present(tf._lf, by),
        as_is=("g",),
    )
    for row in rows:
        levels = _levels(tf, by, row["g"])
        if len(levels) < 2:
            raise ValueError(f"{verb}(): all observations are in the same group{row['__label']}")
        row["levels"] = levels
        row["values"] = [row["y"][row["g"] == level] for level in levels]
    return groups, by, table, rows


def _paired_frame(rows: list[dict[str, Any]], paired: bool, verb: str) -> pl.DataFrame:
    """One block of rows per pair: ``x`` the later level's values, ``y`` the
    earlier one's, padded with missing values to the same length."""
    sizes, xs, ys = [], [], []
    for row in rows:
        for i, j in _pairs(len(row["levels"])):
            x, y = row["values"][i], row["values"][j]
            if paired and len(x) != len(y):
                raise ValueError(
                    f"{verb}(): paired tests need the same number of values in each level{row['__label']}"
                )
            size = max(len(x), len(y))
            xs += [*(None if np.isnan(v) else float(v) for v in x), *[None] * (size - len(x))]
            ys += [*(None if np.isnan(v) else float(v) for v in y), *[None] * (size - len(y))]
            sizes.append(size)
    ids = np.repeat(np.arange(len(sizes)), sizes)
    return pl.DataFrame(
        {"__stats3_pair": ids, "x": pl.Series(xs, dtype=pl.Float64), "y": pl.Series(ys, dtype=pl.Float64)}
    )


def _result(
    tf: Any,
    table: pl.DataFrame,
    groups: list[str],
    rows: list[dict[str, Any]],
    dtype: Any,
    p_values: np.ndarray,
    method: str | list[str],
    p_adjust_method: str,
) -> pl.DataFrame:
    """One row per pair, the p-values adjusted within each group; ``method``
    is one for all groups or one per group."""
    keys = table.select(groups) if groups else None
    methods = [method] * len(rows) if isinstance(method, str) else method
    index, group1, group2, adjusted = [], [], [], []
    start = 0
    for g, row in enumerate(rows):
        pairs = _pairs(len(row["levels"]))
        adjusted.append(adjust_p(p_values[start:start + len(pairs)], p_adjust_method))
        start += len(pairs)
        index += [g] * len(pairs)
        group1 += [row["levels"][i] for i, _ in pairs]
        group2 += [row["levels"][j] for _, j in pairs]
    out = keys[index] if keys is not None else pl.DataFrame({"__stats3_k": [0] * len(index)})
    p = np.concatenate(adjusted) if adjusted else np.array([])
    return _finish(tf, out, groups, {
        "group1": pl.Series(group1, dtype=dtype),
        "group2": pl.Series(group2, dtype=dtype),
        "p_value": pl.Series(p, dtype=pl.Float64).fill_nan(None),
        "method": pl.Series([methods[g] for g in index], dtype=pl.String),
        "p_adjust_method": p_adjust_method,
    })


@step
def pairwise_t_test(
    formula: str,
    *,
    p_adjust_method: str = "holm",
    pool_sd: bool | None = None,
    paired: bool = False,
    alternative: str = "two.sided",
) -> Step:
    """t tests between every pair of levels, as R's
    ``pairwise.t.test(y, g)``, with p-values adjusted by ``p_adjust_method``.

    By default the standard deviation is pooled over all levels (R's
    ``pool.sd = !paired``); ``pool_sd=False`` runs Welch's test on each pair,
    and ``paired=True`` paired tests (values paired in row order). Columns:
    ``group1``, ``group2``, ``p_value``, ``method``, ``p_adjust_method``.
    """
    _check_method("pairwise_t_test", p_adjust_method)
    alternative = _alternative(alternative)
    pool = not paired if pool_sd is None else pool_sd
    if paired and pool:
        raise ValueError("pairwise_t_test(): pooling of SD is incompatible with paired tests")

    def _apply(tf: Any) -> Any:
        from stats3.core import t_test

        special = _special()
        groups, by, table, rows = _by_pairs(tf, formula, "pairwise_t_test")
        if pool:
            p_values = []
            for row in rows:
                values = [v[~np.isnan(v)] for v in row["values"]]
                n = np.array([len(v) for v in values], dtype=float)
                means = np.array([v.mean() if len(v) else np.nan for v in values])
                # R: ifelse(degf, s^2, 0), so a level of one value adds nothing.
                ss = np.array([np.var(v, ddof=1) * (len(v) - 1) if len(v) > 1 else 0.0 for v in values])
                total_df = np.sum(n - 1)
                pooled = np.sqrt(ss.sum() / total_df)
                for i, j in _pairs(len(values)):
                    t = (means[i] - means[j]) / (pooled * np.sqrt(1 / n[i] + 1 / n[j]))
                    if alternative == "two.sided":
                        p_values.append(2 * special.stdtr(total_df, -abs(t)))
                    elif alternative == "less":
                        p_values.append(special.stdtr(total_df, t))
                    else:
                        p_values.append(special.stdtr(total_df, -t))
            p = np.array(p_values, dtype=float)
            method = "t tests with pooled SD"
        else:
            frame = _paired_frame(rows, paired, "pairwise_t_test")
            tests = t_test(frame, "x", "y", paired=paired, alternative=alternative, by="__stats3_pair")
            p = tests["p_value"].to_numpy()
            method = "paired t tests" if paired else "t tests with non-pooled SD"
        dtype = tf._lf.collect_schema()[by]
        return _result(tf, table, groups, rows, dtype, p, method, p_adjust_method)

    return Step(_apply, "pairwise_t_test")


@step
def pairwise_wilcox_test(
    formula: str,
    *,
    p_adjust_method: str = "holm",
    paired: bool = False,
    exact: bool | None = None,
    correct: bool | int = True,
    alternative: str = "two.sided",
) -> Step:
    """Wilcoxon tests between every pair of levels, as R's
    ``pairwise.wilcox.test(y, g)``, with p-values adjusted by
    ``p_adjust_method``. Each pair gets ``wilcox_test`` (rank sum, or signed
    rank with ``paired=True``, values paired in row order), with R 4.6's
    exact tests. ``method`` is each group's first pair's, as in R. Columns:
    ``group1``, ``group2``, ``p_value``, ``method``, ``p_adjust_method``.
    """
    _check_method("pairwise_wilcox_test", p_adjust_method)
    alternative = _alternative(alternative)

    def _apply(tf: Any) -> Any:
        from stats3.ranks import wilcox_test

        groups, by, table, rows = _by_pairs(tf, formula, "pairwise_wilcox_test")
        frame = _paired_frame(rows, paired, "pairwise_wilcox_test")
        tests = wilcox_test(
            frame, "x", "y", paired=paired, exact=exact, correct=correct,
            alternative=alternative, by="__stats3_pair",
        )
        dtype = tf._lf.collect_schema()[by]
        # R names the method after each group's first pair.
        first = np.cumsum([0] + [len(_pairs(len(row["levels"]))) for row in rows])[:-1]
        methods = [tests["method"][int(i)] for i in first]
        return _result(
            tf, table, groups, rows, dtype, tests["p_value"].to_numpy(), methods, p_adjust_method,
        )

    return Step(_apply, "pairwise_wilcox_test")


@step
def pairwise_prop_test(
    x: Any,
    n: Any,
    *,
    labels: Any = None,
    p_adjust_method: str = "holm",
    alternative: str = "two.sided",
    correct: bool = True,
) -> Step:
    """Tests of equal proportions between every pair of samples, as R's
    ``pairwise.prop.test(x, n)``, with p-values adjusted by
    ``p_adjust_method``.

    ``x`` and ``n`` are columns of successes and trials, one row per sample
    (each pair gets ``prop_test``); ``labels`` is a column naming the
    samples, else they are numbered from 1 in row order. Columns:
    ``group1``, ``group2``, ``p_value``, ``method``, ``p_adjust_method``.
    """
    _check_method("pairwise_prop_test", p_adjust_method)
    alternative = _alternative(alternative)

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        columns = [
            _counts_column(tf, x, "x", "pairwise_prop_test"),
            _counts_column(tf, n, "n", "pairwise_prop_test"),
        ]
        if labels is not None:
            columns.append(_counts_column(tf, labels, "labels", "pairwise_prop_test"))
        dtype = pl.Int64
        if labels is not None:
            from stats3._data import expression

            dtype = tf._lf.select(expression(columns[2])).collect_schema().dtypes()[0]
            raw = [columns[0], columns[1]]
            table, samples = _count_lists(tf, groups, raw)
            names = _label_lists(tf, groups, columns[2])
        else:
            table, samples = _count_lists(tf, groups, columns)
            names = None
        rows, p_values = [], []
        for s, sample in enumerate(samples):
            xs, ns = sample["c0"], sample["c1"]
            ok = ~(np.isnan(xs) | np.isnan(ns))
            levels = (
                [v for v, keep in zip(names[s], ok) if keep] if names is not None
                else list(range(1, int(ok.sum()) + 1))
            )
            xs, ns = xs[ok], ns[ok]
            if len(xs) < 2:
                raise ValueError(f"pairwise_prop_test(): too few groups{sample['label']}")
            rows.append({"levels": levels})
            for i, j in _pairs(len(xs)):
                pick = [i, j]
                p_values.append(_prop_row(
                    xs[pick], ns[pick], None, alternative, 0.95, correct, sample["label"], special,
                )["p_value"])
        return _result(
            tf, table, groups, rows, dtype, np.array(p_values, dtype=float),
            "Pairwise comparison of proportions", p_adjust_method,
        )

    return Step(_apply, "pairwise_prop_test")


def _label_lists(tf: Any, groups: list[str], spec: Any) -> list[list[Any]]:
    """Per group (in the same order as ``_count_lists``), a column's values."""
    from stats3._data import expression
    from stats3.core import _sorted_keys

    value = expression(spec).alias("__stats3_label")
    lf = tf._lf
    if groups:
        table = _sorted_keys(tf, lf.group_by(groups, maintain_order=True).agg(value).collect(), groups)
    else:
        table = lf.select(value.implode()).collect()
    return table["__stats3_label"].to_list()
