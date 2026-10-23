"""Tests on counts: R's ``chisq.test``, ``fisher.test``, ``prop.test`` and
``binom.test``.

Contingency tables are counted in one grouped Polars query; each group's
small table is then finished with ``scipy.special`` (and, for Fisher's exact
test, ``scipy.stats``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Step, column_name, expression, step
from stats3.core import (
    _warn,
    _alternative,
    _check_conf_level,
    _column,
    _finish,
    _group_label,
    _level_rank,
    _present,
    _sorted_keys,
    _special,
)

_CHISQ_WARNING = "Chi-squared approximation may be incorrect"


def _tables(
    tf: Any, groups: list[str], names: list[str]
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Per group: the levels of each column present and the cell counts.

    Rows missing any of ``names`` are dropped, and levels are those present,
    in factor order, else sorted: R's ``table(factor(x[OK]), ...)``.
    """
    lf = tf._lf
    present = pl.lit(True)
    for name in names:
        present = present & _present(lf, name)
    counts = (
        lf.filter(present).group_by([*groups, *names]).agg(n=pl.len())
        .sort([*groups, *(_level_rank(tf, name) for name in names)], nulls_last=True)
        .collect()
    )
    if groups:
        keys = _sorted_keys(tf, counts.select(groups).unique(maintain_order=True), groups)
        parts = counts.partition_by(groups, as_dict=True, maintain_order=True)
        frames = [parts[tuple(key)] for key in keys.iter_rows()]
    else:
        keys = pl.DataFrame()
        frames = [counts]
    tables = []
    for frame in frames:
        levels = [list(dict.fromkeys(frame[name].to_list())) for name in names]
        if len(names) == 1:
            levels[0] = frame[names[0]].to_list()
            tables.append({"levels": levels, "counts": frame["n"].cast(pl.Float64).to_numpy()})
            continue
        index = [{v: i for i, v in enumerate(lv)} for lv in levels]
        cells = np.zeros([len(lv) for lv in levels])
        for row in frame.select(*names, "n").iter_rows():
            cells[tuple(index[k][row[k]] for k in range(len(names)))] = row[-1]
        tables.append({"levels": levels, "counts": cells})
    if not groups:
        keys = pl.DataFrame({"__stats3_k": [0]})
    return keys, tables


def _labels(keys: pl.DataFrame, groups: list[str]) -> list[str]:
    rows = keys.select(groups).to_dicts() if groups else [{}] * keys.height
    return [_group_label(row, groups) for row in rows]


# chisq_test ------------------------------------------------------------------


@step
def chisq_test(
    x: Any,
    y: Any = None,
    *,
    p: Sequence[float] | Mapping[Any, float] | None = None,
    correct: bool = True,
    rescale_p: bool = False,
) -> Step:
    """Pearson's chi-squared test, as R's ``chisq.test``.

    - ``chisq_test(species, island)``: independence of two columns, on the
      table of their counts. 2 x 2 tables get Yates' continuity correction
      unless ``correct=False``.
    - ``chisq_test(species, p=[...])``: goodness of fit of one column's counts
      to probabilities ``p`` (equal by default), given in level order or as
      ``{level: probability}``.

    Columns: ``statistic``, ``df``, ``p_value``, ``method``. Warns, as R does,
    when an expected count is below 5.
    """
    if y is not None and p is not None:
        raise ValueError("chisq_test(): p is for one column; pass x and y, or x and p")

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        names = [_column(tf, x, "x", "chisq_test")]
        if y is not None:
            names.append(_column(tf, y, "y", "chisq_test"))
        keys, tables = _tables(tf, groups, names)
        statistic, df, method = [], [], []
        for label, table in zip(_labels(keys, groups), tables):
            observed = table["counts"]
            total = observed.sum()
            if y is not None:
                if min(observed.shape) < 2:
                    raise ValueError(f"chisq_test(): 'x' and 'y' must have at least 2 levels{label}")
                expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
                name = "Pearson's Chi-squared test"
                yates = 0.0
                if correct and observed.shape == (2, 2):
                    yates = float(min(0.5, np.min(np.abs(observed - expected))))
                    if yates > 0:
                        name += " with Yates' continuity correction"
                stat = float(np.sum((np.abs(observed - expected) - yates) ** 2 / expected))
                dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
            else:
                levels = table["levels"][0]
                if len(levels) < 2:
                    raise ValueError(f"chisq_test(): 'x' must at least have 2 elements{label}")
                probs = _probabilities(p, levels, rescale_p)
                expected = total * probs
                name = "Chi-squared test for given probabilities"
                stat = float(np.sum((observed - expected) ** 2 / expected))
                dof = len(levels) - 1
            if np.any(expected < 5):
                _warn(_CHISQ_WARNING)
            statistic.append(stat)
            df.append(float(dof))
            method.append(name)
        stat_arr, df_arr = np.array(statistic), np.array(df)
        return _finish(tf, keys, groups, {
            "statistic": stat_arr,
            "df": df_arr,
            "p_value": special.chdtrc(df_arr, stat_arr),
            "method": pl.Series(method, dtype=pl.String),
        })

    return Step(_apply, "chisq_test")


def _probabilities(p: Any, levels: list[Any], rescale: bool) -> np.ndarray:
    if p is None:
        return np.full(len(levels), 1 / len(levels))
    if isinstance(p, Mapping):
        missing = [level for level in levels if level not in p]
        if missing:
            raise ValueError(f"chisq_test(): p has no probability for levels {missing}")
        probs = np.array([p[level] for level in levels], dtype=float)
    else:
        probs = np.asarray(p, dtype=float)
        if len(probs) != len(levels):
            raise ValueError(
                f"chisq_test(): 'x' and 'p' must have the same number of elements "
                f"({len(levels)} levels: {levels})"
            )
    if np.any(probs < 0):
        raise ValueError("chisq_test(): probabilities must be non-negative.")
    if abs(probs.sum() - 1) > math.sqrt(np.finfo(float).eps):
        if not rescale:
            raise ValueError("chisq_test(): probabilities must sum to 1.")
        probs = probs / probs.sum()
    return probs


# fisher_test -----------------------------------------------------------------


@step
def fisher_test(
    x: Any, y: Any, *, alternative: str = "two.sided", conf_level: float = 0.95
) -> Step:
    """Fisher's exact test, as R's ``fisher.test(x, y)``.

    For a 2 x 2 table, ``estimate`` is the conditional maximum likelihood
    odds ratio with its exact confidence interval, as in R. For larger
    tables the two-sided p-value is exact (R's network algorithm: the tables
    with the same margins no more likely than the observed one), and the
    estimate and interval are missing, as in R. Columns: ``estimate``,
    ``p_value``, ``conf_low``, ``conf_high``, ``method``, ``alternative``.
    """
    alternative = _alternative(alternative)
    _check_conf_level(conf_level)

    def _apply(tf: Any) -> Any:
        from scipy.stats import fisher_exact
        from scipy.stats.contingency import odds_ratio

        groups = list(tf._groups or [])
        names = [_column(tf, x, "x", "fisher_test"), _column(tf, y, "y", "fisher_test")]
        keys, tables = _tables(tf, groups, names)
        side = "two-sided" if alternative == "two.sided" else alternative
        out: dict[str, list[Any]] = {"estimate": [], "p_value": [], "conf_low": [], "conf_high": [], "alternative": []}
        for label, table in zip(_labels(keys, groups), tables):
            cells = table["counts"]
            if min(cells.shape) < 2:
                raise ValueError(f"fisher_test(): 'x' and 'y' must have at least 2 levels{label}")
            counts = cells.astype(np.int64)
            if counts.shape != (2, 2):
                # R tests larger tables two-sided only, without an estimate.
                out["estimate"].append(None)
                out["p_value"].append(_fisher_rxc(counts, label))
                out["conf_low"].append(None)
                out["conf_high"].append(None)
                out["alternative"].append("two.sided")
                continue
            ratio = odds_ratio(counts, kind="conditional")
            ci = ratio.confidence_interval(confidence_level=conf_level, alternative=side)
            out["estimate"].append(float(ratio.statistic))
            out["p_value"].append(float(fisher_exact(counts, alternative=side).pvalue))
            out["conf_low"].append(float(ci.low))
            out["conf_high"].append(float(ci.high))
            out["alternative"].append(alternative)
        columns = {k: pl.Series(v, dtype=pl.Float64) for k, v in out.items() if k != "alternative"}
        return _finish(tf, keys, groups, {
            "estimate": columns["estimate"],
            "p_value": columns["p_value"],
            "conf_low": columns["conf_low"],
            "conf_high": columns["conf_high"],
            "method": "Fisher's Exact Test for Count Data",
            "alternative": pl.Series(out["alternative"], dtype=pl.String),
        })

    return Step(_apply, "fisher_test")


_FISHER_LIMIT = 5_000_000


def _fisher_rxc(table: np.ndarray, label: str = "") -> float:
    """Exact two-sided p-value of an r x c table with fixed margins.

    The probability of a table is K * prod(1 / n_ij!) with K fixed by the
    margins; the p-value adds the tables no more likely than the observed
    one (R's relative slack 1e-7). Tables are built a column at a time; a
    node is the column reached and the row totals still to fill. Exact
    bounds on what the remaining columns can contribute (the most and least
    likely completions) and the closed-form total of all completions let a
    whole node be counted in or left out at once, as in R's network
    algorithm (FEXACT); equal partial paths are merged.
    """
    lf = [math.lgamma(k + 1) for k in range(int(table.sum()) + 1)]
    rows = sorted((int(r) for r in table.sum(axis=1) if r > 0), reverse=True)
    cols = sorted((int(c) for c in table.sum(axis=0) if c > 0), reverse=True)
    observed = -sum(lf[int(v)] for v in table.ravel())
    threshold = observed + math.log1p(1e-7)
    log_k = sum(lf[r] for r in rows) + sum(lf[c] for c in cols) - lf[sum(rows)]
    ncol = len(cols)
    suffix = [sum(cols[k:]) for k in range(ncol + 1)]

    def splits(total: int, caps: tuple[int, ...]):
        """Ways to put ``total`` into rows with room ``caps``."""
        if len(caps) == 1:
            if total <= caps[0]:
                yield (total,)
            return
        room = sum(caps[1:])
        for v in range(max(0, total - room), min(caps[0], total) + 1):
            for rest in splits(total - v, caps[1:]):
                yield (v,) + rest

    def spread(total: int, caps: list[int]) -> float:
        """Largest -sum(log x!) for sum(x) = total, x <= caps: as even as the
        caps allow (the objective is concave)."""
        out, left = 0.0, total
        caps = sorted(caps)
        for i, cap in enumerate(caps):
            share = left // (len(caps) - i)
            if cap <= share:
                out -= lf[cap]
                left -= cap
                continue
            # The rest share evenly: some get one more than the others.
            m = len(caps) - i
            base, extra = divmod(left, m)
            return out - (m - extra) * lf[base] - extra * lf[base + 1]
        return out

    def pack(total: int, caps: list[int]) -> float:
        """Smallest -sum(log x!) for sum(x) = total, x <= caps: filled into
        the largest caps first (the objective is convex)."""
        out, left = 0.0, total
        for cap in sorted(caps, reverse=True):
            take = min(cap, left)
            out -= lf[take]
            left -= take
            if not left:
                break
        return out

    bounds: dict[tuple[int, tuple[int, ...]], tuple[float, float]] = {}

    def bound(k: int, rem: tuple[int, ...]) -> tuple[float, float]:
        """Bounds on what the columns from k can add (sum of -log n_ij!):
        each comes from dropping the row or the column constraints, which
        can only loosen it, so the tighter of the two is still a bound."""
        key = (k, rem)
        if key in bounds:
            return bounds[key]
        rest = cols[k:]
        rem_list = [r for r in rem if r > 0]
        hi = min(sum(spread(c, rem_list) for c in rest), sum(spread(r, rest) for r in rem_list))
        lo = max(sum(pack(c, rem_list) for c in rest), sum(pack(r, rest) for r in rem_list))
        bounds[key] = (hi, lo)
        return bounds[key]

    def log_total(k: int, rem: tuple[int, ...]) -> float:
        """log of the sum of prod(1/n_ij!) over all completions from column k."""
        return lf[suffix[k]] - sum(lf[r] for r in rem) - sum(lf[c] for c in cols[k:])

    pieces: list[float] = []
    level: dict[tuple[tuple[int, ...], int], tuple[float, float]] = {(tuple(rows), 0): (0.0, 1.0)}
    work = 0
    for k in range(ncol):
        nxt: dict[tuple[tuple[int, ...], int], tuple[float, float]] = {}
        for (rem, _), (past, weight) in level.items():
            hi, lo = bound(k, rem)
            if past + hi <= threshold:
                pieces.append(math.log(weight) + past + log_total(k, rem))
                continue
            if past + lo > threshold:
                continue
            for split in splits(cols[k], rem):
                work += 1
                if work > _FISHER_LIMIT:
                    raise ValueError(f"fisher_test(): table too large for the exact test{label}")
                term = -sum(lf[v] for v in split)
                child = tuple(sorted((r - v for r, v in zip(rem, split)), reverse=True))
                value = past + term
                if k == ncol - 1:
                    if value <= threshold:
                        pieces.append(math.log(weight) + value)
                    continue
                key = (child, round(value * 1e9))
                if key in nxt:
                    nxt[key] = (nxt[key][0], nxt[key][1] + weight)
                else:
                    nxt[key] = (value, weight)
        level = nxt
    if not pieces:
        return 0.0
    top = max(pieces)
    total = top + math.log(math.fsum(math.exp(v - top) for v in pieces))
    return min(1.0, math.exp(log_k + total))


# prop_test -------------------------------------------------------------------


def _count_lists(tf: Any, groups: list[str], columns: list[Any]) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Per group, each count column's values in row order."""
    exprs = [expression(c) for c in columns]
    lf = tf._lf
    named = [e.cast(pl.Float64).alias(f"__stats3_c{i}") for i, e in enumerate(exprs)]
    if groups:
        table = lf.group_by(groups, maintain_order=True).agg(named).collect()
        table = _sorted_keys(tf, table, groups)
    else:
        table = lf.select(e.implode() for e in named).collect()
    rows = table.iter_rows(named=True)
    return table, [
        {"label": _group_label(row, groups),
         **{f"c{i}": np.array([np.nan if v is None else v for v in row[f"__stats3_c{i}"]], dtype=float)
            for i in range(len(columns))}}
        for row in rows
    ]


def _counts_column(tf: Any, spec: Any, role: str, verb: str) -> Any:
    """A column's name, checked; or a computed expression as it is."""
    if column_name(spec) is None:
        return spec
    return _column(tf, spec, role, verb)


@step
def prop_test(
    x: Any,
    n: Any,
    *,
    p: float | Sequence[float] | None = None,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    correct: bool = True,
) -> Step:
    """Test of equal or given proportions, as R's ``prop.test(x, n)``.

    ``x`` and ``n`` are columns of successes and trials, one row per sample
    (one test per group after ``group_by``)::

        penguins >> drop_na(sex) >> count(island, male=sex == "male") ...

    One sample tests against ``p`` (0.5 by default) with Wilson's score
    interval; two samples test equality with an interval for the difference;
    more samples test equality (or given ``p``). Yates' continuity correction
    applies to one and two samples unless ``correct=False``. Columns:
    ``estimate`` (one sample) or ``estimate1``..``estimateK``, ``statistic``,
    ``df``, ``p_value``, ``conf_low``, ``conf_high``, ``method``,
    ``alternative``.
    """
    given_alternative = _alternative(alternative)
    _check_conf_level(conf_level)

    def _apply(tf: Any) -> Any:
        special = _special()
        groups = list(tf._groups or [])
        xs = _counts_column(tf, x, "x", "prop_test")
        ns = _counts_column(tf, n, "n", "prop_test")
        table, samples = _count_lists(tf, groups, [xs, ns])
        rows = []
        for sample in samples:
            rows.append(_prop_row(sample["c0"], sample["c1"], p, given_alternative,
                                  conf_level, correct, sample["label"], special))
        width = max((len(r["estimates"]) for r in rows), default=1)
        columns: dict[str, Any] = {}
        if width == 1 and all(len(r["estimates"]) == 1 for r in rows):
            columns["estimate"] = np.array([r["estimates"][0] for r in rows])
        else:
            for i in range(width):
                columns[f"estimate{i + 1}"] = pl.Series(
                    [r["estimates"][i] if i < len(r["estimates"]) else None for r in rows],
                    dtype=pl.Float64,
                )
        for name in ("statistic", "df", "p_value", "conf_low", "conf_high"):
            columns[name] = pl.Series([r[name] for r in rows], dtype=pl.Float64)
        columns["method"] = pl.Series([r["method"] for r in rows], dtype=pl.String)
        columns["alternative"] = pl.Series([r["alternative"] for r in rows], dtype=pl.String)
        return _finish(tf, table, groups, columns)

    return Step(_apply, "prop_test")


def _prop_row(x, n, p, alternative, conf_level, correct, label, special) -> dict[str, Any]:
    ok = ~(np.isnan(x) | np.isnan(n))
    if p is not None:
        probs = np.atleast_1d(np.asarray(p, dtype=float))
        if len(probs) != len(x):
            raise ValueError(f"prop_test(): 'p' must have the same length as 'x' and 'n'{label}")
        probs = probs[ok]
        if np.any((probs <= 0) | (probs >= 1)):
            raise ValueError("prop_test(): elements of 'p' must be in (0,1)")
    else:
        probs = None
    x, n = x[ok], n[ok]
    k = len(x)
    if k < 1:
        raise ValueError(f"prop_test(): not enough data{label}")
    if np.any(n <= 0):
        raise ValueError(f"prop_test(): elements of 'n' must be positive{label}")
    if np.any(x < 0):
        raise ValueError(f"prop_test(): elements of 'x' must be nonnegative{label}")
    if np.any(x > n):
        raise ValueError(f"prop_test(): elements of 'x' must not be greater than those of 'n'{label}")
    if probs is None and k == 1:
        probs = np.array([0.5])
    if k > 2 or (k == 2 and probs is not None):
        alternative = "two.sided"
    estimate = x / n
    yates = 0.5 if (correct and k <= 2) else 0.0
    low = high = None
    delta = None
    if k == 1:
        z = float(special.ndtri((1 + conf_level) / 2 if alternative == "two.sided" else conf_level))
        yates = min(yates, abs(x[0] - n[0] * probs[0]))
        z22n = z**2 / (2 * n[0])
        pc = estimate[0] + yates / n[0]
        pu = 1.0 if pc >= 1 else (pc + z22n + z * math.sqrt(pc * (1 - pc) / n[0] + z22n / (2 * n[0]))) / (1 + 2 * z22n)
        pc = estimate[0] - yates / n[0]
        pl_ = 0.0 if pc <= 0 else (pc + z22n - z * math.sqrt(pc * (1 - pc) / n[0] + z22n / (2 * n[0]))) / (1 + 2 * z22n)
        low, high = {
            "two.sided": (max(pl_, 0.0), min(pu, 1.0)),
            "greater": (max(pl_, 0.0), 1.0),
            "less": (0.0, min(pu, 1.0)),
        }[alternative]
    elif k == 2 and probs is None:
        delta = estimate[0] - estimate[1]
        yates = min(yates, abs(delta) / np.sum(1 / n))
        q = float(special.ndtri((1 + conf_level) / 2 if alternative == "two.sided" else conf_level))
        width = q * math.sqrt(np.sum(estimate * (1 - estimate) / n)) + yates * np.sum(1 / n)
        low, high = {
            "two.sided": (max(delta - width, -1.0), min(delta + width, 1.0)),
            "greater": (max(delta - width, -1.0), 1.0),
            "less": (-1.0, min(delta + width, 1.0)),
        }[alternative]
    what = "1-sample proportions test" if k == 1 else (
        f"{k}-sample test for {'equality of' if probs is None else 'given'} proportions"
    )
    method = f"{what} {'with' if yates else 'without'} continuity correction"
    if probs is None:
        pooled = np.full(k, x.sum() / n.sum())
        df = k - 1
    else:
        pooled = probs
        df = k
    observed = np.column_stack([x, n - x])
    expected = np.column_stack([n * pooled, n * (1 - pooled)])
    if np.any(expected < 5):
        _warn(_CHISQ_WARNING)
    statistic = float(np.sum((np.abs(observed - expected) - yates) ** 2 / expected))
    if alternative == "two.sided":
        p_value = float(special.chdtrc(df, statistic))
    else:
        sign = np.sign(estimate[0] - pooled[0]) if k == 1 else np.sign(delta)
        z = sign * math.sqrt(statistic)
        p_value = float(special.ndtr(z) if alternative == "less" else special.ndtr(-z))
    return {
        "estimates": list(estimate),
        "statistic": statistic,
        "df": float(df),
        "p_value": p_value,
        "conf_low": low,
        "conf_high": high,
        "method": method,
        "alternative": alternative,
    }


# binom_test ------------------------------------------------------------------


@step
def binom_test(
    x: Any,
    n: Any = None,
    *,
    p: float = 0.5,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> Step:
    """Exact binomial test, as R's ``binom.test``, one test per group.

    - ``binom_test(sex == "male")``: a logical column or expression; successes
      are its true values among the non-missing ones.
    - ``binom_test(x, n)``: columns of successes and trials, one row per group.

    The interval is Clopper-Pearson's. Columns: ``estimate``, ``statistic``
    (successes), ``trials``, ``p_value``, ``conf_low``, ``conf_high``,
    ``method``, ``alternative``.
    """
    alternative = _alternative(alternative)
    _check_conf_level(conf_level)
    if not 0 <= p <= 1:
        raise ValueError(f"binom_test(): 'p' must be a single number between 0 and 1, got {p!r}")

    def _apply(tf: Any) -> Any:
        groups = list(tf._groups or [])
        if n is None:
            spec = _counts_column(tf, x, "x", "binom_test")
            value = expression(spec)
            aggs = [value.cast(pl.Boolean).sum().cast(pl.Float64).alias("x"),
                    value.count().cast(pl.Float64).alias("n")]
            lf = tf._lf
            if groups:
                table = _sorted_keys(tf, lf.group_by(groups).agg(aggs).collect(), groups)
            else:
                table = lf.select(aggs).collect()
            pairs = [(row["x"], row["n"], _group_label(row, groups)) for row in table.iter_rows(named=True)]
        else:
            xs = _counts_column(tf, x, "x", "binom_test")
            ns = _counts_column(tf, n, "n", "binom_test")
            table, samples = _count_lists(tf, groups, [xs, ns])
            pairs = []
            for sample in samples:
                if len(sample["c0"]) != 1:
                    raise ValueError(
                        f"binom_test(): x and n need one row per group{sample['label']}, "
                        f"got {len(sample['c0'])}"
                    )
                pairs.append((sample["c0"][0], sample["c1"][0], sample["label"]))
        rows = [_binom_row(xi, ni, p, alternative, conf_level, label) for xi, ni, label in pairs]
        columns = {name: np.array([r[name] for r in rows], dtype=float)
                   for name in ("estimate", "statistic", "trials", "p_value", "conf_low", "conf_high")}
        return _finish(tf, table, groups, {
            **columns, "method": "Exact binomial test", "alternative": alternative,
        })

    return Step(_apply, "binom_test")


def _binom_row(x: float, n: float, p: float, alternative: str, conf_level: float, label: str) -> dict[str, float]:
    from scipy.stats import binom

    special = _special()
    if n is None or np.isnan(n) or n < 1 or n != round(n):
        raise ValueError(f"binom_test(): 'n' must be a positive integer{label}")
    if x is None or np.isnan(x) or x < 0 or x > n or x != round(x):
        raise ValueError(f"binom_test(): 'x' must be an integer between 0 and n{label}")
    x, n = int(x), int(n)
    if alternative == "less":
        p_value = float(binom.cdf(x, n, p))
    elif alternative == "greater":
        p_value = float(binom.sf(x - 1, n, p))
    elif p == 0:
        p_value = float(x == 0)
    elif p == 1:
        p_value = float(x == n)
    else:
        # R's two-sided rule: outcomes at most as likely as x, with 1e-7 slack.
        d = binom.pmf(x, n, p)
        m = n * p
        if x == m:
            p_value = 1.0
        elif x < m:
            i = np.arange(math.ceil(m), n + 1)
            yy = int(np.sum(binom.pmf(i, n, p) <= d * (1 + 1e-7)))
            p_value = float(binom.cdf(x, n, p) + binom.sf(n - yy, n, p))
        else:
            i = np.arange(0, math.floor(m) + 1)
            yy = int(np.sum(binom.pmf(i, n, p) <= d * (1 + 1e-7)))
            p_value = float(binom.cdf(yy - 1, n, p) + binom.sf(x - 1, n, p))
        p_value = min(1.0, p_value)

    def lower(alpha: float) -> float:
        return 0.0 if x == 0 else float(special.betaincinv(x, n - x + 1, alpha))

    def upper(alpha: float) -> float:
        return 1.0 if x == n else float(special.betaincinv(x + 1, n - x, 1 - alpha))

    if alternative == "less":
        low, high = 0.0, upper(1 - conf_level)
    elif alternative == "greater":
        low, high = lower(1 - conf_level), 1.0
    else:
        alpha = (1 - conf_level) / 2
        low, high = lower(alpha), upper(alpha)
    return {
        "estimate": x / n, "statistic": float(x), "trials": float(n),
        "p_value": p_value, "conf_low": low, "conf_high": high,
    }
