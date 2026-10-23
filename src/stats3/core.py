"""Statistical tests computed from Polars aggregations.

Each test reduces the data to its summary statistics (counts, means,
variances, rank sums) in one grouped Polars query, then turns them into
p-values and intervals with ``scipy.special``. R's ``stats`` package is the
reference: argument names, defaults, missing-value handling, warnings and
errors follow it, and ``tests/test_*.py`` check the results against R.

This module holds the shared helpers and the tests built on means:
``t_test``, ``aov`` and ``tukey_hsd``. Rank tests and correlations are in
``ranks``, tests on counts in ``counts``, checks of distribution and spread
in ``checks``, and p-value adjustment, effect sizes and power in
``effects``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from tidy3.expr import Expr
from tidy3.groups import _levels
from tidy3.tidyselect import resolve_selection
from tidy3.verbs import Verb, _factor_rank

_ALTERNATIVES = {
    "two.sided": "two.sided",
    "two_sided": "two.sided",
    "less": "less",
    "greater": "greater",
}

_EPS = np.finfo(float).eps


def _special() -> Any:
    import scipy.special

    return scipy.special


def _alternative(value: str) -> str:
    try:
        return _ALTERNATIVES[value]
    except (KeyError, TypeError):
        raise ValueError(
            "alternative must be 'two.sided', 'less', or 'greater', "
            f"got {value!r}"
        ) from None


def _check_conf_level(conf_level: float) -> None:
    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be between 0 and 1, got {conf_level!r}")


def _check_mu(mu: float) -> None:
    if not math.isfinite(mu):
        raise ValueError(f"mu must be a finite number, got {mu!r}")


def _is_formula(value: Any) -> bool:
    return isinstance(value, str) and "~" in value


def _column(tf: Any, spec: Any, role: str, verb: str = "t_test") -> str:
    if isinstance(spec, Expr) and spec.node[0] in {"col", "name"}:
        spec = spec.node[1]
    names = resolve_selection(tf, [spec])
    if len(names) != 1:
        raise ValueError(f"{verb}(): {role} must name one column, got {names}")
    return names[0]


def _parse_formula(
    tf: Any, formula: str, verb: str = "t_test", *, one_sample: bool = True
) -> tuple[str, str | None]:
    """``"y ~ g"`` gives (y, g); ``"y ~ 1"`` gives (y, None)."""
    lhs, sep, rhs = formula.partition("~")
    lhs, rhs = lhs.strip(), rhs.strip()
    shape = "'y ~ group' or 'y ~ 1'" if one_sample else "'y ~ group'"
    if not sep or not lhs or not rhs or "~" in rhs or (rhs == "1" and not one_sample):
        raise ValueError(f"{verb}(): formula must look like {shape}, got {formula!r}")
    response = _column(tf, lhs, "the formula's left side", verb)
    if rhs == "1":
        return response, None
    if "+" in rhs or "*" in rhs or ":" in rhs:
        raise ValueError(
            f"{verb}(): one grouping column only, got {rhs!r}; "
            "models with several terms come with lm()"
        )
    return response, _column(tf, rhs, "the formula's right side", verb)


def _present(lf: pl.LazyFrame, name: str) -> pl.Expr:
    """Not missing, as R's ``is.na``: NaN counts as missing too."""
    present = pl.col(name).is_not_null()
    if lf.collect_schema()[name].is_float():
        present = present & pl.col(name).is_not_nan()
    return present


def _moments(value: pl.Expr) -> list[pl.Expr]:
    return [
        value.count().alias("n"),
        value.mean().alias("mean"),
        value.var().alias("var"),
    ]


def _summarise(lf: pl.LazyFrame, groups: list[str], aggs: list[pl.Expr]) -> pl.DataFrame:
    if groups:
        return lf.group_by(groups).agg(aggs).collect()
    return lf.select(aggs).collect()


def _keys(table: pl.DataFrame, groups: list[str]) -> list[dict[str, Any]]:
    return table.select(groups).to_dicts() if groups else [{}] * table.height


def _group_label(row: dict[str, Any], groups: list[str]) -> str:
    if not groups:
        return ""
    return " in group " + ", ".join(f"{g} = {row[g]!r}" for g in groups)


def _sorted_keys(tf: Any, frame: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
    """Order groups as dplyr does: factor level order, else value order, missing last."""
    if not groups:
        return frame
    keys = []
    for name in groups:
        levels = _levels(tf, name)
        keys.append(_factor_rank(name, levels) if levels is not None else pl.col(name))
    return frame.sort(keys, nulls_last=True, maintain_order=True)


def _level_rank(tf: Any, by: str) -> pl.Expr:
    levels = _levels(tf, by)
    return _factor_rank(by, levels) if levels is not None else pl.col(by)


def _by_levels(
    tf: Any,
    kept: pl.LazyFrame,
    groups: list[str],
    by: str,
    aggs: list[pl.Expr],
    verb: str,
    *,
    exactly: int | None = None,
) -> pl.DataFrame:
    """Aggregate per (group, level); one row per group with lists in level order.

    Levels are the ones present in each group, in factor order, else sorted,
    as R's ``factor(g)`` after dropping missing rows.
    """
    if by in groups:
        raise ValueError(f"{verb}(): {by!r} is a grouping column; ungroup it first")
    per_level = kept.group_by([*groups, by]).agg(aggs).collect()
    per_level = per_level.sort([*groups, _level_rank(tf, by)], nulls_last=True)
    stacked = (
        per_level.group_by(groups, maintain_order=True).agg(pl.all())
        if groups
        else per_level.select(pl.all().implode())
    )
    for row in stacked.select([*groups, by]).iter_rows(named=True):
        found = row[by]
        label = _group_label(row, groups)
        if exactly is not None and len(found) != exactly:
            raise ValueError(
                f"{verb}(): grouping factor {by!r} must have exactly {exactly} levels"
                f"{label}, found {len(found)}: {found}"
            )
        if len(found) < 2:
            raise ValueError(
                f"{verb}(): all observations are in the same group{label}"
            )
    return _sorted_keys(tf, stacked, groups)


def _finish(tf: Any, table: pl.DataFrame, groups: list[str], columns: dict[str, Any]) -> Any:
    """The result: group keys, then the test's columns; ungrouped."""
    data = {g: table[g] for g in groups}
    for name, values in columns.items():
        if isinstance(values, pl.Series):
            data[name] = values.alias(name)
        elif isinstance(values, str):
            data[name] = [values] * table.height
        else:
            data[name] = values
    result = pl.DataFrame(data) if data else pl.DataFrame()
    return tf._with_lf(result.lazy(), groups=None, rowwise=False)


def _p_normal(z: np.ndarray, alternative: str) -> np.ndarray:
    special = _special()
    if alternative == "less":
        return special.ndtr(z)
    if alternative == "greater":
        return special.ndtr(-z)
    return 2 * special.ndtr(-np.abs(z))


def t_test(
    x: Any,
    y: Any = None,
    *,
    mu: float = 0.0,
    paired: bool = False,
    var_equal: bool = False,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> Verb:
    """Student's and Welch's t-tests, as R's ``t.test``, one row per group.

    ``x`` is a formula or a column:

    - ``t_test("body_mass_g ~ sex")``: two groups, Welch's test unless
      ``var_equal=True``. The first level (factor order, else sorted) is
      ``estimate1``; ``estimate`` is ``estimate1 - estimate2``.
    - ``t_test("flipper_length_mm ~ 1", mu=200)`` or
      ``t_test(flipper_length_mm, mu=200)``: one sample against ``mu``.
    - ``t_test(bill_length_mm, bill_depth_mm, paired=True)``: paired, on rows
      where both are present.
    - ``t_test(bill_length_mm, bill_depth_mm)``: two columns as two samples.

    Missing values are dropped as R drops them. After ``group_by`` the test
    runs once per group; the result keeps the group columns and is ungrouped.
    Columns: ``estimate`` (plus ``estimate1``/``estimate2`` for two samples),
    ``statistic``, ``df``, ``p_value``, ``conf_low``, ``conf_high``,
    ``method``, ``alternative``.
    """
    alternative = _alternative(alternative)
    _check_conf_level(conf_level)
    _check_mu(mu)
    formula = _is_formula(x)
    if formula and paired:
        raise ValueError(
            "t_test(): paired=True takes two columns, "
            "t_test(x, y, paired=True), not a formula"
        )
    if formula and y is not None:
        raise ValueError("t_test(): pass a formula or x and y, not both")
    if paired and y is None:
        raise ValueError("t_test(): paired=True needs two columns, x and y")

    def _apply(tf: Any) -> Any:
        groups = list(tf._groups or [])
        lf = tf._lf
        if formula:
            response, by = _parse_formula(tf, x)
            if by is not None:
                kind, table = "two", _two_sample_by(tf, lf, groups, response, by)
            else:
                kind, table = "one", _one_sample(lf, groups, response)
        elif y is None:
            kind, table = "one", _one_sample(lf, groups, _column(tf, x, "x"))
        elif paired:
            kind = "paired"
            table = _paired(lf, groups, _column(tf, x, "x"), _column(tf, y, "y"))
        else:
            kind = "two"
            table = _two_columns(lf, groups, _column(tf, x, "x"), _column(tf, y, "y"))
        table = _sorted_keys(tf, table, groups)
        columns = _t_statistics(
            table, groups, kind, mu=mu, var_equal=var_equal,
            alternative=alternative, conf_level=conf_level,
        )
        return _finish(tf, table, groups, columns)

    return Verb(_apply, "t_test")


def _one_sample(lf: pl.LazyFrame, groups: list[str], name: str) -> pl.DataFrame:
    value = pl.col(name).filter(_present(lf, name)).cast(pl.Float64)
    return _summarise(lf, groups, _moments(value))


def _paired(lf: pl.LazyFrame, groups: list[str], x: str, y: str) -> pl.DataFrame:
    both = _present(lf, x) & _present(lf, y)
    diff = (pl.col(x).cast(pl.Float64) - pl.col(y).cast(pl.Float64)).filter(both)
    return _summarise(lf, groups, _moments(diff))


def _two_columns(lf: pl.LazyFrame, groups: list[str], x: str, y: str) -> pl.DataFrame:
    aggs = []
    for side, name in (("1", x), ("2", y)):
        value = pl.col(name).filter(_present(lf, name)).cast(pl.Float64)
        aggs += [e.name.suffix(side) for e in _moments(value)]
    return _summarise(lf, groups, aggs)


def _two_sample_by(
    tf: Any, lf: pl.LazyFrame, groups: list[str], response: str, by: str
) -> pl.DataFrame:
    kept = lf.filter(_present(lf, response) & _present(lf, by))
    stacked = _by_levels(
        tf, kept, groups, by, _moments(pl.col(response).cast(pl.Float64)), "t_test",
        exactly=2,
    )
    return stacked.select(
        *groups,
        *(pl.col(stat).list.get(i).alias(f"{stat}{i + 1}")
          for stat in ("n", "mean", "var") for i in (0, 1)),
    )


def _t_statistics(
    table: pl.DataFrame,
    groups: list[str],
    kind: str,
    *,
    mu: float,
    var_equal: bool,
    alternative: str,
    conf_level: float,
) -> dict[str, Any]:
    special = _special()
    keys = _keys(table, groups)

    def arr(name: str) -> np.ndarray:
        return table[name].cast(pl.Float64).fill_null(np.nan).to_numpy()

    if kind == "two":
        n1, n2, m1, m2, v1, v2 = (arr(c) for c in ("n1", "n2", "mean1", "mean2", "var1", "var2"))
        for i, key in enumerate(keys):
            label = _group_label(key, groups)
            if var_equal:
                if n1[i] < 1 or n2[i] < 1 or n1[i] + n2[i] < 3:
                    raise ValueError(f"t_test(): not enough observations{label}")
            else:
                if n1[i] < 2:
                    raise ValueError(f"t_test(): not enough 'x' observations{label}")
                if n2[i] < 2:
                    raise ValueError(f"t_test(): not enough 'y' observations{label}")
        v1 = np.where(n1 < 2, 0.0, v1)
        v2 = np.where(n2 < 2, 0.0, v2)
        if var_equal:
            df = n1 + n2 - 2
            pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
            stderr = np.sqrt(pooled * (1 / n1 + 1 / n2))
            method = "Two Sample t-test"
        else:
            se1, se2 = v1 / n1, v2 / n2
            stderr = np.sqrt(se1 + se2)
            df = stderr**4 / (se1**2 / (n1 - 1) + se2**2 / (n2 - 1))
            method = "Welch Two Sample t-test"
        scale = np.maximum(np.abs(m1), np.abs(m2))
        centre = m1 - m2
        estimates = {"estimate": centre, "estimate1": m1, "estimate2": m2}
    else:
        n, m, v = arr("n"), arr("mean"), arr("var")
        for i, key in enumerate(keys):
            if n[i] < 2:
                raise ValueError(
                    f"t_test(): not enough 'x' observations{_group_label(key, groups)}"
                )
        df = n - 1
        stderr = np.sqrt(v / n)
        scale = np.abs(m)
        centre = m
        method = "Paired t-test" if kind == "paired" else "One Sample t-test"
        estimates = {"estimate": m}

    for i, key in enumerate(keys):
        if stderr[i] < 10 * _EPS * scale[i]:
            raise ValueError(
                f"t_test(): data are essentially constant{_group_label(key, groups)}"
            )

    statistic = (centre - mu) / stderr
    if alternative == "less":
        p_value = special.stdtr(df, statistic)
        q = special.stdtrit(df, conf_level)
        low, high = np.full_like(statistic, -np.inf), statistic + q
    elif alternative == "greater":
        p_value = special.stdtr(df, -statistic)
        q = special.stdtrit(df, conf_level)
        low, high = statistic - q, np.full_like(statistic, np.inf)
    else:
        p_value = 2 * special.stdtr(df, -np.abs(statistic))
        q = special.stdtrit(df, 1 - (1 - conf_level) / 2)
        low, high = statistic - q, statistic + q

    return {
        **{name: pl.Series(name, values, dtype=pl.Float64) for name, values in estimates.items()},
        "statistic": statistic,
        "df": df.astype(float),
        "p_value": p_value,
        "conf_low": mu + low * stderr,
        "conf_high": mu + high * stderr,
        "method": method,
        "alternative": alternative,
    }


def _one_way(tf: Any, formula: str, verb: str) -> tuple[pl.DataFrame, list[str], str]:
    """Per group: level names, sizes, means and variances of the response."""
    groups = list(tf._groups or [])
    lf = tf._lf
    response, by = _parse_formula(tf, formula, verb, one_sample=False)
    kept = lf.filter(_present(lf, response) & _present(lf, by))
    table = _by_levels(
        tf, kept, groups, by, _moments(pl.col(response).cast(pl.Float64)), verb
    )
    return table, groups, by


def _within(n: np.ndarray, v: np.ndarray) -> float:
    """Pooled within-level sum of squares; a one-row level adds nothing."""
    return float(np.sum(np.where(n > 1, (n - 1) * np.nan_to_num(v), 0.0)))


def aov(formula: str) -> Verb:
    """One-way analysis of variance, as ``tidy(aov(y ~ g))`` in R.

    ``penguins >> aov("body_mass_g ~ species")`` gives the ANOVA table: one row
    for the grouping column and one for ``Residuals``, with ``df``, ``sumsq``,
    ``meansq``, ``statistic`` (F) and ``p_value``. After ``group_by`` there is
    a table per group. Rows missing the response or the group are dropped.
    Models with several terms will come with ``lm()``.
    """

    def _apply(tf: Any) -> Any:
        special = _special()
        table, groups, by = _one_way(tf, formula, "aov")
        rows: dict[str, list[Any]] = {
            k: [] for k in ("term", "df", "sumsq", "meansq", "statistic", "p_value")
        }
        for row in table.iter_rows(named=True):
            n = np.array(row["n"], dtype=float)
            m = np.array(row["mean"], dtype=float)
            v = np.array([np.nan if x is None else x for x in row["var"]], dtype=float)
            total = n.sum()
            grand = np.sum(n * m) / total
            between = float(np.sum(n * (m - grand) ** 2))
            within = _within(n, v)
            df_b, df_w = len(n) - 1, total - len(n)
            ms_b = between / df_b
            ms_w = within / df_w if df_w > 0 else np.nan
            f = ms_b / ms_w if df_w > 0 else np.nan
            p = float(special.fdtrc(df_b, df_w, f)) if df_w > 0 else np.nan
            rows["term"] += [by, "Residuals"]
            rows["df"] += [float(df_b), float(df_w)]
            rows["sumsq"] += [between, within]
            rows["meansq"] += [ms_b, ms_w]
            rows["statistic"] += [None if np.isnan(f) else f, None]
            rows["p_value"] += [None if np.isnan(p) else p, None]
        keys = table[np.repeat(np.arange(table.height), 2)]
        columns = {
            name: pl.Series(name, values, dtype=pl.String if name == "term" else pl.Float64)
            for name, values in rows.items()
        }
        return _finish(tf, keys, groups, columns)

    return Verb(_apply, "aov")


def tukey_hsd(formula: str, *, conf_level: float = 0.95) -> Verb:
    """Tukey's honest significant differences, as ``tidy(TukeyHSD(aov(y ~ g)))``.

    One row per pair of levels, in R's order (``"b-a"``, ``"c-a"``, ``"c-b"``):
    ``term``, ``contrast``, ``estimate`` (difference of means), ``conf_low``,
    ``conf_high`` and ``adj_p_value``. Unequal group sizes use the
    Tukey-Kramer standard error, as R does. R's ``qtukey`` is accurate to
    about four digits, so intervals agree with R to about that.
    """
    _check_conf_level(conf_level)

    def _apply(tf: Any) -> Any:
        from scipy.stats import studentized_range

        table, groups, by = _one_way(tf, formula, "tukey_hsd")
        out: dict[str, list[Any]] = {
            k: [] for k in ("term", "contrast", "estimate", "conf_low", "conf_high", "adj_p_value")
        }
        index: list[int] = []
        for i, row in enumerate(table.iter_rows(named=True)):
            levels = [str(level) for level in row[by]]
            n = np.array(row["n"], dtype=float)
            m = np.array(row["mean"], dtype=float)
            v = np.array([np.nan if x is None else x for x in row["var"]], dtype=float)
            k = len(n)
            df = n.sum() - k
            if df <= 0:
                raise ValueError(
                    f"tukey_hsd(): no residual degrees of freedom{_group_label(row, groups)}"
                )
            mse = _within(n, v) / df
            q_crit = float(studentized_range.ppf(conf_level, k, df))
            for b in range(k):
                for a in range(b + 1, k):
                    centre = m[a] - m[b]
                    se = math.sqrt(mse / 2 * (1 / n[a] + 1 / n[b]))
                    out["term"].append(by)
                    out["contrast"].append(f"{levels[a]}-{levels[b]}")
                    out["estimate"].append(centre)
                    out["conf_low"].append(centre - q_crit * se)
                    out["conf_high"].append(centre + q_crit * se)
                    out["adj_p_value"].append(
                        float(studentized_range.sf(abs(centre) / se, k, df))
                    )
                    index.append(i)
        keys = table[index] if index else table.clear()
        columns = {
            name: pl.Series(name, values, dtype=pl.String if name in {"term", "contrast"} else pl.Float64)
            for name, values in out.items()
        }
        return _finish(tf, keys, groups, columns)

    return Verb(_apply, "tukey_hsd")
