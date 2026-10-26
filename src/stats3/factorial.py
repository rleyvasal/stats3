"""Analysis of variance with several terms, fitted with the ``lm`` engine.

``aov("y ~ a * b")`` and ``tukey_hsd("y ~ a + b", which="b")`` come here;
``core`` keeps the one-factor forms, which need only per-level moments.
The ANOVA table is ``summary(aov())``'s sequential (type I) one, which is
``anova(lm())``'s. Tukey's comparisons follow R's ``TukeyHSD.aov``: each
term's means are ``model.tables(fit, "means")``, the cell means of the
projections of the intercept and of every term made of the term's own
factors, so in unbalanced designs a later term's means are adjusted for
the terms before it, as in R.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Data


def check_strata(formula: str, verb: str) -> None:
    if "Error(" in formula.replace(" ", ""):
        raise ValueError(
            f"{verb}(): Error() strata are not supported yet; fit the strata "
            "separately, or use a mixed model"
        )


def _fits(data: Data, formula: str, verb: str) -> list[Any]:
    from stats3.formula import parse_formula
    from stats3.models import LinearModel, _model_frames

    check_strata(formula, verb)
    parsed = parse_formula(formula, verb, data.columns)
    return [LinearModel(parsed, frame, data, verb) for frame in _model_frames(data, parsed, None, None, verb)]


def aov_table(data: Data, formula: str) -> pl.DataFrame:
    """``summary(aov(formula))`` per group: one row per term, then Residuals."""
    return pl.concat([fit.anova_table() for fit in _fits(data, formula, "aov")], how="vertical_relaxed")


def _which(which: Any, labels: list[str]) -> list[str]:
    if which is None:
        return labels
    wanted = [which] if isinstance(which, str) else list(which)
    for name in wanted:
        if name not in labels:
            raise ValueError(f"tukey_hsd(): which= names {name!r}, not a term of the model: {labels}")
    return wanted


def _term_means(fit: Any, term: tuple[str, ...], q: np.ndarray, effects: np.ndarray,
                assign: list[int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cell means, cell sizes and cell names for one factor term, as R's
    ``model.tables(fit, "means")``: the first factor varies fastest, and
    empty cells have no mean."""
    from stats3.formula import _level_name

    own = set(term)
    inside = {0} | {i for i, t in enumerate(fit.formula.terms, start=1) if set(t) <= own}
    cols = [k for k, a in enumerate(assign) if a in inside]
    projected = q[:, cols] @ effects[cols]
    cell = np.zeros(len(projected), dtype=np.int64)
    stride, names = 1, [""]
    for var in term:
        v = fit.frame.variables[var]
        cell += v.codes * stride
        stride *= len(v.levels)
        names = [f"{a}:{_level_name(lv)}" if a else _level_name(lv) for lv in v.levels for a in names]
    n = np.bincount(cell, minlength=stride).astype(float)
    sums = np.bincount(cell, weights=projected, minlength=stride)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(n > 0, sums / n, np.nan)
    return means, np.where(n > 0, n, np.nan), names


def tukey_table(data: Data, formula: str, which: Any, ordered: bool, conf_level: float) -> pl.DataFrame:
    from stats3._tukey import ptukey, qtukey
    from stats3.core import _group_label

    frames = []
    for fit in _fits(data, formula, "tukey_hsd"):
        labels = fit.design.term_labels
        terms = _which(which, labels)
        for name in terms:
            others = [v for v in fit.formula.terms[labels.index(name)] if v not in fit.design.factors]
            if others:
                raise ValueError(
                    f"tukey_hsd(): {name!r} is not a factor term ({', '.join(others)} is numeric); "
                    "choose factor terms with which="
                )
        df = fit.df_residual
        if df <= 0:
            raise ValueError(f"tukey_hsd(): no residual degrees of freedom{_group_label(fit.frame.key, fit.groups)}")
        mse = fit.rss / df
        kept = fit.qr.kept
        q = np.linalg.qr(fit.design.matrix[:, kept], mode="reduced")[0]
        effects = q.T @ fit.y
        assign = [fit.design.assign[j] for j in kept]
        out: dict[str, list[Any]] = {
            k: [] for k in ("term", "contrast", "estimate", "conf_low", "conf_high", "adj_p_value")
        }
        for name in terms:
            means, n, cells = _term_means(fit, fit.formula.terms[labels.index(name)], q, effects, assign)
            if ordered:
                order = np.argsort(np.where(np.isnan(means), np.inf, means), kind="stable")
                means, n, cells = means[order], n[order], [cells[i] for i in order]
            k = len(means)
            q_crit = float(qtukey(conf_level, k, df)) if k > 1 else math.nan
            for b in range(k):
                for a in range(b + 1, k):
                    centre = means[a] - means[b]
                    se = math.sqrt(mse / 2 * (1 / n[a] + 1 / n[b]))
                    known = not math.isnan(centre)
                    out["term"].append(name)
                    out["contrast"].append(f"{cells[a]}-{cells[b]}")
                    out["estimate"].append(centre if known else None)
                    out["conf_low"].append(centre - q_crit * se if known else None)
                    out["conf_high"].append(centre + q_crit * se if known else None)
                    out["adj_p_value"].append(float(ptukey(abs(centre) / se, k, df, lower_tail=False)) if known else None)
        frames.append(fit._with_key(pl.DataFrame(out, schema={
            name: pl.String if name in ("term", "contrast") else pl.Float64 for name in out
        })))
    return pl.concat(frames, how="vertical_relaxed")
