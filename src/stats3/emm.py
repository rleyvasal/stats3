"""Estimated marginal means and their contrasts, as R's emmeans package.

``emmeans(fit, "species")`` builds emmeans' reference grid (every level of
each factor and binary covariate, other numeric covariates at their means
over the model's rows), predicts each cell, and averages the cells over the
factors not in ``specs`` (equally, by default). ``pairs`` and ``contrast``
compare the means, with emmeans' p-value and confidence-level adjustments
(Tukey, Dunnett's approximation, Sidak, Bonferroni and R's ``p.adjust``
methods).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from stats3.core import _check_conf_level, _special, _warn
from stats3.effects import P_ADJUST_METHODS, adjust_p

WEIGHTS = ("equal", "proportional", "cells")
ADJUST = ("tukey", "dunnettx", "sidak", "scheffe", *P_ADJUST_METHODS)
METHODS = ("pairwise", "revpairwise", "trt.vs.ctrl", "eff")


# the reference grid ----------------------------------------------------------------


def _sources(fit: Any) -> tuple[dict[str, list[Any]], list[str]]:
    """The model's source columns: factor ones with their levels, numeric ones."""
    from stats3.formula import _column_names

    factors: dict[str, list[Any]] = {}
    numeric: list[str] = []
    labels = list(fit.formula.variables()) + list(fit.formula.offsets)
    for label in labels:
        names = _column_names(label)
        levels = fit.design.levels.get(label)
        for name in names:
            if levels is not None and len(names) == 1:
                factors.setdefault(name, list(levels))
            elif name not in numeric:
                numeric.append(name)
    clash = [n for n in numeric if n in factors]
    if clash:
        raise ValueError(f"emmeans(): {clash} enter the model both as factors and as numbers")
    return factors, numeric


def _reference_grid(fit: Any, at: Mapping[str, Any]) -> tuple[pl.DataFrame, list[str]]:
    """emmeans' ``ref_grid``: all factor levels crossed with covariate means
    (or the values in ``at``), in emmeans' order (the first factor varies
    fastest)."""
    factors, numeric = _sources(fit)
    rows = fit.frame.rows
    values: dict[str, list[Any]] = {}
    grid_factors = list(factors)
    for name, levels in factors.items():
        values[name] = levels
    for name in numeric:
        unique = rows[name].unique().sort().to_list()
        # R's default cov.keep="2" retains binary numeric covariates.
        values[name] = unique if len(unique) <= 2 else [float(rows[name].cast(pl.Float64).mean())]
        if len(unique) <= 2:
            grid_factors.append(name)
    for name, given in at.items():
        if name not in values:
            raise ValueError(f"emmeans(): at= names {name!r}, which is not in the model")
        given = list(given) if isinstance(given, (list, tuple, np.ndarray, pl.Series)) else [given]
        if name in factors:
            unknown = [g for g in given if g not in factors[name]]
            if unknown:
                raise ValueError(f"emmeans(): {unknown} are not levels of {name!r}")
        else:
            given = [float(g) for g in given]
        values[name] = given
    names = list(values)
    # The first name varies fastest, as R's expand.grid.
    combos = [tuple(reversed(c)) for c in itertools.product(*(values[n] for n in reversed(names)))]
    schema = rows.schema
    grid = pl.DataFrame(
        {n: pl.Series([c[i] for c in combos], dtype=schema[n] if n in factors else pl.Float64)
         for i, n in enumerate(names)}
    )
    return grid, grid_factors


def _cell_counts(fit: Any, grid: pl.DataFrame, factors: list[str]) -> np.ndarray:
    """How many of the model's rows fall in each grid cell (emmeans' .wgt.)."""
    if not factors:
        return np.full(grid.height, float(fit.frame.rows.height))
    counts = fit.frame.rows.group_by(factors).len()
    counts = counts.with_columns(pl.col(n).cast(grid.schema[n]) for n in factors)
    joined = grid.select(factors).with_row_index("__i").join(
        counts, on=factors, how="left", nulls_equal=True
    ).sort("__i")
    return joined["len"].fill_null(0).cast(pl.Float64).to_numpy()


def _numeric_label(levels: Sequence[Any]) -> bool:
    try:
        [float(v) for v in levels]
    except (TypeError, ValueError):
        return False
    return not all(isinstance(v, bool) for v in levels)


class _Estimates:
    """Linear functions of the coefficients, with their table rows."""

    def __init__(self, fit: Any, keys: pl.DataFrame, labels: pl.DataFrame, lin: np.ndarray, by: list[str],
                 vcov: Any, type_: str, level: float):
        self.fit, self.keys, self.labels, self.lin, self.by = fit, keys, labels, lin, by
        self.vcov, self.type, self.level = vcov, type_, level
        self.offset = np.zeros(labels.height)
        kept = fit.qr.kept
        b = np.where(np.isnan(fit.coef), 0.0, fit.coef)
        self.estimate = lin @ b
        if vcov is None:
            v = fit.cov_unscaled * fit.scale
        else:
            from stats3.robust import as_vcov, covariance

            v = covariance(fit, as_vcov(vcov))
        lk = lin[:, kept]
        self.se = np.sqrt(np.einsum("ij,jk,ik->i", lk, v, lk))
        estimable = _estimable(fit, lin)
        self.estimate[~estimable] = np.nan
        self.se[~estimable] = np.nan
        self.df = np.full(len(self.estimate), _df(fit))
        self.df[~np.isfinite(self.estimate)] = np.nan


def _df(fit: Any) -> float:
    from stats3.models import GeneralizedLinearModel

    if isinstance(fit, GeneralizedLinearModel):
        # emmeans: the residual df for gaussian and Gamma glms, else asymptotic.
        return float(fit.df_residual) if fit.family.family in ("gaussian", "Gamma") else math.inf
    return float(fit.df_residual)


def _estimable(fit: Any, lin: np.ndarray) -> np.ndarray:
    """estimability's check: is each row in the row space of the model matrix?"""
    if len(fit.qr.kept) == lin.shape[1]:
        return np.ones(lin.shape[0], dtype=bool)
    x = fit.design.matrix
    _, s, vt = np.linalg.svd(x, full_matrices=True)
    rank = int(np.sum(s > s[0] * 1e-7)) if len(s) else 0
    null = vt[rank:].T
    tol = 1e-8 * np.maximum(1, np.abs(lin).max(axis=1))
    return np.all(np.abs(lin @ null) <= tol[:, None], axis=1)


def _link(fit: Any) -> Any:
    from stats3.models import GeneralizedLinearModel

    return fit.family.link if isinstance(fit, GeneralizedLinearModel) else None


def _critical(level: float, df: np.ndarray, adjust: str, n_contrasts: int, family: float,
              scheffe_rank: int | None = None) -> np.ndarray:
    """emmeans' ``.adj.critval`` for two-sided intervals."""
    from scipy.stats import studentized_range

    special = _special()
    out = np.empty(len(df))
    for i, d in enumerate(df):
        d = math.inf if math.isnan(d) else d
        t = (lambda p: float(special.ndtri(p))) if math.isinf(d) else (lambda p, d=d: float(special.stdtrit(d, p)))
        if adjust == "none" or (n_contrasts == 1 and adjust != "scheffe"):
            out[i] = -t((1 - level) / 2)
        elif adjust == "sidak":
            out[i] = -t((1 - level ** (1 / n_contrasts)) / 2)
        elif adjust == "tukey":
            out[i] = float(studentized_range.ppf(level, family, d)) / math.sqrt(2)
        elif adjust == "scheffe":
            from scipy.stats import f as f_dist

            rank = family - 1 if scheffe_rank is None else scheffe_rank
            out[i] = math.sqrt(rank * f_dist.ppf(level, rank, d))
        elif adjust == "dunnettx":
            out[i] = _q_dunnettx(level, n_contrasts, d)
        else:  # bonferroni, and the p.adjust methods
            out[i] = -t((1 - level) / n_contrasts / 2)
    return out


def _p_dunnettx(x: float, k: int, df: float) -> float:
    """emmeans' ``.pdunnx``: a Tukey-Sidak blend approximating Dunnett."""
    from scipy.stats import f as f_dist, studentized_range

    twt = (k - 1) / k
    tukey = float(studentized_range.cdf(math.sqrt(2) * x, (1 + math.sqrt(1 + 8 * k)) / 2, df))
    sidak = float(f_dist.cdf(x**2, 1, df)) ** k
    return twt * tukey + (1 - twt) * sidak


def _q_dunnettx(p: float, k: int, df: float) -> float:
    from scipy.optimize import brentq
    from scipy.stats import f as f_dist, studentized_range

    special = _special()
    if k < 1.005:
        return float(special.ndtri(1 - 0.5 * (1 - p))) if math.isinf(df) else float(special.stdtrit(df, 1 - 0.5 * (1 - p)))
    xtuk = float(studentized_range.ppf(p, (1 + math.sqrt(1 + 8 * k)) / 2, df)) / math.sqrt(2)
    xsid = math.sqrt(float(f_dist.ppf(p ** (1 / k), 1, df)))
    if abs(xtuk - xsid) < 5e-4:
        return xtuk
    return brentq(lambda x: _p_dunnettx(x, k, df) - p, min(xtuk, xsid), max(xtuk, xsid), xtol=1e-12)


def _p_values(stat: np.ndarray, df: np.ndarray, adjust: str, family: float,
              scheffe_rank: int | None = None) -> np.ndarray:
    """emmeans' ``.adj.p.value`` for two-sided tests within one family."""
    from scipy.stats import f as f_dist, studentized_range

    special = _special()
    df = np.where(np.isnan(df), math.inf, df)
    p = np.array([2 * special.ndtr(-abs(t)) if math.isinf(d) else 2 * special.stdtr(d, -abs(t))
                  for t, d in zip(stat, df)])
    n = int(np.sum(~np.isnan(p)))
    if adjust == "none" or (n <= 1 and adjust != "scheffe"):
        return p
    if adjust in P_ADJUST_METHODS:
        return adjust_p(p, adjust)
    if adjust == "sidak":
        return 1 - (1 - p) ** n
    if adjust == "tukey":
        return np.array([float(studentized_range.sf(math.sqrt(2) * abs(t), family, d)) for t, d in zip(stat, df)])
    if adjust == "scheffe":
        rank = family - 1 if scheffe_rank is None else scheffe_rank
        return np.array([float(f_dist.sf(t**2 / rank, rank, d)) for t, d in zip(stat, df)])
    return np.array([1 - _p_dunnettx(abs(t), n, d) for t, d in zip(stat, df)])


# results -------------------------------------------------------------------------------


class EMMeans:
    """Estimated marginal means; ``tidy()`` gives the table, and ``pairs()``
    and ``contrast()`` compare them."""

    kind = "emmeans"

    def __init__(self, parts: list[_Estimates], specs: list[str], data: Any):
        self.parts, self.specs, self._data = parts, specs, data

    # tables -----------------------------------------------------------------

    def _table(self, part: _Estimates) -> pl.DataFrame:
        crit = _critical(part.level, part.df, "none", 1, 1)
        est, se = part.estimate, part.se
        low, high = est - crit * se, est + crit * se
        link = _link(part.fit)
        if part.type == "response" and link is not None:
            mu_eta = np.abs(link.mu_eta(est))
            est, se = link.linkinv(est), se * mu_eta
            low, high = link.linkinv(low), link.linkinv(high)
            low, high = np.minimum(low, high), np.maximum(low, high)
        cols = {"estimate": est, "std_error": se, "df": part.df, "conf_low": low, "conf_high": high}
        out = pl.concat([part.keys, part.labels, pl.DataFrame(cols)], how="horizontal")
        return out.with_columns(pl.col(c).fill_nan(None) for c in cols)

    def tidy(self) -> pl.DataFrame:
        return pl.concat([self._table(p) for p in self.parts])

    def __tidy__(self, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(f"tidy(): unexpected arguments for emmeans: {sorted(kwargs)}")
        return self._data.write(self.tidy())

    def __repr__(self) -> str:
        from stats3.models import _table_text

        part = self.parts[0]
        note = "estimates on the response scale" if part.type == "response" and _link(part.fit) else "estimates"
        return f"emmeans of {', '.join(self.specs)} ({note}, {part.level:.0%} intervals)\n{_table_text(self.tidy())}"

    # comparisons -----------------------------------------------------------------

    def contrast(self, method: str | Mapping[str, Sequence[float]] = "pairwise", *, adjust: str | None = None,
                 ref: Any = 1, level: float | None = None) -> Contrasts:
        """Compare the means, as emmeans' ``contrast``.

        ``method`` is ``"pairwise"`` (each mean minus each later one; Tukey
        adjustment), ``"revpairwise"``, ``"trt.vs.ctrl"`` (each level minus
        ``ref``, a level or its position from 1; Dunnett's approximation, as
        emmeans' ``dunnettx``), ``"eff"`` (each mean minus their average;
        FDR), or a dict of named weight lists. ``adjust`` overrides the
        method's adjustment: ``"tukey"``, ``"dunnettx"``, ``"sidak"``,
        ``"scheffe"``, ``"bonferroni"``, ``"holm"`` and the other
        ``p_adjust`` methods, or ``"none"``.
        """
        return Contrasts(self, method, adjust, ref, level)

    def pairs(self, *, adjust: str = "tukey", reverse: bool = False, level: float | None = None) -> Contrasts:
        """All pairwise differences, as emmeans' ``pairs``."""
        return self.contrast("revpairwise" if reverse else "pairwise", adjust=adjust, level=level)


def _level_labels(labels: pl.DataFrame, specs: list[str]) -> list[str]:
    columns = []
    for name in specs:
        values = labels[name].to_list()
        prefix = name if _numeric_label(values) else ""
        columns.append([prefix + (f"{v:g}" if isinstance(v, float) else str(v)) for v in values])
    return [" ".join(parts) for parts in zip(*columns)]


def _weights_for(method: Any, k: int, levels: list[str], ref: Any) -> tuple[list[str], np.ndarray, str, str]:
    """Contrast names, the weight matrix (contrasts x means), the default
    adjustment, and the kind ("pairs" for Tukey)."""
    if isinstance(method, Mapping):
        names = list(method)
        weights = np.array([list(map(float, method[n])) for n in names])
        if weights.shape[1] != k:
            raise ValueError(f"contrast(): each weight list needs {k} values, one per mean")
        return names, weights, "none", "custom"
    if method not in METHODS:
        raise ValueError(f"contrast(): method must be one of {list(METHODS)} or a dict of weights, got {method!r}")
    names, rows = [], []
    if method in ("pairwise", "revpairwise"):
        # emmeans' order: pairwise i < j by i then j; revpairwise i > j by i then j.
        order = ([(i, j) for i in range(k - 1) for j in range(i + 1, k)] if method == "pairwise"
                 else [(i, j) for i in range(1, k) for j in range(i)])
        for i, j in order:
            w = np.zeros(k)
            w[i], w[j] = 1, -1
            names.append(f"{levels[i]} - {levels[j]}")
            rows.append(w)
        return names, np.array(rows), "tukey", "pairs"
    if method == "trt.vs.ctrl":
        index = levels.index(ref) if ref in levels else (ref - 1 if isinstance(ref, int) and not isinstance(ref, bool) else None)
        if index is None or not 0 <= index < k:
            raise ValueError(f"contrast(): ref must be a level or a position from 1 to {k}, got {ref!r}")
        for i in range(k):
            if i == index:
                continue
            w = np.zeros(k)
            w[i], w[index] = 1, -1
            names.append(f"{levels[i]} - {levels[index]}")
            rows.append(w)
        return names, np.array(rows), "dunnettx", "trt"
    for i in range(k):
        w = np.full(k, -1 / k)
        w[i] += 1
        names.append(f"{levels[i]} effect")
        rows.append(w)
    return names, np.array(rows), "fdr", "eff"


class Contrasts:
    """Contrasts of estimated marginal means; ``tidy()`` gives the table."""

    kind = "emmeans"

    def __init__(self, emm: EMMeans, method: Any, adjust: str | None, ref: Any, level: float | None):
        if adjust is not None and adjust not in ADJUST:
            raise ValueError(f"contrast(): adjust must be one of {list(ADJUST)}, got {adjust!r}")
        if level is not None:
            _check_conf_level(level)
        self.emm, self.method = emm, method
        self.tables: list[pl.DataFrame] = []
        self.adjust_used: str | None = None
        for part in emm.parts:
            self.tables.append(self._part(part, method, adjust, ref, level or part.level))

    def _part(self, part: _Estimates, method: Any, adjust: str | None, ref: Any, level: float) -> pl.DataFrame:
        by = part.by
        frames = []
        combined = pl.concat([part.labels, pl.DataFrame({"__i": np.arange(part.labels.height)})], how="horizontal")
        by_groups = combined.partition_by(by, maintain_order=True) if by else [combined]
        link = _link(part.fit)
        for chunk in by_groups:
            idx = chunk["__i"].to_numpy()
            levels = _level_labels(chunk, [s for s in self.emm.specs if s not in by])
            names, weights, default, kind = _weights_for(method, len(idx), levels, ref)
            true_contrast = np.all(np.abs(weights.sum(axis=1)) < 1e-6 * (1e-6 + np.abs(weights).max(axis=1)))
            response = part.type == "response" and link is not None and link.name in ("log", "logit") and true_contrast
            use = adjust or default
            if kind != "pairs" and use == "tukey":
                _warn("Tukey adjustment is only for pairwise comparisons; using sidak")
                use = "sidak"
            self.adjust_used = use
            lin = weights @ part.lin[idx]
            b = np.where(np.isnan(part.fit.coef), 0.0, part.fit.coef)
            est = lin @ b + weights @ part.offset[idx]
            kept = part.fit.qr.kept
            if part.vcov is None:
                v = part.fit.cov_unscaled * part.fit.scale
            else:
                from stats3.robust import as_vcov, covariance

                v = covariance(part.fit, as_vcov(part.vcov))
            lk = lin[:, kept]
            se = np.sqrt(np.einsum("ij,jk,ik->i", lk, v, lk))
            bad = ~_estimable(part.fit, lin) | ~np.isfinite(est)
            est[bad], se[bad] = np.nan, np.nan
            df = np.full(len(est), _df(part.fit))
            df[bad] = np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                stat = est / se
            family = len(idx)
            n_contrasts = int(np.sum(~bad))
            if np.any(bad):
                family = (1 + math.sqrt(1 + 8 * n_contrasts)) / 2
            rank = int(np.linalg.matrix_rank(lk[~bad])) if n_contrasts else 0
            p = _p_values(stat, df, use, family, rank)
            ci_adjust = use if use in ("none", "tukey", "sidak", "scheffe", "dunnettx", "bonferroni") else "bonferroni"
            crit = _critical(level, df, ci_adjust, n_contrasts, family, rank)
            low, high = est - crit * se, est + crit * se
            cols: dict[str, Any] = {"contrast": names}
            if response:
                ratio = np.exp(est)
                cols.update(estimate=ratio, std_error=ratio * se, df=df, statistic=stat, p_value=p,
                            conf_low=np.exp(low), conf_high=np.exp(high))
                cols["contrast"] = [n.replace(" - ", " / ") for n in names]
            else:
                cols.update(estimate=est, std_error=se, df=df, statistic=stat, p_value=p,
                            conf_low=low, conf_high=high)
            keys = part.keys.head(1) if part.keys.width else pl.DataFrame()
            head = pl.concat([keys] * len(names)) if keys.width else pl.DataFrame()
            by_cols = chunk.select(by).head(1) if by else pl.DataFrame()
            by_rep = pl.concat([by_cols] * len(names)) if by else pl.DataFrame()
            frame = pl.concat([f for f in (head, by_rep, pl.DataFrame(cols)) if f.width], how="horizontal")
            frames.append(frame)
        out = pl.concat(frames)
        numeric = [c for c in out.columns if out.schema[c] == pl.Float64]
        return out.with_columns(pl.col(c).fill_nan(None) for c in numeric)

    def tidy(self) -> pl.DataFrame:
        return pl.concat(self.tables)

    def __tidy__(self, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(f"tidy(): unexpected arguments for contrasts: {sorted(kwargs)}")
        return self.emm._data.write(self.tidy())

    def __repr__(self) -> str:
        from stats3.models import _table_text

        return f"contrasts (p-value adjustment: {self.adjust_used})\n{_table_text(self.tidy())}"


# the verbs -------------------------------------------------------------------------------


def _one(fit: Any, specs: list[str], by: list[str], at: Mapping[str, Any], weights: str, vcov: Any,
         type_: str, level: float) -> _Estimates:
    grid, factors = _reference_grid(fit, at)
    for name in [*specs, *by]:
        if name not in grid.columns:
            raise ValueError(f"emmeans(): {name!r} is not a variable in the model; its variables are {grid.columns}")
    design_grid = grid
    static_offset = None
    if fit.formula.offset_arg is not None:
        from stats3._data import column_name
        from stats3.formula import variable_expr

        name = column_name(fit.formula.offset_arg)
        if name is None:
            raise ValueError("emmeans(): offset= must name a column; or put offset() in the formula")
        formula_offset = np.zeros(fit.frame.rows.height)
        for label in fit.formula.offsets:
            formula_offset += fit.frame.rows.select(variable_expr(label, "emmeans")[0]).to_series().to_numpy()
        # emmeans treats offset= as a static mean; formula offsets vary on the grid.
        static_offset = float(np.mean(fit.frame.offset - formula_offset))
        if name not in grid.columns:
            design_grid = grid.with_columns(pl.lit(static_offset).alias(name))
    x, offset, _, _ = fit._new_design(design_grid)
    if static_offset is not None:
        offset = offset + static_offset - design_grid[name].to_numpy()
    counts = _cell_counts(fit, grid, factors)
    keep = [*by, *[s for s in specs if s not in by]]
    other = [f for f in factors if f not in keep]
    if weights == "equal":
        w = np.ones(grid.height)
    elif weights == "cells":
        w = counts
    else:
        # emmeans' "proportional": the averaged-over factors' joint frequencies
        # over the whole data, the same for every cell of specs.
        if other:
            joint = fit.frame.rows.group_by(other).len()
            joint = joint.with_columns(pl.col(n).cast(grid.schema[n]) for n in other)
            w = grid.select(other).with_row_index("__i").join(
                joint, on=other, how="left", nulls_equal=True).sort("__i")["len"].fill_null(0).cast(pl.Float64).to_numpy()
        else:
            w = np.ones(grid.height)
    # Spec cells in emmeans' order: the first spec varies fastest, within by.
    cells = grid.select(keep).with_columns(__w=pl.Series(w), __i=pl.int_range(pl.len()))
    labels = grid.select(keep).unique(maintain_order=True)
    rank_exprs = []
    for name in [*by, *reversed([s for s in specs if s not in by])]:
        values = list(dict.fromkeys(grid[name].to_list()))
        rank_exprs.append(pl.col(name).replace_strict({v: i for i, v in enumerate(values)}, return_dtype=pl.Int64))
    labels = labels.sort(rank_exprs) if rank_exprs else labels
    lin = np.zeros((labels.height, x.shape[1]))
    off_cell = np.zeros(labels.height)
    for r, row in enumerate(labels.iter_rows(named=True)):
        mask = np.ones(grid.height, dtype=bool)
        for name, value in row.items():
            mask &= np.array([v == value for v in grid[name].to_list()])
        ww = cells.filter(pl.Series(mask))["__w"].to_numpy()
        if ww.sum() == 0:
            lin[r] = np.nan
            off_cell[r] = np.nan
            continue
        lin[r] = (ww[:, None] * x[mask]).sum(axis=0) / ww.sum()
        off_cell[r] = np.average(offset[mask], weights=ww)
    keys = _keys_frame(fit, labels.height)
    est = _Estimates(fit, keys, labels.select([*by, *[s for s in specs if s not in by]]), lin, by, vcov, type_, level)
    est.offset = off_cell
    est.estimate = est.estimate + off_cell
    return est


def _keys_frame(fit: Any, n: int) -> pl.DataFrame:
    if not fit.groups:
        return pl.DataFrame()
    keys = fit.frame.rows.select(fit.groups).head(1)
    return pl.concat([keys] * n) if n else keys.clear()


def emmeans(
    model: Any,
    specs: str | Sequence[str],
    *,
    by: str | Sequence[str] | None = None,
    at: Mapping[str, Any] | None = None,
    type: str = "link",  # noqa: A002 - emmeans' argument name
    weights: str = "equal",
    vcov: Any = None,
    level: float = 0.95,
) -> EMMeans:
    """Estimated marginal means, as emmeans' ``emmeans(fit, specs)``.

    ``specs`` names the factors (one or several) whose means you want;
    ``by`` splits them (contrasts are then made within each ``by`` group).
    The other factors are averaged over with ``weights``: ``"equal"`` (the
    default), ``"proportional"`` (their frequencies in the data) or
    ``"cells"`` (each cell's frequency). Numeric covariates sit at their
    means, except binary ones keep both values (R's ``cov.keep="2"``).
    ``at`` supplies values (``at={"flipper_length_mm": [190, 210]}``)
    or restricts factors to some levels. Formula offsets are evaluated on
    the grid; an ``offset=`` argument contributes its mean over fitted rows.
    Offsets use the same averaging weights as the design (unlike emmeans
    2.0.4, which averages offsets equally with ``weights="cells"``).
    ``type="response"`` back-transforms a glm's means (the delta-method
    standard error, back-transformed intervals). ``vcov`` takes a robust
    covariance as in ``tidy``. A grouped fit gives means for each group.

    The result prints as a table; ``tidy(emm)`` gives it as a frame, and
    ``emm.pairs()`` / ``emm.contrast(...)`` (or ``pairs(emm)``,
    ``contrast(emm, ...)``) compare the means.
    """
    from stats3.models import _fits

    if any(getattr(f, "kind", None) == "bayesian" for f in _fits(model)):
        raise TypeError("emmeans(): use bayes_emmeans() for posterior marginal means")

    specs_ = [specs] if isinstance(specs, str) else list(specs)
    by_ = [] if by is None else ([by] if isinstance(by, str) else list(by))
    if not specs_:
        raise ValueError("emmeans(): specs needs at least one factor")
    if type not in ("link", "response"):
        raise ValueError(f"emmeans(): type must be 'link' or 'response', got {type!r}")
    if weights not in WEIGHTS:
        raise ValueError(f"emmeans(): weights must be one of {list(WEIGHTS)}, got {weights!r}")
    _check_conf_level(level)
    fits = _fits(model)
    parts = [_one(fit, specs_, by_, at or {}, weights, vcov, type, level) for fit in fits]
    return EMMeans(parts, specs_, fits[0]._data)


def pairs(emm: EMMeans, *, adjust: str | None = None, reverse: bool = False, level: float | None = None) -> Contrasts:
    """All pairwise differences of estimated marginal means, as emmeans' ``pairs``."""
    from stats3.bayes_marginal import BayesianEstimates

    if isinstance(emm, BayesianEstimates):
        return emm.pairs(adjust=adjust, reverse=reverse, level=level)
    if not isinstance(emm, EMMeans):
        raise TypeError("pairs(): expected the result of emmeans()")
    return emm.pairs(adjust="tukey" if adjust is None else adjust, reverse=reverse, level=level)


def contrast(emm: EMMeans, method: Any = "pairwise", *, adjust: str | None = None, ref: Any = 1,
             level: float | None = None) -> Contrasts:
    """Contrasts of estimated marginal means, as emmeans' ``contrast``; see
    ``EMMeans.contrast``."""
    from stats3.bayes_marginal import BayesianEstimates

    if isinstance(emm, BayesianEstimates):
        return emm.contrast(method, adjust=adjust, ref=ref, level=level)
    if not isinstance(emm, EMMeans):
        raise TypeError("contrast(): expected the result of emmeans()")
    return emm.contrast(method, adjust=adjust, ref=ref, level=level)
