"""Marginal response means and contrasts evaluated draw by draw."""

from __future__ import annotations

from collections.abc import Mapping
import itertools

import numpy as np
import polars as pl

from stats3.bayes import _bayesian_fits, _draw_summary
from stats3.core import _check_conf_level


class BayesianEstimates:
    """Joint posterior draws of marginal means or their linear contrasts."""

    def __init__(self, parts, data, level=.95):
        self.parts, self._data, self.level = parts, data, level

    def tidy(self):
        return pl.concat([pl.concat([labels, _draw_summary(values, [""]*labels.height, self.level).drop("term")],
                                   how="horizontal") for labels, values, _ in self.parts], how="diagonal_relaxed")

    def __tidy__(self, **kwargs):
        if kwargs:
            raise TypeError("set the credible interval level when creating Bayesian marginal means/contrasts")
        return self._data.write(self.tidy())

    def __repr__(self):
        from stats3.models import _table_text

        return f"Bayesian marginal estimates ({self.level:.0%} equal-tail credible intervals)\n{_table_text(self.tidy())}"

    def pairs(self, *, reverse=False, level=None, adjust=None):
        return self.contrast("revpairwise" if reverse else "pairwise", level=level, adjust=adjust)

    def contrast(self, method="pairwise", *, level=None, adjust=None, ref=1):
        if adjust not in (None, "none"):
            raise ValueError("Bayesian contrasts summarize joint posterior draws; adjust must be 'none'")
        level = self.level if level is None else level
        _check_conf_level(level)
        parts = []
        for labels, values, keys in self.parts:
            n = labels.height
            if isinstance(method, Mapping):
                names = list(method)
                weights = np.asarray(list(method.values()), dtype=float)
                if weights.shape != (len(names), n) or not np.all(np.isfinite(weights)):
                    raise ValueError(f"each contrast needs {n} finite weights")
            elif method in ("pairwise", "revpairwise"):
                pairs = list(itertools.combinations(range(n), 2))
                if not pairs:
                    raise ValueError("pairwise contrasts need at least two marginal means")
                specs = [c for c in labels.columns if c not in keys]
                text = [" ".join(str(row[c]) for c in specs) for row in labels.iter_rows(named=True)]
                names = [f"{text[j]} - {text[i]}" if method == "revpairwise" else f"{text[i]} - {text[j]}" for i, j in pairs]
                weights = np.zeros((len(pairs), n))
                for r, (i, j) in enumerate(pairs):
                    weights[r, i], weights[r, j] = (1, -1) if method == "pairwise" else (-1, 1)
            else:
                raise ValueError("Bayesian contrasts support 'pairwise', 'revpairwise', or named weight vectors")
            rows = pl.DataFrame({**{k: [labels[k][0]]*len(names) for k in keys}, "contrast": names})
            parts.append((rows, values @ weights.T, keys))
        return BayesianEstimates(parts, self._data, level)


def bayes_emmeans(model, specs, *, by=None, at=None, weights="equal", type="response", level=.95):
    """Marginal means on each joint posterior draw, then credible summaries.

    Grids follow ``emmeans``. ``weights`` is equal, cells, or proportional.
    Response transformations happen *before* averaging. Random effects
    are set to zero (population coefficients); these are not integrals over
    the random-effect distribution. Offset arguments must name a column and
    contribute their fitted mean.

    Ordinal and multinomial fits give one marginal probability per
    reference-grid cell and response ``category``, averaged draw by draw;
    contrasts compare cells within each category. ``type="link"`` follows
    emmeans's ``mode="latent"``: the ordinal linear predictor minus the mean
    cutpoint, or the multinomial logits centred across categories.
    """
    from stats3._data import column_name
    from stats3.emm import _cell_counts, _reference_grid
    from stats3.formula import variable_expr

    _check_conf_level(level)
    specs = [specs] if isinstance(specs, str) else list(specs)
    by = [] if by is None else ([by] if isinstance(by, str) else list(by))
    if not specs or len(set([*by, *specs])) != len([*by, *specs]) or "category" in [*by, *specs]:
        raise ValueError("bayes_emmeans(): specs and by must be nonempty/distinct column names")
    if weights not in ("equal", "cells", "proportional") or type not in ("response", "link"):
        raise ValueError("use weights='equal'/'cells'/'proportional', type='response'/'link'")
    parts = []
    for fit in _bayesian_fits(model):
        grid, factors = _reference_grid(fit, at or {})
        if not set([*specs, *by]) <= set(grid.columns):
            raise ValueError("specs/by must name model variables")
        design_grid = grid
        static_offset = None
        if fit.formula.offset_arg is not None:
            name = column_name(fit.formula.offset_arg)
            if name is None:
                raise ValueError("put computed offsets in offset() in the formula")
            formula_offset = np.zeros(fit.n)
            for label in fit.formula.offsets:
                formula_offset += fit.frame.rows.select(variable_expr(label, "bayes_emmeans")[0]).to_series().to_numpy()
            static_offset = float(np.mean(fit.frame.offset-formula_offset))
            if name not in grid.columns:
                design_grid = grid.with_columns(pl.lit(static_offset).alias(name))
        x, off, rows, _ = fit._new_design(design_grid)
        if static_offset is not None:
            off = off + static_offset - design_grid[name].to_numpy()
        samples = fit._posterior_samples()
        eta = fit._eta(samples, x, off, rows, re_formula="population")
        predictions = eta if type == "link" else fit._mean(samples, eta)
        # Link-scale categorical means follow emmeans's mode="latent".
        if type == "link" and fit.family_name == "ordinal":
            predictions = eta - samples["cutpoints"].mean(axis=-1)[..., None]
        elif type == "link" and fit.family_name == "multinomial":
            logits = np.concatenate([np.zeros((*eta.shape[:-1], 1)), eta], axis=-1)
            predictions = logits - logits.mean(axis=-1, keepdims=True)
        counts = _cell_counts(fit, grid, factors)
        keep = [*by, *specs]
        other = [f for f in factors if f not in keep]
        if weights == "equal":
            w = np.ones(grid.height)
        elif weights == "cells":
            w = counts
        elif other:
            total = grid.select(other).with_columns(__n=pl.Series(counts)).group_by(other).agg(pl.col("__n").sum())
            w = grid.select(other).with_row_index("__i").join(total, on=other, how="left").sort("__i")["__n"].to_numpy()
        else:
            w = np.ones(grid.height)
        labels = grid.select(keep).unique(maintain_order=True)
        values = []
        for row in labels.iter_rows(named=True):
            mask = np.ones(grid.height, dtype=bool)
            for name, value in row.items():
                mask &= np.array([v == value for v in grid[name].to_list()])
            if w[mask].sum() == 0:
                raise ValueError("a requested marginal mean has no observed cells under these weights")
            values.append(np.average(predictions[:, :, mask], weights=w[mask], axis=2))
        values = np.stack(values, axis=2)
        split = list(by)
        if values.ndim == 4:
            # One block of cells per response category, category outermost.
            labels = pl.concat([labels.with_columns(pl.lit(level).alias("category")) for level in fit.response_levels])
            values = np.concatenate(np.moveaxis(values, -1, 0), axis=-1)
            split.append("category")
        for name, value in fit.key.items():
            labels = labels.with_columns(pl.lit(value).alias(name))
        keys = [*fit.groups, *split]
        if split:
            indices = labels.with_row_index("__i").partition_by(split, maintain_order=True)
            for subset in indices:
                selected = subset["__i"].to_numpy()
                parts.append((labels[selected], values[:, :, selected], keys))
        else:
            parts.append((labels, values, keys))
    return BayesianEstimates(parts, _bayesian_fits(model)[0]._data, level)
