"""Survival analysis, following R's survival package.

Formulas take ``Surv(time, status)`` or ``Surv(start, stop, status)`` on
the left. ``status`` is 0/1, logical, or 1/2 (2 = event), as in R.
``survfit`` gives Kaplan-Meier curves, ``survdiff`` the G-rho family of
log-rank tests, ``coxph`` Cox models (with ``strata()`` and ``cluster()``
terms, weights and offsets), ``cox_zph`` the proportional-hazards test,
``survfit(coxfit)`` predicted curves and ``survreg`` parametric
accelerated-failure-time models. Tables follow broom's ``tidy`` and
``glance``.

R's ``timefix`` (merging times equal within rounding error) is not done:
times are compared exactly.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Data, Step, step
from stats3.core import _check_conf_level, _special, _warn
from stats3.formula import Formula, Variable, _compile, _level_name, _python, build_design, parse_formula, variable_expr
from stats3.models import NA_ACTIONS, Models, _contrasts, _Fit, _Frame, _model_frames, _table_text

_HIDDEN = "__surv_"


# formulas and frames ----------------------------------------------------------------


def _call_args(text: str, name: str) -> list[str] | None:
    """The arguments of ``name(...)`` when ``text`` is exactly that call."""
    text = text.strip()
    if not (text.startswith(name + "(") and text.endswith(")")):
        return None
    inner, parts, depth, start = text[len(name) + 1:-1], [], 0, 0
    for i, c in enumerate(inner):
        depth += c == "("
        depth -= c == ")"
        if depth < 0:
            return None
        if c == "," and depth == 0:
            parts.append(inner[start:i].strip())
            start = i + 1
    parts.append(inner[start:].strip())
    return [p for p in parts if p]


def _readable(label: str, verb: str) -> pl.Expr | None:
    """None when the formula grammar reads ``label``; else an expression
    for comparisons such as ``status == 2``, which only Surv() takes."""
    try:
        variable_expr(label, verb)
        return None
    except (ValueError, SyntaxError):
        pass
    tree = ast.parse(_python(label), mode="eval").body
    if isinstance(tree, ast.Compare) and len(tree.ops) == 1:
        a, b = _compile(tree.left, label, verb), _compile(tree.comparators[0], label, verb)
        ops = {ast.Eq: a.__eq__, ast.NotEq: a.__ne__, ast.Gt: a.__gt__, ast.GtE: a.__ge__,
               ast.Lt: a.__lt__, ast.LtE: a.__le__}
        for kind, fn in ops.items():
            if isinstance(tree.ops[0], kind):
                return fn(b)
    raise ValueError(f"{verb}(): cannot read {label!r} in Surv()")


@dataclass
class _Spec:
    """A survival formula: Surv() arguments, specials, and the covariates."""

    formula: Formula  # covariate terms only
    surv: list[str]
    strata: list[str]
    cluster: str | None
    response: str


def _specification(text: str, verb: str, data: Data, contrasts: Any, offset: Any,
                   allow_cluster: bool = True) -> tuple[_Spec, Formula, Data]:
    parsed = parse_formula(text, verb, data.columns)
    surv = _call_args(parsed.response or "", "Surv")
    if surv is None or not 1 <= len(surv) <= 3:
        raise ValueError(f"{verb}(): the response must be Surv(time, status) or Surv(start, stop, status)")
    hidden = {}
    for i, label in enumerate(surv):
        expr = _readable(label, verb)
        if expr is not None:
            hidden[f"{_HIDDEN}{i}"] = expr
            surv[i] = f"{_HIDDEN}{i}"
    strata, cluster, terms = [], None, []
    for term in parsed.terms:
        inner = [_call_args(v, "strata") for v in term] + [_call_args(v, "cluster") for v in term]
        if not any(a is not None for a in inner):
            terms.append(term)
            continue
        if len(term) > 1:
            raise ValueError(f"{verb}(): strata() and cluster() cannot be in an interaction")
        if (args := _call_args(term[0], "strata")) is not None:
            strata.extend(args)
        else:
            if not allow_cluster:
                raise ValueError(f"{verb}(): cluster() is not supported here")
            if cluster is not None:
                raise ValueError(f"{verb}(): a formula cannot have multiple cluster terms")
            (cluster,) = _call_args(term[0], "cluster")
    covariates = replace(parsed, terms=terms, response=None)
    covariates.contrasts = _contrasts(contrasts, covariates, verb)
    covariates.offset_arg = offset
    extra = [*surv[1:], *strata, *([cluster] if cluster else [])]
    seen = set(covariates.variables())
    frame_terms = list(terms) + [(v,) for v in dict.fromkeys(extra) if v not in seen]
    framing = replace(parsed, response=surv[0], terms=frame_terms)
    if hidden:
        data = Data(data._lf.with_columns(**hidden), data._groups, data._levels, data.kind, data.source)
    return _Spec(covariates, surv, strata, cluster, parsed.response), framing, data


def _values(frame: _Frame, label: str, first: bool) -> np.ndarray:
    if first:
        return np.asarray(frame.y, dtype=float)
    v = frame.variables[label]
    if v.is_factor:
        if not all(isinstance(level, (bool, np.bool_)) for level in v.levels):
            raise ValueError(f"Surv(): {label!r} must be numeric or logical")
        return np.array([float(v.levels[c]) for c in v.codes])
    return v.values


@dataclass
class _Response:
    start: np.ndarray  # -inf for right-censored data
    stop: np.ndarray
    status: np.ndarray
    counting: bool


def _response(frame: _Frame, spec: _Spec, verb: str) -> _Response:
    cols = [_values(frame, label, i == 0) for i, label in enumerate(spec.surv)]
    if len(cols) == 1:
        cols.append(np.ones(len(cols[0])))
    status = cols[-1]
    if len(status) and np.max(status) == 2:
        status = status - 1
    if np.any((status != 0) & (status != 1)):
        raise ValueError(f"{verb}(): invalid status values; use 0/1, 1/2 or TRUE/FALSE")
    if len(cols) == 3:
        if np.any(cols[1] <= cols[0]):
            raise ValueError(f"{verb}(): stop time must be greater than start time")
        return _Response(cols[0], cols[1], status, True)
    return _Response(np.full(len(status), -np.inf), cols[0], status, False)


def _level_label(v: Variable, value: Any) -> str:
    return _level_name(v.levels[value] if v.is_factor else (int(value) if float(value).is_integer() else value))


def _combine(frame: _Frame, labels: list[str]) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Codes, R labels ("sex=1, ecog=0") and values for the combinations of
    ``labels`` present, ordered with the first variable slowest."""
    n = len(frame.weights)
    if not labels:
        return np.zeros(n, dtype=np.int64), [], [{}]
    keys = []
    for label in labels:
        v = frame.variables[label]
        if v.is_factor:
            keys.append(v.codes)
        else:
            keys.append(np.unique(v.values, return_inverse=True)[1])
    stacked = np.column_stack(keys)
    unique, codes = np.unique(stacked, axis=0, return_inverse=True)
    names, values = [], []
    for row in unique:
        parts, vals = [], {}
        for label, k in zip(labels, row):
            v = frame.variables[label]
            value = v.levels[k] if v.is_factor else np.unique(v.values)[k]
            vals[label] = value.item() if isinstance(value, np.generic) else value
            parts.append(f"{label}={_level_label(v, k if v.is_factor else value)}")
        names.append(", ".join(parts))
        values.append(vals)
    return codes.reshape(-1).astype(np.int64), names, values


def _frames(verb, formula, weights, offset, contrasts, na_action, data, allow_cluster=True):
    if na_action not in NA_ACTIONS:
        raise ValueError(f"{verb}(): na_action must be one of {list(NA_ACTIONS)}, got {na_action!r}")
    spec, framing, data2 = _specification(formula, verb, data, contrasts, offset, allow_cluster)
    frames = _model_frames(data2, framing, weights, offset, verb, na_action)
    for fr in frames:
        fr.rows = fr.rows.select(c for c in fr.rows.columns if not c.startswith(_HIDDEN))
        if fr.all_rows is not None:
            fr.all_rows = fr.all_rows.select(c for c in fr.all_rows.columns if not c.startswith(_HIDDEN))
    return spec, frames


class _SurvStep(Step):
    __slots__ = ()

    def __rrshift__(self, data: Any) -> Any:
        return self._fn(Data.read(data, self.by, self.name))


def _fitter(verb, build, formula, weights=None, offset=None, contrasts=None, na_action="omit", allow_cluster=True):
    def _apply(data: Data):
        spec, frames = _frames(verb, formula, weights, offset, contrasts, na_action, data, allow_cluster)
        fits = [build(spec, frame, data) for frame in frames]
        return Models(fits, data) if data._groups else fits[0]

    return _SurvStep(_apply, verb)


def _nulls(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col(c).fill_nan(None) for c, t in frame.schema.items() if t.is_float())


def _chisq_p(stat, df):
    return float(_special().chdtrc(df, stat)) if df > 0 and np.isfinite(stat) else None


class _Kept:
    def __init__(self, kept):
        self.kept = kept


# risk sets -----------------------------------------------------------------------------


def _at_risk_sums(y: _Response, times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Sum of ``values`` (rows) over subjects at risk at each time:
    start < t <= stop."""
    order = np.argsort(y.stop, kind="stable")
    stops = y.stop[order]
    suffix = np.concatenate([np.cumsum(values[order][::-1], axis=0)[::-1], np.zeros((1,) + values.shape[1:])])
    total = suffix[np.searchsorted(stops, times, "left")]
    if y.counting:
        order = np.argsort(y.start, kind="stable")
        starts = y.start[order]
        suffix = np.concatenate([np.cumsum(values[order][::-1], axis=0)[::-1], np.zeros((1,) + values.shape[1:])])
        total = total - suffix[np.searchsorted(starts, times, "left")]
    return total


def _subset(y: _Response, keep) -> _Response:
    return _Response(y.start[keep], y.stop[keep], y.status[keep], y.counting)


# Kaplan-Meier ---------------------------------------------------------------------------


def _confint(p, se, conf_type, level):
    """survival's survfit_confint, with se on the log scale."""
    z = float(_special().ndtri(1 - (1 - level) / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        if conf_type == "plain":
            width = se * p * z
            lower, upper = np.maximum(p - width, 0), np.minimum(p + width, 1)
        elif conf_type == "log":
            xx = np.where(p == 0, np.nan, p)
            lower = np.where(se == 0, p, np.exp(np.log(xx) - z * se))
            upper = np.minimum(np.where(se == 0, p, np.exp(np.log(xx) + z * se)), 1)
        elif conf_type == "log-log":
            xx = np.where((p == 0) | (p == 1), np.nan, p)
            w = z * se / np.log(xx)
            lower = np.where(se == 0, p, np.exp(-np.exp(np.log(-np.log(xx)) - w)))
            upper = np.where(se == 0, p, np.exp(-np.exp(np.log(-np.log(xx)) + w)))
        elif conf_type == "logit":
            xx = np.where(p == 0, np.nan, p)
            w = z * se * (1 + xx / (1 - xx))
            lower = np.where(se == 0, p, 1 - 1 / (1 + np.exp(np.log(p / (1 - p)) - w)))
            upper = np.where(se == 0, p, 1 - 1 / (1 + np.exp(np.log(p / (1 - p)) + w)))
        elif conf_type == "arcsin":
            xx = np.where(p == 0, np.nan, p)
            w = .5 * z * se * np.sqrt(xx / (1 - xx))
            lower = np.sin(np.maximum(0, np.arcsin(np.sqrt(xx)) - w)) ** 2
            upper = np.sin(np.minimum(np.pi / 2, np.arcsin(np.sqrt(xx)) + w)) ** 2
        else:
            raise ValueError(conf_type)
    return lower, upper


_CONF_TYPES = ("log", "log-log", "plain", "logit", "arcsin", "none")


def _km(y: _Response, w: np.ndarray) -> dict[str, np.ndarray]:
    times = np.unique(y.stop)
    pos = np.searchsorted(times, y.stop)
    n_event = np.bincount(pos, weights=w * y.status, minlength=len(times))
    n_censor = np.bincount(pos, weights=w * (1 - y.status), minlength=len(times))
    n_risk = _at_risk_sums(y, times, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        surv = np.cumprod(np.where(n_risk > 0, 1 - n_event / n_risk, 1.0))
        var = np.cumsum(np.where(n_event > 0, n_event / (n_risk * (n_risk - n_event)), 0.0))
    return {"time": times, "n_risk": n_risk, "n_event": n_event, "n_censor": n_censor,
            "surv": surv, "std_err": np.sqrt(var)}


def _minmin(y, x):
    tol = np.finfo(float).eps ** .5
    keep = ~np.isnan(y) & (y < .5 + tol)
    if not keep.any():
        return None
    x, y = x[keep], y[keep]
    if abs(y[0] - .5) < tol and np.any(y < y[0]):
        return (x[0] + x[np.argmax(y < y[0])]) / 2
    return float(x[0])


def _survmean(curve, end_time):
    """survival:::survmean on one curve, after survfit0's time-zero row."""
    t0 = min(0.0, float(curve["time"].min()))
    time, surv = curve["time"], curve["surv"]
    n_risk, n_event = curve["n_risk"], curve["n_event"]
    lower, upper = curve.get("lower"), curve.get("upper")
    if time[0] != t0:
        time, surv = np.r_[t0, time], np.r_[1.0, surv]
        n_risk, n_event = np.r_[n_risk[0], n_risk], np.r_[0.0, n_event]
        if lower is not None:
            lower, upper = np.r_[1.0, lower], np.r_[1.0, upper]
    with np.errstate(divide="ignore", invalid="ignore"):
        hh = np.where(n_risk - n_event == 0, 0, n_event / (n_risk * (n_risk - n_event)))
    keep = time <= end_time
    temptime, tempsurv, hh = np.r_[time[keep], end_time], np.r_[surv[keep], surv[keep][-1]], np.r_[hh[keep], 0]
    rectangles = np.diff(np.r_[t0, temptime]) * np.r_[1, tempsurv[:-1]]
    varmean = float(np.sum(np.cumsum(rectangles[1:][::-1]) ** 2 * hh[::-1][1:]))
    out = {"n_max": float(np.max(n_risk)), "n_start": float(n_risk[0]), "events": float(np.sum(n_event)),
           "rmean": float(np.sum(rectangles)), "rmean_std_error": math.sqrt(varmean),
           "median": _minmin(surv, time)}
    if lower is not None:
        out["conf_low"], out["conf_high"] = _minmin(lower, time), _minmin(upper, time)
    return out


class _Curves(_Fit):
    """Survival curves, one per stratum (and per new row for Cox curves)."""

    kind = "survfit"

    def tidy(self, conf_int=False, conf_level=.95, exponentiate=False, vcov=None):
        frames = []
        for name, curve in self.curves:
            cols = {"time": curve["time"], "n_risk": curve["n_risk"], "n_event": curve["n_event"],
                    "n_censor": curve["n_censor"]}
            cols.update(self._estimates(curve))
            frame = pl.DataFrame(cols)
            if name is not None:
                frame = frame.with_columns(strata=pl.lit(name))
            frames.append(frame)
        return self._with_key(_nulls(pl.concat(frames)))

    def _estimates(self, curve):
        cols = {"estimate": curve["surv"], "std_error": curve["std_err"]}
        if "lower" in curve:
            cols.update(conf_high=curve["upper"], conf_low=curve["lower"])
        return cols

    def glance(self):
        rows = []
        ends = max(float(c["time"][-1]) for _, c in self.curves)
        for name, curve in self.curves:
            row = {} if name is None else {"strata": name}
            row["records"] = float(curve["records"])
            row.update(_survmean(curve, ends))
            row["nobs"] = float(curve["records"])
            rows.append(row)
        schema = {k: (pl.String if k == "strata" else pl.Float64) for k in rows[0]}
        return self._with_key(pl.DataFrame(rows, schema=schema))

    def __repr__(self):
        return f"survfit({self.text})\n{_table_text(self.glance())}"


class SurvivalCurves(_Curves):
    """Kaplan-Meier curves from ``survfit(formula)``."""

    def __init__(self, spec, frame, data, conf_type, conf_level):
        super().__init__(spec.formula, None, frame, data, "survfit")
        self.text = f"{spec.response} ~ {' + '.join(spec.formula.term_labels) or '1'}"
        if spec.strata:
            raise ValueError("survfit(): use the grouping variables directly, not strata()")
        w = frame.weights
        if np.any(w != np.floor(w)):
            raise ValueError("survfit(): non-integer weights need R's robust variance, which is not available")
        y = _response(frame, spec, "survfit")
        labels = [t[0] for t in spec.formula.terms]
        if any(len(t) > 1 for t in spec.formula.terms):
            raise ValueError("survfit(): use + between grouping variables")
        codes, names, _ = _combine(frame, labels)
        self.curves = []
        for k in range(max(len(names), 1)):
            keep = codes == k
            curve = _km(_subset(y, keep), w[keep])
            curve["records"] = int(keep.sum())
            if conf_type != "none":
                curve["lower"], curve["upper"] = _confint(curve["surv"], curve["std_err"], conf_type, conf_level)
            self.curves.append((names[k] if names else None, curve))


def survfit(x: Any = None, *args: Any, newdata: Any = None, conf_int: float = .95, conf_type: str = "log",
            weights: Any = None, na_action: str = "omit", by: Any = None, as_: Any = None) -> Any:
    """Survival curves, as R's ``survfit``.

    ``survfit(data, "Surv(time, status) ~ sex")`` (or ``data >> survfit(...)``)
    gives Kaplan-Meier curves, one per combination of the right-side
    variables. ``survfit(coxfit)`` gives the Cox model's curve at the mean
    covariates (per stratum); ``survfit(coxfit, newdata=rows)`` one curve
    per new row. ``tidy`` has one row per time, as broom's
    ``tidy.survfit``; ``glance`` one row per curve with the median and its
    interval, as ``print.survfit`` (and the restricted mean, as
    ``summary.survfit``). Intervals are R's: ``conf_type`` ``"log"``
    (default), ``"log-log"``, ``"plain"``, ``"logit"``, ``"arcsin"`` or
    ``"none"``.
    """
    if conf_type not in _CONF_TYPES:
        raise ValueError(f"survfit(): conf_type must be one of {list(_CONF_TYPES)}")
    if not 0 < conf_int < 1:
        raise ValueError("survfit(): conf_int must be between 0 and 1")
    if isinstance(x, CoxModel):
        if args:
            newdata = args[0]
        return CoxCurves(x, newdata, conf_type, conf_int)
    if isinstance(x, Models):
        raise ValueError("survfit(): use one group's Cox model, e.g. fits[0]")
    return _survfit_formula(x, *args, conf_int=conf_int, conf_type=conf_type, weights=weights,
                            na_action=na_action, by=by, as_=as_)


@step
def _survfit_formula(formula, *, conf_int, conf_type, weights, na_action):
    return _fitter("survfit", lambda s, f, d: SurvivalCurves(s, f, d, conf_type, conf_int),
                   formula, weights, None, None, na_action, allow_cluster=False)


# log-rank tests --------------------------------------------------------------------------


class SurvivalDifference(_Fit):
    """A survdiff test: observed and expected events by group."""

    kind = "survdiff"

    def __init__(self, spec, frame, data, rho):
        super().__init__(spec.formula, None, frame, data, "survdiff")
        y = _response(frame, spec, "survdiff")
        if y.counting:
            raise ValueError("survdiff(): not defined for counting process data")
        if np.any(frame.weights != 1):
            raise ValueError("survdiff(): weights are not supported")
        labels = [t[0] for t in spec.formula.terms]
        if not labels:
            raise ValueError("survdiff(): no groups to test")
        groups, self.names, self.values = _combine(frame, labels)
        self.labels = labels
        strata, _, _ = _combine(frame, spec.strata)
        g = len(self.names)
        if g < 2:
            raise ValueError("survdiff(): there is only 1 group")
        self.n = np.bincount(groups, minlength=g).astype(float)
        obs, exp, var = np.zeros(g), np.zeros(g), np.zeros((g, g))
        for s in np.unique(strata):
            keep = strata == s
            t, d, grp = y.stop[keep], y.status[keep], groups[keep]
            times = np.unique(t[d == 1])
            km = 1.0
            for time in times:
                risk = t >= time
                nrisk = risk.sum()
                dead = (t == time) & (d == 1)
                deaths = dead.sum()
                wt = km ** rho
                n_j = np.bincount(grp[risk], minlength=g)
                d_j = np.bincount(grp[dead], minlength=g)
                obs += wt * d_j
                exp += wt * deaths * n_j / nrisk
                if nrisk > 1:
                    kt = wt * wt * deaths * (nrisk - deaths) / (nrisk * (nrisk - 1))
                    var += kt * (np.diag(n_j) - np.outer(n_j, n_j) / nrisk)
                km *= (nrisk - deaths) / nrisk
        self.obs, self.exp, self.var = obs, exp, var
        df = exp > 0
        if df.sum() < 2:
            self.chisq = 0.0
        else:
            diff = (obs - exp)[df][1:]
            vv = var[np.ix_(df, df)][1:, 1:]
            self.chisq = float(np.linalg.solve(vv, diff) @ diff)
        self.df = int(df.sum()) - 1
        self.rho = rho

    def tidy(self, conf_int=False, conf_level=.95, exponentiate=False, vcov=None):
        cols = {label: [v[label] for v in self.values] for label in self.labels}
        cols.update(N=self.n, obs=self.obs, exp=self.exp)
        return self._with_key(pl.DataFrame(cols))

    def glance(self):
        return self._with_key(pl.DataFrame({"statistic": [self.chisq], "df": [float(self.df)],
                                            "p_value": [_chisq_p(self.chisq, self.df)]},
                                           schema_overrides={"p_value": pl.Float64}))

    def __repr__(self):
        return f"survdiff(rho={self.rho})\n{_table_text(self.tidy())}\n{_table_text(self.glance())}"


@step
def survdiff(formula: str, *, rho: float = 0, na_action: str = "omit"):
    """The G-rho family of tests for survival differences, as R's ``survdiff``.

    ``survdiff(data, "Surv(time, status) ~ sex")``: ``rho=0`` (default) is
    the log-rank test, ``rho=1`` the Peto-Peto Wilcoxon. ``strata()``
    terms stratify the test. ``tidy`` gives N, observed and expected events
    per group (broom's ``tidy.survdiff``); ``glance`` the chi-squared test.
    """
    if not np.isfinite(rho):
        raise ValueError("survdiff(): rho must be finite")
    return _fitter("survdiff", lambda s, f, d: SurvivalDifference(s, f, d, rho), formula,
                   na_action=na_action, allow_cluster=False)


# Cox models ------------------------------------------------------------------------------


@dataclass
class _CoxData:
    y: _Response
    x: np.ndarray  # centred covariates
    w: np.ndarray
    offset: np.ndarray
    strata: np.ndarray
    method: str


def _events(cd: _CoxData, beta: np.ndarray, efron: bool):
    """Per stratum and event time, the risk-set sums R's coxfit6/agfit4 use,
    expanded to one row per Efron step (one per time for Breslow)."""
    eta = cd.x @ beta + cd.offset
    out = []
    for s in np.unique(cd.strata):
        idx = np.flatnonzero(cd.strata == s)
        y = _subset(cd.y, idx)
        dead = y.status == 1
        if not dead.any():
            continue
        x, w, e = cd.x[idx], cd.w[idx], eta[idx]
        shift = float(e.max())
        risk = np.exp(e - shift)
        r = w * risk
        times = np.unique(y.stop[dead])
        s0 = _at_risk_sums(y, times, r)
        s1 = _at_risk_sums(y, times, r[:, None] * x)
        s2 = _at_risk_sums(y, times, r[:, None, None] * x[:, :, None] * x[:, None, :])
        pos = np.searchsorted(times, y.stop)
        pos_d = pos[dead]
        m, p = len(times), x.shape[1]
        d0 = np.bincount(pos_d, r[dead], m)
        d1, d2 = np.zeros((m, p)), np.zeros((m, p, p))
        np.add.at(d1, pos_d, r[dead, None] * x[dead])
        np.add.at(d2, pos_d, r[dead, None, None] * x[dead, :, None] * x[dead, None, :])
        deadwt = np.bincount(pos_d, w[dead], m)
        ndead = np.bincount(pos_d, minlength=m)
        if efron:
            row = np.repeat(np.arange(m), ndead)
            k = np.arange(len(row)) - np.repeat(np.cumsum(ndead) - ndead, ndead)
            frac = k / ndead[row]
            wtave = (deadwt / ndead)[row]
        else:
            row, frac, wtave = np.arange(m), np.zeros(m), deadwt
        den = s0[row] - frac * d0[row]
        xb = (s1[row] - frac[:, None] * d1[row]) / den[:, None]
        out.append(dict(idx=idx, y=y, x=x, w=w, risk=risk, shift=shift, eta=e, dead=dead, times=times,
                        pos=pos, row=row, frac=frac, wtave=wtave, den=den, xb=xb, s2=s2, d2=d2, m=m,
                        deadwt=deadwt, ndead=ndead))
    return out


def _cox_likelihood(cd: _CoxData, beta: np.ndarray):
    if cd.method == "exact":
        return _exact_likelihood(cd, beta)
    p = cd.x.shape[1]
    loglik, u, imat = 0.0, np.zeros(p), np.zeros((p, p))
    for ev in _events(cd, beta, cd.method == "efron"):
        dead, w = ev["dead"], ev["w"]
        loglik += float(np.sum(w[dead] * ev["eta"][dead]) - np.sum(ev["wtave"] * (np.log(ev["den"]) + ev["shift"])))
        u += (w[dead, None] * ev["x"][dead]).sum(0) - (ev["wtave"][:, None] * ev["xb"]).sum(0)
        row, frac = ev["row"], ev["frac"]
        second = (ev["s2"][row] - frac[:, None, None] * ev["d2"][row]) / ev["den"][:, None, None]
        imat += np.einsum("k,kij->ij", ev["wtave"], second - ev["xb"][:, :, None] * ev["xb"][:, None, :])
    return loglik, u, imat


def _exact_likelihood(cd: _CoxData, beta: np.ndarray):
    """The exact discrete partial likelihood, by Gail's recursion over the
    elementary symmetric functions of the risk scores."""
    p = cd.x.shape[1]
    eta = cd.x @ beta + cd.offset
    loglik, u, imat = 0.0, np.zeros(p), np.zeros((p, p))
    for s in np.unique(cd.strata):
        idx = np.flatnonzero(cd.strata == s)
        y = _subset(cd.y, idx)
        x, e = cd.x[idx], eta[idx]
        for t in np.unique(y.stop[y.status == 1]):
            at = (y.start < t) & (y.stop >= t)
            dead = at & (y.stop == t) & (y.status == 1)
            d = int(dead.sum())
            xs, es = x[at], e[at]
            shift = float(es.max())
            r = np.exp(es - shift)
            f = np.zeros(d + 1); f[0] = 1
            g = np.zeros((d + 1, p)); h = np.zeros((d + 1, p, p))
            for j in range(len(r)):
                xj = xs[j]
                f_new, g_new, h_new = f.copy(), g.copy(), h.copy()
                f_new[1:] += r[j] * f[:-1]
                g_new[1:] += r[j] * (xj * f[:-1, None] + g[:-1])
                h_new[1:] += r[j] * (np.outer(xj, xj) * f[:-1, None, None] + xj[None, :, None] * g[:-1, None, :]
                                     + g[:-1, :, None] * xj[None, None, :] + h[:-1])
                f, g, h = f_new, g_new, h_new
            loglik += float(e[dead].sum() - math.log(f[d]) - d * shift)
            mean = g[d] / f[d]
            u += x[dead].sum(0) - mean
            imat += h[d] / f[d] - np.outer(mean, mean)
    return loglik, u, imat


def _solve(imat, u):
    from scipy.linalg import cho_factor, cho_solve

    return cho_solve(cho_factor(imat), u)


def _cox_newton(cd: _CoxData, init, maxit, eps):
    """R's coxfit6 Newton-Raphson: same start, steps, halving and stopping."""
    p = cd.x.shape[1]
    beta = np.zeros(p) if init is None else np.asarray(init, dtype=float)
    loglik0, u, imat = _cox_likelihood(cd, beta)
    if p == 0:
        return dict(beta=beta, loglik=[loglik0, loglik0], score=None, imat=imat, iter=0, converged=True)
    step_ = _solve(imat, u)
    score = float(u @ step_)
    current, newbeta, halving, iteration, converged = loglik0, beta + step_, 0, 0, False
    newlk = loglik0
    for iteration in range(1, maxit + 1):
        newlk, u, imat = _cox_likelihood(cd, newbeta)
        if abs(1 - current / newlk) <= eps and halving == 0:
            converged = True
            break
        if iteration == maxit:
            break
        if newlk < current:
            halving += 1
            newbeta = (newbeta + beta) / 2
        else:
            halving = 0
            current = newlk
            beta = newbeta
            newbeta = newbeta + _solve(imat, u)
    if not converged and maxit > 0:
        _warn("coxph(): ran out of iterations and did not converge")
    if maxit == 0:
        newbeta, newlk = beta, loglik0
    return dict(beta=newbeta, loglik=[loglik0, newlk], score=score, imat=imat, iter=iteration,
                converged=converged)


def _cox_residuals(cd: _CoxData, beta: np.ndarray, efron: bool):
    """Martingale and score residuals (R's coxmart and coxscore2)."""
    n, p = cd.x.shape
    expected, score = np.zeros(n), np.zeros((n, p))
    for ev in _events(cd, beta, efron):
        m, row, frac = ev["m"], ev["row"], ev["frac"]
        haz = ev["wtave"] / ev["den"]
        h = np.bincount(row, haz, m)
        hx = np.zeros((m, p)); np.add.at(hx, row, haz[:, None] * ev["xb"])
        hd = np.bincount(row, frac * haz, m)
        hdx = np.zeros((m, p)); np.add.at(hdx, row, (frac * haz)[:, None] * ev["xb"])
        xbar = np.zeros((m, p)); np.add.at(xbar, row, ev["xb"])
        xbar /= np.bincount(row, minlength=m)[:, None]
        cum_h, cum_hx = np.r_[0, np.cumsum(h)], np.vstack([np.zeros(p), np.cumsum(hx, 0)])
        y, times = ev["y"], ev["times"]
        a = np.searchsorted(times, y.start, "right")
        b = np.searchsorted(times, y.stop, "right")
        hi, hxi = cum_h[b] - cum_h[a], cum_hx[b] - cum_hx[a]
        dead = ev["dead"]
        j = ev["pos"][dead]
        hi[dead] -= hd[j]
        hxi[dead] -= hdx[j]
        x, risk = ev["x"], ev["risk"]
        res = -risk[:, None] * (x * hi[:, None] - hxi)
        res[dead] += x[dead] - xbar[j]
        expected[ev["idx"]] = risk * hi
        score[ev["idx"]] = res
    return expected, score


def _concordance(y: _Response, lp, strata, w, cluster):
    """survival's concordancefit(reverse=TRUE): counts and the infinitesimal
    jackknife standard error."""
    n = len(lp)
    counts = np.zeros(5)
    influence = np.zeros((n, 3))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        st, sp, ev, x, ww = y.start[idx], y.stop[idx], y.status[idx], lp[idx], w[idx]
        for i in np.flatnonzero(ev == 1):
            t = sp[i]
            comp = (st < t) & ((sp > t) | ((sp == t) & (ev == 0)))
            tied_y = (sp == t) & (ev == 1) & (np.arange(len(idx)) > i)
            weight = ww[i] * ww
            conc = comp & (x[i] > x)
            disc = comp & (x[i] < x)
            tie = comp & (x[i] == x)
            counts[0] += weight[conc].sum()
            counts[1] += weight[disc].sum()
            counts[2] += weight[tie].sum()
            counts[3] += weight[tied_y & (x != x[i])].sum()
            counts[4] += weight[tied_y & (x == x[i])].sum()
            for k, mask in enumerate((conc, disc, tie)):
                influence[idx[i], k] += ww[mask].sum()
                influence[idx[mask], k] += ww[i]
    npair = counts[:3].sum()
    somer = (counts[0] - counts[1]) / npair if npair > 0 else np.nan
    dfbeta = ((influence[:, 0] - influence[:, 1]) - influence.sum(1) * somer) * w / (2 * npair)
    if cluster is not None:
        dfbeta = np.bincount(cluster, dfbeta)
    return counts, (somer + 1) / 2, math.sqrt(float(np.sum(dfbeta ** 2)))


def _wtest(var, b):
    if len(b) == 0:
        return np.nan
    return float(b @ np.linalg.solve(var, b))


class CoxModel(_Fit):
    """A Cox proportional hazards fit, as R's ``coxph``."""

    kind = "coxph"

    def __init__(self, spec, frame, data, ties, robust, init, maxit, eps, nocenter=(-1, 0, 1)):
        n = len(frame.weights)
        design = build_design(replace(spec.formula, intercept=True), frame.variables, n)
        super().__init__(spec.formula, design, frame, data, "coxph")
        self.spec, self.method = spec, ties
        self.y = _response(frame, spec, "coxph")
        w = frame.weights
        if np.any(w <= 0):
            raise ValueError("coxph(): invalid weights, must be > 0")
        if ties == "exact" and np.any(w != 1):
            raise ValueError("coxph(): case weights are not supported with ties='exact'")
        self.prior_weights, self.n, self.nevent = w, n, float(np.sum(self.y.status))
        self.strata, self.strata_names, self.strata_values = _combine(frame, spec.strata)
        self.cluster = None
        if spec.cluster is not None:
            v = frame.variables[spec.cluster]
            self.cluster = v.codes if v.is_factor else np.unique(v.values, return_inverse=True)[1]
        has_rwt = bool(np.any(w != np.floor(w)))
        self.robust = (self.cluster is not None or has_rwt) if robust is None else bool(robust)
        if self.robust and self.cluster is None:
            self.cluster = np.arange(n)
        if self.robust and ties == "exact":
            raise ValueError("coxph(): robust variances are not available with ties='exact'")
        full = design.matrix[:, 1:]
        self.names = design.names[1:]
        self.assign = design.assign[1:]
        self.means_all = (w @ full) / w.sum() if full.shape[1] else np.zeros(0)
        # R's nocenter=c(-1, 0, 1): indicator-like columns are not centred.
        self.means_all[np.all(np.isin(full, nocenter), axis=0)] = 0
        centred = full - self.means_all
        from stats3.models import _qr

        kept = _qr(centred * np.sqrt(w)[:, None], 1e-9, np.zeros(n)).kept if full.shape[1] else []
        if len(kept) < full.shape[1]:
            _warn("coxph(): X matrix deemed to be singular; coefficients set to NA")
        self.kept = kept
        self.qr = _Kept(list(range(len(design.names))))
        self.cd = _CoxData(self.y, centred[:, kept], w, frame.offset, self.strata, ties)
        if init is not None:
            init = np.asarray(init, dtype=float)
            if init.shape != (full.shape[1],):
                raise ValueError("coxph(): wrong length for init")
            init = init[kept]
        fit = _cox_newton(self.cd, init, maxit, eps)
        self.iterations, self.converged = fit["iter"], fit["converged"]
        self.coef = np.full(full.shape[1], np.nan)
        self.coef[kept] = fit["beta"]
        self.beta = fit["beta"]
        self.loglik = fit["loglik"]
        self.score_test = fit["score"]
        p = len(kept)
        self.naive_var = np.linalg.inv(fit["imat"]) if p else np.zeros((0, 0))
        self.var = self.naive_var
        self.linear_predictors = self.cd.x @ self.beta + frame.offset
        resid_efron = ties == "efron"
        expected, score = _cox_residuals(self.cd, self.beta, resid_efron)
        self.residuals = self.y.status - expected
        self.robust_score = None
        if self.robust and p:
            dfbeta = np.zeros((self.cluster.max() + 1, p))
            np.add.at(dfbeta, self.cluster, (score @ self.naive_var) * w[:, None])
            self.var = dfbeta.T @ dfbeta
            beta0 = np.zeros(p) if init is None else init
            zero = replace(self.cd, offset=-(self.cd.x @ beta0))
            _, score0 = _cox_residuals(zero, beta0, resid_efron)
            collapsed = np.zeros((self.cluster.max() + 1, p))
            np.add.at(collapsed, self.cluster, score0 * w[:, None])
            u = collapsed.sum(0)
            self.robust_score = _wtest(collapsed.T @ collapsed, u)
        self.wald = _wtest(self.var, self.beta - (0 if init is None else init)) if p else None
        self.concordance = _concordance(self.y, self.linear_predictors, self.strata, w,
                                        self.cluster if spec.cluster is not None else None)

    # -- tables --------------------------------------------------------------------------

    @property
    def coefficients(self):
        return dict(zip(self.names, self.coef.tolist()))

    def _full(self, matrix):
        k = len(self.names)
        out = np.full((k, k), np.nan)
        out[np.ix_(self.kept, self.kept)] = matrix
        return out

    def tidy(self, conf_int=False, conf_level=.95, exponentiate=False, vcov=None):
        if vcov is not None:
            raise ValueError("coxph(): vcov= is for lm/glm; use cluster() or robust=True")
        s = _special()
        se = np.sqrt(np.diag(self._full(self.naive_var)))
        rse = np.sqrt(np.diag(self._full(self.var)))
        stat = self.coef / rse
        cols = {"term": self.names, "estimate": self.coef, "std_error": se}
        if self.robust:
            cols["robust_se"] = rse
        cols.update(statistic=stat, p_value=2 * s.ndtr(-np.abs(stat)))
        if conf_int:
            _check_conf_level(conf_level)
            z = float(s.ndtri((1 + conf_level) / 2))
            cols.update(conf_low=self.coef - z * rse, conf_high=self.coef + z * rse)
        if exponentiate:
            for name in ("estimate", "conf_low", "conf_high"):
                if name in cols:
                    cols[name] = np.exp(cols[name])
        return self._with_key(_nulls(pl.DataFrame(cols)))

    def vcov_table(self, vcov=None):
        if vcov is not None:
            raise ValueError("coxph(): vcov= is for lm/glm; use cluster() or robust=True")
        full = self._full(self.var)
        return self._with_key(_nulls(pl.DataFrame({"term": self.names, **{n: full[:, i] for i, n in enumerate(self.names)}})))

    @property
    def df(self):
        return len(self.kept)

    def log_lik(self):
        return self.loglik[1]

    def glance(self):
        df = self.df
        logtest = -2 * (self.loglik[0] - self.loglik[1])
        counts, conc, conc_se = self.concordance
        ll = self.log_lik()
        row = {"n": self.n, "nevent": self.nevent,
               "statistic_log": logtest, "p_value_log": _chisq_p(logtest, df),
               "statistic_sc": self.score_test, "p_value_sc": _chisq_p(self.score_test, df) if df else None,
               "statistic_wald": self.wald, "p_value_wald": _chisq_p(self.wald, df) if df else None,
               "statistic_robust": self.robust_score,
               "p_value_robust": None if self.robust_score is None else _chisq_p(self.robust_score, df),
               "r_squared": 1 - math.exp(-logtest / self.n), "r_squared_max": 1 - math.exp(2 * self.loglik[0] / self.n),
               "concordance": conc, "std_error_concordance": conc_se,
               "logLik": ll, "AIC": -2 * ll + 2 * df, "BIC": -2 * ll + math.log(self.nevent) * df,
               "nobs": self.nevent}
        schema = {k: (pl.Int64 if k == "n" else pl.Float64) for k in row}
        return self._with_key(pl.DataFrame([row], schema=schema))

    # -- prediction ---------------------------------------------------------------------

    def _new(self, newdata):
        x, offset, raw, data = self._new_design(newdata)
        return x[:, 1:] if x.shape[1] else x, offset, raw, data

    def _new_strata(self, raw_data):
        if not self.spec.strata:
            return np.zeros(raw_data.height, dtype=np.int64)
        out = []
        for row in raw_data.select(_strata_exprs(self.spec)).iter_rows(named=True):
            key = {k: row[k] for k in self.spec.strata}
            matches = [i for i, v in enumerate(self.strata_values) if all(_same(v[k], key[k]) for k in key)]
            if not matches:
                raise ValueError("predict(): new data has a stratum not found in the original model")
            out.append(matches[0])
        return np.array(out, dtype=np.int64)

    def _prediction(self, newdata, type, se_fit, reference):
        if type not in ("lp", "risk", "expected"):
            raise ValueError("predict(): type must be 'lp', 'risk' or 'expected'")
        if reference not in ("strata", "sample", "zero"):
            raise ValueError("predict(): reference must be 'strata', 'sample' or 'zero'")
        beta = np.nan_to_num(self.coef)
        w = self.prior_weights
        full = self.design.matrix[:, 1:]
        has_strata = bool(self.spec.strata)
        se = None
        if type == "expected":
            if se_fit:
                raise ValueError("predict(): se_fit is not available for type='expected'")
            if newdata is None:
                pred = self.y.status - self.residuals
            else:
                pred = self._expected_new(newdata)
            out = pl.DataFrame({"fit": pred})
        else:
            if newdata is None:
                x, offset, strata = full, self.frame.offset - self.frame.offset.mean(), self.strata
                use_x = se_fit or (has_strata and reference == "strata") or (reference == "zero" and np.any(self.means_all != 0))
            else:
                x, offset, raw, _ = self._new(newdata)
                offset = offset - self.frame.offset.mean()
                strata = self._new_strata(raw) if has_strata and reference == "strata" else None
                use_x = True
            if not use_x:
                pred = self.linear_predictors.copy()
            else:
                if has_strata and reference == "strata":
                    sums = np.zeros((len(self.strata_names), full.shape[1]))
                    np.add.at(sums, self.strata, full * w[:, None])
                    means = sums / np.bincount(self.strata, w)[:, None]
                    centred = x - means[strata]
                elif reference == "zero":
                    centred = x
                else:
                    centred = x - self.means_all
                pred = centred @ beta + offset
                if se_fit:
                    c = centred[:, self.kept]
                    se = np.sqrt(np.einsum("ij,jk,ik->i", c, self.var, c))
            if type == "risk":
                pred = np.exp(pred)
                if se is not None:
                    se = se * np.sqrt(pred)
            out = pl.DataFrame({"fit": pred} | ({} if se is None else {"se_fit": se}))
        out = _nulls(out)
        if newdata is None:
            return self._padded(out), self._data
        return out, Data.read(newdata, None, "predict")

    def _baseline(self, efron):
        """agsurv's hazard pieces per stratum: time, hazard, varhaz, xbar,
        with risk exp(lp) and raw covariates."""
        full = self.design.matrix[:, 1:][:, self.kept]
        risk = np.exp(self.linear_predictors)
        w = self.prior_weights
        out = []
        for s in range(max(len(self.strata_names), 1)):
            keep = self.strata == s
            y = _subset(self.y, keep)
            x, wr, ww = full[keep], (w * risk)[keep], w[keep]
            times = np.unique(y.stop)
            pos = np.searchsorted(times, y.stop)
            dead = y.status == 1
            nevent = np.bincount(pos, ww * dead, len(times))
            ncens = np.bincount(pos, ww * (~dead), len(times))
            ndeath = np.bincount(pos, dead.astype(float), len(times))
            nrisk = _at_risk_sums(y, times, wr)
            irisk = _at_risk_sums(y, times, ww)
            xsum = _at_risk_sums(y, times, wr[:, None] * x)
            p = x.shape[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                if efron:
                    erisk = np.bincount(pos, wr * dead, len(times))
                    xsum2 = np.zeros((len(times), p)); np.add.at(xsum2, pos, (wr * dead)[:, None] * x)
                    sum1, sum2, xb = np.zeros(len(times)), np.zeros(len(times)), np.zeros((len(times), p))
                    for i in range(len(times)):
                        d = int(ndeath[i])
                        if d <= 1:
                            sum1[i], sum2[i] = 1 / nrisk[i], 1 / nrisk[i] ** 2
                            xb[i] = xsum[i] / nrisk[i] ** 2
                        else:
                            for j in range(d):
                                den = nrisk[i] - erisk[i] * j / d
                                sum1[i] += 1 / den / d
                                sum2[i] += 1 / den ** 2 / d
                                xb[i] += (xsum[i] - xsum2[i] * j / d) / den ** 2 / d
                    haz, varhaz, xbar = nevent * sum1, nevent * sum2, nevent[:, None] * xb
                else:
                    haz = nevent / nrisk
                    varhaz = nevent / nrisk ** 2
                    xbar = xsum / nrisk[:, None] * haz[:, None]
            out.append(dict(time=times, n_risk=irisk, n_event=nevent, n_censor=ncens, hazard=haz,
                            varhaz=varhaz, xbar=xbar, records=int(keep.sum())))
        return out

    def _expected_new(self, newdata):
        data = Data.read(newdata, None, "predict")
        raw = data._lf.collect()
        x, offset, _, _ = self._new(newdata)
        cols = raw.select([variable_expr(lbl, "predict")[0].cast(pl.Float64).alias(f"__s{i}")
                           for i, lbl in enumerate(self.spec.surv[:-1])])
        stop = cols[f"__s{len(self.spec.surv) - 2}"].to_numpy()
        start = cols["__s0"].to_numpy() if self.y.counting else None
        strata = self._new_strata(raw)
        risk = np.exp((x - self.means_all)[:, self.kept] @ self.beta + offset)
        base = self._baseline(self.method == "efron")
        pred = np.zeros(len(stop))
        for i, s in enumerate(strata):
            b = base[s]
            cum = np.r_[0, np.cumsum(b["hazard"])]
            chaz = cum[np.searchsorted(b["time"], stop[i], "right")]
            if start is not None:
                chaz -= cum[np.searchsorted(b["time"], start[i], "right")]
            pred[i] = chaz * risk[i]
        return pred

    def predict(self, newdata=None, *, type="lp", se_fit=False, reference="strata"):
        out, data = self._prediction(newdata, type, se_fit, reference)
        return data.write(out)

    def augment(self, newdata=None):
        if newdata is not None:
            out, data = self._prediction(newdata, "lp", True, "strata")
            raw = data._lf.collect()
            return data.write(pl.concat([raw, out.rename({"fit": ".fitted", "se_fit": ".se_fit"})], how="horizontal"))
        out, _ = self._prediction(None, "lp", True, "strata")
        resid = self._padded(pl.DataFrame({".resid": self.residuals}))
        return pl.concat([self._rows(), out.rename({"fit": ".fitted", "se_fit": ".se_fit"}), resid], how="horizontal")

    # -- comparisons --------------------------------------------------------------------

    def _refit_loglik(self, columns):
        cd = replace(self.cd, x=(self.design.matrix[:, 1:] - self.means_all)[:, columns])
        return _cox_newton(cd, None, 20, 1e-9)["loglik"][1]

    def anova_table(self, test=None):
        if self.robust:
            raise ValueError("anova(): can't do anova tables with robust variances")
        terms = sorted(set(self.assign))
        labels = ["NULL"] + [self.formula.term_labels[t - 1] for t in terms]
        loglik, dfs = [self.loglik[0]], [0]
        for k in range(1, len(terms) + 1):
            cols = [j for j in self.kept if self.assign[j] in terms[:k]]
            loglik.append(self._refit_loglik(cols) if k < len(terms) else self.loglik[1])
            dfs.append(len(cols))
        stat = [None] + [2 * (b - a) for a, b in zip(loglik[:-1], loglik[1:])]
        df = [None] + [b - a for a, b in zip(dfs[:-1], dfs[1:])]
        p = [None] + [_chisq_p(s, d) for s, d in zip(stat[1:], df[1:])]
        return self._with_key(pl.DataFrame({"term": labels, "logLik": loglik, "statistic": stat, "df": df, "p_value": p},
                                           schema_overrides={"statistic": pl.Float64, "df": pl.Float64, "p_value": pl.Float64}))

    def anova_type(self, kind, test=None):
        raise ValueError("anova(): coxph fits have sequential tables only (type=1)")

    @staticmethod
    def _anova_models(fits, test):
        return _anova_cox(fits, test)

    def __repr__(self):
        return (f"coxph({self.formula.text}, ties={self.method!r})\n{_table_text(self.tidy())}\n"
                f"n={self.n}, number of events={self.nevent:g}")


def _same(a, b):
    try:
        return a == b or float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _strata_exprs(spec):
    return [variable_expr(lbl, "predict")[0].alias(lbl) for lbl in spec.strata]


def _anova_cox(fits, test):
    if any(f.robust for f in fits):
        raise ValueError("anova(): can't do anova tables with robust variances")
    if any(f.method != fits[0].method for f in fits):
        raise ValueError("anova(): all models must have the same ties option")
    if any(f.spec.response != fits[0].spec.response for f in fits) or any(f.n != fits[0].n for f in fits):
        raise ValueError("anova(): models must have the same response and rows")
    loglik = [f.loglik[1] for f in fits]
    df = [f.df for f in fits]
    stat = [None] + [abs(2 * (b - a)) for a, b in zip(loglik[:-1], loglik[1:])]
    ddf = [None] + [float(abs(b - a)) for a, b in zip(df[:-1], df[1:])]
    cols = {"term": [f.formula.text for f in fits], "logLik": loglik, "statistic": stat, "df": ddf}
    if test not in ("none", False):
        cols["p_value"] = [None] + [_chisq_p(s, d) for s, d in zip(stat[1:], ddf[1:])]
    return pl.DataFrame(cols, schema_overrides={"statistic": pl.Float64, "df": pl.Float64, "p_value": pl.Float64})


@step
def coxph(formula: str, *, ties: str = "efron", weights: Any = None, offset: Any = None, robust: Any = None,
          init: Any = None, iter_max: int = 20, eps: float = 1e-9, contrasts: Any = None, na_action: str = "omit"):
    """Fit a Cox proportional hazards model, as R's ``coxph``.

    ``coxph(lung, "Surv(time, status) ~ age + sex + strata(inst)")``;
    ``Surv(start, stop, status)`` gives counting-process data. ``ties`` is
    ``"efron"`` (default), ``"breslow"`` or ``"exact"``. ``cluster(id)``
    (or ``robust=True``, or non-integer weights) gives robust sandwich
    standard errors, as in R. ``tidy`` and ``glance`` have broom's
    columns (with the likelihood-ratio, score, Wald and robust score
    tests, and the concordance). ``predict`` takes ``type="lp"``, ``"risk"``
    or ``"expected"``; ``anova`` gives sequential and nested comparisons;
    ``cox_zph`` tests proportional hazards; ``survfit(fit, newdata)`` gives
    the predicted survival curves.
    """
    if ties not in ("efron", "breslow", "exact"):
        raise ValueError("coxph(): ties must be 'efron', 'breslow' or 'exact'")
    if isinstance(iter_max, bool) or not isinstance(iter_max, (int, np.integer)) or iter_max < 0:
        raise ValueError("coxph(): iter_max must be a non-negative integer")
    if not eps > 0:
        raise ValueError("coxph(): eps must be positive")
    return _fitter("coxph", lambda s, f, d: CoxModel(s, f, d, ties, robust, init, iter_max, eps),
                   formula, weights, offset, contrasts, na_action)


class CoxCurves(_Curves):
    """Predicted survival curves from a Cox model (R's ``survfit.coxph``)."""

    def __init__(self, fit: CoxModel, newdata, conf_type, conf_level):
        self.formula, self.design, self.frame, self._data = fit.formula, None, fit.frame, fit._data
        self.key, self.groups, self.verb, self.text = fit.key, fit.groups, "survfit", fit.formula.text
        efron = fit.method == "efron"
        base = fit._baseline(efron)
        beta, w = fit.beta, fit.prior_weights
        offset = fit.frame.offset
        xcenter = float(fit.means_all[fit.kept] @ beta + (offset * w).sum() / w.sum())
        if newdata is None:
            x2 = fit.means_all[fit.kept][None, :]
            risk2 = np.ones(1)
        else:
            if fit.spec.strata:
                raise ValueError("survfit(): new data for a stratified Cox model is not supported")
            x, off2, _, _ = fit._new(newdata)
            x2 = x[:, fit.kept]
            risk2 = np.exp(x2 @ beta + off2 - xcenter)
        # agsurv's risk is exp(X beta + offset - xcenter); the fit's lp centres
        # X only, so the hazards scale by exp(xcenter - means beta).
        shift = math.exp(xcenter - float(fit.means_all[fit.kept] @ beta))
        self.columns = len(risk2) if newdata is not None else 0
        self.curves = []
        names = fit.strata_names or [None]
        for name, b in zip(names, base):
            haz = b["hazard"] * shift
            varhaz = b["varhaz"] * shift ** 2
            xbar = b["xbar"] * shift
            cumhaz = np.cumsum(haz)
            curve = {k: b[k] for k in ("time", "n_risk", "n_event", "n_censor", "records")}
            surv, se, lower, upper = [], [], [], []
            for i in range(len(risk2)):
                dt = np.cumsum(haz[:, None] * x2[i] - xbar, axis=0)
                v = (np.cumsum(varhaz) + np.einsum("ij,jk,ik->i", dt, fit.var, dt)) * risk2[i] ** 2
                s = np.exp(-cumhaz * risk2[i])
                surv.append(s); se.append(np.sqrt(v))
                if conf_type != "none":
                    lo, hi = _confint(s, np.sqrt(v), conf_type, conf_level)
                    lower.append(lo); upper.append(hi)
            curve["surv_all"], curve["se_all"] = surv, se
            curve["lower_all"], curve["upper_all"] = lower, upper
            self.curves.append((name, curve))

    def _estimates(self, curve):
        k = len(curve["surv_all"])
        suffix = (lambda i: "") if self.columns == 0 or k == 1 else (lambda i: f"_{i + 1}")
        cols = {}
        for i in range(k):
            cols[f"estimate{suffix(i)}"] = curve["surv_all"][i]
        for i in range(k):
            cols[f"std_error{suffix(i)}"] = curve["se_all"][i]
        if curve["lower_all"]:
            for i in range(k):
                cols[f"conf_high{suffix(i)}"] = curve["upper_all"][i]
            for i in range(k):
                cols[f"conf_low{suffix(i)}"] = curve["lower_all"][i]
        return cols

    def glance(self):
        rows = []
        ends = max(float(c["time"][-1]) for _, c in self.curves)
        for name, curve in self.curves:
            for i in range(len(curve["surv_all"])):
                one = dict(curve, surv=curve["surv_all"][i])
                if curve["lower_all"]:
                    one.update(lower=curve["lower_all"][i], upper=curve["upper_all"][i])
                row = {} if name is None else {"strata": name}
                if self.columns > 1:
                    row["curve"] = float(i + 1)
                row["records"] = float(curve["records"])
                row.update(_survmean(one, ends))
                row["nobs"] = float(curve["records"])
                rows.append(row)
        schema = {k: (pl.String if k == "strata" else pl.Float64) for k in rows[0]}
        return self._with_key(pl.DataFrame(rows, schema=schema))


def cox_zph(fit: Any, transform: str = "km", terms: bool = True, global_: bool = True) -> Any:
    """The test of proportional hazards, as R's ``cox.zph``: one row per
    term (and ``GLOBAL``) with ``statistic``, ``df`` and ``p_value``.

    ``transform`` is ``"km"`` (default), ``"rank"``, ``"identity"`` or
    ``"log"``; ``terms=False`` tests each coefficient separately.
    """
    if isinstance(fit, Models):
        return fit._stack([_zph(f, transform, terms, global_) for f in fit.fits])
    if not isinstance(fit, CoxModel):
        raise TypeError("cox_zph(): expected a coxph fit")
    return fit._write(_zph(fit, transform, terms, global_))


def _zph(fit: CoxModel, transform, terms, global_):
    if fit.df == 0:
        raise ValueError("cox_zph(): there are no score residuals for a null model")
    y = fit.y
    event = y.status == 1
    times = y.stop
    if transform == "km":
        km = _km(y, np.ones(len(times)))
        pos = np.searchsorted(km["time"], times, "left")
        ttimes = 1 - np.r_[1, km["surv"]][pos]
    elif transform == "rank":
        from scipy.stats import rankdata

        ttimes = rankdata(times)
    elif transform == "identity":
        ttimes = times.astype(float)
    elif transform == "log":
        ttimes = np.log(times)
    else:
        raise ValueError("cox_zph(): transform must be 'km', 'rank', 'identity' or 'log'")
    gtime = ttimes - ttimes[event].mean()
    p = fit.df
    cd = fit.cd
    u2 = np.zeros(p)
    imat = np.zeros((2 * p, 2 * p))
    for ev in _events(cd, fit.beta, fit.method == "efron"):
        dead, w = ev["dead"], ev["w"]
        g_time = np.zeros(ev["m"])
        g_time[ev["pos"][dead]] = gtime[ev["idx"]][dead]
        gr = g_time[ev["row"]]
        contrib = np.zeros((ev["m"], p))
        np.add.at(contrib, ev["pos"][dead], w[dead, None] * ev["x"][dead])
        np.add.at(contrib, ev["row"], -ev["wtave"][:, None] * ev["xb"])
        u2 += (g_time[:, None] * contrib).sum(0)
        row, frac = ev["row"], ev["frac"]
        second = (ev["s2"][row] - frac[:, None, None] * ev["d2"][row]) / ev["den"][:, None, None]
        v = ev["wtave"][:, None, None] * (second - ev["xb"][:, :, None] * ev["xb"][:, None, :])
        a, b, c = v.sum(0), np.einsum("k,kij->ij", gr, v), np.einsum("k,kij->ij", gr ** 2, v)
        imat += np.block([[a, b], [b, c]])
    kept_assign = [fit.assign[j] for j in fit.kept]
    if terms:
        groups = {}
        for j, t in enumerate(kept_assign):
            groups.setdefault(t, []).append(j)
        labels = [fit.formula.term_labels[t - 1] for t in groups]
        sets = list(groups.values())
    else:
        labels, sets = [fit.names[j] for j in fit.kept], [[j] for j in range(p)]
    stats, dfs = [], []
    for jj in sets:
        kk = list(range(p)) + [j + p for j in jj]
        u = np.r_[np.zeros(p), u2[jj]]
        stats.append(float(np.linalg.solve(imat[np.ix_(kk, kk)], u) @ u))
        dfs.append(float(len(jj)))
    if global_:
        u = np.r_[np.zeros(p), u2]
        stats.append(float(np.linalg.solve(imat, u) @ u))
        dfs.append(float(p))
        labels = labels + ["GLOBAL"]
    out = pl.DataFrame({"term": labels, "statistic": stats, "df": dfs,
                        "p_value": [_chisq_p(s, d) for s, d in zip(stats, dfs)]})
    return fit._with_key(out)


# parametric models -----------------------------------------------------------------------

# name: (base distribution, log-transformed time, fixed scale)
_DISTRIBUTIONS = {
    "weibull": ("extreme", True, None), "exponential": ("extreme", True, 1.0), "rayleigh": ("extreme", True, .5),
    "lognormal": ("gaussian", True, None), "loggaussian": ("gaussian", True, None),
    "loglogistic": ("logistic", True, None), "gaussian": ("gaussian", False, None),
    "logistic": ("logistic", False, None), "extreme": ("extreme", False, None),
}


def _pieces(base, z, exact):
    """log-likelihood, dl/dz and d2l/dz2 per observation on the z scale."""
    s = _special()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if base == "extreme":
            w = np.exp(z)
            ll = np.where(exact, z - w, -w)
            g1 = np.where(exact, 1 - w, -w)
            g2 = -w
        elif base == "logistic":
            w = np.exp(z)
            log1p = np.logaddexp(0, z)
            ll = np.where(exact, z - 2 * log1p, -log1p)
            g1 = np.where(exact, (1 - w) / (1 + w), -w / (1 + w))
            g1 = np.where(np.isfinite(w), g1, np.where(exact, -1.0, -1.0))
            g2 = np.where(exact, -2 * w / (1 + w) ** 2, -w / (1 + w) ** 2)
            g2 = np.where(np.isfinite(w), g2, 0.0)
        else:
            surv = s.log_ndtr(-z)
            ll = np.where(exact, -z * z / 2 - .5 * math.log(2 * math.pi), surv)
            h = np.exp(-z * z / 2 - .5 * math.log(2 * math.pi) - surv)
            g1 = np.where(exact, -z, -h)
            g2 = np.where(exact, -1.0, h * z - h * h)
    return ll, g1, g2


def _init(base, y, w):
    mean = np.sum(y * w) / np.sum(w)
    var = np.sum(w * (y - mean) ** 2) / np.sum(w)
    return {"extreme": (mean + .572, var / 1.64), "logistic": (mean, var / 3.2), "gaussian": (mean, var)}[base]


def _survreg_p(stat, df):
    # broom's 1 - pchisq(stat, df); with no slopes the statistic is 0 on 0 df.
    return (1.0 if stat <= 0 else 0.0) if df == 0 else _chisq_p(stat, df)


class _Survreg:
    """R's survreg6: Newton-Raphson on (beta, log scale), with step halving."""

    def __init__(self, base, y, status, x, w, offset, scale):
        self.base, self.y, self.exact, self.x, self.w, self.offset, self.scale = base, y, status == 1, x, w, offset, scale

    def derivatives(self, theta):
        p = self.x.shape[1]
        beta = theta[:p]
        s = math.log(self.scale) if self.scale else theta[p]
        sigma = math.exp(s)
        z = (self.y - self.x @ beta - self.offset) / sigma
        ll, g1, g2 = _pieces(self.base, z, self.exact)
        w = self.w
        loglik = float(np.sum(w * (ll - self.exact * s)))
        d_eta, dd_eta = -g1 / sigma, g2 / sigma ** 2
        u = self.x.T @ (w * d_eta)
        h = (self.x * (w * dd_eta)[:, None]).T @ self.x
        if not self.scale:
            d_s = -z * g1 - self.exact
            d_es = (z * g2 + g1) / sigma
            d_ss = z * g1 + z * z * g2
            u = np.r_[u, np.sum(w * d_s)]
            cross = self.x.T @ (w * d_es)
            h = np.block([[h, cross[:, None]], [cross[None, :], np.array([[np.sum(w * d_ss)]])]])
        return loglik, u, -h

    def fit(self, theta, maxit, eps):
        from scipy.linalg import LinAlgError

        def step(imat, u):
            try:
                return _solve(imat, u)
            except LinAlgError:
                return np.linalg.lstsq(imat, u, rcond=None)[0]

        theta = np.asarray(theta, dtype=float)
        loglik, u, imat = self.derivatives(theta)
        newtheta = theta + step(imat, u)
        halving, converged, iteration = 0, False, 0
        for iteration in range(1, maxit + 1):
            newlk, u, imat = self.derivatives(newtheta)
            if np.isfinite(newlk) and abs(1 - loglik / newlk) <= eps and halving == 0:
                converged = True
                break
            if iteration == maxit:
                break
            if not np.isfinite(newlk) or newlk < loglik:
                halving += 1
                newtheta = (newtheta + theta) / 2
            else:
                halving = 0
                loglik = newlk
                theta = newtheta
                newtheta = newtheta + step(imat, u)
        return dict(theta=newtheta, loglik=newlk, imat=imat, iter=iteration, converged=converged)


class SurvregModel(_Fit):
    """A parametric survival regression, as R's ``survreg``."""

    kind = "survreg"

    def __init__(self, spec, frame, data, dist, scale, maxit, eps):
        n = len(frame.weights)
        if spec.strata:
            raise ValueError("survreg(): strata() (one scale per stratum) is not supported")
        design = build_design(spec.formula, frame.variables, n)
        super().__init__(spec.formula, design, frame, data, "survreg")
        self.spec, self.dist = spec, dist
        self.base, self.logged, fixed = _DISTRIBUTIONS[dist]
        if fixed is not None:
            if scale:
                _warn(f"survreg(): the {dist} distribution has a fixed scale, user specified value ignored")
            scale = fixed
        self.scale_fixed = scale or None
        y = _response(frame, spec, "survreg")
        if y.counting:
            raise ValueError("survreg(): counting process data are not supported")
        w = frame.weights
        if np.any(w <= 0):
            raise ValueError("survreg(): invalid weights, must be > 0")
        time = y.stop
        exact = y.status == 1
        if self.logged:
            if np.any(time <= 0):
                raise ValueError("survreg(): invalid survival times for this distribution")
            logcorrect = -float(np.sum(w[exact] * np.log(time[exact])))
            ty = np.log(time)
        else:
            logcorrect, ty = 0.0, time.astype(float)
        self.time, self.status, self.prior_weights, self.n = time, y.status, w, n
        x = design.matrix
        from stats3.models import _qr

        self.qr = _qr(x, 1e-7, np.zeros(n))
        if len(self.qr.kept) < x.shape[1]:
            raise ValueError("survreg(): X matrix is singular")
        offset = frame.offset
        mean0, var0 = _init(self.base, ty, w)
        vars_ = math.log(scale) if scale else math.log(4 * var0) / 2
        meanonly = x.shape[1] == 1 and np.all(x == 1)

        def start_eta(sigma):
            _, g1, g2 = _pieces(self.base, np.zeros(n), exact)
            return -g1 / sigma, g2 / sigma ** 2

        if not meanonly:
            dg, ddg = start_eta(math.exp(vars_))
            wt = -ddg * w
            c0 = np.sum(w * dg + wt * (ty - offset)) / np.sum(wt)
            null = _Survreg(self.base, ty, y.status, np.ones((n, 1)), w, offset, scale)
            fit0 = null.fit(np.r_[c0, [] if scale else [vars_]], 20, eps)
            if not np.all(np.isfinite(fit0["theta"])):
                raise ValueError("survreg(): initial iteration failed")
            vars_ = math.log(scale) if scale else fit0["theta"][1]
        dg, ddg = start_eta(math.exp(vars_))
        wt = -ddg * w
        coef = np.linalg.solve(x.T @ (wt[:, None] * x), x.T @ (wt * (ty - offset) + w * dg))
        model = _Survreg(self.base, ty, y.status, x, w, offset, scale)
        fit = model.fit(np.r_[coef, [] if scale else [vars_]], maxit, eps)
        if not fit["converged"]:
            _warn("survreg(): ran out of iterations and did not converge")
        p = x.shape[1]
        theta = fit["theta"]
        self.coef = theta[:p]
        self.scale = scale if scale else math.exp(theta[p])
        self.theta, self.var = theta, np.linalg.inv(fit["imat"])
        self.iterations, self.converged = fit["iter"], fit["converged"]
        null_ll = fit["loglik"] if meanonly else fit0["loglik"]
        self.loglik = [null_ll + logcorrect, fit["loglik"] + logcorrect]
        self.df = len(theta)
        self.idf = 1 if scale else 2
        self.linear_predictors = x @ self.coef + offset
        self.parameter_names = list(design.names) + ([] if scale else ["Log(scale)"])

    @property
    def coefficients(self):
        return dict(zip(self.design.names, self.coef.tolist()))

    def tidy(self, conf_int=False, conf_level=.95, exponentiate=False, vcov=None):
        if vcov is not None:
            raise ValueError("survreg(): vcov= is for lm/glm")
        if exponentiate:
            raise TypeError("tidy(): exponentiate is not available for survreg fits")
        s = _special()
        se = np.sqrt(np.diag(self.var))
        stat = self.theta / se
        cols = {"term": self.parameter_names, "estimate": self.theta, "std_error": se,
                "statistic": stat, "p_value": 2 * s.ndtr(-np.abs(stat))}
        if conf_int:
            _check_conf_level(conf_level)
            z = float(s.ndtri((1 + conf_level) / 2))
            p = len(self.coef)
            low, high = self.theta - z * se, self.theta + z * se
            low[p:], high[p:] = np.nan, np.nan
            cols.update(conf_low=low, conf_high=high)
        return self._with_key(_nulls(pl.DataFrame(cols)))

    def vcov_table(self, vcov=None):
        if vcov is not None:
            raise ValueError("survreg(): vcov= is for lm/glm")
        names = self.parameter_names
        return self._with_key(pl.DataFrame({"term": names, **{n: self.var[:, i] for i, n in enumerate(names)}}))

    def log_lik(self):
        return self.loglik[1]

    def glance(self):
        stat = 2 * (self.loglik[1] - self.loglik[0])
        ll = self.log_lik()
        return self._with_key(pl.DataFrame([{
            "iter": self.iterations, "df": self.df, "statistic": stat, "logLik": ll,
            "AIC": -2 * ll + 2 * self.df, "BIC": -2 * ll + math.log(self.n) * self.df,
            "df_residual": self.n - self.df, "nobs": self.n,
            "p_value": _survreg_p(stat, self.df - self.idf)}],
            schema={"iter": pl.Int64, "df": pl.Int64, "statistic": pl.Float64, "logLik": pl.Float64,
                    "AIC": pl.Float64, "BIC": pl.Float64, "df_residual": pl.Int64, "nobs": pl.Int64,
                    "p_value": pl.Float64}))

    def _prediction(self, newdata, type, p, se_fit):
        if type not in ("response", "lp", "link", "linear", "quantile", "uquantile"):
            raise ValueError("predict(): type must be 'response', 'lp', 'quantile' or 'uquantile'")
        type = "lp" if type in ("link", "linear") else type
        k = len(self.coef)
        vv = self.var[:k, :k]
        if newdata is None:
            x, lp = self.design.matrix, self.linear_predictors
        else:
            x, _, _, _ = self._new_design(newdata)
            lp = x @ self.coef  # R's predict.survreg drops offsets for new data
        itrans = np.exp if self.logged else (lambda v: v)
        if type in ("lp", "response"):
            pred = lp
            se = np.sqrt(np.einsum("ij,jk,ik->i", x, vv, x)) if se_fit else None
            if type == "response":
                pred = itrans(pred)
                if se is not None and self.logged:
                    se = se * pred
            cols = {"fit": pred} | ({} if se is None else {"se_fit": se})
        else:
            probs = np.atleast_1d(np.asarray(p, dtype=float))
            from scipy.stats import gumbel_l, logistic, norm

            q = {"extreme": gumbel_l, "logistic": logistic, "gaussian": norm}[self.base].ppf(probs)
            if newdata is None:
                lp = self.linear_predictors
            cols = {}
            for j, (prob, qq) in enumerate(zip(probs, q)):
                name = "fit" if len(probs) == 1 else f"fit_{prob:g}"
                pred = lp + qq * self.scale
                se = None
                if se_fit:
                    if self.scale_fixed:
                        se = np.sqrt(np.einsum("ij,jk,ik->i", x, vv, x))
                    else:
                        temp = np.column_stack([x, np.full(len(x), qq * self.scale)])
                        se = np.sqrt(np.einsum("ij,jk,ik->i", temp, self.var, temp))
                if type == "quantile":
                    pred = itrans(pred)
                    if se is not None and self.logged:
                        se = se * pred
                cols[name] = pred
                if se is not None:
                    cols["se_" + name] = se
        out = _nulls(pl.DataFrame(cols))
        if newdata is None:
            return self._padded(out), self._data
        return out, Data.read(newdata, None, "predict")

    def predict(self, newdata=None, *, type="response", p=(.1, .9), se_fit=False):
        out, data = self._prediction(newdata, type, p, se_fit)
        return data.write(out)

    def augment(self, newdata=None):
        out, data = self._prediction(newdata, "response", None, False)
        if newdata is not None:
            return data.write(pl.concat([data._lf.collect(), out.rename({"fit": ".fitted"})], how="horizontal"))
        return pl.concat([self._rows(), out.rename({"fit": ".fitted"})], how="horizontal")

    def anova_table(self, test=None):
        raise TypeError("anova(): compare nested survreg fits")

    def anova_type(self, kind, test=None):
        return self.anova_table(test)

    @staticmethod
    def _anova_models(fits, test):
        ll = [f.loglik[1] for f in fits]
        df = [f.df for f in fits]
        stat = [None] + [abs(2 * (b - a)) for a, b in zip(ll[:-1], ll[1:])]
        ddf = [None] + [float(abs(b - a)) for a, b in zip(df[:-1], df[1:])]
        return pl.DataFrame({"term": [f.formula.text for f in fits], "df_residual": [float(f.n - f.df) for f in fits],
                             "logLik": ll, "statistic": stat, "df": ddf,
                             "p_value": [None] + [_chisq_p(s, d) for s, d in zip(stat[1:], ddf[1:])]},
                            schema_overrides={"statistic": pl.Float64, "df": pl.Float64, "p_value": pl.Float64})

    def __repr__(self):
        return f"survreg({self.formula.text}, dist={self.dist!r})\n{_table_text(self.tidy())}\nscale = {self.scale:.6g}"


@step
def survreg(formula: str, *, dist: str = "weibull", scale: float = 0, weights: Any = None, offset: Any = None,
            iter_max: int = 30, rel_tolerance: float = 1e-9, contrasts: Any = None, na_action: str = "omit"):
    """Parametric survival regression, as R's ``survreg``.

    ``dist`` is ``"weibull"`` (default), ``"exponential"``, ``"lognormal"``,
    ``"loglogistic"``, ``"gaussian"``, ``"logistic"``, ``"extreme"`` or
    ``"rayleigh"``. Coefficients are on R's accelerated-failure-time
    scale, with ``Log(scale)`` estimated unless ``scale=`` fixes it.
    ``tidy`` and ``glance`` follow broom; ``predict`` takes
    ``type="response"`` (default), ``"lp"``, ``"quantile"`` or
    ``"uquantile"`` (with ``p=``) and ``se_fit=``. Right-censored data
    only; strata (separate scales) and robust variances are not supported.
    """
    if dist not in _DISTRIBUTIONS:
        raise ValueError(f"survreg(): dist must be one of {sorted(_DISTRIBUTIONS)}")
    if not scale >= 0:
        raise ValueError("survreg(): invalid scale")
    return _fitter("survreg", lambda s, f, d: SurvregModel(s, f, d, dist, scale, iter_max, rel_tolerance),
                   formula, weights, offset, contrasts, na_action, allow_cluster=False)
