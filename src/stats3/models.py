"""Linear and generalized linear models: R's ``lm`` and ``glm``, read the way
broom reads them.

``lm`` solves weighted least squares by QR, dropping columns that add nothing
to the ones before them as R does (tolerance 1e-7), and reports them as
aliased. ``glm`` is a port of R's ``glm.fit``: the same starting values,
iteratively reweighted least squares with the same convergence rule and step
halving, and the standard errors from the last iteration's weights. Model
frames come from Polars: one query selects and evaluates the variables, drops
incomplete rows (R's ``na.omit``), and splits the groups.

``tidy``, ``glance`` and ``augment`` give broom's tables (with ``_`` for
broom's ``.``: ``std_error``, ``p_value``, ``r_squared``); ``anova`` gives
R's sequential tables and model comparisons; ``predict`` and
``augment(newdata=)`` predict new rows. A grouped model (``by=`` or tidy3's
``group_by``) holds one fit per group, and its tables stack the groups.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import polars as pl

from stats3._data import Data, Step, expression, step
from stats3.core import _check_conf_level, _group_label, _sorted_keys, _special, _warn
from stats3.emm import Contrasts, EMMeans
from stats3.robust import as_vcov, covariance
from stats3.formula import (
    Design,
    Formula,
    Variable,
    build_design,
    encode,
    is_categorical,
    parse_formula,
    variable_expr,
)

_EPS = np.finfo(float).eps


# families and links ------------------------------------------------------------


@dataclass
class Link:
    name: str
    linkfun: Callable[[np.ndarray], np.ndarray]
    linkinv: Callable[[np.ndarray], np.ndarray]
    mu_eta: Callable[[np.ndarray], np.ndarray]
    valideta: Callable[[np.ndarray], bool] = lambda eta: True


def _link(name: str) -> Link:
    special = _special()
    if name == "identity":
        return Link(name, lambda m: m, lambda e: e, lambda e: np.ones_like(e))
    if name == "log":
        return Link(name, np.log, lambda e: np.maximum(np.exp(e), _EPS),
                    lambda e: np.maximum(np.exp(e), _EPS))
    if name == "logit":
        thresh = -math.log(_EPS)

        def inv(e):
            e = np.clip(e, -thresh, thresh)
            return 1 / (1 + np.exp(-e))

        def deriv(e):
            opexp = np.exp(np.abs(e))
            out = opexp / (1 + opexp) ** 2
            return np.where(np.abs(e) > thresh, _EPS, out)

        return Link(name, lambda m: np.log(m / (1 - m)), inv, deriv)
    if name == "probit":
        thresh = -float(special.ndtri(_EPS))
        return Link(name, special.ndtri,
                    lambda e: special.ndtr(np.clip(e, -thresh, thresh)),
                    lambda e: np.maximum(np.exp(-e * e / 2) / math.sqrt(2 * math.pi), _EPS))
    if name == "cloglog":
        return Link(name, lambda m: np.log(-np.log1p(-m)),
                    lambda e: np.clip(-np.expm1(-np.exp(e)), _EPS, 1 - _EPS),
                    lambda e: np.maximum(np.exp(np.minimum(e, 700)) * np.exp(-np.exp(np.minimum(e, 700))), _EPS))
    if name == "inverse":
        return Link(name, lambda m: 1 / m, lambda e: 1 / e, lambda e: -1 / e**2,
                    lambda e: bool(np.all(np.isfinite(e)) and np.all(e != 0)))
    if name == "sqrt":
        return Link(name, np.sqrt, lambda e: e**2, lambda e: 2 * e,
                    lambda e: bool(np.all(np.isfinite(e)) and np.all(e > 0)))
    if name == "1/mu^2":
        return Link(name, lambda m: 1 / m**2, lambda e: 1 / np.sqrt(e), lambda e: -1 / (2 * e**1.5),
                    lambda e: bool(np.all(np.isfinite(e)) and np.all(e > 0)))
    raise ValueError(f"unknown link {name!r}")


def _xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x == 0, 0.0, x * np.log(np.where(x == 0, 1.0, y)))


@dataclass
class Family:
    """A GLM family, as R's ``binomial()``, ``poisson()``, ..."""

    family: str
    link: Link
    variance: Callable[[np.ndarray], np.ndarray]
    dev_resids: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    aic: Callable[..., float]
    validmu: Callable[[np.ndarray], bool]
    dispersion: float | None  # fixed at 1, or None when estimated

    def __repr__(self) -> str:
        return f"{self.family}(link = {self.link.name!r})"


def _binomial_dev(y, mu, wt):
    return 2 * wt * (_xlogy(y, y / mu) + _xlogy(1 - y, (1 - y) / (1 - mu)))


def _family(name: str, link: str | None) -> Family:
    from scipy import stats

    def finite_positive(mu):
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    if name == "gaussian":
        return Family(name, _link(link or "identity"), lambda mu: np.ones_like(mu),
                      lambda y, mu, wt: wt * (y - mu) ** 2,
                      lambda y, n, mu, wt, dev: len(y) * (math.log(dev / len(y) * 2 * math.pi) + 1) + 2 - np.sum(np.log(wt)),
                      lambda mu: True, None)
    if name in ("binomial", "quasibinomial"):
        def aic(y, n, mu, wt, dev):
            m = n if np.any(n > 1) else wt
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(m > 0, wt / np.where(m > 0, m, 1), 0.0)
            return -2 * float(np.sum(scale * stats.binom.logpmf(np.round(m * y), np.round(m), mu)))

        return Family(name, _link(link or "logit"), lambda mu: mu * (1 - mu), _binomial_dev,
                      aic if name == "binomial" else (lambda *a: float("nan")),
                      lambda mu: bool(np.all(np.isfinite(mu)) and np.all((mu > 0) & (mu < 1))),
                      1.0 if name == "binomial" else None)
    if name in ("poisson", "quasipoisson"):
        def dev(y, mu, wt):
            return 2 * wt * (_xlogy(y, y / mu) - (y - mu))

        return Family(name, _link(link or "log"), lambda mu: mu, dev,
                      (lambda y, n, mu, wt, d: -2 * float(np.sum(stats.poisson.logpmf(y, mu) * wt)))
                      if name == "poisson" else (lambda *a: float("nan")),
                      finite_positive, 1.0 if name == "poisson" else None)
    if name == "Gamma":
        def aic(y, n, mu, wt, dev):
            disp = dev / np.sum(wt)
            return -2 * float(np.sum(stats.gamma.logpdf(y, 1 / disp, scale=mu * disp) * wt)) + 2

        return Family(name, _link(link or "inverse"), lambda mu: mu**2,
                      lambda y, mu, wt: -2 * wt * (np.log(np.where(y == 0, 1, y / mu)) - (y - mu) / mu),
                      aic, finite_positive, None)
    if name == "inverse.gaussian":
        return Family(name, _link(link or "1/mu^2"), lambda mu: mu**3,
                      lambda y, mu, wt: wt * (y - mu) ** 2 / (y * mu**2),
                      lambda y, n, mu, wt, dev: float(np.sum(wt) * (1 + math.log(dev / np.sum(wt) * 2 * math.pi))
                                                      + 3 * np.sum(np.log(y) * wt) + 2),
                      lambda mu: True, None)
    raise ValueError(
        "family must be gaussian, binomial, poisson, Gamma, inverse.gaussian, "
        f"quasibinomial, or quasipoisson; got {name!r}"
    )


_LINKS = {
    "gaussian": {"identity", "log", "inverse"},
    "binomial": {"logit", "probit", "cloglog", "log"},
    "quasibinomial": {"logit", "probit", "cloglog", "log"},
    "poisson": {"log", "identity", "sqrt"},
    "quasipoisson": {"log", "identity", "sqrt"},
    "Gamma": {"inverse", "identity", "log"},
    "inverse.gaussian": {"1/mu^2", "inverse", "identity", "log"},
}


def _make_family(name: str, link: str | None) -> Family:
    if link is not None and link not in _LINKS.get(name, set()):
        raise ValueError(f"link {link!r} is not available for the {name} family")
    return _family(name, link)


def gaussian(link: str = "identity") -> Family:
    return _make_family("gaussian", link)


def binomial(link: str = "logit") -> Family:
    return _make_family("binomial", link)


def quasibinomial(link: str = "logit") -> Family:
    return _make_family("quasibinomial", link)


def poisson(link: str = "log") -> Family:
    return _make_family("poisson", link)


def quasipoisson(link: str = "log") -> Family:
    return _make_family("quasipoisson", link)


def Gamma(link: str = "inverse") -> Family:  # noqa: N802 - R's name
    return _make_family("Gamma", link)


def inverse_gaussian(link: str = "1/mu^2") -> Family:
    return _make_family("inverse.gaussian", link)


def _as_family(family: Any) -> Family:
    if isinstance(family, Family):
        return family
    if callable(family) and getattr(family, "__module__", "") == __name__:
        return family()
    if isinstance(family, str):
        name = {"inverse_gaussian": "inverse.gaussian", "gamma": "Gamma"}.get(str(family), str(family))
        return _make_family(name, None)
    raise TypeError(f"family must be a family like binomial() or its name, got {family!r}")


# least squares --------------------------------------------------------------------


class _QR:
    """A QR of the kept (weighted) columns: R, Q'z for the response it was
    built with, and Q only when asked for (hat values)."""

    def __init__(self, kept: list[int], a: np.ndarray, r: np.ndarray, qtz: np.ndarray | None):
        self.kept, self._a, self.r, self.qtz = kept, a, r, qtz
        self._q: np.ndarray | None = None

    @property
    def q(self) -> np.ndarray:
        if self._q is None:
            self._q = np.linalg.qr(self._a, mode="reduced")[0] if self.kept else np.zeros((self._a.shape[0], 0))
        return self._q

    def coefficients(self) -> np.ndarray:
        return _solve_upper(self.r, self.qtz) if self.kept else np.empty(0)

    def cov_unscaled(self) -> np.ndarray:
        if not self.kept:
            return np.empty((0, 0))
        rinv = _solve_upper(self.r, np.eye(len(self.kept)))
        return rinv @ rinv.T

    def hat(self) -> np.ndarray:
        return np.sum(self.q**2, axis=1) if self.kept else np.zeros(self._a.shape[0])


def _solve_upper(r: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_triangular

    return solve_triangular(r, b, lower=False)


def _qr(x: np.ndarray, tol: float, z: np.ndarray) -> _QR:
    """QR of the columns R's limited-pivoting QR (dqrdc2) keeps: a column whose
    part not explained by the columns before it is below ``tol`` of its own
    norm is aliased. Without pivoting, |R[j, j]| of a Householder QR is
    exactly that part (an aliased column adds nothing to the span), so one
    LAPACK QR decides every column; it is redone only without the aliased
    ones. Q'z comes from LAPACK without forming Q."""
    from scipy.linalg import qr_multiply

    n, p = x.shape
    if p == 0:
        return _QR([], x, np.zeros((0, 0)), np.zeros(0))
    qtz, r = qr_multiply(x, z, mode="right")
    norms = np.linalg.norm(x, axis=0)
    diag = np.abs(np.diag(r))
    if len(diag) < p:
        diag = np.r_[diag, np.zeros(p - len(diag))]
    kept = [j for j in range(p) if norms[j] > 0 and diag[j] >= tol * norms[j]]
    if len(kept) == p:
        return _QR(kept, x, r[:p, :p], qtz[:p])
    if not kept:
        return _QR([], x[:, []], np.zeros((0, 0)), np.zeros(0))
    a = x[:, kept]
    qtz, r = qr_multiply(a, z, mode="right")
    k = len(kept)
    return _QR(kept, a, r[:k, :k], qtz[:k])


# model frames -------------------------------------------------------------------------


@dataclass
class _Frame:
    """One group's model frame: response, variables, weights, offset."""

    key: dict[str, Any]
    rows: pl.DataFrame  # the source columns the model used, complete rows only
    y: np.ndarray  # (n,) or (n, 2) for cbind
    variables: dict[str, Variable]
    weights: np.ndarray
    offset: np.ndarray
    response_levels: list[Any] | None = None
    index: np.ndarray | None = None  # the rows' positions in the input
    all_rows: pl.DataFrame | None = None  # with na_action="exclude": every row of the group
    all_index: np.ndarray | None = None


def _response_parts(response: str) -> list[str]:
    if response.startswith("cbind(") and response.endswith(")"):
        inner = response[len("cbind("):-1]
        parts, depth, start = [], 0, 0
        for i, c in enumerate(inner):
            depth += c == "("
            depth -= c == ")"
            if c == "," and depth == 0:
                parts.append(inner[start:i].strip())
                start = i + 1
        parts.append(inner[start:].strip())
        if len(parts) != 2:
            raise ValueError(f"cbind() in a response takes two columns: {response!r}")
        return parts
    return [response]


def _source_columns(labels: list[str], columns: list[str]) -> list[str]:
    from stats3.formula import _column_names

    out: dict[str, None] = {}
    for label in labels:
        for name in _column_names(label):
            if name not in columns:
                raise KeyError(f"column {name!r} does not exist. Columns: {', '.join(repr(c) for c in columns[:20])}")
            out.setdefault(name, None)
    return list(out)


NA_ACTIONS = ("omit", "exclude", "fail")


def _model_frames(
    data: Data, formula: Formula, weights: Any, offset: Any, verb: str, na_action: str = "omit"
) -> list[_Frame]:
    groups = list(data._groups)
    if formula.response is None:
        raise ValueError(f"{verb}(): the formula needs a response: 'y ~ ...'")
    responses = _response_parts(formula.response)
    variables = formula.variables()
    exprs: dict[str, pl.Expr] = {}
    factor_vars: set[str] = set()
    for i, label in enumerate(responses):
        exprs[f"__y{i}"] = variable_expr(label, verb)[0]
    for label in variables:
        expr, as_factor = variable_expr(label, verb)
        exprs[label] = expr
        if as_factor:
            factor_vars.add(label)
    offsets = list(formula.offsets)
    for i, label in enumerate(offsets):
        exprs[f"__offset{i}"] = variable_expr(label, verb)[0].cast(pl.Float64)
    if offset is not None:
        exprs["__offset_arg"] = expression(offset).cast(pl.Float64)
    if weights is not None:
        exprs["__weights"] = expression(weights).cast(pl.Float64)
    for g in groups:
        if g in variables or g in responses:
            raise ValueError(f"{verb}(): {g!r} is a grouping column; it cannot be in the formula")
    labels = responses + variables + offsets
    sources = _source_columns(labels, data.columns)
    frame = data._lf.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("__row"),
        *[pl.col(g) for g in groups],
        *[pl.col(c).alias(f"__src_{c}") for c in sources],
        **{k: v for k, v in exprs.items()},
    ).collect()
    # R's na.omit: rows missing (NA or NaN) in any model variable go.
    keep = pl.lit(True)
    for name in exprs:
        dtype = frame.schema[name]
        ok = pl.col(name).is_not_null()
        if dtype.is_float():
            ok = ok & pl.col(name).is_not_nan()
        keep = keep & ok
    full = frame if na_action == "exclude" else None
    height = frame.height
    frame = frame.filter(keep)
    dropped = frame.height < height
    if na_action == "fail" and dropped:
        raise ValueError(f"{verb}(): missing values in the model's variables (na_action='fail')")
    for name in exprs:
        if frame.schema[name].is_float() and frame.select(pl.col(name).is_infinite().any()).item():
            raise ValueError(f"{verb}(): infinite values in {name.removeprefix('__')!r}")
    if groups:
        keys = _sorted_keys(data, frame.select(groups).unique(maintain_order=True), groups)
        parts = frame.partition_by(groups, as_dict=True, maintain_order=True)
        pieces = [(dict(zip(groups, key)), parts[tuple(key)]) for key in keys.iter_rows()]
    else:
        pieces = [({}, frame)]
    out = []
    for key, part in pieces:
        label = _group_label(key, groups)
        n = part.height
        if n == 0:
            raise ValueError(f"{verb}(): 0 (non-NA) cases{label}")
        ys = [part[f"__y{i}"] for i in range(len(responses))]
        response_levels = None
        if len(ys) == 1 and is_categorical(ys[0].dtype):
            y_var = encode(part.rename({"__y0": "y"}), "y", True, _levels_for(data, responses[0]))
            response_levels = y_var.levels
            y = y_var.codes.astype(float)
        else:
            y = np.column_stack([s.cast(pl.Float64).to_numpy() for s in ys]) if len(ys) == 2 else ys[0].cast(pl.Float64).to_numpy()
        encoded = {
            label_: encode(part, label_, label_ in factor_vars, _levels_for(data, label_))
            for label_ in variables
        }
        w = part["__weights"].to_numpy() if weights is not None else np.ones(n)
        if np.any(w < 0):
            raise ValueError(f"{verb}(): negative weights not allowed{label}")
        off = np.zeros(n)
        for i in range(len(offsets)):
            off = off + part[f"__offset{i}"].to_numpy()
        if offset is not None:
            off = off + part["__offset_arg"].to_numpy()
        source = [*groups, *[pl.col(f"__src_{c}").alias(c) for c in sources]]
        rows = part.select(source)
        all_rows = all_index = None
        if full is not None:
            whole = full
            for g in groups:
                whole = whole.filter(pl.col(g).is_null() if key[g] is None else pl.col(g) == key[g])
            all_rows, all_index = whole.select(source), whole["__row"].to_numpy()
        out.append(_Frame(key, rows, y, encoded, w, off, response_levels, part["__row"].to_numpy(),
                          all_rows, all_index))
    return out


def _levels_for(data: Data, label: str) -> list[Any] | None:
    """Factor levels for a variable that is a column, or ``factor(column)``."""
    from stats3.formula import _column_names

    names = _column_names(label)
    if len(names) == 1 and names[0] in data.columns:
        bare = label.strip()
        if bare in (names[0], f"`{names[0]}`") or bare.startswith(("factor(", "as.factor(", "as_factor(")):
            return data.levels_of(names[0])
    return None


# fits -------------------------------------------------------------------------------


class _Fit:
    """What every fit shares: coefficients, their covariance, the design."""

    kind = "model"

    def __init__(self, formula: Formula, design: Design, frame: _Frame, data: Data, verb: str):
        self.formula = formula
        self.design = design
        self.frame = frame
        self._data = data
        self.verb = verb
        self.key = frame.key
        self.groups = list(data._groups)

    # -- shared tables ----------------------------------------------------------

    def __tidy__(self, **kwargs: Any) -> Any:
        """The coefficient table for any ``tidy()``: tidy3's calls this, so
        ``tidy(fit)`` works whichever package's ``tidy`` was imported last."""
        return tidy(self, **kwargs)

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.design.names, self.coef.tolist()))

    def _coef_table(self, conf_int: bool, conf_level: float, exponentiate: bool,
                    vcov: Any = None) -> pl.DataFrame:
        """The coefficient table; with ``vcov``, tested with that covariance
        as lmtest's ``coeftest`` and ``coefci`` do (t on the residual df for
        ``lm``, normal for ``glm``)."""
        names = self.design.names
        est = self.coef.copy()
        se = np.full(len(names), np.nan)
        spec = as_vcov(vcov)
        if spec is None:
            se[self.qr.kept] = np.sqrt(np.diag(self.cov_unscaled) * self.scale)
        else:
            se[self.qr.kept] = np.sqrt(np.diag(covariance(self, spec)))
        stat = est / se
        normal = spec is not None and isinstance(self, GeneralizedLinearModel)
        p = 2 * _special().ndtr(-np.abs(stat)) if normal else self._p_values(stat)
        table: dict[str, Any] = {"term": names, "estimate": est, "std_error": se, "statistic": stat, "p_value": p}
        if conf_int:
            _check_conf_level(conf_level)
            q = self._quantile((1 + conf_level) / 2)
            table["conf_low"] = est - q * se
            table["conf_high"] = est + q * se
        if exponentiate:
            table["estimate"] = np.exp(est)
            if conf_int:
                table["conf_low"], table["conf_high"] = np.exp(table["conf_low"]), np.exp(table["conf_high"])
        frame = pl.DataFrame(table)
        return frame.with_columns(pl.col(c).fill_nan(None) for c in frame.columns if c != "term")

    def vcov_table(self, vcov: Any = None) -> pl.DataFrame:
        spec = as_vcov(vcov)
        kept = self.qr.kept
        if spec is None:
            names = self.design.names
            full = np.full((len(names), len(names)), np.nan)
            full[np.ix_(kept, kept)] = self.cov_unscaled * self.scale
        else:
            names = [self.design.names[i] for i in kept]
            full = covariance(self, spec)
        frame = pl.DataFrame({"term": names, **{n: full[:, i] for i, n in enumerate(names)}})
        return self._with_key(frame.with_columns(pl.col(n).fill_nan(None) for n in names))

    def _x(self, frame: _Frame) -> np.ndarray:
        return build_design(self.formula, frame.variables, len(frame.weights)).matrix

    def _new_design(self, newdata: Any) -> tuple[np.ndarray, np.ndarray, pl.DataFrame, Data]:
        """The model matrix and offset for new rows, with the fit's levels."""
        data = Data.read(newdata, None, "predict")
        exprs = {}
        for label in self.formula.variables():
            exprs[label] = variable_expr(label, "predict")[0]
        for i, label in enumerate(self.formula.offsets):
            exprs[f"__offset{i}"] = variable_expr(label, "predict")[0].cast(pl.Float64)
        if self.formula.offset_arg is not None:
            exprs["__offset_arg"] = expression(self.formula.offset_arg).cast(pl.Float64)
        frame = data._lf.select(**exprs).collect() if exprs else data._lf.select(pl.len()).collect()
        n = frame.height if exprs else int(frame.item())
        variables = {}
        for label in self.formula.variables():
            levels = self.design.levels.get(label)
            if levels is not None:
                values = frame[label].to_list()
                unknown = sorted({v for v in values if v not in set(levels)}, key=str)
                if unknown:
                    raise ValueError(f"predict(): factor {label!r} has new levels {unknown}")
                index = {lv: i for i, lv in enumerate(levels)}
                variables[label] = Variable(label, codes=np.array([index[v] for v in values]), levels=levels)
            else:
                variables[label] = Variable(label, values=frame[label].cast(pl.Float64).to_numpy())
        x = build_design(self.formula, variables, n).matrix
        offset = np.zeros(n)
        for i in range(len(self.formula.offsets)):
            offset = offset + frame[f"__offset{i}"].to_numpy()
        if self.formula.offset_arg is not None:
            offset = offset + frame["__offset_arg"].to_numpy()
        if self.qr.kept != list(range(len(self.design.names))):
            _warn("prediction from rank-deficient fit; attr(*, \"non-estim\") has doubtful cases")
        return x, offset, data._lf.collect(), data

    def _design_for(self, newdata: Any) -> tuple[np.ndarray, np.ndarray, pl.DataFrame, Data]:
        """The model matrix and offset to predict: new rows, or the fit's own."""
        if newdata is None:
            return self.design.matrix, self.frame.offset, self.frame.rows, self._data
        return self._new_design(newdata)

    def _rows(self) -> pl.DataFrame:
        """The fit's rows; with na_action="exclude", every row, as R's na.exclude."""
        return self.frame.rows if self.frame.all_rows is None else self.frame.all_rows

    def _padded(self, cols: pl.DataFrame) -> pl.DataFrame:
        """Per-row results on the fit's rows, with missing values for the rows
        left out when na_action="exclude" (R's ``naresid``)."""
        if self.frame.all_rows is None:
            return cols
        where = np.searchsorted(self.frame.all_index, self.frame.index)
        used = cols.with_columns(__stats3_at=pl.Series(where, dtype=pl.Int64))
        at = pl.DataFrame({"__stats3_at": np.arange(len(self.frame.all_index), dtype=np.int64)})
        return at.join(used, on="__stats3_at", how="left", maintain_order="left").drop("__stats3_at")

    def _write(self, frame: pl.DataFrame, data: Data | None = None) -> Any:
        return (data or self._data).write(frame)

    def _with_key(self, frame: pl.DataFrame) -> pl.DataFrame:
        if not self.groups:
            return frame
        keys = self.frame.rows.select(self.groups).head(1)
        return pl.concat([pl.concat([keys] * frame.height) if frame.height else keys.clear(), frame], how="horizontal")

    def _aliased_note(self) -> str:
        aliased = [n for i, n in enumerate(self.design.names) if i not in self.qr.kept]
        if not aliased:
            return ""
        return f"\n{len(aliased)} not estimated (aliased): {', '.join(aliased)}"


def _hypotheses(design: Design, formula: Formula, kind: int) -> list[tuple[str, list[int], list[int]]]:
    """For each tested term: its label, the columns dropped in the smaller
    model, and the columns dropped in the larger one (car's definitions).

    Type II tests a term after every other term except its higher-order
    relatives (the terms containing it); type III after all other terms,
    the intercept included.
    """
    assign = np.array(design.assign)
    terms = [frozenset(t) for t in formula.terms]
    out = []
    if kind == 3 and formula.intercept:
        out.append(("(Intercept)", list(np.where(assign == 0)[0]), []))
    for index, (label, term) in enumerate(zip(design.term_labels, terms), start=1):
        own = list(np.where(assign == index)[0])
        if kind == 3:
            out.append((label, own, []))
            continue
        relatives = [j for j, other in enumerate(terms, start=1) if term < other]
        rel_cols = list(np.where(np.isin(assign, relatives))[0])
        out.append((label, rel_cols + own, rel_cols))
    return out


def _check_type(type_: Any) -> int:
    kinds = {1: 1, 2: 2, 3: 3, "I": 1, "II": 2, "III": 3}
    if type_ not in kinds:
        raise ValueError(f"type must be 1, 2, or 3 (or 'I', 'II', 'III'), got {type_!r}")
    return kinds[type_]


class LinearModel(_Fit):
    """A fitted ``lm``."""

    def __init__(self, formula: Formula, frame: _Frame, data: Data, verb: str = "lm"):
        design = build_design(formula, frame.variables, len(frame.weights))
        super().__init__(formula, design, frame, data, verb)
        if frame.y.ndim != 1:
            raise ValueError(f"{verb}(): the response must be one numeric column")
        if frame.response_levels is not None:
            raise ValueError(f"{verb}(): the response {formula.response!r} must be numeric")
        x, y, w, off = design.matrix, frame.y, frame.weights, frame.offset
        self.n = int(np.sum(w > 0))
        sw = np.sqrt(w)
        self.qr = _qr(x * sw[:, None], 1e-7, (y - off) * sw)
        coef = np.full(x.shape[1], np.nan)
        coef[self.qr.kept] = self.qr.coefficients()
        self.coef = coef
        self.rank = len(self.qr.kept)
        self.fitted = x[:, self.qr.kept] @ coef[self.qr.kept] + off
        self.residuals = y - self.fitted
        self.weights = w
        self.y = y
        self.df_residual = self.n - self.rank
        self.rss = float(np.sum(w * self.residuals**2))
        self.sigma = math.sqrt(self.rss / self.df_residual) if self.df_residual > 0 else float("nan")
        self.scale = self.sigma**2
        self.cov_unscaled = self.qr.cov_unscaled()
        self.effects = self.qr.qtz

    def _p_values(self, stat):
        return 2 * _special().stdtr(self.df_residual, -np.abs(stat))

    def _quantile(self, p):
        return float(_special().stdtrit(self.df_residual, p))

    # -- broom ----------------------------------------------------------------------

    def tidy(self, conf_int: bool = False, conf_level: float = 0.95, vcov: Any = None) -> pl.DataFrame:
        return self._with_key(self._coef_table(conf_int, conf_level, False, vcov))

    def _r_squared(self) -> tuple[float, float, float, float, float]:
        w, f = self.weights, self.fitted - self.frame.offset
        df_int = 1 if self.formula.intercept else 0
        if self.formula.intercept:
            m = np.sum(w * f) / np.sum(w)
            mss = float(np.sum(w * (f - m) ** 2))
        else:
            mss = float(np.sum(w * f**2))
        r2 = mss / (mss + self.rss)
        adj = 1 - (1 - r2) * ((self.n - df_int) / self.df_residual)
        num_df = self.rank - df_int
        fstat = (mss / num_df) / self.scale if num_df > 0 else float("nan")
        p = float(_special().fdtrc(num_df, self.df_residual, fstat)) if num_df > 0 else float("nan")
        return r2, adj, fstat, p, num_df

    def log_lik(self) -> float:
        ok = self.weights > 0
        w, res = self.weights[ok], self.residuals[ok]
        n = len(res)
        return 0.5 * (np.sum(np.log(w)) - n * (math.log(2 * math.pi) + 1 - math.log(n) + math.log(np.sum(w * res**2))))

    def glance(self, vcov: Any = None) -> pl.DataFrame:
        r2, adj, fstat, p, num_df = self._r_squared()
        if vcov is not None and num_df > 0:
            fstat, p = self._wald_f(covariance(self, as_vcov(vcov)))
        ll = self.log_lik()
        k = self.rank + 1
        frame = pl.DataFrame({
            "r_squared": [r2], "adj_r_squared": [adj], "sigma": [self.sigma],
            "statistic": [fstat], "p_value": [p], "df": [float(num_df)],
            "logLik": [ll], "AIC": [-2 * ll + 2 * k], "BIC": [-2 * ll + math.log(self.n) * k],
            "deviance": [self.rss], "df_residual": [float(self.df_residual)], "nobs": [self.n],
        })
        return self._with_key(frame.with_columns(pl.col(pl.Float64).fill_nan(None)))

    def _wald_f(self, cov: np.ndarray) -> tuple[float, float]:
        """lmtest's ``waldtest``: all coefficients but the intercept are 0."""
        kept = self.qr.kept
        slopes = [i for i, j in enumerate(kept) if not (self.formula.intercept and j == 0)]
        b = self.coef[kept][slopes]
        v = cov[np.ix_(slopes, slopes)]
        f = float(b @ np.linalg.solve(v, b)) / len(slopes)
        return f, float(_special().fdtrc(len(slopes), self.df_residual, f))

    def _influence(self) -> dict[str, np.ndarray]:
        h = self.qr.hat()
        e = np.sqrt(self.weights) * self.residuals
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_i = _leave_one_out_sigma(e, h, self.rank)
            std = e / (self.sigma * np.sqrt(1 - h))
            cooks = (e / (self.sigma * (1 - h))) ** 2 * h / self.rank
        return {".fitted": self.fitted, ".resid": self.residuals, ".hat": h,
                ".sigma": sigma_i, ".cooksd": cooks, ".std_resid": std}

    def augment(self, newdata: Any = None, *, interval: str = "none", conf_level: float = 0.95,
                se_fit: bool = False) -> pl.DataFrame:
        if newdata is None:
            cols = self._influence()
            frame = pl.concat([self._rows(), self._padded(pl.DataFrame(cols))], how="horizontal")
            return frame.with_columns(pl.col(c).fill_nan(None) for c in cols)
        return self._predict_frame(newdata, interval, conf_level, se_fit, augment=True)

    def predict(self, newdata: Any = None, *, interval: str = "none", level: float = 0.95,
                se_fit: bool = False) -> Any:
        """Predictions for ``newdata``, or for the rows the model was fitted
        to; ``interval`` and ``se_fit`` apply to both, as in R."""
        return self._predict_frame(newdata, interval, level, se_fit, augment=False)

    def _predict_frame(self, newdata, interval, level, se_fit, augment):
        if interval not in ("none", "confidence", "prediction"):
            raise ValueError("interval must be 'none', 'confidence', or 'prediction'")
        x, off, rows, data = self._design_for(newdata)
        kept = self.qr.kept
        xk = x[:, kept]
        fit = xk @ self.coef[kept] + off
        se = np.sqrt(np.sum((xk @ _solve_upper(self.qr.r, np.eye(len(kept)))) ** 2, axis=1)) * self.sigma if kept else np.zeros(len(fit))
        names = (".fitted", ".lower", ".upper", ".se_fit") if augment else ("fit", "lwr", "upr", "se_fit")
        cols = {names[0]: fit}
        if interval != "none":
            q = self._quantile((1 + level) / 2)
            weighted = bool(np.any(self.weights != 1))
            pred_var = np.full(len(fit), self.scale)
            if interval == "prediction":
                # R's predict.lm: on the fitted rows the variance of a new
                # response uses the fitting weights; on new rows it does not.
                if newdata is None:
                    _warn("predictions on current data refer to _future_ responses")
                    if weighted:
                        _warn("assuming prediction variance inversely proportional to weights used for fitting")
                        pred_var = self.scale / self.weights
                elif weighted:
                    _warn("Assuming constant prediction variance even though model fit is weighted")
            width = q * (se if interval == "confidence" else np.sqrt(se**2 + pred_var))
            cols[names[1]], cols[names[2]] = fit - width, fit + width
        if se_fit:
            cols[names[3]] = se
        out = pl.DataFrame(cols)
        if newdata is None:
            out, rows = self._padded(out), self._rows()
        if augment:
            out = pl.concat([rows, out], how="horizontal")
        return data.write(out)

    def anova_table(self) -> pl.DataFrame:
        """Sequential sums of squares by term, as ``anova(lm)``."""
        assign = [self.design.assign[j] for j in self.qr.kept]
        terms, dfs, ss = [], [], []
        for index, label in enumerate(self.design.term_labels, start=1):
            cols = [k for k, a in enumerate(assign) if a == index]
            if not cols:
                continue
            terms.append(label)
            dfs.append(float(len(cols)))
            ss.append(float(np.sum(self.effects[cols] ** 2)))
        special = _special()
        ms = [s / d for s, d in zip(ss, dfs)]
        res_ms = self.rss / self.df_residual if self.df_residual > 0 else float("nan")
        fval = [m / res_ms for m in ms]
        p = [float(special.fdtrc(d, self.df_residual, f)) for d, f in zip(dfs, fval)]
        frame = pl.DataFrame({
            "term": terms + ["Residuals"],
            "df": dfs + [float(self.df_residual)],
            "sumsq": ss + [self.rss],
            "meansq": ms + [res_ms],
            "statistic": pl.Series(fval + [None], dtype=pl.Float64),
            "p_value": pl.Series(p + [None], dtype=pl.Float64),
        })
        return self._with_key(frame)

    def _rss_without(self, drop: list[int]) -> tuple[float, int]:
        """Residual sum of squares and rank with some columns dropped."""
        keep = [j for j in range(self.design.matrix.shape[1]) if j not in set(drop)]
        sw = np.sqrt(self.weights)
        yw = (self.y - self.frame.offset) * sw
        if not keep:
            return float(np.sum(yw**2)), 0
        xw = self.design.matrix[:, keep] * sw[:, None]
        qr = _qr(xw, 1e-7, yw)
        resid = yw - xw[:, qr.kept] @ qr.coefficients()
        return float(np.sum(resid**2)), len(qr.kept)

    def anova_type(self, kind: int) -> pl.DataFrame:
        """Type II or III sums of squares, as car's ``Anova(type=)``."""
        special = _special()
        res_ms = self.rss / self.df_residual
        rows: dict[str, list[Any]] = {k: [] for k in ("term", "df", "sumsq", "meansq", "statistic", "p_value")}
        for label, smaller, larger in _hypotheses(self.design, self.formula, kind):
            rss_small, rank_small = self._rss_without(smaller)
            rss_large, rank_large = self._rss_without(larger)
            df = rank_large - rank_small
            if df == 0:
                continue
            ss = rss_small - rss_large
            fval = (ss / df) / res_ms
            rows["term"].append(label)
            rows["df"].append(float(df))
            rows["sumsq"].append(ss)
            rows["meansq"].append(ss / df)
            rows["statistic"].append(fval)
            rows["p_value"].append(float(special.fdtrc(df, self.df_residual, fval)))
        rows["term"].append("Residuals")
        rows["df"].append(float(self.df_residual))
        rows["sumsq"].append(self.rss)
        rows["meansq"].append(res_ms)
        rows["statistic"].append(None)
        rows["p_value"].append(None)
        return self._with_key(pl.DataFrame(rows, schema_overrides={"statistic": pl.Float64, "p_value": pl.Float64}))

    # -- printing ---------------------------------------------------------------------

    def __repr__(self) -> str:
        table = self._coef_table(True, 0.95, False).select("term", "estimate", "conf_low", "conf_high", "std_error", "p_value")
        r2, adj, fstat, p, num_df = self._r_squared()
        head = f"lm({self.formula.text})" + _key_text(self.key) + f"   n = {self.n}"
        foot = (f"residual SE {self.sigma:.4g} on {self.df_residual} df · R² {r2:.4g} · adjusted R² {adj:.4g}\n"
                f"F({num_df}, {self.df_residual}) = {fstat:.4g}, {_p_text(p)}")
        return f"{head}\n{_table_text(table)}\n{foot}{self._aliased_note()}"


class GeneralizedLinearModel(_Fit):
    """A fitted ``glm``."""

    def __init__(self, formula: Formula, frame: _Frame, data: Data, family: Family,
                 maxit: int = 25, epsilon: float = 1e-8, verb: str = "glm"):
        design = build_design(formula, frame.variables, len(frame.weights))
        super().__init__(formula, design, frame, data, verb)
        self.family = family
        y, w, n_trials = self._prepare(frame, family)
        self.y, self.prior_weights, self.n_trials = y, w, n_trials
        fit = _glm_fit(design.matrix, y, w, frame.offset, family, n_trials, maxit, epsilon, verb)
        self.__dict__.update(fit)
        if formula.intercept and np.any(frame.offset != 0):
            # R refits the intercept-only model when there is an offset.
            ones = np.ones((len(y), 1))
            self.null_deviance = _glm_fit(ones, y, w, frame.offset, family, n_trials, maxit, epsilon, verb)["deviance"]
        self.cov_unscaled = self.qr.cov_unscaled()
        if family.dispersion is not None:
            self.scale = family.dispersion
        elif self.df_residual > 0:
            ok = self.working_weights > 0
            self.scale = float(np.sum((self.working_weights * self.working_residuals**2)[ok]) / self.df_residual)
        else:
            self.scale = float("nan")

    def _prepare(self, frame: _Frame, family: Family):
        y = frame.y
        w = frame.weights.astype(float).copy()
        n_trials = np.ones(len(w))
        if family.family in ("binomial", "quasibinomial"):
            if y.ndim == 2:
                if np.any(np.abs(y - np.round(y)) > 0.001):
                    _warn("non-integer counts in a binomial glm!")
                n_trials = y[:, 0] + y[:, 1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = np.where(n_trials == 0, 0.0, y[:, 0] / np.where(n_trials == 0, 1, n_trials))
                w = w * n_trials
            else:
                y = y.astype(float).copy()
                y[w == 0] = 0
                if np.any((y < 0) | (y > 1)):
                    raise ValueError(f"{self.verb}(): y values must be 0 <= y <= 1")
                m = w * y
                if np.any(np.abs(m - np.round(m)) > 0.001):
                    _warn("non-integer #successes in a binomial glm!")
        elif y.ndim == 2:
            raise ValueError(f"{self.verb}(): cbind() responses are for binomial models")
        elif frame.response_levels is not None:
            raise ValueError(f"{self.verb}(): the response must be numeric for the {family.family} family")
        return y, w, n_trials

    def _p_values(self, stat):
        special = _special()
        if self.family.dispersion is not None:
            return 2 * special.ndtr(-np.abs(stat))
        return 2 * special.stdtr(self.df_residual, -np.abs(stat))

    def _quantile(self, p):
        # Wald intervals use the normal quantile whatever the family, as
        # R's confint.default does.
        return float(_special().ndtri(p))

    def tidy(self, conf_int: bool = False, conf_level: float = 0.95, exponentiate: bool = False,
             conf_method: str | None = None, vcov: Any = None) -> pl.DataFrame:
        """Coefficients. ``conf_int=True`` adds intervals: by default profile
        likelihood intervals as R's ``confint`` (``profile.glm``'s grid,
        spline and inversion), or ``conf_method="wald"`` for estimate ± z
        times the standard error, R's ``confint.default``. With ``vcov``
        the intervals are Wald intervals with the robust standard errors."""
        if conf_method is None:
            conf_method = "profile" if vcov is None else "wald"
        if conf_method not in ("profile", "wald"):
            raise ValueError("conf_method must be 'profile' or 'wald'")
        if conf_method == "profile" and vcov is not None:
            raise ValueError("profile intervals don't use vcov; use conf_method='wald'")
        table = self._coef_table(conf_int, conf_level, False, vcov)
        if conf_int and conf_method == "profile":
            low, high = self._profile_intervals(conf_level)
            table = table.with_columns(conf_low=pl.Series(low).fill_nan(None), conf_high=pl.Series(high).fill_nan(None))
        if exponentiate:
            cols = ["estimate"] + (["conf_low", "conf_high"] if conf_int else [])
            table = table.with_columns(pl.col(c).exp() for c in cols)
        return self._with_key(table)

    def _profile_intervals(self, conf_level: float) -> tuple[np.ndarray, np.ndarray]:
        """R's ``confint(glm)``: ``profile.glm`` then ``confint.profile.glm``.

        Each coefficient is fixed at up to 9 points either side of its
        estimate (steps of zmax/5 standard errors), the others refitted from
        the previous point's linear predictor; the signed root deviance
        profile is splined (FMM) at three times as many points and inverted
        by linear interpolation at the normal quantiles, as R does.
        """
        from scipy import stats

        special = _special()
        alpha = (1 - conf_level) / 4
        x = self.design.matrix
        n, p = x.shape
        if self.family.family in ("binomial", "poisson"):
            zmax = math.sqrt(float(stats.chi2.isf(alpha, 1)))
        else:
            zmax = math.sqrt(float(stats.f.isf(alpha, 1, n - p)))
        delta = zmax / 5
        se = np.full(p, np.nan)
        se[self.qr.kept] = np.sqrt(np.diag(self.cov_unscaled) * self.scale)
        cut = special.ndtri(np.array([(1 - conf_level) / 2, 1 - (1 - conf_level) / 2]))
        offset = self.frame.offset
        nona = np.zeros(p, dtype=bool)
        nona[self.qr.kept] = True
        low, high = np.full(p, np.nan), np.full(p, np.nan)
        b0 = np.nan_to_num(self.coef)
        for i in range(p):
            if not nona[i]:
                continue
            others = nona.copy()
            others[i] = False
            xi = x[:, others]
            zs, values = [0.0], [b0[i]]
            for sign in (-1, 1):
                lp = x[:, nona] @ b0[nona] + offset
                step, z = 0, 0.0
                while (step := step + 1) < 10 and abs(z) < zmax:
                    bi = b0[i] + sign * step * delta * se[i]
                    o = offset + x[:, i] * bi
                    fit = _glm_fit(xi, self.y, self.prior_weights, o, self.family, self.n_trials,
                                   25, 1e-8, self.verb, etastart=lp)
                    coef = np.nan_to_num(fit["coef"])
                    lp = xi @ coef + o
                    zz = (fit["deviance"] - self.deviance) / self.scale
                    if zz <= -0.001:
                        raise ValueError("profiling has found a better solution, so original fit had not converged")
                    z = sign * math.sqrt(max(zz, 0.0))
                    zs.append(z)
                    values.append(bi)
            order = np.argsort(zs, kind="stable")
            sx, sy = _spline(np.array(values)[order], np.array(zs)[order])
            low[i], high[i] = _approx(sy, sx, cut)
        return low, high

    def log_lik(self) -> float:
        p = self.rank + (1 if self.family.family in ("gaussian", "Gamma", "inverse.gaussian") else 0)
        return p - self.aic / 2

    def glance(self) -> pl.DataFrame:
        ll = self.log_lik()
        k = self.rank + (1 if self.family.family in ("gaussian", "Gamma", "inverse.gaussian") else 0)
        nobs = int(np.sum(self.prior_weights != 0))
        frame = pl.DataFrame({
            "null_deviance": [self.null_deviance], "df_null": [float(self.df_null)],
            "logLik": [ll], "AIC": [self.aic], "BIC": [-2 * ll + math.log(nobs) * k],
            "deviance": [self.deviance], "df_residual": [float(self.df_residual)], "nobs": [nobs],
        })
        return self._with_key(frame.with_columns(pl.col(pl.Float64).fill_nan(None)))

    def _deviance_residuals(self) -> np.ndarray:
        d = self.family.dev_resids(self.y, self.mu, self.prior_weights)
        return np.sign(self.y - self.mu) * np.sqrt(np.maximum(d, 0))

    def augment(self, newdata: Any = None, *, type_predict: str = "link", se_fit: bool = False) -> pl.DataFrame:
        if newdata is not None:
            return self._predict(newdata, type_predict, se_fit, augment=True)
        h = self.qr.hat()
        hat = np.zeros(len(self.y))
        hat[self.good] = h
        dev = self._deviance_residuals()
        pearson = (self.y - self.mu) * np.sqrt(self.prior_weights) / np.sqrt(self.family.variance(self.mu))
        # R's weighted.residuals for a glm: working residuals times the
        # square root of the working weights.
        e = (self.working_residuals * np.sqrt(self.working_weights))[self.good]
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_i = np.full(len(self.y), np.nan)
            sigma_i[self.good] = _leave_one_out_sigma(e, h, self.rank)
            cooks = (pearson / (1 - hat)) ** 2 * hat / (self.scale * self.rank)
            # R 4.6's rstandard() for a glm: Pearson residuals over sigma(),
            # the square root of the dispersion.
            std = pearson / np.sqrt(self.scale * (1 - hat))
        fitted = self.eta if type_predict == "link" else self.mu
        cols = {".fitted": fitted, ".resid": dev, ".hat": hat, ".sigma": sigma_i, ".cooksd": cooks, ".std_resid": std}
        frame = pl.concat([self._rows(), self._padded(pl.DataFrame(cols))], how="horizontal")
        return frame.with_columns(pl.col(c).fill_nan(None) for c in cols)

    def predict(self, newdata: Any = None, *, type: str = "link", se_fit: bool = False) -> Any:  # noqa: A002
        """Predictions for ``newdata``, or for the rows the model was fitted
        to; ``type`` and ``se_fit`` apply to both, as in R."""
        return self._predict(newdata, type, se_fit, augment=False)

    def _predict(self, newdata, type_, se_fit, augment):
        if type_ not in ("link", "response"):
            raise ValueError("type must be 'link' or 'response'")
        x, off, rows, data = self._design_for(newdata)
        kept = self.qr.kept
        eta = x[:, kept] @ self.coef[kept] + off
        se = np.sqrt(np.sum((x[:, kept] @ _solve_upper(self.qr.r, np.eye(len(kept)))) ** 2, axis=1) * self.scale)
        fit = eta
        if type_ == "response":
            fit = self.family.link.linkinv(eta)
            se = se * np.abs(self.family.link.mu_eta(eta))
        names = (".fitted", ".se_fit") if augment else ("fit", "se_fit")
        cols = {names[0]: fit}
        if se_fit:
            cols[names[1]] = se
        out = pl.DataFrame(cols)
        if newdata is None:
            out, rows = self._padded(out), self._rows()
        if augment:
            out = pl.concat([rows, out], how="horizontal")
        return data.write(out)

    def _default_test(self) -> str:
        return "Chisq" if self.family.dispersion is not None else "F"

    def anova_table(self, test: str | None = None) -> pl.DataFrame:
        """Analysis of deviance, adding terms in order, as ``anova(glm)``."""
        test = test or self._default_test()
        x = self.design.matrix
        assign = np.array(self.design.assign)
        labels = self.design.term_labels
        resdf, resdev = [float(self.df_null)], [self.null_deviance]
        for index in range(1, len(labels)):
            cols = np.where(assign <= index)[0]
            fit = _glm_fit(x[:, cols], self.y, self.prior_weights, self.frame.offset, self.family,
                           self.n_trials, 25, 1e-8, self.verb)
            resdf.append(float(fit["df_residual"]))
            resdev.append(fit["deviance"])
        if labels:
            resdf.append(float(self.df_residual))
            resdev.append(self.deviance)
        table = {
            "term": ["NULL"] + labels,
            "df": [None] + [resdf[i - 1] - resdf[i] for i in range(1, len(resdf))],
            "deviance": [None] + [resdev[i - 1] - resdev[i] for i in range(1, len(resdev))],
            "df_residual": resdf,
            "residual_deviance": resdev,
        }
        table.update(_deviance_tests(table["df"], table["deviance"], self.scale, self.df_residual, test, self.family))
        return self._with_key(pl.DataFrame(table, schema_overrides={"df": pl.Float64, "deviance": pl.Float64}))

    def _deviance_without(self, drop: list[int]) -> tuple[float, int]:
        keep = [j for j in range(self.design.matrix.shape[1]) if j not in set(drop)]
        x = self.design.matrix[:, keep]
        fit = _glm_fit(x, self.y, self.prior_weights, self.frame.offset, self.family, self.n_trials,
                       25, 1e-8, self.verb)
        return fit["deviance"], fit["rank"]

    def anova_type(self, kind: int, test: str | None = None) -> pl.DataFrame:
        """Type II or III tests, as car's ``Anova(type=)``: likelihood ratio
        chi-squared tests for binomial and Poisson models, F tests on the
        deviance (with the model's dispersion) for the others."""
        special = _special()
        test = test or ("LR" if self.family.dispersion is not None else "F")
        if test not in ("LR", "F"):
            raise ValueError("test must be 'LR' or 'F' for type II and III tables")
        rows: dict[str, list[Any]] = {k: [] for k in ("term", "df", "statistic", "p_value")}
        for label, smaller, larger in _hypotheses(self.design, self.formula, kind):
            dev_small, rank_small = self._deviance_without(smaller)
            dev_large, rank_large = self._deviance_without(larger)
            df = rank_large - rank_small
            if df == 0:
                continue
            diff = dev_small - dev_large
            if test == "LR":
                stat = diff / self.scale
                p = float(special.chdtrc(df, stat))
            else:
                stat = (diff / df) / self.scale
                p = float(special.fdtrc(df, self.df_residual, stat))
            rows["term"].append(label)
            rows["df"].append(float(df))
            rows["statistic"].append(stat)
            rows["p_value"].append(p)
        return self._with_key(pl.DataFrame(rows))

    def __repr__(self) -> str:
        table = self._coef_table(False, 0.95, False).select("term", "estimate", "std_error", "statistic", "p_value")
        head = f"glm({self.formula.text}, family = {self.family!r})" + _key_text(self.key)
        disp = (f"dispersion {self.scale:.4g} (estimated)" if self.family.dispersion is None
                else f"dispersion {self.scale:g}")
        foot = (f"null deviance {self.null_deviance:.5g} on {self.df_null} df · residual deviance "
                f"{self.deviance:.5g} on {self.df_residual} df\n"
                f"AIC {self.aic:.5g} · {disp} · {self.iter} Fisher scoring iterations")
        return f"{head}\n{_table_text(table)}\n{foot}{self._aliased_note()}"


def _ties_mean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R's ``regularize.values(ties = mean)``: sort by x, average tied x."""
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    ux, start = np.unique(x, return_index=True)
    sums = np.add.reduceat(y, start)
    counts = np.diff(np.r_[start, len(x)])
    return ux, sums / counts


def _spline(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R's ``spline(x, y)``: the FMM cubic spline through the points,
    evaluated at ``3 * length(x)`` equally spaced points."""
    x, y = _ties_mean(x, y)
    n = len(x)
    xout = np.linspace(x[0], x[-1], 3 * n)
    b, c, d = np.zeros(n), np.zeros(n), np.zeros(n)
    if n < 2:
        return xout, np.full(len(xout), y[0] if n else np.nan)
    if n < 3:
        b[:] = (y[1] - y[0]) / (x[1] - x[0])
    else:
        # Forsythe, Malcolm & Moler, as R's fmm_spline (0-based here).
        d[0] = x[1] - x[0]
        c[1] = (y[1] - y[0]) / d[0]
        for i in range(1, n - 1):
            d[i] = x[i + 1] - x[i]
            b[i] = 2.0 * (d[i - 1] + d[i])
            c[i + 1] = (y[i + 1] - y[i]) / d[i]
            c[i] = c[i + 1] - c[i]
        b[0], b[n - 1] = -d[0], -d[n - 2]
        c[0] = c[n - 1] = 0.0
        if n > 3:
            c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
            c[n - 1] = c[n - 2] / (x[n - 1] - x[n - 3]) - c[n - 3] / (x[n - 2] - x[n - 4])
            c[0] = c[0] * d[0] * d[0] / (x[3] - x[0])
            c[n - 1] = -c[n - 1] * d[n - 2] * d[n - 2] / (x[n - 1] - x[n - 4])
        for i in range(1, n):
            t = d[i - 1] / b[i - 1]
            b[i] -= t * d[i - 1]
            c[i] -= t * c[i - 1]
        c[n - 1] /= b[n - 1]
        for i in range(n - 2, -1, -1):
            c[i] = (c[i] - d[i] * c[i + 1]) / b[i]
        b[n - 1] = (y[n - 1] - y[n - 2]) / d[n - 2] + d[n - 2] * (c[n - 2] + 2.0 * c[n - 1])
        for i in range(n - 1):
            b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
            d[i] = (c[i + 1] - c[i]) / d[i]
            c[i] = 3.0 * c[i]
        c[n - 1] = 3.0 * c[n - 1]
        d[n - 1] = d[n - 2]
    idx = np.clip(np.searchsorted(x, xout, side="right") - 1, 0, n - 1)
    dx = xout - x[idx]
    return xout, y[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx]))


def _approx(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
    """R's ``approx(x, y, xout)``: linear interpolation, NA outside."""
    x, y = _ties_mean(x, y)
    out = np.interp(xout, x, y)
    return np.where((xout < x[0]) | (xout > x[-1]), np.nan, out)


def _leave_one_out_sigma(e: np.ndarray, h: np.ndarray, rank: int) -> np.ndarray:
    """R's ``influence()$sigma``: the residual scale without each row."""
    e = np.where(np.abs(e) < 100 * _EPS * np.median(np.abs(e)), 0.0, e)
    return np.sqrt((np.sum(e**2) - e**2 / (1 - h)) / (len(e) - rank - 1))


def _deviance_tests(df, dev, scale, df_scale, test, family) -> dict[str, list]:
    special = _special()
    if test is False or test is None:
        return {}
    if test == "Chisq":
        p = [None if d is None or d == 0 else float(special.chdtrc(d, dv / scale)) for d, dv in zip(df, dev)]
        return {"p_value": p}
    if test == "F":
        if family.dispersion is not None:
            _warn(f"using F test with a '{family.family}' family is inappropriate")
        fval = [None if d is None or d == 0 else (dv / d) / scale for d, dv in zip(df, dev)]
        p = [None if f is None else float(special.fdtrc(d, df_scale, f)) for d, f in zip(df, fval)]
        return {"statistic": fval, "p_value": p}
    raise ValueError("test must be 'Chisq', 'F', or False")


def _glm_fit(x, y, w, offset, family, n_trials, maxit, epsilon, verb, etastart=None) -> dict[str, Any]:
    """R's ``glm.fit``: IRLS from R's starting values (or ``etastart``);
    standard errors from the last iteration's QR."""
    link = family.link
    nobs, nvars = x.shape
    if family.family in ("binomial", "quasibinomial"):
        mustart = (w * y + 0.5) / (w + 1) if np.all(n_trials == 1) else (n_trials * y + 0.5) / (n_trials + 1)
    elif family.family in ("poisson", "quasipoisson"):
        if np.any(y < 0):
            raise ValueError(f"{verb}(): negative values not allowed for the 'Poisson' family")
        mustart = y + 0.1
    elif family.family == "Gamma":
        if np.any(y <= 0):
            raise ValueError(f"{verb}(): non-positive values not allowed for the 'Gamma' family")
        mustart = y.copy()
    elif family.family == "inverse.gaussian":
        if np.any(y <= 0):
            raise ValueError(f"{verb}(): positive values only are allowed for the 'inverse.gaussian' family")
        mustart = y.copy()
    else:
        if (link.name == "inverse" and np.any(y == 0)) or (link.name == "log" and np.any(y <= 0)):
            raise ValueError(f"{verb}(): cannot find valid starting values: please specify some")
        mustart = y.copy()
    eta = link.linkfun(mustart) if etastart is None else np.asarray(etastart, dtype=float)
    mu = link.linkinv(eta)
    if not (link.valideta(eta) and family.validmu(mu)):
        raise ValueError(f"{verb}(): cannot find valid starting values: please specify some")
    devold = float(np.sum(family.dev_resids(y, mu, w)))
    coefold = None
    conv = boundary = False
    start = np.zeros(nvars)
    good = w > 0
    qr = _QR([], np.zeros((nobs, 0)), np.zeros((0, 0)), np.zeros(0))
    ww = np.zeros(0)
    it = 0
    for it in range(1, maxit + 1):
        mu_eta = link.mu_eta(eta)
        good = (w > 0) & (mu_eta != 0)
        if not np.any(good):
            conv = False
            _warn(f"no observations informative at iteration {it}")
            break
        var = family.variance(mu)
        if np.any(var[good] == 0):
            raise ValueError(f"{verb}(): 0s in V(mu)")
        z = (eta - offset)[good] + (y - mu)[good] / mu_eta[good]
        ww = np.sqrt(w[good] * mu_eta[good] ** 2 / var[good])
        qr = _qr(x[good] * ww[:, None], min(1e-7, epsilon / 1000), z * ww)
        start = np.zeros(nvars)
        start[qr.kept] = qr.coefficients()
        eta = x @ start + offset
        mu = link.linkinv(eta)
        dev = float(np.sum(family.dev_resids(y, mu, w)))
        boundary = False
        if not math.isfinite(dev):
            if coefold is None:
                raise ValueError(f"{verb}(): no valid set of coefficients has been found: please supply starting values")
            _warn("step size truncated due to divergence")
            for _ in range(maxit):
                start = (start + coefold) / 2
                eta = x @ start + offset
                mu = link.linkinv(eta)
                dev = float(np.sum(family.dev_resids(y, mu, w)))
                if math.isfinite(dev):
                    break
            boundary = True
        if not (link.valideta(eta) and family.validmu(mu)):
            if coefold is None:
                raise ValueError(f"{verb}(): no valid set of coefficients has been found: please supply starting values")
            _warn("step size truncated: out of bounds")
            for _ in range(maxit):
                start = (start + coefold) / 2
                eta = x @ start + offset
                mu = link.linkinv(eta)
                if link.valideta(eta) and family.validmu(mu):
                    break
            boundary = True
            dev = float(np.sum(family.dev_resids(y, mu, w)))
        if abs(dev - devold) / (0.1 + abs(dev)) < epsilon:
            conv = True
            break
        devold = dev
        coefold = start.copy()
    if not conv:
        _warn("glm.fit: algorithm did not converge")
    if boundary:
        _warn("glm.fit: algorithm stopped at boundary value")
    if family.family == "binomial" and (np.any(mu > 1 - 10 * _EPS) or np.any(mu < 10 * _EPS)):
        _warn("glm.fit: fitted probabilities numerically 0 or 1 occurred")
    if family.family == "poisson" and np.any(mu < 10 * _EPS):
        _warn("glm.fit: fitted rates numerically 0 occurred")
    coef = start.copy()
    coef[[j for j in range(nvars) if j not in qr.kept]] = np.nan
    working_weights = np.zeros(nobs)
    working_weights[good] = ww**2
    intercept = bool(nvars and np.all(x[:, 0] == 1))
    wtdmu = np.sum(w * y) / np.sum(w) if intercept else link.linkinv(offset)
    n_ok = nobs - int(np.sum(w == 0))
    rank = len(qr.kept)
    return {
        "coef": coef, "qr": qr, "rank": rank, "eta": eta, "mu": mu, "deviance": dev,
        "null_deviance": float(np.sum(family.dev_resids(y, wtdmu, w))),
        "working_weights": working_weights, "working_residuals": (y - mu) / link.mu_eta(eta),
        "good": good, "df_residual": n_ok - rank, "df_null": n_ok - int(intercept),
        "aic": family.aic(y, n_trials, mu, w, dev) + 2 * rank, "iter": it, "converged": conv,
    }


# printing ---------------------------------------------------------------------------


def _table_text(frame: pl.DataFrame) -> str:
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_rows=-1, fmt_str_lengths=80, tbl_width_chars=160):
        return str(frame)


def _p_text(p: float) -> str:
    return "p < 2.2e-16" if p < 2.2e-16 else f"p = {p:.4g}"


def _key_text(key: dict[str, Any]) -> str:
    return "" if not key else "   [" + ", ".join(f"{k} = {v!r}" for k, v in key.items()) + "]"


# grouped models -------------------------------------------------------------------------


class Models:
    """One model per group; its tables stack the groups' tables."""

    def __init__(self, fits: list[_Fit], data: Data):
        self.fits = fits
        self._data = data
        self.groups = list(data._groups)

    def __len__(self) -> int:
        return len(self.fits)

    def __iter__(self):
        return iter(self.fits)

    def __getitem__(self, i: int) -> _Fit:
        return self.fits[i]

    def __tidy__(self, **kwargs: Any) -> Any:
        return tidy(self, **kwargs)

    def _stack(self, frames: list[pl.DataFrame]) -> Any:
        return self._data.write(pl.concat(frames, how="diagonal_relaxed"))

    def __repr__(self) -> str:
        return "\n\n".join(repr(f) for f in self.fits)


def _is_model(value: Any) -> bool:
    return isinstance(value, (_Fit, Models))


def _fits(value: Any) -> list[_Fit]:
    return value.fits if isinstance(value, Models) else [value]


class _ModelStep(Step):
    """A model waiting for data: ``data >> lm("y ~ x")`` returns the fit."""

    __slots__ = ()

    def __rrshift__(self, data: Any) -> Any:
        frame = Data.read(data, self.by, self.name)
        return self._fn(frame)


def _contrasts(contrasts: Any, parsed: Formula, verb: str) -> dict[str, str]:
    if contrasts is None:
        return {}
    variables = parsed.variables()
    spec = {v: contrasts for v in variables} if isinstance(contrasts, str) else dict(contrasts)
    for name, kind in spec.items():
        if kind not in ("treatment", "sum"):
            raise ValueError(f"{verb}(): contrasts must be 'treatment' or 'sum', got {kind!r}")
        if name not in variables:
            raise ValueError(f"{verb}(): contrasts for {name!r}, which is not in the formula")
    return spec


def _model(verb: str, build: Callable[[Formula, _Frame, Data], _Fit], formula: str, weights: Any, offset: Any,
           contrasts: Any = None, na_action: str = "omit"):
    if na_action not in NA_ACTIONS:
        raise ValueError(f"{verb}(): na_action must be one of {list(NA_ACTIONS)}, got {na_action!r}")

    def _apply(data: Data) -> Any:
        parsed = parse_formula(formula, verb, data.columns)
        parsed.contrasts = _contrasts(contrasts, parsed, verb)
        parsed.offset_arg = offset
        frames = _model_frames(data, parsed, weights, offset, verb, na_action)
        fits = [build(parsed, frame, data) for frame in frames]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, verb)


@step
def lm(formula: str, *, weights: Any = None, offset: Any = None, contrasts: Any = None,
       na_action: str = "omit") -> Step:
    """Fit a linear model, as R's ``lm``.

    ``lm(penguins, "body_mass_g ~ flipper_length_mm + species")`` or
    ``penguins >> lm(...)``; ``by=`` (or tidy3's ``group_by``) fits one model
    per group. Rows missing any model variable are dropped. Factors use
    treatment contrasts (first level as reference); ``contrasts="sum"`` (or
    ``{"species": "sum"}``) uses R's ``contr.sum``, as type III ANOVA
    wants. Print the fit for a summary; ``tidy``, ``glance``, ``augment``,
    ``anova`` and ``predict`` give tables.

    ``na_action`` is R's: ``"omit"`` (the default) leaves the dropped rows
    out of ``augment`` and ``predict``; ``"exclude"`` keeps every row, with
    missing values where a row was not used, so the results line up with
    the data; ``"fail"`` raises if any row has a missing value.
    """
    return _model("lm", lambda f, fr, d: LinearModel(f, fr, d), formula, weights, offset, contrasts, na_action)


@step
def glm(formula: str, family: Any = "gaussian", *, weights: Any = None, offset: Any = None,
        contrasts: Any = None, maxit: int = 25, epsilon: float = 1e-8, na_action: str = "omit") -> Step:
    """Fit a generalized linear model, as R's ``glm``.

    ``family`` is ``binomial()``, ``poisson()``, ``Gamma()``,
    ``inverse_gaussian()``, ``gaussian()``, ``quasibinomial()`` or
    ``quasipoisson()`` (with ``link=``), or one of their names. Binomial
    responses are 0/1, logical, a factor (first level is failure), or
    ``cbind(successes, failures)``; proportions take ``weights=`` trials.
    ``na_action`` is as in ``lm``.
    """
    fam = _as_family(family)
    return _model("glm", lambda f, fr, d: GeneralizedLinearModel(f, fr, d, fam, maxit, epsilon),
                  formula, weights, offset, contrasts, na_action)


# broom verbs ------------------------------------------------------------------------------


class _ModelVerb:
    """``fit >> tidy()``: a table from a model on the left of ``>>``."""

    def __init__(self, fn: Callable[[Any], Any], name: str):
        self._fn, self.name = fn, name

    def __rrshift__(self, model: Any) -> Any:
        return self._fn(model)

    def __call__(self, model: Any) -> Any:
        return self._fn(model)


def _table(model: Any, method: str, **kwargs: Any) -> Any:
    fits = _fits(model)
    frames = [getattr(f, method)(**kwargs) for f in fits]
    if isinstance(model, Models):
        return model._stack(frames)
    return fits[0]._write(frames[0])


def tidy(x: Any = None, *args: Any, conf_int: bool = False, conf_level: float = 0.95,
         exponentiate: bool = False, **kwargs: Any) -> Any:
    """A model's coefficients, as broom's ``tidy``: ``term``, ``estimate``,
    ``std_error``, ``statistic``, ``p_value`` (``conf_int=True`` adds
    ``conf_low``/``conf_high``). ``fit >> tidy()`` works too.

    ``vcov=`` uses a robust covariance, as lmtest's ``coeftest(fit, vcov)``:
    a type such as ``"HC3"`` or ``"HC1"``, ``vcov_cl("island")`` for
    clustered, or ``vcov_nw()`` for Newey-West standard errors. ``lm`` fits
    keep t tests on the residual degrees of freedom; ``glm`` fits get z tests.

    Marginal means and contrasts give their tables too, including through
    ``emm >> tidy()``. Set their intervals and scale when creating them.
    Anything else goes to tidy3's ``tidy()`` when tidy3 is
    loaded, so ``from stats3 import *`` keeps tidy3's constructor working.
    """
    def run(model):
        if isinstance(model, (EMMeans, Contrasts)):
            if conf_int or conf_level != 0.95 or exponentiate:
                raise TypeError("tidy(): set intervals and scale in emmeans() or contrast()")
            return model.__tidy__(**kwargs)
        opts = {"conf_int": conf_int, "conf_level": conf_level, **kwargs}
        if exponentiate:
            opts["exponentiate"] = True
        unknown = set(kwargs) - {"conf_method", "vcov"}
        if unknown:
            raise TypeError(f"tidy(): unexpected arguments for a model: {sorted(unknown)}")
        if "conf_method" in opts and not all(isinstance(f, GeneralizedLinearModel) for f in _fits(model)):
            raise TypeError("tidy(): conf_method is for glm fits")
        if opts.get("vcov") is not None:
            as_vcov(opts["vcov"])
        return _table(model, "tidy", **opts)

    if x is None and not args and set(kwargs) <= {"conf_method", "vcov"}:
        return _ModelVerb(run, "tidy")
    if _is_model(x) or isinstance(x, (EMMeans, Contrasts)):
        return run(x)
    if "tidy3" in sys.modules:
        return sys.modules["tidy3"].tidy(x, *args, **kwargs)
    raise TypeError("tidy(): expected a model from lm() or glm(); for data, use tidy3's tidy()")


def glance(x: Any = None, *, vcov: Any = None) -> Any:
    """One row of fit statistics per model, as broom's ``glance``. For an
    ``lm``, ``vcov=`` (as in ``tidy``) makes ``statistic`` and ``p_value`` the
    robust Wald F test of all slopes, as lmtest's ``waldtest(fit, vcov=)``."""
    opts = {} if vcov is None else {"vcov": vcov}
    if vcov is not None:
        as_vcov(vcov)

    def run(model):
        if vcov is not None and not all(isinstance(f, LinearModel) for f in _fits(model)):
            raise TypeError("glance(): vcov is for lm fits")
        return _table(model, "glance", **opts)

    if x is None:
        return _ModelVerb(run, "glance")
    if not _is_model(x):
        raise TypeError("glance(): expected a model from lm() or glm()")
    return run(x)


def vcov(x: Any, vcov: Any = None) -> Any:  # noqa: A002 - R's name
    """The coefficients' covariance matrix, as R's ``vcov`` (or with
    ``vcov=``, the robust one, as sandwich's ``vcovHC``, ``vcovCL`` and
    ``NeweyWest``): a ``term`` column, then one column per coefficient.
    Aliased coefficients are missing, as in R; robust matrices leave them
    out, as sandwich does."""
    if not _is_model(x):
        raise TypeError("vcov(): expected a model from lm() or glm()")
    return _table(x, "vcov_table", vcov=vcov)


def augment(x: Any = None, newdata: Any = None, **kwargs: Any) -> Any:
    """The model's rows with ``.fitted``, ``.resid``, ``.hat``, ``.sigma``,
    ``.cooksd`` and ``.std_resid``, as broom's ``augment``; with ``newdata``,
    the new rows with ``.fitted`` (and ``interval=`` bounds for ``lm``)."""
    def run(model):
        if newdata is not None:
            if isinstance(model, Models):
                raise ValueError("augment(newdata=): predict with one group's model")
            return model.augment(newdata, **kwargs)
        return _table(model, "augment", **kwargs)

    if x is None:
        return _ModelVerb(run, "augment")
    if not _is_model(x):
        raise TypeError("augment(): expected a model from lm() or glm()")
    return run(x)


def predict(model: Any, newdata: Any = None, **kwargs: Any) -> Any:
    """Predictions, as R's ``predict``: ``fit`` (``lwr``/``upr`` with
    ``interval=`` for ``lm``; ``type="response"`` and ``se_fit=`` for
    ``glm``), in the new data's frame type."""
    if isinstance(model, Models):
        raise ValueError("predict(): predict with one group's model, e.g. models[0]")
    if not isinstance(model, _Fit):
        raise TypeError("predict(): expected a model from lm() or glm()")
    return model.predict(newdata, **kwargs)


def anova(*models: Any, test: Any = None, type: Any = 1) -> Any:  # noqa: A002
    """Analysis of variance or deviance, as R's ``anova`` and car's ``Anova``.

    One model: ``type=1`` (default) is R's sequential table by term;
    ``type=2`` and ``type=3`` are car's type II and III tests (for type III
    use ``contrasts="sum"`` in the model, as car advises). Several nested
    models fitted to the same rows: one row per model, with the F test (lm),
    or the chi-squared test for binomial and Poisson glms and the F test for
    the others (R's defaults; ``test="Chisq"`` / ``"F"`` / ``False``).
    ``fit >> anova()`` works too.
    """
    kind = _check_type(type)
    if not models:
        return _ModelVerb(lambda m: anova(m, test=test, type=type), "anova")
    if len(models) == 1:
        model = models[0]
        if not _is_model(model):
            raise TypeError("anova(): expected models from lm() or glm()")
        fits = _fits(model)
        if kind == 1:
            frames = [f.anova_table() if isinstance(f, LinearModel) else f.anova_table(test) for f in fits]
        else:
            frames = [f.anova_type(kind) if isinstance(f, LinearModel) else f.anova_type(kind, test) for f in fits]
        return model._stack(frames) if isinstance(model, Models) else fits[0]._write(frames[0])
    if kind != 1:
        raise ValueError("anova(): type 2 and 3 tables are for one model; compare models with type=1")
    fits = []
    for m in models:
        if isinstance(m, Models) or not isinstance(m, _Fit):
            raise TypeError("anova(): compare ungrouped models from lm() or glm()")
        fits.append(m)
    nobs = {f.n if isinstance(f, LinearModel) else int(np.sum(f.prior_weights != 0)) for f in fits}
    if len(nobs) > 1:
        raise ValueError("anova(): models were not all fitted to the same size of dataset")
    if all(isinstance(f, LinearModel) for f in fits):
        return fits[0]._write(_anova_lm(fits))
    if all(isinstance(f, GeneralizedLinearModel) for f in fits):
        return fits[0]._write(_anova_glm(fits, test))
    raise TypeError("anova(): compare lm fits with lm fits, glm fits with glm fits")


def _anova_lm(fits: list[LinearModel]) -> pl.DataFrame:
    special = _special()
    resdf = [float(f.df_residual) for f in fits]
    rss = [f.rss for f in fits]
    big = int(np.argmin(resdf))
    scale = rss[big] / resdf[big]
    df = [None] + [resdf[i - 1] - resdf[i] for i in range(1, len(fits))]
    ss = [None] + [rss[i - 1] - rss[i] for i in range(1, len(fits))]
    fval = [None if d is None or d == 0 else (s / abs(d)) / scale for d, s in zip(df, ss)]
    p = [None if f is None else float(special.fdtrc(abs(d), resdf[big], f)) for d, f in zip(df, fval)]
    return pl.DataFrame({
        "term": [f.formula.text for f in fits], "df_residual": resdf, "rss": rss,
        "df": df, "sumsq": ss, "statistic": fval, "p_value": p,
    }, schema_overrides={"df": pl.Float64, "sumsq": pl.Float64, "statistic": pl.Float64, "p_value": pl.Float64})


def _anova_glm(fits: list[GeneralizedLinearModel], test: Any) -> pl.DataFrame:
    test = test or fits[0]._default_test()
    resdf = [float(f.df_residual) for f in fits]
    resdev = [f.deviance for f in fits]
    big = int(np.argmin(resdf))
    df = [None] + [resdf[i - 1] - resdf[i] for i in range(1, len(fits))]
    dev = [None] + [resdev[i - 1] - resdev[i] for i in range(1, len(fits))]
    table = {"term": [f.formula.text for f in fits], "df_residual": resdf, "residual_deviance": resdev,
             "df": df, "deviance": dev}
    table.update(_deviance_tests(
        [None if d is None else abs(d) for d in df], [None if d is None else abs(v) if d < 0 else v for d, v in zip(df, dev)],
        fits[big].scale, resdf[big], test, fits[0].family))
    return pl.DataFrame(table, schema_overrides={"df": pl.Float64, "deviance": pl.Float64})
