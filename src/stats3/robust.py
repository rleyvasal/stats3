"""Robust covariance matrices for ``lm`` and ``glm`` fits, as R's sandwich
package: ``vcovHC`` (heteroskedasticity-consistent, HC0-HC5), ``vcovCL``
(clustered, one or more clustering columns) and ``NeweyWest`` (heteroskedasticity
and autocorrelation consistent).

Each is ``bread @ meat @ bread / n`` with sandwich's bread (the model's
unscaled covariance times n) and the meat built from the estimating
functions: the working residuals times the working weights times the model
matrix. ``tidy(fit, vcov=...)`` then tests the coefficients with it, as
lmtest's ``coeftest`` and ``coefci`` do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from stats3._data import expression

HC_TYPES = ("const", "HC", "HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5")
CL_TYPES = ("HC", "HC0", "HC1", "HC2", "HC3")
_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class Vcov:
    """How to estimate a fit's coefficient covariance; see ``vcov_hc``,
    ``vcov_cl`` and ``vcov_nw``."""

    kind: str  # "HC", "CL" or "NW"
    type: str | None = None
    cluster: tuple[Any, ...] = ()
    cadjust: bool = True
    lag: int | None = None
    prewhite: bool = True
    adjust: bool = False
    order_by: Any = None

    @property
    def label(self) -> str:
        if self.kind == "HC":
            return self.type
        if self.kind == "CL":
            return "cluster"
        return "Newey-West"


def vcov_hc(type: str = "HC3") -> Vcov:  # noqa: A002 - R's argument name
    """Heteroskedasticity-consistent covariance, as sandwich's ``vcovHC``.

    ``type`` is ``"HC3"`` (sandwich's default), ``"HC0"`` (White's),
    ``"HC1"`` (Stata's ``robust``), ``"HC2"``, ``"HC4"``, ``"HC4m"``,
    ``"HC5"``, or ``"const"`` (the usual covariance, as sandwich computes it).
    ``tidy(fit, vcov="HC1")`` is short for ``tidy(fit, vcov=vcov_hc("HC1"))``.
    """
    if type not in HC_TYPES:
        raise ValueError(f"vcov_hc(): type must be one of {list(HC_TYPES)}, got {type!r}")
    return Vcov("HC", "HC0" if type == "HC" else type)


def vcov_cl(cluster: Any, type: str | None = None, *, cadjust: bool = True) -> Vcov:  # noqa: A002
    """Clustered covariance, as sandwich's ``vcovCL(fit, cluster = ~ g)``.

    ``cluster`` is a column (or expression), or a list of them for multi-way
    clustering. ``type`` is ``"HC1"`` for ``lm`` and ``"HC0"`` for ``glm`` by
    default, as in sandwich; ``"HC2"`` and ``"HC3"`` are the bias-reduced
    versions. ``cadjust=True`` multiplies by G / (G - 1) for G clusters.
    """
    if type is not None and type not in CL_TYPES:
        raise ValueError(f"vcov_cl(): type must be one of {list(CL_TYPES)}, got {type!r}")
    clusters = tuple(cluster) if isinstance(cluster, (list, tuple)) else (cluster,)
    if not clusters:
        raise ValueError("vcov_cl(): give at least one clustering column")
    return Vcov("CL", "HC0" if type == "HC" else type, clusters, cadjust)


def vcov_nw(
    lag: int | None = None, *, prewhite: bool = True, adjust: bool = False, order_by: Any = None
) -> Vcov:
    """Newey-West covariance, as sandwich's ``NeweyWest``: Bartlett weights
    up to ``lag`` (by default chosen by ``bwNeweyWest``), after VAR(1)
    prewhitening of the estimating functions unless ``prewhite=False``.
    ``adjust=True`` adds the n / (n - k) factor. Rows are taken in the
    data's order, or sorted by ``order_by`` (a column or expression).
    """
    if lag is not None and (isinstance(lag, bool) or not isinstance(lag, int) or lag < 0):
        raise ValueError(f"vcov_nw(): lag must be a whole number >= 0, got {lag!r}")
    return Vcov("NW", None, (), True, lag, bool(prewhite), bool(adjust), order_by)


def as_vcov(value: Any) -> Vcov | None:
    if value is None or isinstance(value, Vcov):
        return value
    if isinstance(value, str):
        if value in ("NW", "NeweyWest"):
            return vcov_nw()
        return vcov_hc(value)
    raise TypeError(
        "vcov must be a type such as 'HC3', or vcov_hc(), vcov_cl() or vcov_nw(); "
        f"got {value!r}"
    )


# the pieces --------------------------------------------------------------------


def _pieces(fit: Any) -> dict[str, Any]:
    """sandwich's ingredients for one fit, over all its rows (zero weights too):
    the model matrix without aliased columns, the working residuals times
    the working weights (``res``), the estimating functions, the hat values,
    the bread, and the weights used for clustered HC2/HC3."""
    from stats3.models import GeneralizedLinearModel

    kept = fit.qr.kept
    x = fit.design.matrix[:, kept]
    n_used = fit.rank + fit.df_residual
    if isinstance(fit, GeneralizedLinearModel):
        wres = fit.working_residuals * fit.working_weights
        fixed = fit.family.family in ("poisson", "binomial")
        dispersion = 1.0 if fixed else float(np.sum(wres**2) / np.sum(fit.working_weights))
        res = wres / dispersion
        hat = np.zeros(len(res))
        hat[fit.good] = fit.qr.hat()
        bread = fit.cov_unscaled * n_used * dispersion
        weights = fit.working_weights
        glm = True
    else:
        res = fit.residuals * fit.weights
        hat = fit.qr.hat()
        bread = fit.cov_unscaled * n_used
        weights = fit.weights
        dispersion = 1.0
        glm = False
    ef = res[:, None] * x
    zero = np.all(np.abs(ef) < _EPS, axis=1)
    res = np.where(zero, 0.0, res)
    return {"x": x, "res": res, "ef": ef, "hat": hat, "bread": bread, "weights": weights,
            "zero": zero, "glm": glm, "fit": fit, "dispersion": dispersion}


def _sandwich(parts: dict[str, Any], meat: np.ndarray) -> np.ndarray:
    bread = parts["bread"]
    return bread @ meat @ bread / parts["x"].shape[0]


def _meat_hc(parts: dict[str, Any], type_: str) -> np.ndarray:
    x, res, h = parts["x"], parts["res"].copy(), parts["hat"]
    n, k = x.shape
    df = n - k
    if type_ == "const" and np.any(parts["zero"]):
        # sandwich's fallback when some rows have no information.
        fit = parts["fit"]
        if parts["glm"]:
            res = fit.working_residuals * fit.working_weights
            if fit.family.family not in ("poisson", "binomial"):
                res = res * np.sum(fit.working_weights) / np.sum(res**2)
        else:
            res = fit.residuals * fit.weights
    p = int(round(float(np.sum(h))))
    with np.errstate(divide="ignore", invalid="ignore"):
        if type_ == "const":
            omega = np.full(n, np.sum(res**2) / df)
        elif type_ == "HC0":
            omega = res**2
        elif type_ == "HC1":
            omega = res**2 * n / df
        elif type_ == "HC2":
            omega = res**2 / (1 - h)
        elif type_ == "HC3":
            omega = res**2 / (1 - h) ** 2
        elif type_ == "HC4":
            omega = res**2 / (1 - h) ** np.minimum(4, n * h / p)
        elif type_ == "HC4m":
            delta = np.minimum(1, n * h / p) + np.minimum(1.5, n * h / p)
            omega = res**2 / (1 - h) ** delta
        else:  # HC5
            delta = np.minimum(n * h / p, np.maximum(4, n * 0.7 * np.max(h) / p))
            omega = res**2 / np.sqrt((1 - h) ** delta)
    if type_ != "const":
        close = h > 1 - math.sqrt(_EPS)
        if np.any(close):
            from stats3.core import _warn

            rows = ", ".join(str(i + 1) for i in np.flatnonzero(close)[:10])
            what = ("covariances become (close to) singular if hat values are (close to) 1"
                    if type_ in ("HC0", "HC1") else
                    "covariances are numerically unstable for hat values close to 1 "
                    "(and undefined if exactly 1)")
            _warn(f"{type_} {what} as for observation(s) {rows}")
    scaled = np.sqrt(omega)[:, None] * x
    return scaled.T @ scaled / n


def _codes(values: Sequence[Any]) -> np.ndarray:
    """Integer codes for a cluster's values (any type)."""
    index: dict[Any, int] = {}
    return np.array([index.setdefault(v, len(index)) for v in values])


def _power_i_minus_h(xj: np.ndarray, xx1: np.ndarray, wj: np.ndarray, power: float) -> np.ndarray:
    """(I - H W)^power for one cluster, H = X (X'WX)^-1 X', as sandwich's
    ``matrixpower``, with eigenvalues below eps^(1/1.3) set to 0. I - H W
    is similar to the symmetric I - W^1/2 H W^1/2, whose eigendecomposition
    is exact; a general one of I - H W can fail when eigenvalues repeat."""
    sw = np.sqrt(wj)
    s = np.eye(len(wj)) - (sw[:, None] * xj) @ xx1 @ (xj.T * sw[None, :])
    if s.shape == (1, 1):
        return s**power
    values, vectors = np.linalg.eigh(s)
    values = np.where(values < _EPS ** (1 / 1.3), 0.0, values)
    with np.errstate(divide="ignore"):
        powered = values**power
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 / sw)[:, None] * ((vectors * powered) @ vectors.T) * sw[None, :]


def _meat_cl(parts: dict[str, Any], clusters: list[np.ndarray], type_: str | None, cadjust: bool) -> np.ndarray:
    import itertools

    x, ef, res = parts["x"], parts["ef"], parts["res"]
    n, k = ef.shape
    if type_ is None:
        type_ = "HC0" if parts["glm"] else "HC1"
    p = len(clusters)
    combos = [c for size in range(1, p + 1) for c in itertools.combinations(range(p), size)]
    signs = [(-1) ** (len(c) + 1) for c in combos]
    codes = []
    for combo in combos:
        if len(combo) == 1:
            codes.append(clusters[combo[0]])
        else:
            codes.append(_codes(list(zip(*(clusters[i] for i in combo)))))
    g = [len(np.unique(c)) for c in codes]
    if type_ in ("HC2", "HC3") and not all(gi == n for gi in g):
        xx1 = parts["fit"].cov_unscaled
        w = parts["weights"]
    meat = np.zeros((k, k))
    for combo_codes, gi, sign in zip(codes, g, signs):
        efi = ef
        adj = gi / (gi - 1) if cadjust else 1.0
        if type_ in ("HC2", "HC3"):
            if gi == n:
                h = parts["hat"]
                efi = efi / np.sqrt(1 - h)[:, None] if type_ == "HC2" else efi / (1 - h)[:, None]
            else:
                efi = efi.copy()
                for j in np.unique(combo_codes):
                    ij = np.flatnonzero(combo_codes == j)
                    xj = x[ij]
                    if type_ == "HC2":
                        hij = _power_i_minus_h(xj, xx1, w[ij], -0.5)
                    else:
                        eye = np.eye(len(ij))
                        hij = np.linalg.solve(eye - xj @ xx1 @ xj.T * w[ij][None, :], eye)
                    efi[ij] = (hij @ res[ij])[:, None] * xj
            efi = math.sqrt((gi - 1) / gi) * efi
        if gi < n:
            sums = np.zeros((gi, k))
            _, inverse = np.unique(combo_codes, return_inverse=True)
            np.add.at(sums, inverse, efi)
            efi = sums
        meat = meat + sign * adj * (efi.T @ efi) / n
    if type_ == "HC1":
        meat = (n - 1) / (n - k) * meat
    return meat


def _var1(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R's ``ar(u, order.max = 1, aic = FALSE, demean = FALSE, method = "ols")``:
    the VAR(1) coefficient matrix and the residuals (first row dropped)."""
    lagged, current = u[:-1], u[1:]
    coef = np.linalg.lstsq(lagged, current, rcond=None)[0]  # current ~ lagged @ coef
    return coef.T, current - lagged @ coef


def _bw_newey_west(parts: dict[str, Any], u: np.ndarray, prewhite: bool) -> float:
    """sandwich's ``bwNeweyWest`` for the Bartlett kernel."""
    n, k = u.shape
    names = [parts["fit"].design.names[i] for i in parts["fit"].qr.kept]
    weights = np.ones(k)
    if "(Intercept)" in names:
        weights[names.index("(Intercept)")] = 0
    else:
        weights[np.sum((u - parts["res"][:, None]) ** 2, axis=0) < 1e-16] = 0
    if np.all(weights <= 0):
        weights = np.ones(k)
    if k < 2:
        weights = np.ones(1)
    m = math.floor((3 if prewhite else 4) * (n / 100) ** (2 / 9))
    if prewhite:
        u = _var1(u)[1]
        n -= 1
    hw = u @ weights
    sigma = np.array([np.sum(hw[: n - j] * hw[j:]) / n for j in range(m + 1)])
    s0 = sigma[0] + 2 * np.sum(sigma[1:])
    s1 = 2 * np.sum(np.arange(1, m + 1) * sigma[1:])
    return 1.1447 * ((s1 / s0) ** 2) ** (1 / 3) * (n + int(prewhite)) ** (1 / 3)


def _meat_nw(parts: dict[str, Any], spec: Vcov, order: np.ndarray | None) -> np.ndarray:
    u = parts["ef"] if order is None else parts["ef"][order]
    lag = spec.lag
    if lag is None:
        # As sandwich, the no-intercept check compares the sorted estimating
        # functions with the residuals in the fit's own order.
        lag = math.floor(_bw_newey_west(parts, u, spec.prewhite))
    weights = 1 - np.arange(lag + 2) / (lag + 1)
    n_orig, k = u.shape
    if spec.prewhite:
        coef, u = _var1(u)
        d = np.linalg.inv(np.eye(k) - coef)
    n = u.shape[0]
    weights = weights[:n]
    utu = 0.5 * (u.T @ u) * weights[0]
    for ii in range(1, len(weights)):
        utu = utu + weights[ii] * (u[: n - ii].T @ u[ii:])
    utu = utu + utu.T
    if spec.adjust:
        utu = n_orig / (n_orig - k) * utu
    if spec.prewhite:
        utu = d @ utu @ d.T
    return utu / n_orig


def _column_values(fit: Any, spec: Any, verb: str) -> list[Any]:
    """A column or expression's values on the rows the fit used."""
    values = fit._data._lf.select(expression(spec).alias("__v")).collect()["__v"]
    values = values.gather(fit.frame.index)
    if values.null_count() or (values.dtype.is_float() and values.is_nan().any()):
        raise ValueError(
            f"{verb}: cannot handle missing values in the clustering or ordering column; "
            "refit the model without those rows"
        )
    return values.to_list()


def covariance(fit: Any, spec: Vcov) -> np.ndarray:
    """The robust covariance of the estimated (non-aliased) coefficients."""
    parts = _pieces(fit)
    if spec.kind == "HC":
        meat = _meat_hc(parts, spec.type)
    elif spec.kind == "CL":
        clusters = [_codes(_column_values(fit, c, "vcov_cl()")) for c in spec.cluster]
        meat = _meat_cl(parts, clusters, spec.type, spec.cadjust)
    else:
        order = None
        if spec.order_by is not None:
            z = _column_values(fit, spec.order_by, "vcov_nw()")
            order = np.argsort(np.array(z), kind="stable")
        meat = _meat_nw(parts, spec, order)
    return _sandwich(parts, meat)
