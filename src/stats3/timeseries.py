"""Base R's time series core: ``acf``, ``pacf``, ``ccf``, ``Box.test`` and ``arima``.

A series is a column, in the frame's row order or ordered by ``order_by=``;
``by=`` (or tidy3's ``group_by``) gives one series per group. Missing values
stay in place, as in a ``ts``: what happens to them is ``na_action``'s
choice, as in R.

``arima`` is a port of R's: the Gardner/Jones state-space form of
``stats:::makeARIMA``, its Kalman filter likelihood (``ARIMA_Like``), its
conditional sum of squares, parameter transformations and forecasts. The
stationary covariance that R's ``getQ0`` computes is solved exactly here.
R's ``optim`` minimises with finite-difference gradients and stops at its
own tolerance; here the optimum is found with complex-step gradients to full
precision. The standard errors follow R's recipe at that optimum: the
Hessian of ``optimHess`` (central differences of a central-difference
gradient, step ``1e-3`` on R's parameter scale) and R's forward-difference
Jacobian of the AR transformation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Data, Step, expression, is_frame, step
from stats3.checks import _group_values
from stats3.core import _check_conf_level, _column, _warn
from stats3.models import Models, _Fit, _table_text

NA_ACTIONS = ("fail", "pass", "omit", "contiguous")
DOUBLE_XMAX = np.finfo(float).max


# series ------------------------------------------------------------------------------


def _ordered(tf: Data, order_by: Any, verb: str) -> pl.LazyFrame:
    """The input rows in time order: as they are, or sorted by ``order_by``."""
    if order_by is None:
        return tf._lf
    name = _column(tf, order_by, "order_by", verb)
    if tf._lf.select(pl.col(name).null_count()).collect().item():
        raise ValueError(f"{verb}(): order_by column {name!r} has missing values")
    return tf._lf.sort(name, maintain_order=True)


def _series(tf: Data, columns: dict[str, Any], order_by: Any, verb: str):
    """Per group (in dplyr order): each column's values in time order, with
    missing values as NaN."""
    groups = list(tf._groups or [])
    lf = _ordered(tf, order_by, verb)
    exprs = {role: expression(spec).cast(pl.Float64) for role, spec in columns.items()}
    return groups, *_group_values(tf, groups, exprs, pl.lit(True), lf)


def _na_action(x: np.ndarray, na_action: str, verb: str, label: str = "") -> np.ndarray:
    """R's na.fail, na.pass, na.omit (on a ts) and na.contiguous, on the rows
    of a series or of a matrix of series."""
    if na_action not in NA_ACTIONS:
        raise ValueError(f"{verb}(): na_action must be one of {list(NA_ACTIONS)}, got {na_action!r}")
    missing = np.isnan(x) if x.ndim == 1 else np.isnan(x).any(axis=1)
    if na_action == "pass" or not missing.any():
        return x
    if na_action == "fail":
        raise ValueError(f"{verb}(): missing values in object{label}")
    good = np.flatnonzero(~missing)
    if not len(good):
        raise ValueError(f"{verb}(): all times contain an NA{label}")
    if na_action == "omit":
        x = x[good[0]:good[-1] + 1]
        if (np.isnan(x) if x.ndim == 1 else np.isnan(x).any(axis=1)).any():
            raise ValueError(f"{verb}(): time series contains internal NAs{label}")
        return x
    edges = np.flatnonzero(np.r_[True, missing, True])
    lengths = np.diff(edges) - 1
    first = int(np.argmax(lengths))  # the first longest, as R
    return x[edges[first]:edges[first] + lengths[first]]


def _rows_per_group(table: pl.DataFrame, groups: list[str], parts: list[dict[str, Any]]) -> pl.DataFrame:
    """Stack per-group tables, each led by its group's key columns."""
    frames = []
    for i, part in enumerate(parts):
        frame = pl.DataFrame(part)
        if groups:
            keys = table.select(groups).slice(i, 1)
            frame = pl.concat([pl.concat([keys] * frame.height) if frame.height else keys.clear(), frame],
                              how="horizontal")
        frames.append(frame)
    return pl.concat(frames, how="vertical_relaxed")


# correlation functions -------------------------------------------------------------


def _check_lag_max(lag_max: Any, verb: str) -> None:
    if lag_max is not None and (isinstance(lag_max, bool) or not isinstance(lag_max, (int, np.integer))):
        raise ValueError(f"{verb}(): lag_max must be a whole number")


def _mean(x: np.ndarray) -> np.ndarray:
    """Column means of the values present, correctly rounded (R's colMeans
    sums in long double; a level far from zero shows the difference)."""
    return np.array([math.fsum(c[~np.isnan(c)]) / np.sum(~np.isnan(c)) for c in np.atleast_2d(x.T)])


def _acf_matrix(x: np.ndarray, lag_max: int, correlation: bool, demean: bool) -> np.ndarray:
    """R's C_acf: lagged products over the pairs present, divided by the pair
    count plus the lag (n when nothing is missing). ``out[k, u, v]`` pairs
    ``x[t + k, u]`` with ``x[t, v]``."""
    n, nser = x.shape
    if demean:
        x = x - _mean(x)
    out = np.empty((lag_max + 1, nser, nser))
    present = ~np.isnan(x)
    filled = np.where(present, x, 0.0)
    for lag in range(lag_max + 1):
        for u in range(nser):
            for v in range(nser):
                both = present[lag:, u] & present[:n - lag, v]
                count = int(both.sum())
                total = float(filled[lag:, u] @ filled[:n - lag, v])
                out[lag, u, v] = total / (count + lag) if count else np.nan
    if correlation:
        if n == 1:
            out[0] = np.eye(nser)
        else:
            se = np.sqrt(np.diagonal(out[0]))
            out = np.clip(out / np.outer(se, se), -1.0, 1.0)
    return out


def _pacf_values(cor: np.ndarray, lag_max: int) -> np.ndarray:
    """R's uni_pacf: the Durbin-Levinson recursion on the autocorrelations."""
    p = np.empty(lag_max)
    w = np.empty(lag_max)
    w[0] = p[0] = cor[1]
    for ll in range(1, lag_max):
        a = cor[ll + 1] - w[:ll] @ cor[ll:0:-1]
        b = 1.0 - w[:ll] @ cor[1:ll + 1]
        p[ll] = c = a / b
        if ll + 1 == lag_max:
            break
        w[ll] = c
        w[:ll] = w[:ll] - c * w[:ll][::-1]
    return p


def _bounds(n_used: int, conf_level: float, values: np.ndarray, ci_type: str):
    """plot.acf's white-noise bounds, or its MA(k - 1) bounds at lag k."""
    from scipy.special import ndtri

    width = ndtri((1 + conf_level) / 2) / math.sqrt(n_used)
    if ci_type == "white":
        bound = np.full(len(values), width)
    else:
        bound = np.r_[np.nan, width * np.sqrt(np.cumsum(np.r_[1.0, 2 * values[1:-1] ** 2]))]
    return -bound, bound


def _check_ci(ci_type: str, conf_level: float, type_: str, verb: str) -> None:
    if ci_type not in ("white", "ma"):
        raise ValueError(f"{verb}(): ci_type must be 'white' or 'ma', got {ci_type!r}")
    if ci_type == "ma" and type_ != "correlation":
        raise ValueError(f"{verb}(): ci_type='ma' is only for type='correlation'")
    _check_conf_level(conf_level)


@step
def acf(x: Any, *, lag_max: int | None = None, type: str = "correlation",  # noqa: A002 - R's name
        na_action: str = "fail", demean: bool = True, ci_type: str = "white",
        conf_level: float = 0.95, order_by: Any = None) -> Step:
    """Auto-correlation (or covariance, or partial auto-correlation), as R's
    ``acf(x, plot=FALSE)``, one block of rows per series.

    Columns: ``lag`` (in observations: R's lag times the frequency),
    ``acf``, and the bounds ``plot.acf`` draws, ``conf_low``/``conf_high``:
    white noise, ``qnorm((1 + conf_level)/2)/sqrt(n)``, or with
    ``ci_type="ma"`` those for an MA(k - 1) process at lag k. Covariances
    have no bounds. ``lag_max`` defaults to R's ``10 * log10(n)``.
    ``type="partial"`` is ``pacf``. ``na_action`` is R's: ``"fail"`` (the
    default) raises on a missing value, ``"pass"`` uses the pairs present,
    ``"omit"`` drops leading and trailing missing values (internal ones
    raise, as ``na.omit`` on a ts), ``"contiguous"`` keeps the longest
    stretch without them.
    """
    if type not in ("correlation", "covariance", "partial"):
        raise ValueError(f"acf(): type must be 'correlation', 'covariance' or 'partial', got {type!r}")
    _check_lag_max(lag_max, "acf")
    _check_ci(ci_type, conf_level, type, "acf")
    if type == "partial":
        return _pacf_step(x, lag_max, na_action, ci_type, conf_level, order_by, "acf")

    def _apply(tf: Data) -> pl.DataFrame:
        groups, table, rows = _series(tf, {"x": x}, order_by, "acf")
        parts = []
        for row in rows:
            values = _na_action(row["x"], na_action, "acf", row["__label"])
            n = len(values)
            lags = int(math.floor(10 * math.log10(n))) if lag_max is None else int(lag_max)
            lags = min(lags, n - 1)
            if lags < 0:
                raise ValueError(f"acf(): 'lag_max' must be at least 0{row['__label']}")
            out = _acf_matrix(values[:, None], lags, type == "correlation", demean)[:, 0, 0]
            low, high = _bounds(n, conf_level, out, ci_type)
            if type == "covariance":
                low = high = np.full(len(out), np.nan)
            parts.append({"lag": np.arange(lags + 1), "acf": out, "conf_low": low, "conf_high": high})
        return _finish_nan(_rows_per_group(table, groups, parts))

    return Step(_apply, "acf")


def _finish_nan(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col(c).fill_nan(None) for c in ("acf", "conf_low", "conf_high"))


def _pacf_step(x, lag_max, na_action, ci_type, conf_level, order_by, verb) -> Step:
    def _apply(tf: Data) -> pl.DataFrame:
        groups, table, rows = _series(tf, {"x": x}, order_by, verb)
        parts = []
        for row in rows:
            values = _na_action(row["x"], na_action, verb, row["__label"])
            n = len(values)
            lags = int(math.floor(10 * math.log10(n))) if lag_max is None else int(lag_max)
            lags = min(lags, n - 1)
            if lags < 1:
                raise ValueError(f"{verb}(): 'lag_max' must be at least 1{row['__label']}")
            cor = _acf_matrix((values - _mean(values[:, None]))[:, None], lags, True, True)[:, 0, 0]
            out = _pacf_values(cor, lags)
            low, high = _bounds(n, conf_level, np.r_[1.0, out], ci_type)
            parts.append({"lag": np.arange(1, lags + 1), "acf": out, "conf_low": low[1:], "conf_high": high[1:]})
        return _finish_nan(_rows_per_group(table, groups, parts))

    return Step(_apply, verb)


@step
def pacf(x: Any, *, lag_max: int | None = None, na_action: str = "fail", ci_type: str = "white",
         conf_level: float = 0.95, order_by: Any = None) -> Step:
    """Partial auto-correlations, as R's ``pacf(x, plot=FALSE)``: lags 1 to
    ``lag_max`` (R's ``10 * log10(n)``) from the Durbin-Levinson recursion.
    Columns and ``na_action`` as in ``acf``."""
    _check_lag_max(lag_max, "pacf")
    _check_ci(ci_type, conf_level, "partial", "pacf")
    return _pacf_step(x, lag_max, na_action, ci_type, conf_level, order_by, "pacf")


@step
def ccf(x: Any, y: Any, *, lag_max: int | None = None, type: str = "correlation",  # noqa: A002
        na_action: str = "fail", conf_level: float = 0.95, order_by: Any = None) -> Step:
    """Cross-correlation (or covariance) of two series, as R's
    ``ccf(x, y, plot=FALSE)``: at lag k, the correlation of ``x[t + k]`` with
    ``y[t]``, for k from ``-lag_max`` to ``lag_max`` (by default R's
    ``10 * log10(n / 2)``). Columns as in ``acf``, with white-noise bounds;
    ``na_action`` applies to the rows of the pair."""
    if type not in ("correlation", "covariance"):
        raise ValueError(f"ccf(): type must be 'correlation' or 'covariance', got {type!r}")
    _check_lag_max(lag_max, "ccf")
    _check_conf_level(conf_level)

    def _apply(tf: Data) -> pl.DataFrame:
        groups, table, rows = _series(tf, {"x": x, "y": y}, order_by, "ccf")
        parts = []
        for row in rows:
            values = _na_action(np.column_stack([row["x"], row["y"]]), na_action, "ccf", row["__label"])
            n = len(values)
            lags = int(math.floor(10 * (math.log10(n) - math.log10(2)))) if lag_max is None else int(lag_max)
            lags = min(lags, n - 1)
            if lags < 0:
                raise ValueError(f"ccf(): 'lag_max' must be at least 0{row['__label']}")
            out = _acf_matrix(values, lags, type == "correlation", True)
            both = np.r_[out[:0:-1, 1, 0], out[:, 0, 1]]
            low, high = _bounds(n, conf_level, both, "white")
            if type == "covariance":
                low = high = np.full(len(both), np.nan)
            parts.append({"lag": np.arange(-lags, lags + 1), "acf": both, "conf_low": low, "conf_high": high})
        return _finish_nan(_rows_per_group(table, groups, parts))

    return Step(_apply, "ccf")


# Box.test ----------------------------------------------------------------------------


def _box(values: np.ndarray, lag: int, type_: str, fitdf: float) -> dict[str, Any]:
    from scipy.special import chdtrc

    n_obs = len(values)
    if lag > n_obs - 1:
        raise ValueError(f"box_test(): lag must be less than the series length ({n_obs})")
    cor = _acf_matrix(values[:, None], lag, True, True)[1:lag + 1, 0, 0]
    n = float(np.sum(~np.isnan(values)))
    if type_ == "Box-Pierce":
        statistic = n * np.sum(cor ** 2)
    else:
        statistic = n * (n + 2) * np.sum(cor ** 2 / (n - np.arange(1, lag + 1)))
    df = lag - fitdf
    p = float(chdtrc(df, statistic)) if df > 0 else np.nan
    return {"statistic": float(statistic), "df": float(df), "p_value": p}


def box_test(x: Any = None, *args: Any, lag: int = 1, type: str = "Box-Pierce",  # noqa: A002
             fitdf: float = 0, order_by: Any = None, by: Any = None, as_: str | None = None) -> Any:
    """Box-Pierce or Ljung-Box test of no autocorrelation, as R's
    ``Box.test``, one row per series: ``statistic`` (X-squared), ``df``
    (``lag - fitdf``), ``p_value``, ``method``. Missing values are passed,
    as in R. Give a column (``box_test(frame, "x", lag=10)`` or
    ``frame >> box_test("x")``), or an ``arima`` fit to test its residuals
    (one row per group of a grouped fit), as ``tsdiag`` does with
    ``type="Ljung-Box"``; ``fitdf=`` is usually the number of ARMA
    coefficients.
    """
    if type not in ("Box-Pierce", "Ljung-Box"):
        raise ValueError(f"box_test(): type must be 'Box-Pierce' or 'Ljung-Box', got {type!r}")
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError("box_test(): lag must be a positive whole number")
    method = "Box-Pierce test" if type == "Box-Pierce" else "Box-Ljung test"
    if isinstance(x, (ArimaModel, Models)):
        fits = list(x) if isinstance(x, Models) else [x]
        if not all(isinstance(f, ArimaModel) for f in fits):
            raise TypeError("box_test(): expected an arima() fit or a column")
        frames = [f._with_key(pl.DataFrame({**_box(f.residuals, int(lag), type, fitdf), "method": [method]}))
                  for f in fits]
        frame = pl.concat(frames, how="vertical_relaxed").with_columns(pl.col("p_value").fill_nan(None))
        return fits[0]._data.write(frame, as_)

    def make(column: Any) -> Step:
        def _apply(tf: Data) -> pl.DataFrame:
            groups, table, rows = _series(tf, {"x": column}, order_by, "box_test")
            parts = [{**_box(row["x"], int(lag), type, fitdf), "method": [method]} for row in rows]
            return _rows_per_group(table, groups, parts).with_columns(pl.col("p_value").fill_nan(None))

        return Step(_apply, "box_test", by, as_)

    if x is not None and is_frame(x) and not isinstance(x, str):
        if len(args) != 1:
            raise TypeError("box_test(data, x): name the series column")
        return make(args[0])(x)
    if args:
        raise TypeError("box_test(): give one series column")
    return make(x)


# ARIMA: parameters -----------------------------------------------------------------


def _partrans(raw: np.ndarray) -> np.ndarray:
    """R's partrans: tanh to partial autocorrelations, then Durbin-Levinson
    to the coefficients of a stationary AR polynomial."""
    new = np.tanh(raw)
    for j in range(1, len(raw)):
        new[:j] = new[:j] - new[j] * new[:j][::-1]
    return new


def _invpartrans(phi: np.ndarray) -> np.ndarray:
    new = np.array(phi, dtype=float)
    for j in range(len(new) - 1, 0, -1):
        a = new[j]
        new[:j] = (new[:j] + a * new[:j][::-1]) / (1 - a * a)
    return np.arctanh(new)


def _trans_pars(par: np.ndarray, arma: list[int], trans: bool) -> tuple[np.ndarray, np.ndarray]:
    """R's ARIMA_transPars: (phi, theta) of the multiplied-out seasonal model."""
    mp, mq, msp, msq, ns = arma[:5]
    p, q = mp + ns * msp, mq + ns * msq
    params = np.array(par[:mp + mq + msp + msq])
    if trans:
        if mp > 0:
            params[:mp] = _partrans(par[:mp])
        v = mp + mq
        if msp > 0:
            params[v:v + msp] = _partrans(par[v:v + msp])
    phi = np.zeros(p, dtype=params.dtype)
    theta = np.zeros(q, dtype=params.dtype)
    phi[:mp] = params[:mp]
    theta[:mq] = params[mp:mp + mq]
    for j in range(msp):
        s = params[j + mp + mq]
        phi[(j + 1) * ns - 1] += s
        phi[(j + 1) * ns:(j + 1) * ns + mp] -= params[:mp] * s
    for j in range(msq):
        s = params[j + mp + mq + msp]
        theta[(j + 1) * ns - 1] += s
        theta[(j + 1) * ns:(j + 1) * ns + mq] += params[mp:mp + mq] * s
    return phi, theta


def _undo_pars(par: np.ndarray, arma: list[int]) -> np.ndarray:
    mp, mq, msp = arma[:3]
    out = np.array(par, dtype=float)
    if mp > 0:
        out[:mp] = _partrans(par[:mp])
    if msp > 0:
        out[mp + mq:mp + mq + msp] = _partrans(par[mp + mq:mp + mq + msp])
    return out


def _invtrans(par: np.ndarray, arma: list[int]) -> np.ndarray:
    mp, mq, msp = arma[:3]
    out = np.array(par, dtype=float)
    if mp > 0:
        out[:mp] = _invpartrans(par[:mp])
    if msp > 0:
        out[mp + mq:mp + mq + msp] = _invpartrans(par[mp + mq:mp + mq + msp])
    return out


def _grad_trans(par: np.ndarray, arma: list[int]) -> np.ndarray:
    """R's ARIMA_Gradtrans: forward differences, step 1e-3, of the AR
    transformation; ``A[i, j]`` is d new[j] / d raw[i]."""
    eps = 1e-3
    mp, mq, msp = arma[:3]
    a = np.eye(len(par))
    for start, size in ((0, mp), (mp + mq, msp)):
        if size == 0:
            continue
        w1 = np.array(par[start:start + size], dtype=float)
        w2 = _partrans(w1)
        for i in range(size):
            w1[i] += eps
            a[start + i, start:start + size] = (_partrans(w1) - w2) / eps
            w1[i] -= eps
    return a


def _polyroot_moduli(coefs: np.ndarray) -> np.ndarray:
    """Moduli of the roots of ``coefs[0] + coefs[1] z + ...`` (R's polyroot)."""
    return np.abs(np.roots(np.asarray(coefs, dtype=float)[::-1]))


def _ar_check(ar: np.ndarray) -> bool:
    poly = np.r_[1.0, -np.asarray(ar, dtype=float)]
    nonzero = np.flatnonzero(poly != 0)
    p = nonzero[-1]
    if not p:
        return True
    return bool(np.all(_polyroot_moduli(poly[:p + 1]) > 1))


def _ma_invert(ma: np.ndarray) -> np.ndarray:
    """R's maInvert: reflect MA roots inside the unit circle."""
    ma = np.asarray(ma, dtype=float)
    q = len(ma)
    q0 = np.flatnonzero(np.r_[1.0, ma] != 0)[-1]
    if not q0:
        return ma
    roots = np.roots(np.r_[1.0, ma[:q0]][::-1])
    inside = np.abs(roots) < 1
    if not inside.any():
        return ma
    if q0 == 1:
        return np.r_[1 / ma[0], np.zeros(q - q0)]
    roots[inside] = 1 / roots[inside]
    x = np.array([1.0 + 0j])
    for r in roots:
        x = np.r_[x, 0] - np.r_[0, x] / r
    return np.r_[x[1:].real, np.zeros(q - q0)]


# ARIMA: state space ------------------------------------------------------------------


def _q0(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The stationary state covariance (R's getQ0): P = T P T' + V, solved
    exactly. Doubling for large states; works on complex input too."""
    r = t.shape[0]
    if r <= 8:
        system = np.eye(r * r) - np.kron(t, t)
        return np.linalg.solve(system, v.reshape(-1)).reshape(r, r)
    p, a = v.copy(), t.copy()
    for _ in range(80):
        p = p + a @ p @ a.T
        a = a @ a
        if np.max(np.abs(a.real)) < 1e-18:
            return p
    system = np.eye(r * r) - np.kron(t, t)
    return np.linalg.solve(system, v.reshape(-1)).reshape(r, r)


def _make_arima(phi: np.ndarray, theta: np.ndarray, delta: np.ndarray, kappa: float) -> dict[str, Any]:
    """R's makeARIMA."""
    dtype = np.result_type(phi, theta, float)
    p, q = len(phi), len(theta)
    r = max(p, q + 1)
    d = len(delta)
    rd = r + d
    z = np.r_[1.0, np.zeros(r - 1), delta]
    t = np.zeros((rd, rd), dtype=dtype)
    t[:p, 0] = phi
    if r > 1:
        t[np.arange(r - 1), np.arange(1, r)] = 1
    if d > 0:
        t[r, :] = z
        if d > 1:
            t[r + np.arange(1, d), r + np.arange(d - 1)] = 1
    theta = np.r_[theta, np.zeros(max(r - 1 - q, 0), dtype=dtype)]
    rr = np.r_[np.ones(1, dtype=dtype), theta, np.zeros(d, dtype=dtype)]
    v = np.outer(rr, rr)
    pn = np.zeros((rd, rd), dtype=dtype)
    if r > 1:
        pn[:r, :r] = _q0(t[:r, :r], v[:r, :r])
    else:
        pn[0, 0] = 1 / (1 - phi[0] ** 2) if p > 0 else 1.0
    if d > 0:
        pn[r + np.arange(d), r + np.arange(d)] = kappa
    return {"phi": phi, "theta": theta[:q], "Delta": delta, "Z": z, "a": np.zeros(rd, dtype=dtype),
            "P": np.zeros((rd, rd), dtype=dtype), "T": t, "V": v, "h": 0.0, "Pn": pn}


_FIXED_POINT = 4 * float(np.finfo(float).eps)


def _converged(new: np.ndarray, old: np.ndarray) -> bool:
    """Whether a prediction covariance has reached its fixed point to within
    a few units in the last place, in its real and (for complex-step
    derivatives) its imaginary part separately."""
    if not np.iscomplexobj(new):
        return float(np.abs(new - old).max()) <= _FIXED_POINT * float(np.abs(new).max())
    change = new - old
    return (float(np.abs(change.real).max()) <= _FIXED_POINT * float(np.abs(new.real).max())
            and float(np.abs(change.imag).max()) <= _FIXED_POINT * float(np.abs(new.imag).max()))


def _arima_like(y: np.ndarray, mod: dict[str, Any], fast: bool = False) -> tuple[Any, Any, int, np.ndarray]:
    """R's ARIMA_Like: the Kalman filter, with the first prediction variance
    the initial Pn; observations with gain of 1e4 or more (the diffuse
    start of differenced models) are left out of the likelihood. Updates
    the model's state (a, P, Pn) as R does in place, for forecasting.

    ``fast`` (for the optimizer only): once the prediction covariance stops
    changing (after the last missing value), the filter's gain is fixed and
    its innovations obey the model's own recursion,
    theta(L) e = phi(L) Delta(L) y, so the rest of the series goes through
    scipy's lfilter in one call. The innovations agree with the loop's to
    rounding; the state is then not updated, so the final fit uses the loop."""
    from scipy.signal import lfilter, lfiltic

    t, v, z = mod["T"], mod["V"], mod["Z"]
    a, p, pnew = mod["a"].copy(), mod["P"].copy(), mod["Pn"].copy()
    ssq = sumlog = 0.0
    nu = 0
    resid = np.full(len(y), np.nan, dtype=np.result_type(t, y))
    raw = np.zeros(len(y), dtype=resid.dtype)  # innovations before scaling
    if fast:
        missing = np.flatnonzero(np.isnan(np.real(y)))
        after = int(missing[-1]) + 1 if len(missing) else 0
        b_poly = np.convolve(np.r_[1.0, -mod["phi"]], np.r_[1.0, -mod["Delta"]])
        a_poly = np.r_[1.0, mod["theta"]]
        warm = max(len(b_poly), len(a_poly))
    previous = None
    for i, value in enumerate(y):
        anew = t @ a
        if i > 0:
            pnew = t @ p @ t.T + v
            if fast and i >= after + warm and previous is not None and _converged(pnew, previous):
                m = pnew @ z
                gain = z @ m
                if gain.real < 1e4:
                    zi = lfiltic(b_poly, a_poly, raw[i - 1::-1][:len(a_poly) - 1], y[i - 1::-1][:len(b_poly) - 1])
                    tail = lfilter(b_poly, a_poly, y[i:], zi=zi)[0]
                    count = len(tail)
                    nu += count
                    ssq = ssq + np.sum(tail * tail) / gain
                    sumlog = sumlog + count * np.log(gain)
                    resid[i:] = tail / np.sqrt(gain)
                    return ssq, sumlog, nu, resid
            previous = pnew
        if not np.isnan(value):
            resid0 = value - z @ anew
            m = pnew @ z
            gain = z @ m
            if gain.real < 1e4:
                nu += 1
                ssq = ssq + resid0 * resid0 / gain
                sumlog = sumlog + np.log(gain)
            resid[i] = resid0 / np.sqrt(gain)
            raw[i] = resid0
            a = anew + m * (resid0 / gain)
            p = pnew - np.outer(m, m) / gain
        else:
            a, p = anew, pnew
    mod["a"], mod["P"], mod["Pn"] = a, p, pnew
    return ssq, sumlog, nu, resid


def _arima_css(y: np.ndarray, arma: list[int], phi: np.ndarray, theta: np.ndarray, ncond: int):
    """R's ARIMA_CSS: differenced series, then the ARMA recursion from
    ``ncond`` on; residuals before it are 0. Returns (ssq / n, residuals)."""
    w = np.array(y, dtype=np.result_type(y, phi, theta))
    n = len(w)
    for _ in range(arma[5]):
        w[1:] = w[1:] - w[:-1]
    ns = arma[4]
    for _ in range(arma[6]):
        w[ns:] = w[ns:] - w[:-ns]
    from scipy.signal import lfilter, lfiltic

    p, q = len(phi), len(theta)
    resid = np.zeros(n, dtype=w.dtype)
    if ncond < n:
        # The recursion resid[i] = w[i] - phi . w[i-1..] - theta . resid[i-1..]
        # from ncond on, with resid before ncond zero, is one linear filter.
        b_poly, a_poly = np.r_[1.0, -phi], np.r_[1.0, theta]
        zi = lfiltic(b_poly, a_poly, np.zeros(q, dtype=w.dtype), w[ncond - 1::-1][:p] if ncond else np.zeros(0))
        resid[ncond:] = lfilter(b_poly, a_poly, w[ncond:], zi=zi)[0]
    tail = resid[ncond:]
    ok = ~np.isnan(np.real(tail))
    nu = int(ok.sum())
    ssq = np.sum(tail[ok] * tail[ok])
    return ssq / nu, resid


def _kalman_forecast(n_ahead: int, mod: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """R's KalmanForecast: predictions and their variances (in sigma2 units)."""
    t, v, z = mod["T"].real, mod["V"].real, mod["Z"]
    a, p = mod["a"].real.copy(), mod["P"].real.copy()
    forecasts, variances = np.empty(n_ahead), np.empty(n_ahead)
    for i in range(n_ahead):
        a = t @ a
        forecasts[i] = z @ a
        p = t @ p @ t.T + v
        variances[i] = mod["h"] + z @ p @ z
    return forecasts, variances


# ARIMA: optimisation ---------------------------------------------------------------


def _complex_gradient(f, x: np.ndarray) -> np.ndarray:
    """Gradient to machine precision by complex steps."""
    h = 1e-20
    out = np.empty(len(x))
    for j in range(len(x)):
        z = x.astype(complex)
        z[j] += 1j * h
        out[j] = f(z).imag / h
    return out


def _minimize(f, start: np.ndarray, parscale: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """The minimum of ``f`` to full precision: BFGS with exact gradients on
    R's parameter scale, then Newton steps on a differenced gradient."""
    from scipy.optimize import minimize

    if not len(start):
        return start, float(f(start).real), True
    scale = np.asarray(parscale, dtype=float)

    def fs(u):
        value = f(u * scale)
        return float(value.real) if np.isfinite(value.real) else DOUBLE_XMAX

    def gs(u):
        return _complex_gradient(f, u * scale) * scale

    result = minimize(fs, np.asarray(start, dtype=float) / scale, jac=gs, method="BFGS",
                      options={"gtol": 1e-12, "maxiter": 2000})
    u, gradient = result.x, gs(result.x)
    for _ in range(8):
        size = np.max(np.abs(gradient))
        if size < 1e-13:
            break
        step = 1e-5 * np.maximum(1, np.abs(u))
        hessian = np.column_stack([(gs(u + s * e) - gs(u - s * e)) / (2 * s)
                                   for s, e in zip(step, np.eye(len(u)))])
        try:
            proposal = u - np.linalg.solve((hessian + hessian.T) / 2, gradient)
        except np.linalg.LinAlgError:
            break
        new_gradient = gs(proposal)
        if not (np.max(np.abs(new_gradient)) < size and fs(proposal) <= fs(u) + 1e-14 * abs(fs(u))):
            break
        u, gradient = proposal, new_gradient
    converged = bool(np.max(np.abs(gradient)) < 1e-6 * max(1, abs(fs(u))))
    return u * scale, fs(u), converged


def _optim_hess(f, par: np.ndarray, parscale: np.ndarray, ndeps: float = 1e-3) -> np.ndarray:
    """R's optimHess without a gradient: central differences (step ndeps
    in each parameter) of optim's central-difference gradient (step
    ndeps * parscale), symmetrised."""
    n = len(par)

    def gradient(x):
        out = np.empty(n)
        for j in range(n):
            up, down = x.copy(), x.copy()
            up[j] += ndeps * parscale[j]
            down[j] -= ndeps * parscale[j]
            out[j] = (f(up) - f(down)) / (2 * ndeps * parscale[j])
        return out

    h = np.empty((n, n))
    for i in range(n):
        up, down = par.copy(), par.copy()
        up[i] += ndeps
        down[i] -= ndeps
        h[i] = (gradient(up) - gradient(down)) / (2 * ndeps)
    return (h + h.T) / 2


# ARIMA: the fit --------------------------------------------------------------------


def _as_order(value: Any, name: str) -> list[int]:
    try:
        order = [int(v) for v in value]
    except TypeError:
        order = []
    if len(order) != 3 or any(v < 0 for v in order) or any(int(v) != v for v in value):
        raise ValueError(f"arima(): '{name}' must be a non-negative numeric vector of length 3")
    return order


def _diff(x: np.ndarray, lag: int, times: int) -> np.ndarray:
    for _ in range(times):
        x = x[lag:] - x[:-lag]
    return x


def _lm_coef(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """lm(y ~ x - 1, na.action=na.omit): coefficients, standard errors, rank."""
    keep = ~(np.isnan(y) | np.isnan(x).any(axis=1))
    x, y = x[keep], y[keep]
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    df = len(y) - rank
    if rank < x.shape[1] or df <= 0:
        se = np.full(x.shape[1], np.nan)
    else:
        resid = y - x @ coef
        se = np.sqrt(np.diag(np.linalg.inv(x.T @ x)) * (resid @ resid) / df)
    return coef, se, int(rank)


class ArimaModel(_Fit):
    """An ARIMA fit: R's ``arima``, by exact likelihood, CSS, or both."""

    kind = "arima"

    def __init__(self, x: np.ndarray, xreg: np.ndarray | None, xreg_names: list[str], key: dict[str, Any],
                 data: Data, order, seasonal, period, include_mean, transform_pars, fixed, init,
                 method, n_cond, kappa, series: str, label: str):
        self._data, self.key, self.groups = data, key, list(data._groups)
        self.verb, self.series, self.method = "arima", series, method
        self.x = x
        n = len(x)
        self.arma = arma = [order[0], order[2], seasonal[0], seasonal[2], period, order[1], seasonal[1]]
        narma = sum(arma[:4])
        delta = np.array([1.0])
        for _ in range(order[1]):
            delta = np.convolve(delta, [1.0, -1.0])
        for _ in range(seasonal[1]):
            delta = np.convolve(delta, np.r_[1.0, np.zeros(period - 1), -1.0])
        delta = -delta[1:]
        self.delta, self.kappa = delta, kappa
        nd = order[1] + seasonal[1]
        n_used = int(np.sum(~np.isnan(x))) - len(delta)
        xreg = np.empty((n, 0)) if xreg is None else np.asarray(xreg, dtype=float)
        names = list(xreg_names)
        if include_mean and nd == 0:
            xreg = np.column_stack([np.ones(n), xreg])
            names = ["intercept", *names]
        ncxreg = xreg.shape[1]
        if method == "CSS-ML" and (np.isnan(x).any() or np.isnan(xreg).any()):
            method = "ML"
        self.fit_method = method
        if method in ("CSS", "CSS-ML"):
            ncond = order[1] + seasonal[1] * period
            ncond1 = order[0] + period * seasonal[0]
            ncond += max(n_cond, ncond1) if n_cond is not None else ncond1
        else:
            ncond = 0
        if fixed is None:
            fixed = np.full(narma + ncxreg, np.nan)
        else:
            fixed = np.array([np.nan if v is None else v for v in fixed], dtype=float)
            if len(fixed) != narma + ncxreg:
                raise ValueError(f"arima(): wrong length for 'fixed'{label}")
        mask = np.isnan(fixed)
        no_optim = not mask.any()
        if no_optim:
            transform_pars = False
        if transform_pars:
            ar_ind = list(range(arma[0])) + list(range(arma[0] + arma[1], arma[0] + arma[1] + arma[2]))
            if not mask[ar_ind].all():
                _warn("arima(): some AR parameters were fixed: setting transform_pars = False")
                transform_pars = False
        init0 = np.zeros(narma)
        parscale = np.ones(narma)
        rotation = None
        if ncxreg:
            orig_xreg = ncxreg == 1 or not mask[narma:].all()
            if not orig_xreg:
                complete = xreg[~np.isnan(xreg).any(axis=1)]
                rotation = np.linalg.svd(complete, full_matrices=False)[2].T
                xreg = xreg @ rotation
            dx, dxreg = x, xreg
            if order[1] > 0:
                dx, dxreg = _diff(dx, 1, order[1]), _diff(dxreg, 1, order[1])
            if period > 1 and seasonal[1] > 0:
                dx, dxreg = _diff(dx, period, seasonal[1]), _diff(dxreg, period, seasonal[1])
            fitted = _lm_coef(dxreg, dx) if len(dx) > dxreg.shape[1] else None
            if fitted is None or fitted[2] == 0:
                fitted = _lm_coef(xreg, x)
            isna = np.isnan(x) | np.isnan(xreg).any(axis=1)
            n_used = int(np.sum(~isna)) - len(delta)
            init0 = np.r_[init0, fitted[0]]
            parscale = np.r_[parscale, 10 * fitted[1]]
        if n_used <= 0:
            raise ValueError(f"arima(): too few non-missing observations{label}")
        if init is not None:
            init = np.array([np.nan if v is None else v for v in init], dtype=float)
            if len(init) != len(init0):
                raise ValueError(f"arima(): 'init' is of the wrong length{label}")
            init = np.where(np.isnan(init), init0, init)
            if method == "ML":
                if arma[0] > 0 and not _ar_check(init[:arma[0]]):
                    raise ValueError(f"arima(): non-stationary AR part{label}")
                if arma[2] > 0 and not _ar_check(init[arma[0] + arma[1]:arma[0] + arma[1] + arma[2]]):
                    raise ValueError(f"arima(): non-stationary seasonal AR part{label}")
                if transform_pars:
                    init = _invtrans(init, arma)
        else:
            init = init0
        coef = fixed.copy()
        scale = parscale[mask]

        def regressed(par):
            return x - xreg @ par[narma:] if ncxreg else x

        def arma_css(p):
            par = fixed.astype(np.result_type(p, float))
            par[mask] = p
            phi, theta = _trans_pars(par, arma, False)
            return 0.5 * np.log(_arima_css(regressed(par), arma, phi, theta, ncond)[0])

        fast = True  # the steady-state shortcut while optimizing; the reported fit uses the full filter

        def arma_fn(p, trans):
            par = coef.astype(np.result_type(p, float))
            par[mask] = p
            phi, theta = _trans_pars(par, arma, trans)
            try:
                mod = _make_arima(phi, theta, delta, kappa)
            except np.linalg.LinAlgError:
                return DOUBLE_XMAX
            ssq, sumlog, nu, _ = _arima_like(regressed(par), mod, fast)
            return 0.5 * (np.log(ssq / nu) + sumlog / nu)

        self.converged = True
        if method == "CSS":
            par, value, ok = _minimize(arma_css, init[mask], scale)
            self.converged = ok
            if not ok:
                _warn(f"arima(): possible convergence problem{label}")
            coef[mask] = par
            hessian = None if no_optim else _optim_hess(lambda p: float(arma_css(p)), par, scale)
            phi, theta = _trans_pars(coef, arma, False)
            mod = _make_arima(phi, theta, delta, kappa)
            _arima_like(regressed(coef), mod)
            sigma2, resid = _arima_css(regressed(coef), arma, phi, theta, ncond)
            var = None if no_optim else np.linalg.inv(hessian * n_used)
        else:
            if method == "CSS-ML":
                par, _, ok = _minimize(arma_css, init[mask], scale)
                if ok:
                    init = init.copy()
                    init[mask] = par
                if arma[0] > 0 and not _ar_check(init[:arma[0]]):
                    raise ValueError(f"arima(): non-stationary AR part from CSS{label}")
                if arma[2] > 0 and not _ar_check(init[arma[0] + arma[1]:arma[0] + arma[1] + arma[2]]):
                    raise ValueError(f"arima(): non-stationary seasonal AR part from CSS{label}")
                ncond = 0
            if transform_pars:
                init = _invtrans(init, arma)
                if arma[1] > 0:
                    init[arma[0]:arma[0] + arma[1]] = _ma_invert(init[arma[0]:arma[0] + arma[1]])
                if arma[3] > 0:
                    s = sum(arma[:3])
                    init[s:s + arma[3]] = _ma_invert(init[s:s + arma[3]])
            objective = lambda p: arma_fn(p, transform_pars)  # noqa: E731
            par, value, ok = _minimize(objective, init[mask], scale)
            self.converged = ok
            if not ok:
                _warn(f"arima(): possible convergence problem{label}")
            coef[mask] = par
            real_objective = lambda p: float(np.real(objective(p)))  # noqa: E731
            if transform_pars:
                for start, size in ((arma[0], arma[1]), (sum(arma[:3]), arma[3])):
                    if size > 0 and mask[start:start + size].all():
                        coef[start:start + size] = _ma_invert(coef[start:start + size])
                if np.any(coef[mask] != par):
                    par = coef[mask]
                    value = real_objective(par)
                hessian = _optim_hess(real_objective, coef[mask].copy(), scale)
                a = _grad_trans(coef, arma)[np.ix_(mask, mask)]
                var = a.T @ np.linalg.solve(hessian * n_used, a)
                coef = _undo_pars(coef, arma)
            else:
                var = None if no_optim else np.linalg.inv(_optim_hess(real_objective, par, scale) * n_used)
            # The reported value comes from the full filter (the shortcut is used
            # for the search and the Hessian, where it agrees to rounding).
            fast = False
            value = real_objective(par)
            phi, theta = _trans_pars(coef, arma, False)
            mod = _make_arima(phi, theta, delta, kappa)
            ssq, _, _, resid = _arima_like(regressed(coef), mod)
            sigma2 = float(np.real(ssq)) / n_used
        value = float(np.real(value))
        self.minus2loglik = 2 * n_used * value + n_used + n_used * math.log(2 * math.pi)
        self.aic = self.minus2loglik + 2 * mask.sum() + 2 if method != "CSS" else np.nan
        names_arma = ([f"ar{i}" for i in range(1, arma[0] + 1)] + [f"ma{i}" for i in range(1, arma[1] + 1)]
                      + [f"sar{i}" for i in range(1, arma[2] + 1)] + [f"sma{i}" for i in range(1, arma[3] + 1)])
        if rotation is not None:
            coef[narma:] = rotation @ coef[narma:]
            full = np.eye(narma + ncxreg)
            full[narma:, narma:] = rotation
            full = full[np.ix_(mask, mask)]
            var = full @ var @ full.T
        self.coef = np.asarray(coef.real, dtype=float)
        self.names = names_arma + names
        self.mask, self.n_used, self.ncond, self.narma = mask, n_used, ncond, narma
        self.var = np.zeros((0, 0)) if var is None else np.asarray(var, dtype=float)
        self.sigma2 = float(np.real(sigma2))
        self.residuals = np.asarray(np.real(resid), dtype=float)
        self.model = mod
        self.xreg_names = list(xreg_names)
        self.include_intercept = "intercept" in names

    # -- tables -------------------------------------------------------------------------

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.names, self.coef.tolist()))

    @property
    def loglik(self) -> float:
        return -0.5 * self.minus2loglik

    def _with_key(self, frame: pl.DataFrame) -> pl.DataFrame:
        if not self.groups:
            return frame
        keys = pl.DataFrame({g: [self.key[g]] * frame.height for g in self.groups})
        return pl.concat([keys, frame], how="horizontal")

    def _write(self, frame: pl.DataFrame, data: Data | None = None) -> Any:
        return (data or self._data).write(frame)

    def _check_vcov(self, vcov: Any) -> None:
        if vcov is not None:
            raise ValueError("arima(): vcov= is for lm/glm; this model supplies its own covariance")

    def tidy(self, conf_int: bool = False, conf_level: float = 0.95, exponentiate: bool = False,
             vcov: Any = None) -> pl.DataFrame:
        from scipy.special import ndtri

        self._check_vcov(vcov)
        se = np.full(len(self.coef), np.nan)
        if self.var.size:
            se[self.mask] = np.sqrt(np.diag(self.var))
        cols: dict[str, Any] = {"term": self.names, "estimate": self.coef, "std_error": se,
                                "statistic": self.coef / se,
                                "p_value": pl.Series([None] * len(se), dtype=pl.Float64)}
        if conf_int:
            _check_conf_level(conf_level)
            width = ndtri((1 + conf_level) / 2) * se
            cols.update(conf_low=self.coef - width, conf_high=self.coef + width)
        if exponentiate:
            for name in ("estimate", "conf_low", "conf_high"):
                if name in cols:
                    cols[name] = np.exp(cols[name])
        frame = pl.DataFrame(cols)
        return self._with_key(frame.with_columns(pl.col(c).fill_nan(None) for c in frame.columns
                                                 if c not in ("term", "p_value")))

    def vcov_table(self, vcov: Any = None) -> pl.DataFrame:
        self._check_vcov(vcov)
        names = [n for n, m in zip(self.names, self.mask) if m]
        return self._with_key(pl.DataFrame({"term": names, **{n: self.var[:, j] for j, n in enumerate(names)}}))

    def log_lik(self) -> float:
        return self.loglik

    def glance(self) -> pl.DataFrame:
        """sigma2 and the log likelihood as R's arima reports them; AIC as
        R's, BIC as ``BIC(fit)``, AICc as forecast's, with k = the estimated
        coefficients plus sigma2. A CSS fit has only its part likelihood."""
        k = int(self.mask.sum()) + 1
        css = self.fit_method == "CSS"
        n = self.n_used
        aicc = self.aic + 2 * k * (k + 1) / (n - k - 1) if not css and n - k - 1 > 0 else None
        return self._with_key(pl.DataFrame({
            "sigma2": [self.sigma2], "logLik": [self.loglik],
            "AIC": [None if css else self.aic], "AICc": [aicc],
            "BIC": [None if css else self.minus2loglik + math.log(n) * k],
            "nobs": [n], "converged": [self.converged]},
            schema_overrides={"AIC": pl.Float64, "AICc": pl.Float64, "BIC": pl.Float64}))

    def augment(self) -> pl.DataFrame:
        """The series with ``.resid`` (R's residuals), ``.fitted`` (the
        series less them, as forecast's ``fitted``) and ``.std_resid``
        (residuals over sigma, as ``tsdiag`` plots)."""
        resid = self.residuals
        return self._with_key(pl.DataFrame({
            "t": np.arange(1, len(resid) + 1), self.series: self.x, ".fitted": self.x - resid,
            ".resid": resid, ".std_resid": resid / math.sqrt(self.sigma2)}).with_columns(
            pl.col(c).fill_nan(None) for c in (self.series, ".fitted", ".resid", ".std_resid")))

    def _forecast(self, newxreg: np.ndarray | None, n_ahead: int, se_fit: bool) -> pl.DataFrame:
        if self.arma[1] > 0 and np.any(_polyroot_moduli(np.r_[1, self.coef[self.arma[0]:sum(self.arma[:2])]]) < 1):
            _warn("predict(): MA part of model is not invertible")
        if self.arma[3] > 0:
            s = sum(self.arma[:3])
            if np.any(_polyroot_moduli(np.r_[1, self.coef[s:s + self.arma[3]]]) < 1):
                _warn("predict(): seasonal MA part of model is not invertible")
        ncx = len(self.xreg_names)
        if ncx:
            if newxreg is None:
                raise ValueError("predict(): this arima fit has xreg; give newdata with the xreg columns")
            if len(newxreg) != n_ahead:
                raise ValueError(f"predict(): newdata has {len(newxreg)} rows for n_ahead={n_ahead}")
        elif newxreg is not None and newxreg.shape[1]:
            raise ValueError("predict(): 'xreg' and 'newxreg' have different numbers of columns")
        regressors = np.empty((n_ahead, 0)) if newxreg is None else newxreg
        if self.include_intercept:
            regressors = np.column_stack([np.ones(n_ahead), regressors])
        xm = regressors @ self.coef[self.narma:] if regressors.shape[1] else 0.0
        mean, variance = _kalman_forecast(n_ahead, self.model)
        cols: dict[str, Any] = {"horizon": np.arange(1, n_ahead + 1), "fit": mean + xm}
        if se_fit:
            cols["se_fit"] = np.sqrt(variance * self.sigma2)
        return pl.DataFrame(cols)

    def _newxreg(self, newdata: Any) -> tuple[np.ndarray | None, int | None]:
        if newdata is None or not self.xreg_specs:
            return None, None
        data = Data.read(newdata, None, "predict")
        lf = data._lf
        if self.order_by is not None and self.order_by in data.columns:
            lf = lf.sort(self.order_by, maintain_order=True)
        if self.groups and all(g in data.columns for g in self.groups):
            lf = lf.filter(pl.all_horizontal(pl.col(g) == self.key[g] for g in self.groups))
        frame = lf.select(expression(spec).cast(pl.Float64).alias(f"__x{i}")
                          for i, spec in enumerate(self.xreg_specs)).collect()
        return frame.to_numpy() if frame.width else np.empty((frame.height, 0)), frame.height

    def predict_table(self, newdata: Any = None, n_ahead: int | None = None, se_fit: bool = True) -> pl.DataFrame:
        newxreg, rows = self._newxreg(newdata)
        if n_ahead is None:
            n_ahead = rows if rows is not None else 1
        if isinstance(n_ahead, bool) or not isinstance(n_ahead, (int, np.integer)) or n_ahead < 1:
            raise ValueError("predict(): n_ahead must be a positive whole number")
        return self._with_key(self._forecast(newxreg, int(n_ahead), se_fit))

    def predict(self, newdata: Any = None, *, n_ahead: int | None = None, se_fit: bool = True) -> Any:
        """Forecasts ``n_ahead`` steps past the series' end, as R's
        ``predict(fit, n.ahead=)``: ``horizon``, ``fit`` and ``se_fit``. A
        fit with ``xreg`` needs ``newdata`` holding those columns for the
        forecast periods (its row count is the default ``n_ahead``)."""
        return self._write(self.predict_table(newdata, n_ahead, se_fit))

    def anova_table(self, test: Any = None):
        raise TypeError("anova(): arima fits have no ANOVA table; compare them with glance()")

    def anova_type(self, kind: int, test: Any = None):
        return self.anova_table(test)

    def __repr__(self) -> str:
        head = f"arima({self.series}, order=({self.arma[0]}, {self.arma[5]}, {self.arma[1]})"
        if any(self.arma[i] for i in (2, 3, 6)):
            head += f", seasonal=({self.arma[2]}, {self.arma[6]}, {self.arma[3]}), period={self.arma[4]}"
        head += f", method={self.method!r})"
        if self.key:
            head += "  " + ", ".join(f"{k} = {v}" for k, v in self.key.items())
        likelihood = "part log likelihood" if self.fit_method == "CSS" else "log likelihood"
        tail = f"sigma^2 estimated as {self.sigma2:.4g}: {likelihood} = {self.loglik:.2f}"
        if self.fit_method != "CSS":
            tail += f", aic = {self.aic:.2f}"
        return f"{head}\n{_table_text(self.tidy().drop(self.groups) if self.groups else self.tidy())}\n{tail}"


class _ArimaStep(Step):
    """``data >> arima(...)`` returns the fit."""

    __slots__ = ()

    def __rrshift__(self, data: Any) -> Any:
        frame = Data.read(data, self.by, self.name)
        return self._fn(frame)


@step
def arima(x: Any, order: Any = (0, 0, 0), *, seasonal: Any = (0, 0, 0), period: int | None = None,
          xreg: Any = None, include_mean: bool = True, transform_pars: bool = True, fixed: Any = None,
          init: Any = None, method: str = "CSS-ML", n_cond: int | None = None, kappa: float = 1e6,
          order_by: Any = None) -> Step:
    """Fit an ARIMA model to a series, as R's ``arima``.

    ``arima(frame, "value", order=(1, 0, 1))``; ``by=`` fits one model per
    series. ``seasonal=(P, D, Q)`` needs ``period=`` (R's frequency).
    ``xreg=`` names one or more regressor columns; a mean (``intercept``) is
    added when there is no differencing and ``include_mean``. ``method`` is
    ``"CSS-ML"`` (CSS to start, then exact likelihood; ML when anything is
    missing), ``"ML"`` or ``"CSS"``. ``fixed`` and ``init`` take one value
    per coefficient, ``None`` for free. Missing values in the series are
    kept in place and skipped by the Kalman filter, as in R.

    ``tidy`` gives the coefficients with R's standard errors (p-values
    missing, as in R's print), ``glance`` sigma2, logLik, AIC, AICc, BIC,
    ``predict(fit, n_ahead=)`` forecasts with standard errors, ``augment``
    the residuals, ``vcov`` the coefficient covariance, and
    ``box_test(fit, lag=, type="Ljung-Box")`` tests the residuals.
    """
    order = _as_order(order, "order")
    seasonal = _as_order(seasonal, "seasonal")
    if method not in ("CSS-ML", "ML", "CSS"):
        raise ValueError(f"arima(): method must be 'CSS-ML', 'ML' or 'CSS', got {method!r}")
    if period is None or period == 0:
        if any(seasonal):
            raise ValueError("arima(): a seasonal model needs period= (R's frequency)")
        period = 1
    if isinstance(period, bool) or int(period) != period or period < 1:
        raise ValueError("arima(): period must be a positive whole number")
    period = int(period)
    if n_cond is not None and (isinstance(n_cond, bool) or int(n_cond) != n_cond or n_cond < 0):
        raise ValueError("arima(): n_cond must be a non-negative whole number")
    xreg_specs = [] if xreg is None else list(xreg) if isinstance(xreg, (list, tuple)) else [xreg]

    def _apply(tf: Data) -> Any:
        series = _column(tf, x, "x", "arima")
        names = [_column(tf, spec, "xreg", "arima") for spec in xreg_specs]
        order_name = None if order_by is None else _column(tf, order_by, "order_by", "arima")
        columns = {"x": series, **{f"__xreg{i}": name for i, name in enumerate(names)}}
        groups, table, rows = _series(tf, columns, order_by, "arima")
        keys = table.select(groups).to_dicts() if groups else [{}]
        fits = []
        for key, row in zip(keys, rows):
            regressors = np.column_stack([row[f"__xreg{i}"] for i in range(len(names))]) if names else None
            fit = ArimaModel(row["x"], regressors, names, key, tf, order, seasonal, period, include_mean,
                             transform_pars, fixed, init, method, n_cond, kappa, series, row["__label"])
            fit.order_by, fit.xreg_specs = order_name, names
            fits.append(fit)
        return Models(fits, tf) if groups else fits[0]

    return _ArimaStep(_apply, "arima")
