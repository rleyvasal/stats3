"""Smooth bases for ``gam``: ports of mgcv's smooth constructors.

Each constructor follows mgcv 1.9's R and C code: the thin plate regression
spline (``tp``, ``smooth.construct.tp.smooth.spec`` and ``tprs.c``), cubic
regression splines (``cr``, ``cs``; ``crspl``/``getFS``), P-splines (``ps``),
simple random effects (``re``) and tensor products of these (``te``).
``smooth_con`` then does what mgcv's ``smoothCon`` does: it rescales the
penalties, expands factor ``by`` variables and absorbs the centring
constraint by a Householder QR.

Thin plate bases come from an eigen-decomposition, so their columns are
mgcv's up to sign. The signs never reach fitted values, effective degrees of
freedom or smoothing parameters (they only flip columns through the
Householder steps that follow), but they do flip individual basis
coefficients: compare those up to sign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from stats3.core import _warn

_EPS = np.finfo(float).eps


@dataclass
class SmoothSpec:
    """One ``s()`` or ``te()`` term of a gam formula."""

    kind: str  # "s" or "te"
    terms: list[str]
    bs: list[str]
    k: list[int]
    m: list[Any]
    by: str | None = None
    fx: bool = False
    text: str = ""

    @property
    def label(self) -> str:
        return f"{self.kind}({','.join(self.terms)})"


@dataclass
class Basis:
    """A raw (unconstrained, unscaled) basis: mgcv's smooth object."""

    X: np.ndarray
    S: list[np.ndarray]
    rank: list[int]
    null_space_dim: int
    predict: Callable[[dict[str, Any]], np.ndarray]
    constrain: bool = True  # False: mgcv's 0-row C (no centring)
    noterp: bool = False  # te: no reparameterization of this margin
    random: bool = False
    dim: int = 1


@dataclass
class Smooth:
    """A smooth ready for the model matrix, with its constraint absorbed."""

    label: str
    spec: SmoothSpec
    X: np.ndarray
    S: list[np.ndarray]
    rank: list[int]
    null_space_dim: int
    random: bool
    fixed: bool
    predict: Callable[[dict[str, Any]], np.ndarray]
    first: int = 0
    last: int = 0
    by_level: Any = None
    S_scale: list[float] = field(default_factory=list)


# thin plate regression splines ------------------------------------------------


def null_space_dimension(d: int, m: int) -> int:
    """mgcv's null.space.dimension (m reset when 2m <= d)."""
    if 2 * m < d + 1:
        m = 1
        while 2 * m < d + 2:
            m += 1
    out = 1
    for i in range(d):
        out *= d + m - 1 - i
    for i in range(2, d + 1):
        out //= i
    return out


def _tps_order(m: int, d: int) -> int:
    if 2 * m <= d:
        m = 0
        while 2 * m < d + 2:
            m += 1
    return m


def _eta_const(m: int, d: int) -> float:
    pi = math.pi
    d2, m2 = d // 2, 2 * m
    if m2 <= d:
        raise ValueError("gam(): you must have 2m > d for a thin plate spline")
    if d % 2 == 0:
        f = -1.0 if (m + 1 + d2) % 2 else 1.0
        for _ in range(m2 - 1):
            f /= 2
        for _ in range(d2):
            f /= pi
        for i in range(2, m):
            f /= i
        for i in range(2, m - d2 + 1):
            f /= i
    else:
        f = math.sqrt(pi)
        for i in range(m - (d - 1) // 2):
            f /= -0.5 - i
        for _ in range(m):
            f /= 4
        for _ in range(d2):
            f /= pi
        f /= math.sqrt(pi)
        for i in range(2, m):
            f /= i
    return f


def _eta(r2: np.ndarray, m: int, d: int, f: float) -> np.ndarray:
    """mgcv's fast_eta on squared distances."""
    d2 = d // 2
    out = np.zeros_like(r2)
    pos = r2 > 0
    r = r2[pos]
    if d % 2 == 0:
        out[pos] = f * np.log(r) * 0.5 * r ** (m - d2)
    else:
        out[pos] = f * r ** (m - d2 - 1) * np.sqrt(r)
    return out


def _poly_powers(M: int, m: int, d: int) -> np.ndarray:
    """gen_tps_poly_powers: the null space polynomials' powers, in mgcv's order."""
    out = np.zeros((M, d), dtype=int)
    index = [0] * d
    for i in range(M):
        out[i] = index
        total = sum(index)
        if total < m - 1:
            index[0] += 1
        else:
            total -= index[0]
            index[0] = 0
            for j in range(1, d):
                index[j] += 1
                total += 1
                if total == m:
                    total -= index[j]
                    index[j] = 0
                else:
                    break
    return out


def _householder_rows(A: np.ndarray) -> list[np.ndarray]:
    """mgcv's QT: Householder vectors u_i with A Q = [0, T], Q = H_1 H_2 ...,
    H_i = I - u_i u_i', each row rotated into its last free element."""
    A = A.astype(float).copy()
    rows, cols = A.shape
    us = []
    for i in range(rows):
        n = cols - i
        p = A[i, :n].copy()
        scale = np.max(np.abs(p))
        if scale:
            p = p / scale
        lsq = math.sqrt(float(p @ p))
        if p[n - 1] < 0:
            lsq = -lsq
        p[n - 1] += lsq
        g = 1 / (lsq * p[n - 1]) if lsq else 0.0
        for j in range(i + 1, rows):
            x = g * float(p @ A[j, :n])
            A[j, :n] -= x * p
        u = np.zeros(cols)
        u[:n] = p * math.sqrt(g)
        us.append(u)
    return us


def _apply_right(C: np.ndarray, us: list[np.ndarray]) -> np.ndarray:
    """C H_1 H_2 ... (HQmult with p = 0, t = 0)."""
    C = C.copy()
    for u in us:
        C -= np.outer(C @ u, u)
    return C


def _eigh_lapack(A: np.ndarray):
    """Symmetric eigen-decomposition by LAPACK's dsyevr, as R's eigen(),
    returned in decreasing order."""
    from scipy.linalg import eigh

    values, vectors = eigh(A, driver="evr")
    return values[::-1], vectors[:, ::-1]


def tp_basis(xs: list[np.ndarray], k: int, m: Any, max_knots: int = 2000, seed: int = 1) -> Basis:
    """smooth.construct.tp.smooth.spec with construct_tprs (no supplied knots)."""
    d = len(xs)
    n = len(xs[0])
    m = 0 if m is None else int(m)
    M = null_space_dimension(d, m)
    if k < 0:
        k = M + (8, 27, 100)[min(d, 3) - 1]
    if k < M + 1:
        k = M + 1
        _warn("basis dimension, k, increased to minimum possible")
    shift = np.array([float(np.mean(x)) for x in xs])
    x0 = np.column_stack([x - s for x, s in zip(xs, shift)])
    mm = _tps_order(m, d)
    f = _eta_const(mm, d)
    powers = _poly_powers(M, mm, d)

    def poly(pts):
        return np.column_stack([np.prod(pts ** powers[j], axis=1) for j in range(M)])

    xu = np.unique(x0, axis=0)
    knots_given = False
    if n > max_knots and len(xu) > max_knots:
        if d > 1:  # uniquecombs(ordered=TRUE): rows sorted as "x1*x2" text, C locale
            text = ["*".join(_r_number(v) for v in row).encode() for row in xu]
            xu = xu[sorted(range(len(xu)), key=text.__getitem__)]
        xu = np.unique(xu[_r_sample(len(xu), max_knots, seed)], axis=0)
        knots_given = True
    nu = len(xu)
    if nu < k:
        raise ValueError("gam(): a term has fewer unique covariate combinations than specified maximum degrees of freedom")
    r2 = np.sum((xu[:, None, :] - xu[None, :, :]) ** 2, axis=2)
    E = _eta(r2, mm, d, f)
    T = poly(xu)
    if nu == k:  # pure knot basis: no eigen step
        A, D = np.eye(nu), E
    else:
        values, vectors = _eigh_lapack(E)
        top = np.argsort(-np.abs(values), kind="stable")[:k]
        top = top[np.argsort(-values[top], kind="stable")]  # Lanczos order: descending
        A, D = vectors[:, top], np.diag(values[top])
    us = _householder_rows(T.T @ A)
    Q = _apply_right(np.eye(A.shape[1]), us)
    UZ = np.zeros((nu + M, k))
    UZ[:nu, :k - M] = (A @ Q)[:, :k - M]
    UZ[nu:, k - M:] = np.eye(M)
    S = Q.T @ D @ Q
    S[k - M:, :] = 0
    S[:, k - M:] = 0

    def raw(pts):
        r2 = np.sum((pts[:, None, :] - xu[None, :, :]) ** 2, axis=2)
        return np.column_stack([_eta(r2, mm, d, f), poly(pts)]) @ UZ

    if knots_given:
        X = raw(x0)
    else:
        X1 = np.column_stack([(A @ D @ Q)[:, :k - M], T])
        _, where = np.unique(x0, axis=0, return_inverse=True)
        X = X1[where.reshape(-1)]
    w = np.sqrt(np.sum(X**2, axis=0) / n)
    X, UZ = X / w, UZ / w
    S = S / np.outer(w, w)
    S = (S + S.T) / 2

    def predict(values):
        pts = np.column_stack([np.asarray(values[t], dtype=float) - s for t, s in zip(range(d), shift)])
        return raw(pts)

    return Basis(X, [S], [k - M], M, predict, dim=d)


def _r_number(v: float) -> str:
    """as.character() of a double: 15 significant digits."""
    text = f"{v:.15g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}e{exponent[0]}{int(exponent[1:]):02d}"
    return text


def _r_sample(n: int, size: int, seed: int) -> np.ndarray:
    """R's sample(n, size) after set.seed(seed): Mersenne-Twister with
    rejection sampling (R >= 3.6 defaults). Zero-based indices."""
    s = seed & 0xFFFFFFFF
    for _ in range(50):
        s = (69069 * s + 1) & 0xFFFFFFFF
    mt = []
    for _ in range(625):
        s = (69069 * s + 1) & 0xFFFFFFFF
        mt.append(s)
    state = mt[1:]
    pos = [624]

    def genrand():
        N, Mm = 624, 397
        if pos[0] >= N:
            for kk in range(N):
                y = (state[kk] & 0x80000000) | (state[(kk + 1) % N] & 0x7FFFFFFF)
                state[kk] = state[(kk + Mm) % N] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            pos[0] = 0
        y = state[pos[0]]
        pos[0] += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        value = y * 2.3283064365386963e-10
        return min(max(value, 0.5 * 2.328306437080797e-10), 1 - 0.5 * 2.328306437080797e-10)

    def unif_index(dn):
        bits = int(math.ceil(math.log2(dn)))
        while True:
            v = 0
            for _ in range(0, bits + 1, 16):
                v = 65536 * v + int(math.floor(genrand() * 65536))
            v &= (1 << bits) - 1
            if v < dn:
                return v

    pool = list(range(n))
    out = []
    remaining = n
    for _ in range(size):
        j = unif_index(remaining)
        out.append(pool[j])
        remaining -= 1
        pool[j] = pool[remaining]
    return np.array(out)


# cubic regression splines -------------------------------------------------------


def _cr_FS(xk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """getFS: the penalty S and the map F from knot values to second derivatives."""
    n = len(xk)
    h = np.diff(xk)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i], D[i, i + 2] = 1 / h[i], 1 / h[i + 1]
        D[i, i + 1] = -1 / h[i] - 1 / h[i + 1]
    B = np.diag((h[:-1] + h[1:]) / 3)
    for i in range(1, n - 2):
        B[i - 1, i] = B[i, i - 1] = h[i] / 6
    BinvD = np.linalg.solve(B, D)
    F = np.zeros((n, n))
    F[:, 1:-1] = BinvD.T
    S = D.T @ BinvD
    return (S + S.T) / 2, F


def _cr_design(x: np.ndarray, xk: np.ndarray, F: np.ndarray) -> np.ndarray:
    nk = len(xk)
    X = np.zeros((len(x), nk))
    kmin, kmax = xk[0], xk[-1]
    for i, xi in enumerate(x):
        if xi < kmin:
            h = xk[1] - kmin
            xik = xi - kmin
            X[i] = -xik * h / 3 * F[:, 0] - xik * h / 6 * F[:, 1]
            X[i, 0] += 1 - xik / h
            X[i, 1] += xik / h
        elif xi > kmax:
            j = nk - 1
            h = kmax - xk[j - 1]
            xik = xi - kmax
            X[i] = xik * h / 6 * F[:, j - 1] + xik * h / 3 * F[:, j]
            X[i, nk - 2] += -xik / h
            X[i, nk - 1] += 1 + xik / h
        else:
            j = min(max(int(np.searchsorted(xk, xi, side="left")) - 1, 0), nk - 2)
            xj, xj1 = xk[j], xk[j + 1]
            h = xj1 - xj
            ajm, ajp = xj1 - xi, xi - xj
            cjm = ajm * (ajm * ajm / h - h) / 6
            cjp = ajp * (ajp * ajp / h - h) / 6
            X[i] = cjm * F[:, j] + cjp * F[:, j + 1]
            X[i, j] += ajm / h
            X[i, j + 1] += ajp / h
    return X


def cr_basis(x: np.ndarray, k: int, shrink: float | None = None, term: str = "x") -> Basis:
    if k < 0:
        k = 10
    if k < 3:
        k = 3
        _warn("basis dimension, k, increased to minimum possible")
    xu = np.unique(x)
    if len(xu) < k:
        raise ValueError(f"gam(): {term} has insufficient unique values to support {k} knots: reduce k.")
    xk = np.quantile(xu, np.linspace(0, 1, k))
    S, F = _cr_FS(xk)
    X = _cr_design(x, xk, F)
    if shrink is not None:
        values, vectors = _eigh_lapack(S)
        values = values.copy()
        values[k - 2] = values[k - 3] * shrink
        values[k - 1] = values[k - 2] * shrink
        S = (vectors * values) @ vectors.T
    rank, nsd = (k - 2, 2) if shrink is None else (k, 0)
    return Basis(X, [S], [rank], nsd, lambda v: _cr_design(_values(v[0]), xk, F), noterp=True)


# P-splines ----------------------------------------------------------------------


def _bspline(x: np.ndarray, knots: np.ndarray, order: int, deriv: int = 0) -> np.ndarray:
    from scipy.interpolate import BSpline

    nb = len(knots) - order
    spline = BSpline(knots, np.eye(nb), order - 1, extrapolate=False)
    if deriv:
        spline = spline.derivative(deriv)
    out = spline(x)
    return np.nan_to_num(out)


def ps_basis(x: np.ndarray, k: int, m: Any) -> Basis:
    m = [2, 2] if m is None else (list(m) * 2 if np.ndim(m) == 0 else list(m))
    m = [2 if v is None or (isinstance(v, float) and math.isnan(v)) else int(v) for v in m[:2]]
    if k < 0:
        k = max(10, m[0] + 1)
    nk = k - m[0]
    if nk <= 0:
        raise ValueError("gam(): basis dimension too small for b-spline order")
    xl, xu = float(np.min(x)), float(np.max(x))
    xr = xu - xl
    xl, xu = xl - xr * 0.001, xu + xr * 0.001
    dx = (xu - xl) / (nk - 1)
    knots = np.linspace(xl - dx * (m[0] + 1), xu + dx * (m[0] + 1), nk + 2 * m[0] + 2)
    order = m[0] + 2
    X = _bspline(x, knots, order)
    if len(np.unique(x)) < k:
        _warn("basis dimension is larger than number of unique covariates")
    D = np.diff(np.eye(k), n=m[1], axis=0) if m[1] > 0 else np.eye(k)
    S = D.T @ D
    ll, ul = knots[m[0] + 1], knots[len(knots) - m[0] - 2]

    def predict(values):
        x = _values(values[0])
        inside = (x <= ul) & (x >= ll)
        out = np.zeros((len(x), k))
        if inside.any():
            out[inside] = _bspline(x[inside], knots, order)
        if not inside.all():
            ends = np.array([ll, ul])
            value, slope = _bspline(ends, knots, order), _bspline(ends, knots, order, 1)
            low, high = x < ll, x > ul
            out[low] = value[0] + np.outer(x[low] - ll, slope[0])
            out[high] = value[1] + np.outer(x[high] - ul, slope[1])
        return out

    return Basis(X, [S], [k - m[1]], m[1], predict)


# random effects -----------------------------------------------------------------


def _re_columns(variables: list[Any]) -> np.ndarray:
    """model.matrix(~ a:b - 1): factor indicators (first factor fastest) times numerics."""
    cols = [np.ones(len(variables[0].codes if variables[0].is_factor else variables[0].values))]
    for v in variables:
        if v.is_factor:
            cols = [c * (v.codes == j) for j in range(len(v.levels)) for c in cols]
        else:
            cols = [c * v.values for c in cols]
    return np.column_stack(cols)


def re_basis(variables: list[Any]) -> Basis:
    X = _re_columns(variables)
    p = X.shape[1]

    return Basis(X, [np.eye(p)], [p], 0, _re_columns, constrain=False, noterp=True, random=True,
                 dim=len(variables))


# tensor products ----------------------------------------------------------------


def _row_tensor(Xs: list[np.ndarray]) -> np.ndarray:
    out = Xs[0]
    for X in Xs[1:]:
        out = (out[:, :, None] * X[:, None, :]).reshape(len(out), -1)
    return out


def te_basis(margins: list[Basis], data: list[Any]) -> Basis:
    """smooth.construct.tensor.smooth.spec with np=TRUE."""
    Xm, Sm, preds = [], [], []
    for b, x in zip(margins, data):
        X, S, predict = b.X, b.S[0], b.predict
        if b.dim == 1 and not b.noterp:
            pts = np.linspace(np.min(x), np.max(x), X.shape[1])
            u, d, vt = np.linalg.svd(predict([pts]), full_matrices=False)
            if d[-1] / d[0] < _EPS**0.66:
                _warn("reparameterization unstable for margin: not done")
            else:
                XP = vt.T @ (u.T / d[:, None])
                X, S = X @ XP, XP.T @ S @ XP
                predict = (lambda f, P: lambda v: f(v) @ P)(b.predict, XP)
        S = S / np.linalg.eigvalsh((S + S.T) / 2)[-1]
        Xm.append(X)
        Sm.append(S)
        preds.append(predict)
    dims = [X.shape[1] for X in Xm]
    S = []
    for i in range(len(Sm)):
        out = np.ones((1, 1))
        for j, d in enumerate(dims):
            out = np.kron(out, Sm[i] if i == j else np.eye(d))
        S.append((out + out.T) / 2)
    total = int(np.prod(dims))
    rank = [int(total * b.rank[0] / d) for b, d in zip(margins, dims)]
    nsd = int(np.prod([b.null_space_dim for b in margins]))
    constrain = all(b.constrain for b in margins)

    def predict(values):
        return _row_tensor([f([v]) for f, v in zip(preds, values)])

    return Basis(_row_tensor(Xm), S, rank, nsd, predict, constrain=constrain,
                 dim=sum(b.dim for b in margins))


# smoothCon -------------------------------------------------------------------------


def _values(v) -> np.ndarray:
    """Numeric values of a Variable, or an array as given."""
    return np.asarray(v.values if hasattr(v, "values") and not isinstance(v, np.ndarray) else v, dtype=float)


def _numeric(variables, name, verb="gam"):
    v = variables[name]
    if v.is_factor:
        raise ValueError(f"{verb}(): {name!r} must be numeric for this smooth basis")
    return v.values


def construct(spec: SmoothSpec, variables: dict[str, Any]) -> Basis:
    if spec.kind == "te":
        margins, data = [], []
        for t, bs, k, m in zip(spec.terms, spec.bs, spec.k, spec.m):
            if bs == "re":
                margins.append(re_basis([variables[t]]))
                data.append(None)
                continue
            x = _numeric(variables, t)
            margins.append(_margin(bs, [x], k, m, [variables[t]], t))
            data.append(x)
        return te_basis(margins, data)
    bs = spec.bs[0]
    if bs == "re":
        return re_basis([variables[t] for t in spec.terms])
    xs = [_numeric(variables, t) for t in spec.terms]
    if bs in ("cr", "cs", "ps") and len(xs) != 1:
        raise ValueError(f"gam(): bs={bs!r} only handles 1D smooths; use te() for more variables")
    return _margin(bs, xs, spec.k[0], spec.m[0], [variables[t] for t in spec.terms], spec.terms[0])


def _margin(bs, xs, k, m, variables, term) -> Basis:
    if bs == "tp":
        basis = tp_basis(xs, k, m)
        shift_predict = basis.predict
        basis.predict = lambda v: shift_predict({i: _values(v[i]) for i in range(len(xs))})
        return basis
    if bs in ("cr", "cs"):
        return cr_basis(xs[0], k, 0.1 if bs == "cs" else None, term)
    if bs == "ps":
        return ps_basis(xs[0], k, m)
    if bs == "re":
        return re_basis(variables)
    raise ValueError(f"gam(): bs must be 'tp', 'cr', 'cs', 'ps' or 're', got {bs!r}")


def smooth_con(spec: SmoothSpec, variables: dict[str, Any], n: int) -> list[Smooth]:
    """mgcv's smoothCon with absorb.cons=TRUE and scale.penalty=TRUE."""
    basis = construct(spec, variables)
    X = basis.X
    C = X.mean(axis=0)[None, :] if basis.constrain else np.zeros((0, X.shape[1]))
    S = [] if spec.fx else list(basis.S)
    rank = [] if spec.fx else list(basis.rank)
    scales = []
    if S:
        maXX = np.max(np.sum(np.abs(X), axis=1)) ** 2
        for i, Si in enumerate(S):
            maS = np.max(np.sum(np.abs(Si), axis=0)) / maXX
            S[i] = Si / maS
            scales.append(maS)
    pieces: list[tuple[str, np.ndarray, Any]] = []
    by = variables.get(spec.by) if spec.by else None
    label = spec.label
    if by is None:
        pieces.append((label, X, None))
    elif by.is_factor:
        for j, level in enumerate(by.levels):
            pieces.append((f"{label}:{spec.by}{_level_text(level)}", X * (by.codes == j)[:, None], j))
    else:
        if np.std(by.values, ddof=1) > np.mean(by.values) * _EPS * 1000:
            C = np.zeros((0, X.shape[1]))
        pieces.append((f"{label}:{spec.by}", X * by.values[:, None], None))
    Z = None
    if C.shape[0] > 0:
        Z = _constraint_null_space(C)
    nsd = basis.null_space_dim
    out = []
    for lab, Xi, level in pieces:
        Si = [Z.T @ s @ Z for s in S] if Z is not None else [s.copy() for s in S]
        Si = [(s + s.T) / 2 for s in Si]
        out.append(Smooth(lab, spec, Xi @ Z if Z is not None else Xi, Si,
                          [min(r, X.shape[1] - C.shape[0]) for r in rank],
                          max(0, nsd - C.shape[0]), basis.random, spec.fx,
                          _predictor(basis.predict, spec, Z, level),
                          by_level=None if level is None else by.levels[level], S_scale=scales))
    return out


def _level_text(level: Any) -> str:
    from stats3.formula import _level_name

    return _level_name(level)


def _constraint_null_space(C: np.ndarray) -> np.ndarray:
    """Columns j+1..k of the Householder Q of t(C): mgcv's qr.qty constraint absorption."""
    j, k = C.shape
    indi = np.flatnonzero(C.sum(axis=0) != 0)
    if len(indi) < k:  # mgcv's partial path: only constrained columns rotate
        nz = len(indi) - j
        q, _ = np.linalg.qr(C[:, indi].T, mode="complete")
        Z = np.zeros((k, k - j))
        keep = [c for c in range(k) if c not in set(indi[nz:])]
        # columns indi[:nz] become rotated combinations; free columns pass through
        for col_out, c in enumerate(keep):
            if c in set(indi[:nz]):
                pos = list(indi[:nz]).index(c)
                Z[indi, col_out] = q[:, j + pos]
            else:
                Z[c, col_out] = 1.0
        return Z
    q, _ = np.linalg.qr(C.T, mode="complete")
    return q[:, j:]


def _predictor(raw, spec: SmoothSpec, Z, level):
    def predict(variables: dict[str, Any]) -> np.ndarray:
        X = raw([variables[t] for t in spec.terms])
        if spec.by:
            by = variables[spec.by]
            if by.is_factor:
                X = X * (by.codes == level)[:, None]
            else:
                X = X * by.values[:, None]
        return X @ Z if Z is not None else X

    return predict
