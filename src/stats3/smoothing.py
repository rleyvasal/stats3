"""Scatterplot smoothers: R's loess, lowess, smooth.spline and ksmooth.

``loess`` ports the netlib dloess code R uses (``loessf.f``, ``loessc.c``):
the k-d tree of cells and vertices, local fits at the vertices with their
slopes, cubic Hermite interpolation with blending in two dimensions, and the
approximations for the trace and the residual degrees of freedom. The local
least squares use the same equilibrated pseudoinverse. ``smooth_spline``
ports ``sbart.c`` and its Fortran (B-spline Gram matrix, banded Cholesky,
leverages, Brent's search for ``spar``), ``lowess`` is ``clowess`` and
``ksmooth`` is ``BDRksmooth``.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Data, Step, step
from stats3.core import _special, _warn
from stats3.formula import _backtick, parse_formula, variable_expr
from stats3.models import _Fit, _model, _model_frames, _table_text

_EPS = 2.0**-52  # d1mach(4)


# loess: the k-d tree ------------------------------------------------------------------


def _select(lo: int, hi: int, k: int, values: list, pi: list) -> None:
    """ehg106: Floyd and Rivest's selection, permuting ``pi`` so that
    ``values[pi[k]]`` is the k-th smallest of ``pi[lo..hi]``."""
    l, r = lo, hi
    while l < r:
        t = values[pi[k]]
        i, j = l, r
        pi[l], pi[k] = pi[k], pi[l]
        if t < values[pi[r]]:
            pi[l], pi[r] = pi[r], pi[l]
        while i < j:
            pi[i], pi[j] = pi[j], pi[i]
            i += 1
            j -= 1
            while values[pi[i]] < t:
                i += 1
            while t < values[pi[j]]:
                j -= 1
        if values[pi[l]] == t:
            pi[l], pi[j] = pi[j], pi[l]
        else:
            j += 1
            pi[r], pi[j] = pi[j], pi[r]
        if j <= k:
            l = j + 1
        if k <= j:
            r = j - 1


class _Tree:
    """Cells (split dimension ``a``, -1 for a leaf, at ``xi``), their
    children, their points ``pi[pl..pu]``, their corner vertices ``c``."""

    def __init__(self, x: np.ndarray, fc: int, nvmax: int):
        n, d = x.shape
        vc = 2**d
        lower, upper = [], []
        for k in range(d):  # ehg126: the bounding box, expanded a little
            alpha, beta = float(x[:, k].min()), float(x[:, k].max())
            mu = 0.005 * max(beta - alpha, 1e-10 * max(abs(alpha), abs(beta)) + 1e-30)
            lower.append(alpha - mu)
            upper.append(beta + mu)
        vertices = [[upper[k] if (b >> k) & 1 else lower[k] for k in range(d)] for b in range(vc)]
        index = {tuple(v): i for i, v in enumerate(vertices)}
        cols = [x[:, k].tolist() for k in range(d)]
        pi = list(range(n))
        a, xi, lo, hi, pl_, pu, c = [-1], [0.0], [-1], [-1], [0], [n - 1], [list(range(vc))]
        p = 0
        while p < len(c):  # ehg124
            l, u = pl_[p], pu[p]
            first, last = vertices[c[p][0]], vertices[c[p][-1]]
            diam = math.sqrt(sum((last[i] - first[i])**2 for i in range(d)))
            leaf = u - l + 1 <= fc or diam <= 0.0
            if not leaf:
                leaf = nvmax < len(c) + 2 or nvmax < len(vertices) + vc / 2
            if not leaf:
                spread = []
                for k in range(d):
                    vals = [cols[k][pi[i]] for i in range(l, u + 1)]
                    spread.append(max(vals) - min(vals))
                k = spread.index(max(spread))
                xk = cols[k]
                m = (l + u) // 2
                _select(l, u, m, xk, pi)
                offset = 0  # all ties go with the hi son
                while l <= m + offset < u:
                    if offset < 0:
                        lower_, check, upper_ = l, m + offset, m + offset
                    else:
                        lower_ = check = m + offset + 1
                        upper_ = u
                    _select(lower_, upper_, check, xk, pi)
                    if xk[pi[m + offset]] == xk[pi[m + offset + 1]]:
                        offset = -offset
                        if offset >= 0:
                            offset += 1
                    else:
                        m += offset
                        break
                split = xk[pi[m]]
                leaf = first[k] == split or last[k] == split
            if not leaf:
                a[p], xi[p] = k, split
                kids = []
                for bounds in ((l, m), (m + 1, u)):
                    a.append(-1), xi.append(0.0), lo.append(-1), hi.append(-1)
                    pl_.append(bounds[0]), pu.append(bounds[1]), c.append([0] * vc)
                    kids.append(len(c) - 1)
                lo[p], hi[p] = kids
                r, s = 2**k, 2**(d - k - 1)  # ehg125: vertices on the cut
                f = c[p]
                for i in range(r):
                    for j in range(s):
                        i0, i1 = i + 2 * r * j, i + r + 2 * r * j
                        coords = list(vertices[f[i0]])
                        coords[k] = split
                        key = tuple(coords)
                        v = index.get(key)
                        if v is None:
                            v = len(vertices)
                            vertices.append(coords)
                            index[key] = v
                        c[lo[p]][i0], c[lo[p]][i1] = f[i0], v
                        c[hi[p]][i0], c[hi[p]][i1] = v, f[i1]
            p += 1
        self.a, self.xi, self.lo, self.hi, self.pl, self.pu, self.c = a, xi, lo, hi, pl_, pu, c
        self.v = np.array(vertices)
        self.pi = pi
        self.d = d

    def leaves(self, z) -> list[int]:
        """ehg137: the leaf cells whose closed boxes contain ``z``."""
        stack, p, out = [], 0, []
        while True:
            if self.a[p] < 0:
                out.append(p)
                if not stack:
                    return out
                p = stack.pop()
            elif z[self.a[p]] == self.xi[p]:
                stack.append(self.hi[p])
                p = self.lo[p]
            else:
                p = self.lo[p] if z[self.a[p]] <= self.xi[p] else self.hi[p]

    def interpolate(self, z, vval: np.ndarray) -> np.ndarray:
        """ehg128: the surface at ``z`` from vertex values and slopes
        ``vval`` (vertex, 0..d, ...), linear in ``vval``."""
        a, xi, lo, hi, c, v, d = self.a, self.xi, self.lo, self.hi, self.c, self.v, self.d
        path, j = [0], 0
        while a[j] >= 0:
            j = lo[j] if z[a[j]] <= xi[j] else hi[j]
            path.append(j)
        cell = c[j]
        ll, ur = cell[0], cell[-1]
        g = vval[cell]
        lg = len(cell)
        for i in range(d - 1, -1, -1):
            width = v[ur, i] - v[ll, i]
            h = (z[i] - v[ll, i]) / width
            if not -.001 <= h <= 1.001:
                _warn("extrapolation not allowed with blending")
            lg //= 2
            phi0, phi1, psi0, psi1 = (1 - h)**2 * (1 + 2 * h), h**2 * (3 - 2 * h), h * (1 - h)**2, h**2 * (h - 1)
            low, up = g[:lg], g[lg:2 * lg]
            new = phi0 * low + phi1 * up
            new[:, 0] = phi0 * low[:, 0] + phi1 * up[:, 0] + (psi0 * low[:, i + 1] + psi1 * up[:, i + 1]) * width
            g = new
        s = g[0, 0]
        if d != 2:
            return s

        def edge(k_split, xibar, go_hi, own, other, axis, along, across):
            v0, v1 = v[ll, axis], v[ur, axis]
            g0, g1 = vval[cell[own[0]]], vval[cell[own[1]]]
            m = len(path) - 2
            while m >= 0 and not (a[path[m]] == k_split and xi[path[m]] == xibar):
                m -= 1
            if m >= 0:
                m = hi[path[m]] if go_hi else lo[path[m]]
                while a[m] >= 0:
                    m = lo[m] if z[a[m]] <= xi[m] else hi[m]
                if v0 < v[c[m][other[0]], axis]:
                    v0, g0 = v[c[m][other[0]], axis], vval[c[m][other[0]]]
                if v[c[m][other[1]], axis] < v1:
                    v1, g1 = v[c[m][other[1]], axis], vval[c[m][other[1]]]
            h = (z[axis] - v0) / (v1 - v0)
            phi0, phi1, psi0, psi1 = (1 - h)**2 * (1 + 2 * h), h**2 * (3 - 2 * h), h * (1 - h)**2, h**2 * (h - 1)
            return (phi0 * g0[0] + phi1 * g1[0] + (psi0 * g0[along] + psi1 * g1[along]) * (v1 - v0),
                    phi0 * g0[across] + phi1 * g1[across])

        gn, gpn = edge(1, v[ur, 1], True, (2, 3), (0, 1), 0, 1, 2)
        gs, gps = edge(1, v[ll, 1], False, (0, 1), (2, 3), 0, 1, 2)
        ge, gpe = edge(0, v[ur, 0], True, (1, 3), (0, 2), 1, 2, 1)
        gw, gpw = edge(0, v[ll, 0], False, (0, 2), (1, 3), 1, 2, 1)

        def hermite(h, left, right, dleft, dright, width):
            return ((1 - h)**2 * (1 + 2 * h) * left + h**2 * (3 - 2 * h) * right
                    + (h * (1 - h)**2 * dleft + h**2 * (h - 1) * dright) * width)

        w1, w0 = v[ur, 1] - v[ll, 1], v[ur, 0] - v[ll, 0]
        sns = hermite((z[1] - v[ll, 1]) / w1, gs, gn, gps, gpn, w1)
        sew = hermite((z[0] - v[ll, 0]) / w0, gw, ge, gpw, gpe, w0)
        return (sns + sew) - s


# ehg176's fixed one-dimensional surface, used by the delta approximations
_T176 = None


def _ehg176(z: float) -> float:
    global _T176
    if _T176 is None:
        t = object.__new__(_Tree)
        t.d = 1
        t.a = [0] * 7 + [-1] * 7 + [0, -1, -1]
        t.xi = [.3705, .2017, .5591, .1204, .2815, .4536, .7132] + [0.0] * 7 + [.8751, 0.0, 0.0]
        t.lo = [1, 3, 5, 7, 9, 11, 13] + [-1] * 7 + [15, -1, -1]
        t.hi = [2, 4, 6, 8, 10, 12, 14] + [-1] * 7 + [16, -1, -1]
        pairs = [(1, 2), (1, 3), (3, 2), (1, 4), (4, 3), (3, 5), (5, 2), (1, 6), (6, 4), (4, 7),
                 (7, 3), (3, 8), (8, 5), (5, 9), (9, 2), (9, 10), (10, 2)]
        t.c = [[p - 1, q - 1] for p, q in pairs]
        t.v = np.array([[-.005], [1.005], [.3705], [.2017], [.5591], [.1204], [.2815], [.4536], [.7132], [.8751]])
        t.vval = np.array([[-9.0572e-2, 4.4844], [-1.0856e-2, -.7736], [-5.3718e-2, -.3495], [2.6152e-2, -.7286],
                           [-5.8387e-2, .1611], [9.5807e-2, -.7978], [-3.1926e-2, -.4457], [-6.4170e-2, 3.2813e-2],
                           [-2.0636e-2, .3350], [4.0172e-2, -4.1032e-2]])
        _T176 = t
    return float(_T176.interpolate([z], _T176.vval))


_C141 = [.2971620, .3802660, .5886043, .4263766, .3346498, .6271053, .5241198, .3484836, .6687687, .6338795,
         .4076457, .7207693, .1611761, .3091323, .4401023, .2939609, .3580278, .5555741, .3972390, .4171278,
         .6293196, .4675173, .4699070, .6674802, .2848308, .2254512, .2914126, .5393624, .2517230, .3898970,
         .7603231, .2969113, .4740130, .9664956, .3629838, .5348889, .2075670, .2822574, .2369957, .3911566,
         .2981154, .3623232, .5508869, .3501989, .4371032, .7002667, .4291632, .4930370]


def _dk(degree: int, d: int) -> int:
    return {0: 1, 1: d + 1, 2: int((d + 2) * (d + 1) / 2)}[degree]


def _ehg141(trl, n, deg, k, d, nsing):
    corx = math.sqrt(k / n)
    z = (math.sqrt(k / trl) - corx) / (1 - corx)
    if nsing == 0 and 1 < z:
        _warn(f"Chernobyl! trL<k {trl:.5g}")
    elif z < 0:
        _warn(f"Chernobyl! trL>n {trl:.5g}")
    z = min(1.0, max(0.0, z))
    c4 = math.exp(_ehg176(z))
    out = []
    for i in (3 * (min(d, 4) - 1 + 4 * (deg - 1)), 3 * (min(d, 4) - 1 + 4 * (deg - 1)) + 24):
        if d <= 4:
            c1, c2, c3 = _C141[i:i + 3]
        else:
            c1, c2, c3 = (_C141[i + j] + (d - 4) * (_C141[i + j] - _C141[i + j - 3]) for j in range(3))
        out.append(n - trl * math.exp(c1 * z**c2 * (1 - z)**c3 * c4))
    return out


def _lowesa(trl, n, d, tau, nsing):
    """Approximate delta1 and delta2 from the trace (Cleveland and Grosse)."""
    d1a, d2a = _ehg141(trl, n, 1, tau, d, nsing)
    d1b, d2b = _ehg141(trl, n, 2, tau, d, nsing)
    alpha = (tau - _dk(1, d)) / (_dk(2, d) - _dk(1, d))
    return (1 - alpha) * d1a + alpha * d1b, (1 - alpha) * d2a + alpha * d2b


def _ehg196(tau, d, f):
    """The trace approximated from the span alone (trace_hat='approximate')."""
    g1 = (-0.08125 * d + 0.13) * d + 1.05
    trla, trlb = (_dk(deg, d) * (1 + max(0.0, (g1 - f) / f)) for deg in (1, 2))
    alpha = (tau - _dk(1, d)) / (_dk(2, d) - _dk(1, d))
    return (1 - alpha) * trla + alpha * trlb


def _lowesc(L: np.ndarray):
    """Exact trace, delta1 and delta2 from the operator matrix."""
    resid = np.eye(len(L)) - L
    ll = resid @ resid.T
    return float(np.trace(L)), float(np.trace(ll)), float(np.sum(ll * ll.T))


# loess: local fits ------------------------------------------------------------------------


class _Local:
    """Weighted local regressions (ehg127) for one fit's settings."""

    def __init__(self, x, rw, span, degree):
        self.x, self.rw, self.span, self.degree = x, rw, span, degree
        n, d = x.shape
        self.nf = min(n, math.floor(n * span + 1e-5))
        if self.nf <= 0:
            raise ValueError("loess(): span is too small")
        self.k = _dk(degree, d)
        if self.k > self.nf - 1:
            _warn("span too small.   fewer data values than degrees of freedom.")
        self.singular = 0

    def operator(self, q):
        """The rows mapping the neighbours' responses to the local value
        and slopes at ``q``: (neighbour indices, (d + 1, m) rows)."""
        x, d = self.x, self.x.shape[1]
        diff = x - q
        dist = diff[:, 0]**2
        for j in range(1, d):
            dist = dist + diff[:, j]**2
        rho = float(np.partition(dist, self.nf - 1)[self.nf - 1]) * max(1.0, self.span)
        if rho <= 0:
            raise ValueError("loess(): zero-width neighborhood. make span bigger")
        near = np.flatnonzero(dist < rho)
        r = np.sqrt(dist[near] / rho)
        w = np.sqrt(self.rw[near] * (1 - r**3)**3)
        if not np.any(w):
            _warn("all data on boundary of neighborhood. make span bigger")
        dx = diff[near]
        cols = [w]
        if self.degree >= 1:
            cols += [w * dx[:, j] for j in range(d)]
        if self.degree >= 2:
            for j in range(d):
                cols.append(w * dx[:, j]**2)
                cols += [w * dx[:, j] * dx[:, jj] for jj in range(j + 1, d)]
        b = np.column_stack(cols)
        norm = np.sqrt(np.sum(b**2, axis=0))
        norm[~(norm > 0)] = 1.0
        u, s, vt = np.linalg.svd(b / norm, full_matrices=False)
        k = b.shape[1]
        sigma = np.zeros(k)
        sigma[:len(s)] = s
        tol = sigma[0] * (100 * _EPS)
        if sigma[k - 1] <= tol:
            self.singular += 1
            if self.singular == 1:
                _warn(f"pseudoinverse used at {' '.join(f'{v:.5g}' for v in q)}")
            elif self.singular == 2:
                _warn("There are other near singularities as well.")
        inv = np.where(s > tol, 1 / np.where(s > 0, s, 1), 0.0)
        op = ((vt.T * inv) @ u.T) * w / norm[:, None]
        rows = np.zeros((d + 1, len(near)))
        rows[:min(k, d + 1)] = op[:d + 1]
        return near, rows

    def vertex_operators(self, tree):
        n, d = self.x.shape
        out = np.zeros((len(tree.v), d + 1, n))
        for i, q in enumerate(tree.v):
            near, rows = self.operator(q)
            out[i][:, near] = rows
        return out

    def direct(self, z, y, hat=False):
        """Fits at the rows of ``z`` (lowesf): values, and the operator rows."""
        n = len(self.x)
        fit, L = np.empty(len(z)), np.zeros((len(z), n)) if hat else None
        for i, q in enumerate(z):
            near, rows = self.operator(q)
            fit[i] = rows[0] @ y[near]
            if hat:
                L[i, near] = rows[0]
        return fit, L


def _robustness_weights(resid):
    """lowesw: bisquare weights on six times the median absolute residual."""
    r = np.abs(resid)
    n = len(r)
    ordered = np.sort(r)
    nh = n // 2 + 1
    cmad = 3 * (ordered[nh - 1] + ordered[nh - 2]) if n - nh + 1 < nh else 6 * ordered[nh - 1]
    if cmad < 2.2250738585072014e-308:
        return np.ones(n)
    return np.where(cmad * .999 < r, 0.0, np.where(cmad * .001 < r, (1 - (r / cmad)**2)**2, 1.0))


def _pseudovalues(y, fitted, pw, rw):
    """lowesp: pseudovalues for the robust fit's residual scale."""
    n = len(y)
    ordered = np.sort(np.abs(y - fitted) * np.sqrt(pw))
    m = n // 2 + 1
    mad = (ordered[m - 2] + ordered[m - 1]) / 2 if n - m + 1 < m else ordered[m - 1]
    c = (6 * mad)**2 / 5
    t = (1 - ((y - fitted)**2 * pw) / c) * np.sqrt(rw)
    total = 0.0
    for value in t[::-1]:
        total = value + total
    return fitted + (n / total * rw) * (y - fitted)


class _Raw:
    """One pass of loess_raw: the surface, fitted values and statistics."""

    def __init__(self, y, x, rw, opts, statistics, tree):
        self.local = _Local(x, rw, opts["span"], opts["degree"])
        n, d = x.shape
        self.trace_hat = self.one_delta = self.two_delta = float("nan")
        tau = _dk(2, d) if opts["degree"] > 1 else d + 1  # loess_workspace's tau
        if opts["surface"] == "interpolate":
            self.lv = self.local.vertex_operators(tree)
            self.vval = self.lv @ y
            self.fitted = np.array([tree.interpolate(x[i], self.vval) for i in range(n)])
            if statistics == "1.approx":
                member = np.zeros((len(tree.v), n), dtype=bool)
                for vertex in range(len(tree.v)):
                    for leaf in tree.leaves(tree.v[vertex]):
                        member[vertex, tree.pi[tree.pl[leaf]:tree.pu[leaf] + 1]] = True
                diag = [tree.interpolate(x[i], self.lv[:, :, i] * member[:, i, None]) for i in range(n)]
                self.trace_hat = float(np.sum(diag))
            elif statistics == "2.approx":
                self.trace_hat = _ehg196(tau, d, opts["span"])
            elif statistics == "exact":
                L = np.array([tree.interpolate(x[i], self.lv) for i in range(n)])
                self.trace_hat, self.one_delta, self.two_delta = _lowesc(L)
            if statistics in ("1.approx", "2.approx"):
                self.one_delta, self.two_delta = _lowesa(self.trace_hat, n, d, tau, self.local.singular)
        else:
            hat = statistics in ("approximate", "exact")
            self.fitted, L = self.local.direct(x, y, hat)
            if statistics == "approximate":
                self.trace_hat = float(np.trace(L))
                self.one_delta, self.two_delta = _lowesa(self.trace_hat, n, d, tau, self.local.singular)
            elif statistics == "exact":
                self.trace_hat, self.one_delta, self.two_delta = _lowesc(L)


# loess: the model --------------------------------------------------------------------------


class LoessModel(_Fit):
    """A local regression surface, as R's ``loess`` object."""

    kind = "loess"

    def __init__(self, formula, frame, data, opts):
        super().__init__(formula, None, frame, data, "loess")
        labels = formula.variables()
        if not labels:
            raise ValueError("loess(): the formula needs at least one predictor")
        if any(frame.variables[label].is_factor for label in labels):
            raise ValueError("loess(): predictors must all be numeric")
        if frame.y.ndim != 1 or frame.response_levels is not None:
            raise ValueError("loess(): the response must be one numeric column")
        if len(labels) > 4:
            raise ValueError("loess(): only 1-4 predictors are allowed")
        self.labels = labels
        x = np.column_stack([frame.variables[label].values.astype(float) for label in labels])
        y, w = frame.y.astype(float), frame.weights.astype(float)
        n, d = x.shape
        self.n, self.d, self.y, self.weights, self.prior_weights = n, d, y, w, w
        self.opts = opts
        if opts["normalize"] and d > 1:
            trim = math.ceil(0.1 * n)
            self.divisor = np.sqrt(np.var(np.sort(x, axis=0)[trim:n - trim], axis=0, ddof=1))
        else:
            self.divisor = np.ones(d)
        self.x = x / self.divisor
        statistics = opts["statistics"]
        if opts["surface"] == "interpolate" and statistics == "approximate":
            statistics = "1.approx" if opts["trace_hat"] == "exact" else "2.approx"
        self.tree = None
        if opts["surface"] == "interpolate":
            self.tree = _Tree(self.x, math.floor(n * (opts["span"] * opts["cell"])), max(200, n))
        iterations = 1 if opts["family"] == "gaussian" else opts["iterations"]
        robust = np.ones(n)
        for j in range(1, iterations + 1):
            raw = _Raw(y, self.x, w * robust, opts, statistics if j == 1 else "none", self.tree)
            fitted = raw.fitted
            resid = y - fitted
            if j < iterations:
                robust = _robustness_weights(resid)
            if j == 1:
                self.trace_hat, self.one_delta, self.two_delta = raw.trace_hat, raw.one_delta, raw.two_delta
        self._raw = raw
        if iterations > 1:
            pseudo = _pseudovalues(y, fitted, w, robust)
            again = _Raw(pseudo, self.x, w, opts, "none", self.tree)
            ss = float(np.sum(w * (pseudo - again.fitted)**2))
        else:
            ss = float(np.sum(w * resid**2))
        self.fitted, self.residuals, self.robust_weights = fitted, resid, robust
        self.enp = self.one_delta + 2 * self.trace_hat - n
        self.sigma = math.sqrt(ss / self.one_delta)
        self._interp_ops = None

    # -- tables --------------------------------------------------------------------------

    def tidy(self, **kwargs):
        raise TypeError("tidy(): loess fits have no coefficients; use augment() or predict()")

    def vcov_table(self, vcov=None):
        raise TypeError("vcov(): loess fits have no coefficients")

    def anova_table(self, test=None):
        raise TypeError("anova(): compare loess fits with anova(fit1, fit2)")

    def anova_type(self, kind, test=None):
        raise TypeError("anova(): type II/III tables are not available for loess fits")

    def glance(self):
        return self._with_key(pl.DataFrame({
            "nobs": [self.n], "enp": [self.enp], "sigma": [self.sigma], "trace_hat": [self.trace_hat],
            "one_delta": [self.one_delta], "two_delta": [self.two_delta]}))

    def _new_x(self, newdata):
        data = Data.read(newdata, None, "predict")
        frame = data._lf.select(**{label: variable_expr(label, "predict")[0].cast(pl.Float64)
                                   for label in self.labels}).collect()
        x = np.column_stack([frame[label].fill_null(float("nan")).to_numpy() for label in self.labels])
        return x, data

    def _evaluate(self, x, se):
        """predLoess: the fit (and standard errors) at unscaled rows ``x``."""
        z = x / self.divisor
        m = len(z)
        fit, se_fit = np.full(m, np.nan), np.full(m, np.nan)
        w, robust = self.weights, self.robust_weights
        if self.opts["surface"] == "direct":
            ok = ~np.isnan(z).any(axis=1)
            if se:
                local = _Local(self.x, w * robust, self.opts["span"], self.opts["degree"])
                values, L = local.direct(z[ok], self.y, True)
                if self.opts["family"] != "gaussian":
                    # R's loess_dfitse fits with the robustness weights alone.
                    values, _ = _Local(self.x, robust, self.opts["span"], self.opts["degree"]).direct(z[ok], self.y)
                fit[ok] = values
                se_fit[ok] = self.sigma * np.sqrt(np.sum(L**2 / w, axis=1))
            else:
                fit[ok] = _Local(self.x, w * robust, self.opts["span"], self.opts["degree"]).direct(z[ok], self.y)[0]
            return fit, se_fit
        low, high = self.x.min(axis=0), self.x.max(axis=0)
        with np.errstate(invalid="ignore"):
            inside = np.all((z <= high) & (z >= low), axis=1)
        vval = self._raw.vval
        for i in np.flatnonzero(inside):
            fit[i] = self.tree.interpolate(z[i], vval)
        if se and inside.any():
            if self._interp_ops is None:  # loess_ise refits with the prior weights alone
                same = self.opts["family"] == "gaussian"
                self._interp_ops = self._raw.lv if same else _Local(
                    self.x, w, self.opts["span"], self.opts["degree"]).vertex_operators(self.tree)
            L = np.array([self.tree.interpolate(z[i], self._interp_ops) for i in np.flatnonzero(inside)])
            se_fit[inside] = self.sigma * np.sqrt(np.sum(L**2 / w, axis=1))
        return fit, se_fit

    @property
    def df(self):
        return self.one_delta**2 / self.two_delta

    def predict(self, newdata=None, *, se=False):
        if newdata is None and not se:
            out = self._padded(pl.DataFrame({"fit": self.fitted}))
            return self._data.write(out)
        x, data = (self.x * self.divisor, self._data) if newdata is None else self._new_x(newdata)
        fit, se_fit = self._evaluate(x, se)
        cols = {"fit": fit}
        if se:
            cols.update(se_fit=se_fit, residual_scale=np.full(len(fit), self.sigma), df=np.full(len(fit), self.df))
        out = pl.DataFrame(cols).with_columns(pl.col(c).fill_nan(None) for c in cols)
        if newdata is None:
            out = self._padded(out)
        return data.write(out)

    def augment(self, newdata=None, *, se_fit=False):
        if newdata is None:
            cols = {".fitted": self.fitted, ".resid": self.residuals}
            if se_fit:
                cols[".se_fit"] = self._evaluate(self.x * self.divisor, True)[1]
            return pl.concat([self._rows(), self._padded(pl.DataFrame(cols))], how="horizontal")
        x, data = self._new_x(newdata)
        fit, se = self._evaluate(x, se_fit)
        cols = {".fitted": fit, **({".se_fit": se} if se_fit else {})}
        out = pl.DataFrame(cols).with_columns(pl.col(c).fill_nan(None) for c in cols)
        return data.write(pl.concat([data._lf.collect(), out], how="horizontal"))

    def __repr__(self):
        family = self.opts["family"]
        return (f"loess({self.formula.text}, span={self.opts['span']:g}, degree={self.opts['degree']})\n"
                f"Number of Observations: {self.n}\n"
                f"Equivalent Number of Parameters: {self.enp:.2f}\n"
                f"Residual {'Standard Error' if family == 'gaussian' else 'Scale Estimate'}: {self.sigma:.4g}")


@step
def loess(formula: str, *, span=0.75, enp_target=None, degree=2, family="gaussian", normalize=True,
          surface="interpolate", statistics="approximate", trace_hat="exact", cell=0.2, iterations=4,
          weights=None, na_action="omit"):
    """Local polynomial regression, as R's ``loess`` (netlib dloess).

    ``loess(cars, "dist ~ speed")``; one to four numeric predictors.
    ``span``, ``degree`` (0, 1 or 2), ``family`` (``"gaussian"`` or
    ``"symmetric"`` for bisquare robustness iterations) and ``normalize`` are
    R's; ``surface``, ``statistics``, ``trace_hat``, ``cell`` and
    ``iterations`` are ``loess.control``'s. The default surface interpolates
    vertex fits over R's k-d tree, so values match R's, not an exact local fit
    at every point (``surface="direct"``). ``glance`` gives the equivalent
    number of parameters, residual standard error and the trace of the hat
    matrix; ``predict(fit, newdata, se=True)``, ``augment`` and
    ``anova(fit1, fit2)`` (``anova.loess``) work as in R. Conditionally
    parametric terms and ``drop.square`` are not included.
    """
    if family not in ("gaussian", "symmetric"):
        raise ValueError("loess(): family must be 'gaussian' or 'symmetric'")
    if degree not in (0, 1, 2):
        raise ValueError("loess(): 'degree' must be 0, 1 or 2")
    if surface not in ("interpolate", "direct") or statistics not in ("approximate", "exact") \
            or trace_hat not in ("exact", "approximate"):
        raise ValueError("loess(): invalid surface, statistics or trace_hat")
    if not (isinstance(iterations, int) and iterations > 0):
        raise ValueError("loess(): iterations must be a positive integer")

    def build(f, fr, d):
        s = span
        if enp_target is not None:
            p = len(f.variables())
            s = 1.2 * {0: 1, 1: p + 1, 2: (p + 1) * (p + 2) / 2}[degree] / enp_target
        if not (math.isfinite(s) and s > 0):
            raise ValueError("loess(): span must be positive")
        opts = dict(span=float(s), degree=degree, family=family, normalize=normalize, surface=surface,
                    statistics=statistics, trace_hat=trace_hat, cell=float(cell), iterations=iterations)
        return LoessModel(f, fr, d, opts)

    return _model("loess", build, formula, weights, None, None, na_action)


def _anova_loess(fits):
    """anova.loess: F tests from the approximate residual degrees of freedom."""
    responses = [f.formula.response for f in fits]
    kept = [f for f, r in zip(fits, responses) if r == responses[0]]
    if len(kept) < len(fits):
        _warn("models with response removed because response differs from model 1")
    if len(kept) <= 1:
        raise ValueError("anova(): no models to compare")
    d1 = np.array([f.one_delta for f in kept])
    d2 = np.array([f.two_delta for f in kept])
    s = np.array([f.sigma for f in kept])
    enp = np.array([f.enp for f in kept])
    rss = s**2 * d1
    big = int(np.argsort(enp, kind="stable")[-1])
    d1diff = np.abs(np.diff(d1))
    dfnum = d1diff**2 / np.abs(np.diff(d2))
    dfden = (d1**2 / d2)[big]
    fvalue = (np.abs(np.diff(rss)) / d1diff) / s[big]**2
    p = _special().fdtrc(dfnum, dfden, fvalue)
    return pl.DataFrame({
        "term": [f.formula.text for f in kept], "enp": enp, "rss": rss,
        "statistic": [None, *fvalue.tolist()], "p_value": [None, *p.tolist()],
        "df": [None, *dfnum.tolist()], "df_residual": [dfden] * len(kept),
    }, schema_overrides={"statistic": pl.Float64, "p_value": pl.Float64, "df": pl.Float64})


# lowess and ksmooth -----------------------------------------------------------------------


def _xy_parts(data: Data, x, y, w, verb):
    for name, spec in (("x", x), ("y", y)):
        if not isinstance(spec, str):
            raise TypeError(f"{verb}(): {name} must name a column")
    formula = parse_formula(f"{_backtick(y)} ~ {_backtick(x)}", verb, data.columns)
    frames = _model_frames(data, formula, w, None, verb)
    label = formula.variables()[0]
    for frame in frames:
        if frame.response_levels is not None or frame.variables[label].is_factor:
            raise ValueError(f"{verb}(): x and y must be numeric")
        yield frame, frame.variables[label].values.astype(float), frame.y.astype(float)


def _keyed(frame, groups, cols: dict) -> pl.DataFrame:
    out = pl.DataFrame(cols)
    if not groups:
        return out
    keys = frame.rows.select(groups).head(1)
    return pl.concat([pl.concat([keys] * out.height) if out.height else keys.clear(), out], how="horizontal")


def _clowess(x, y, f, nsteps, delta):
    n = len(x)
    if n < 2:
        return np.array(y, dtype=float)
    x, y = [0.0, *x.tolist()], [0.0, *y.tolist()]  # one-based, as clowess
    ys, rw, res = [0.0] * (n + 1), [0.0] * (n + 1), [0.0] * (n + 1)
    ns = max(2, min(n, int(f * n + 1e-7)))

    def lowest(xs, nleft, nright, userw):
        rng = x[n] - x[1]
        h = max(xs - x[nleft], x[nright] - xs)
        h9, h1 = .999 * h, .001 * h
        a, j = 0.0, nleft
        while j <= n:
            res[j] = 0.0
            r = abs(x[j] - xs)
            if r <= h9:
                res[j] = 1.0 if r <= h1 else (1. - (r / h)**3)**3
                if userw:
                    res[j] *= rw[j]
                a += res[j]
            elif x[j] > xs:
                break
            j += 1
        nrt = j - 1
        if a <= 0:
            return None
        for j in range(nleft, nrt + 1):
            res[j] /= a
        if h > 0:
            a = 0.0
            for j in range(nleft, nrt + 1):
                a += res[j] * x[j]
            b = xs - a
            c = 0.0
            for j in range(nleft, nrt + 1):
                c += res[j] * (x[j] - a) * (x[j] - a)
            if math.sqrt(c) > .001 * rng:
                b /= c
                for j in range(nleft, nrt + 1):
                    res[j] *= b * (x[j] - a) + 1.
        out = 0.0
        for j in range(nleft, nrt + 1):
            out += res[j] * y[j]
        return out

    it = 1
    while it <= nsteps + 1:
        nleft, nright, last, i = 1, ns, 0, 1
        while True:
            if nright < n:
                if x[i] - x[nleft] > x[nright + 1] - x[i]:
                    nleft += 1
                    nright += 1
                    continue
            value = lowest(x[i], nleft, nright, it > 1)
            ys[i] = y[i] if value is None else value
            if last < i - 1:
                denom = x[i] - x[last]
                for j in range(last + 1, i):
                    alpha = (x[j] - x[last]) / denom
                    ys[j] = alpha * ys[i] + (1. - alpha) * ys[last]
            last = i
            cut = x[last] + delta
            i = last + 1
            while i <= n:
                if x[i] > cut:
                    break
                if x[i] == x[last]:
                    ys[i] = ys[last]
                    last = i
                i += 1
            i = max(last + 1, i - 1)
            if last >= n:
                break
        resid = [y[i] - ys[i] for i in range(1, n + 1)]
        sc = sum(abs(r) for r in resid) / n
        if it > nsteps:
            break
        ordered = sorted(abs(r) for r in resid)
        m1 = n // 2
        cmad = 3. * (ordered[m1] + ordered[n - m1 - 1]) if n % 2 == 0 else 6. * ordered[m1]
        if cmad < 1e-7 * sc:
            break
        c9, c1 = .999 * cmad, .001 * cmad
        for i in range(n):
            r = abs(resid[i])
            rw[i + 1] = 1. if r <= c1 else (1. - (r / cmad)**2)**2 if r <= c9 else 0.
        it += 1
    return np.array(ys[1:])


@step
def lowess(x: str, y: str, *, f=2 / 3, iter=3, delta=None):  # noqa: A002 - R's name
    """Robust locally weighted scatterplot smoothing, as R's ``lowess``.

    ``lowess(cars, "speed", "dist")`` gives the sorted ``x`` and the smooth
    ``y`` (``clowess``): ``f`` is the span, ``iter`` the robustness
    iterations, ``delta`` (by default 1% of the range of ``x``) the distance
    within which fits are interpolated. One smooth per ``by=`` group; rows
    missing ``x`` or ``y`` are dropped.
    """
    if not (math.isfinite(f) and f > 0):
        raise ValueError("lowess(): 'f' must be finite and > 0")
    if not (isinstance(iter, int) and iter >= 0):
        raise ValueError("lowess(): 'iter' must be finite and >= 0")

    def run(data: Data):
        parts = []
        for frame, xs, ys in _xy_parts(data, x, y, None, "lowess"):
            order = np.argsort(xs, kind="stable")
            xs, ys = xs[order], ys[order]
            dl = 0.01 * (xs[-1] - xs[0]) if delta is None else float(delta)
            if not (math.isfinite(dl) and dl >= 0):
                raise ValueError("lowess(): 'delta' must be finite and > 0")
            parts.append(_keyed(frame, data._groups, {"x": xs, "y": _clowess(xs, ys, float(f), iter, dl)}))
        return pl.concat(parts)

    return Step(run, "lowess")


@step
def ksmooth(x: str, y: str, *, kernel="box", bandwidth=0.5, range_x=None, n_points=None, x_points=None):
    """The Nadaraya-Watson kernel regression estimate, as R's ``ksmooth``.

    ``kernel`` is ``"box"`` or ``"normal"``, scaled so their quartiles are at
    ``+/- 0.25 * bandwidth``. Evaluated at ``x_points``, or at ``n_points``
    (by default ``max(100, n)``) equally spaced points over ``range_x`` (by
    default the range of ``x``). ``y`` is missing where no point is in range.
    """
    if kernel not in ("box", "normal"):
        raise ValueError("ksmooth(): kernel must be 'box' or 'normal'")

    def run(data: Data):
        parts = []
        for frame, xs, ys in _xy_parts(data, x, y, None, "ksmooth"):
            order = np.argsort(xs, kind="stable")
            xs, ys = xs[order], ys[order]
            if x_points is None:
                lo, hi = (xs[0], xs[-1]) if range_x is None else map(float, range_x)
                count = max(100, len(xs)) if n_points is None else int(n_points)
                points = lo + np.arange(count) * ((hi - lo) / (count - 1)) if count > 2 else np.array([lo, hi][:count])
                if count > 1:
                    points[-1] = hi
            else:
                points = np.sort(np.asarray([p for p in x_points if p is not None and not math.isnan(p)], dtype=float))
            bw = float(bandwidth) * (.5 if kernel == "box" else .3706506)
            cutoff = bw if kernel == "box" else 4 * bw
            fit = np.full(len(points), np.nan)
            for j, x0 in enumerate(points):
                inside = ~(xs < x0 - cutoff) & ~(xs > x0 + cutoff)
                weight = np.ones(inside.sum()) if kernel == "box" else np.exp(-.5 * (np.abs(xs[inside] - x0) / bw)**2)
                den = weight.sum()
                if den > 0:
                    fit[j] = (weight * ys[inside]).sum() / den
            out = pl.DataFrame({"x": points, "y": fit}).with_columns(pl.col("y").fill_nan(None))
            parts.append(_keyed(frame, data._groups, out.to_dict()))
        return pl.concat(parts)

    return Step(run, "ksmooth")


# smooth.spline -----------------------------------------------------------------------------


def _interv(xt, lxt, x):
    """R's interv on one-based ``xt[1..lxt]``: (left, mflag)."""
    if x < xt[1]:
        return 0, -1
    if x >= xt[lxt]:
        return lxt, 1
    return bisect_right(xt, x, 1, lxt + 1) - 1, 0


class _Bsplvb:
    def __init__(self):
        self.j, self.deltal, self.deltar = 1, [0.0] * 21, [0.0] * 21

    def __call__(self, t, jhigh, index, x, left, biatx):
        if index != 2:
            self.j = 1
            biatx[1] = 1.0
            if self.j >= jhigh:
                return
        while True:
            j = self.j
            jp1 = j + 1
            self.deltar[j] = t[left + j] - x
            self.deltal[j] = x - t[left + 1 - j]
            saved = 0.0
            for i in range(1, j + 1):
                term = biatx[i] / (self.deltar[i] + self.deltal[jp1 - i])
                biatx[i] = saved + self.deltar[i] * term
                saved = self.deltal[jp1 - i] * term
            biatx[jp1] = saved
            self.j = jp1
            if self.j >= jhigh:
                return


def _bsplvd(t, k, x, left, nderiv):
    """de Boor's bsplvd: ``db[m][i]``, the (m-1)-th derivative of the i-th
    B-spline nonzero at ``x`` (one-based)."""
    mhigh = max(min(nderiv, k), 1)
    kp1 = k + 1
    db = [[0.0] * (k + 1) for _ in range(nderiv + 1)]
    bsplvb = _Bsplvb()
    bsplvb(t, kp1 - mhigh, 1, x, left, db[1])
    if mhigh == 1:
        return db
    ideriv = mhigh
    for m in range(2, mhigh + 1):
        jp1mid = 1
        for j in range(ideriv, k + 1):
            db[ideriv][j] = db[1][jp1mid]
            jp1mid += 1
        ideriv -= 1
        bsplvb(t, kp1 - ideriv, 2, x, left, db[1])
    a = [[0.0] * (k + 1) for _ in range(k + 1)]  # a[j][i]
    for i in range(1, k + 1):
        a[i][i] = 1.0
    for m in range(2, mhigh + 1):
        kp1mm = kp1 - m
        fkp1mm = float(kp1mm)
        il, i = left, k
        for _ in range(kp1mm):
            factor = fkp1mm / (t[il + kp1mm] - t[il])
            for j in range(1, i + 1):
                a[i][j] = (a[i][j] - a[i - 1][j]) * factor
            il -= 1
            i -= 1
        for i in range(1, k + 1):
            s = 0.0
            for j in range(max(i, m), k + 1):
                s = a[j][i] * db[m][j] + s
            db[m][i] = s
    return db


def _bvalue(t, bcoef, n, k, x, jderiv):
    """de Boor's bvalue on one-based knots and coefficients."""
    if jderiv >= k:
        return 0.0
    if x != t[n + 1] or t[n + 1] != t[n + k]:
        i, mflag = _interv(t, n + k, x)
        if mflag != 0:
            return 0.0
    else:
        i = n
    km1 = k - 1
    if km1 <= 0:
        return bcoef[i]
    aj, dm, dp = [0.0] * 21, [0.0] * 21, [0.0] * 21
    jcmin = 1
    imk = i - k
    if imk >= 0:
        for j in range(1, km1 + 1):
            dm[j] = x - t[i + 1 - j]
    else:
        jcmin = 1 - imk
        for j in range(1, i + 1):
            dm[j] = x - t[i + 1 - j]
        for j in range(i, km1 + 1):
            aj[k - j] = 0.0
            dm[j] = dm[i]
    jcmax = k
    nmi = n - i
    if nmi >= 0:
        for j in range(1, km1 + 1):
            dp[j] = t[i + j] - x
    else:
        jcmax = k + nmi
        for j in range(1, jcmax + 1):
            dp[j] = t[i + j] - x
        for j in range(jcmax, km1 + 1):
            aj[j + 1] = 0.0
            dp[j] = dp[jcmax]
    for jc in range(jcmin, jcmax + 1):
        aj[jc] = bcoef[imk + jc]
    for j in range(1, jderiv + 1):
        kmj = k - j
        fkmj = float(kmj)
        ilo = kmj
        for jj in range(1, kmj + 1):
            aj[jj] = ((aj[jj + 1] - aj[jj]) / (dm[ilo] + dp[jj])) * fkmj
            ilo -= 1
    for j in range(jderiv + 1, km1 + 1):
        kmj = k - j
        ilo = kmj
        for jj in range(1, kmj + 1):
            aj[jj] = (aj[jj + 1] * dm[ilo] + aj[jj] * dp[jj]) / (dm[ilo] + dp[jj])
            ilo -= 1
    return aj[1]


def _intervals(knot: np.ndarray, lxt: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``_interv`` for every x at once, on 0-based ``knot``: one-based left
    knot positions and R's mflag (-1 below the knots, 1 at or past the last)."""
    left = np.searchsorted(knot[:lxt], x, "right")
    mflag = np.where(x < knot[0], -1, np.where(x >= knot[lxt - 1], 1, 0))
    left = np.where(mflag == 1, lxt, left)
    return left, mflag


def _cubic_basis(knot: np.ndarray, x: np.ndarray, left: np.ndarray) -> np.ndarray:
    """de Boor's bsplvb for the four cubic B-splines nonzero at each ``x``
    (one-based ``left``), column ``q - 1`` holding ``biatx[q]``: the
    recurrence of ``_Bsplvb`` on every point at once, in its order of
    operations."""
    t = np.r_[0.0, knot]  # one-based
    biatx = [np.ones_like(x), None, None, None]
    deltar, deltal = [None] * 4, [None] * 4
    for j in range(1, 4):
        deltar[j] = t[left + j] - x
        deltal[j] = x - t[left + 1 - j]
        saved = np.zeros_like(x)
        for i in range(1, j + 1):
            term = biatx[i - 1] / (deltar[i] + deltal[j + 1 - i])
            biatx[i - 1] = saved + deltar[i] * term
            saved = deltal[j + 1 - i] * term
        biatx[j] = saved
    return np.column_stack(biatx)


def _bvalues(knot: np.ndarray, bcoef: np.ndarray, n: int, x: np.ndarray, jderiv: int) -> np.ndarray:
    """``_bvalue`` with k = 4 at every ``x`` inside the knots (0-based
    ``knot`` and ``bcoef``), where no boundary branch applies."""
    k = 4
    if jderiv >= k:
        return np.zeros_like(x)
    t = np.r_[0.0, knot]  # one-based
    i, mflag = _intervals(knot, n + k, x)
    at_end = (x == t[n + 1]) & (t[n + 1] == t[n + k])
    i = np.where(at_end, n, i)
    outside = ~at_end & (mflag != 0)
    i = np.where(outside, k, i)  # a valid gather; the value is zeroed below
    dm = [None] + [x - t[i + 1 - j] for j in range(1, k)]
    dp = [None] + [t[i + j] - x for j in range(1, k)]
    aj = [None] + [bcoef[i - k + jc - 1] for jc in range(1, k + 1)]
    for j in range(1, jderiv + 1):
        kmj = k - j
        ilo = kmj
        for jj in range(1, kmj + 1):
            aj[jj] = ((aj[jj + 1] - aj[jj]) / (dm[ilo] + dp[jj])) * float(kmj)
            ilo -= 1
    for j in range(jderiv + 1, k):
        kmj = k - j
        ilo = kmj
        for jj in range(1, kmj + 1):
            aj[jj] = (aj[jj + 1] * dm[ilo] + aj[jj] * dp[jj]) / (dm[ilo] + dp[jj])
            ilo -= 1
    return np.where(outside, 0.0, aj[1])


def _sequential_sum(values: np.ndarray) -> float:
    """The values added first to last, as a Fortran loop adds them."""
    return float(np.cumsum(values)[-1]) if len(values) else 0.0


def _sgram(tb, nb):
    """The banded Gram matrix of B-spline second derivatives (with R's
    .3330 in place of 1/3)."""
    sg = [[0.0] * (nb + 1) for _ in range(4)]
    for i in range(1, nb + 1):
        ileft, _ = _interv(tb, nb + 1, tb[i])
        yw1 = _bsplvd(tb, 4, tb[i], ileft, 3)[3]
        right = _bsplvd(tb, 4, tb[i + 1], ileft, 3)[3]
        yw2 = [0.0] + [right[ii] - yw1[ii] for ii in range(1, 5)]
        wpt = tb[i + 1] - tb[i]
        if ileft >= 4:
            for ii in range(1, 5):
                for off in range(4):
                    jj = ii + off
                    if jj <= 4:
                        sg[off][ileft - 4 + ii] += wpt * (yw1[ii] * yw1[jj] + (yw2[ii] * yw1[jj] + yw2[jj] * yw1[ii]) * .50
                                                          + yw2[ii] * yw2[jj] * .3330)
    return sg


def _dpbfa(abd, n, m):
    for j in range(1, n + 1):
        s = 0.0
        ik, jk, mu = m + 1, max(j - m, 1), max(m + 2 - j, 1)
        for k in range(mu, m + 1):
            dot = 0.0
            for q in range(k - mu):
                dot += abd[ik + q][jk] * abd[mu + q][j]
            t = (abd[k][j] - dot) / abd[m + 1][jk]
            abd[k][j] = t
            s += t * t
            ik -= 1
            jk += 1
        s = abd[m + 1][j] - s
        if s <= 0:
            return j
        abd[m + 1][j] = math.sqrt(s)
    return 0


def _dpbsl(abd, n, m, b):
    for k in range(1, n + 1):
        lm = min(k - 1, m)
        la, lb = m + 1 - lm, k - lm
        t = 0.0
        for q in range(lm):
            t += abd[la + q][k] * b[lb + q]
        b[k] = (b[k] - t) / abd[m + 1][k]
    for kb in range(1, n + 1):
        k = n + 1 - kb
        lm = min(k - 1, m)
        la, lb = m + 1 - lm, k - lm
        b[k] = b[k] / abd[m + 1][k]
        t = -b[k]
        for q in range(lm):
            b[lb + q] += t * abd[la + q][k]


def _sinerp(abd, nk):
    """Inner products of the columns of R^{-1}, within the band (pass 1)."""
    p1 = [[0.0] * (nk + 1) for _ in range(5)]
    wjm3, wjm2, wjm1 = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]
    for i in range(1, nk + 1):
        j = nk - i + 1
        c0 = 1.0 / abd[4][j]
        c1 = abd[1][j + 3] * c0 if j <= nk - 3 else 0.0
        c2 = abd[2][j + 2] * c0 if j <= nk - 2 else 0.0
        c3 = abd[3][j + 1] * c0 if j <= nk - 1 else 0.0
        p1[1][j] = 0.0 - (c1 * wjm3[1] + c2 * wjm3[2] + c3 * wjm3[3])
        p1[2][j] = 0.0 - (c1 * wjm3[2] + c2 * wjm2[1] + c3 * wjm2[2])
        p1[3][j] = 0.0 - (c1 * wjm3[3] + c2 * wjm2[2] + c3 * wjm1[1])
        p1[4][j] = (c0**2 + c1**2 * wjm3[1] + 2. * c1 * c2 * wjm3[2] + 2. * c1 * c3 * wjm3[3]
                    + c2**2 * wjm2[1] + 2. * c2 * c3 * wjm2[2] + c3**2 * wjm1[1])
        wjm3[1], wjm3[2], wjm3[3] = wjm2[1], wjm2[2], p1[2][j]
        wjm2[1], wjm2[2] = wjm1[1], p1[3][j]
        wjm1[1] = p1[4][j]
    return p1


class _Sbart:
    """sbart(): the penalized B-spline fit for one data set. The sums over
    the points (stxwx, the fit, leverages and criterion) run on arrays, in
    the order R's loops add them."""

    def __init__(self, xs, ys, ws, ssw, knot, nk, penalty, dofoff, icrit):
        self.n, self.nk, self.knot = len(xs), nk, [0.0, *knot]
        self.x, self.y = [0.0, *xs], [0.0, *ys]
        self.w = [0.0] + [math.sqrt(v) if v > 0 else v for v in ws]
        self.ssw, self.penalty, self.dofoff, self.icrit = ssw, penalty, dofoff, icrit
        t, n = self.knot, self.n
        self.sg = _sgram(t, nk)
        knot0 = np.asarray(knot, dtype=float)
        x, y, w = (np.asarray(v, dtype=float) for v in (xs, ys, self.w[1:]))
        self._x, self._y, self._w = x, y, w
        # stxwx: x is sorted, so R's break past the last knot leaves out the
        # points after it; a point at the last knot moves one interval back.
        ileft, mflag = _intervals(knot0, nk + 1, x)
        past = mflag == 1
        back = past & (x <= np.r_[0.0, knot0][ileft] + .1e-9)
        keep = ~past | back
        ileft = np.where(back, ileft - 1, ileft)
        v = _cubic_basis(knot0, x[keep], ileft[keep])
        ww = w[keep] ** 2
        yk, left_k = y[keep], ileft[keep]
        # One running sum per bin in R's order: point by point, then the four splines.
        j = (left_k[:, None] - 4 + np.arange(1, 5)[None, :])
        xwy = np.bincount(j.ravel(), ((ww * yk)[:, None] * v).ravel(), minlength=nk + 1)
        hs = [np.zeros(nk + 1) for _ in range(4)]
        for off in range(4):
            qs = np.arange(1, 5 - off)
            weights = (ww[:, None] * v[:, qs - 1]) * v[:, qs - 1 + off]
            hs[off] = np.bincount(j[:, qs - 1].ravel(), weights.ravel(), minlength=nk + 1)
        self.xwy, self.hs = xwy, hs
        t1 = _sequential_sum(hs[0][3:nk - 2])
        t2 = _sequential_sum(np.asarray(self.sg[0])[3:nk - 2])
        self.ratio = t1 / t2
        # The leverage basis does not depend on lambda.
        ileft, mflag = _intervals(knot0, nk + 1, x)
        xv = np.where(mflag == -1, t[4] + 1e-11, np.where(mflag == 1, t[nk + 1] - 1e-11, x))
        ileft = np.where(mflag == -1, 4, np.where(mflag == 1, nk, ileft))
        self.basis_j, self.basis = ileft - 3, _cubic_basis(knot0, xv, ileft)

    def solve(self, lam):
        """sslvrg(): coefficients, fit, leverages and criterion at lambda."""
        nk, hs, sg = self.nk, self.hs, self.sg
        abd = [[0.0] * (nk + 1) for _ in range(5)]
        coef = list(self.xwy)
        for i in range(1, nk + 1):
            abd[4][i] = hs[0][i] + lam * sg[0][i]
        for i in range(1, nk):
            abd[3][i + 1] = hs[1][i] + lam * sg[1][i]
        for i in range(1, nk - 1):
            abd[2][i + 2] = hs[2][i] + lam * sg[2][i]
        for i in range(1, nk - 2):
            abd[1][i + 3] = hs[3][i] + lam * sg[3][i]
        info = _dpbfa(abd, nk, 3)
        if info:
            return dict(ier=info, coef=coef[1:], sz=None, lev=None, crit=float("nan"))
        _dpbsl(abd, nk, 3, coef)
        n, w, x, y = self.n, self._w, self._x, self._y
        sz = _bvalues(np.asarray(self.knot[1:]), np.asarray(coef[1:]), nk, x, 0)
        lev, crit = None, float("nan")
        if self.icrit >= 1:
            p1 = np.asarray(_sinerp(abd, nk))
            j, b = self.basis_j, self.basis
            b0, b1, b2, b3 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            lev = (p1[4][j] * b0**2 + 2. * p1[3][j] * b0 * b1 + 2. * p1[2][j] * b0 * b2
                   + 2. * p1[1][j] * b0 * b3 + p1[4][j + 1] * b1**2 + 2. * p1[3][j + 1] * b1 * b2
                   + 2. * p1[2][j + 1] * b1 * b3 + p1[4][j + 2] * b2**2 + 2. * p1[3][j + 2] * b2 * b3
                   + p1[4][j + 3] * b3**2) * w**2
            if self.icrit == 1:
                rss = self.ssw + _sequential_sum(((y - sz) * w)**2)
                df = _sequential_sum(lev)
                sumw = _sequential_sum(w**2)
                crit = (rss / sumw) / ((1. - (self.dofoff + self.penalty * df) / sumw)**2)
            elif self.icrit == 2:
                crit = _sequential_sum((((y - sz) * w) / (1 - lev))**2) / n
            else:
                df = _sequential_sum(lev)
                crit = 3 + (self.dofoff - df)**2
            lev = lev.tolist()
        return dict(ier=0, coef=coef[1:], sz=sz.tolist(), lev=lev, crit=crit)

    def lam(self, spar, is_lambda):
        return spar if is_lambda else self.ratio * 16.0**(spar * 6. - 2.)

    def search(self, lower, upper, tol, eps, maxit):
        """Brent's minimization of the criterion over spar, as sbart();
        the fit returned is the last one evaluated, as R's."""
        gold = 0.381966011250105151795413165634
        big = 1e100
        a, b = lower, upper
        v = a + gold * (b - a)
        w = x = v
        d = e = 0.0
        last = self.solve(self.lam(x, False))
        last_lam = self.lam(x, False)
        fx = fv = fw = last["crit"]
        it = 0
        while last["ier"] == 0:
            xm = (a + b) * .5
            tol1 = eps * abs(x) + tol / 3.
            tol2 = tol1 * 2.
            it += 1
            if abs(x - xm) <= tol2 - (b - a) * .5 or it > maxit:
                break
            golden = abs(e) <= tol1 or fx >= big or fv >= big or fw >= big
            if not golden:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = (q - r) * 2.
                if q > 0.:
                    p = -p
                q = abs(q)
                r = e
                e = d
                if abs(p) >= abs(.5 * q * r) or q == 0. or p <= q * (a - x) or p >= q * (b - x):
                    golden = True
                else:
                    d = p / q
                    u = x + d
                    if u - a < tol2 or b - u < tol2:
                        d = abs(tol1) if xm - x >= 0 else -abs(tol1)
            if golden:
                e = (a - x) if x >= xm else (b - x)
                d = gold * e
            u = x + (d if abs(d) >= tol1 else (abs(tol1) if d >= 0 else -abs(tol1)))
            last_lam = self.lam(u, False)
            last = self.solve(last_lam)
            fu = last["crit"]
            if not math.isfinite(fu):
                fu = 2. * big
            if fu <= fx:
                if u >= x:
                    a = x
                else:
                    b = x
                v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v, fv, w, fw = w, fw, u, fu
                elif fu <= fv or v == x or v == w:
                    v, fv = u, fu
        return x, fx, last, last_lam, it


def _nknots(n: int) -> int:
    """R's .nknots.smspl: the number of inner knots."""
    if n < 50:
        return n
    a1, a2, a3, a4 = math.log2(50), math.log2(100), math.log2(140), math.log2(200)
    if n < 200:
        return int(2**(a1 + (a2 - a1) * (n - 50) / 150))
    if n < 800:
        return int(2**(a2 + (a3 - a2) * (n - 200) / 600))
    if n < 3200:
        return int(2**(a3 + (a4 - a3) * (n - 800) / 2400))
    return int(200 + (n - 3200)**0.2)


class SmoothingSpline(_Fit):
    """A cubic smoothing spline, as R's ``smooth.spline`` object."""

    kind = "smooth.spline"

    def __init__(self, formula, frame, data, opts, has_weights):
        super().__init__(formula, None, frame, data, "smooth_spline")
        label = formula.variables()[0]
        if frame.response_levels is not None or frame.variables[label].is_factor:
            raise ValueError("smooth_spline(): x and y must be numeric")
        x, y = frame.variables[label].values.astype(float), frame.y.astype(float)
        n = len(x)
        self.n, self.prior_weights = n, frame.weights
        if has_weights:
            w = frame.weights.astype(float)
            if np.all(w == 0):
                raise ValueError("smooth_spline(): some weights should be positive")
            w = (w * np.sum(w > 0)) / np.sum(w)
        else:
            w = None
        tol = opts["tol"]
        if tol is None:
            q1, q3 = np.quantile(x, [.25, .75])
            tol = 1e-6 * (q3 - q1)
        if not (math.isfinite(tol) and tol > 0):
            raise ValueError("smooth_spline(): 'tol' must be strictly positive and finite")
        xx = np.round((x - x.mean()) / tol)
        order = np.argsort(x, kind="stable")
        xxs = xx[order]
        first = np.r_[True, xxs[:-1] < xxs[1:]]
        ux = x[order][first]
        nx = len(ux)
        if nx <= 3:
            raise ValueError("smooth_spline(): need at least four unique 'x' values")
        codes = np.searchsorted(xxs[first], xx)
        wi = np.ones(n) if w is None else w
        # bincount adds row by row, as R's sequential group sums do.
        wbar = np.bincount(codes, wi, minlength=nx)
        sy = np.bincount(codes, wi * y, minlength=nx)
        syy = np.bincount(codes, wi * y**2, minlength=nx)
        ybar = sy / np.where(wbar > 0, wbar, 1)
        yssw = _sequential_sum(syy - wbar * ybar**2)
        cv = opts["cv"]
        if cv and nx < n:
            _warn("cross-validation with non-unique 'x' values seems doubtful")
        r_ux = ux[-1] - ux[0]
        xbar = (ux - ux[0]) / r_ux
        all_knots, nknots = opts["all_knots"], opts["nknots"]
        if not isinstance(all_knots, bool):
            inner = np.asarray(all_knots, dtype=float)
            if np.any(np.diff(inner) <= 0):
                raise ValueError("smooth_spline(): numeric 'all_knots' must be strictly increasing")
            if 0 < inner[0] or inner[-1] < 1:
                raise ValueError("smooth_spline(): numeric 'all_knots' must cover [0,1]")
            nknots = len(inner)
        else:
            if all_knots:
                nknots = nx
            elif nknots is None:
                nknots = _nknots(nx)
            else:
                nknots = int(nknots(nx)) if callable(nknots) else int(nknots)
                if nknots < 1:
                    raise ValueError("smooth_spline(): 'nknots' must be at least 1")
                if nknots > nx:
                    raise ValueError("smooth_spline(): cannot use more inner knots than unique 'x' values")
            if all_knots:
                inner = xbar
            else:
                at = np.empty(nknots)
                at[0] = 1.0
                if nknots > 1:
                    at[-1] = float(nx)
                if nknots > 2:
                    by = (nx - 1.0) / (nknots - 1)
                    at[1:-1] = 1.0 + np.arange(1, nknots - 1) * by
                inner = xbar[at.astype(int) - 1]
        knot = np.r_[[inner[0]] * 3, inner, [inner[-1]] * 3]
        nk = nknots + 2
        penalty, dofoff = float(opts["penalty"]), float(opts["df_offset"])
        icrit = 0 if cv is None else 2 if cv else 1
        df = opts["df"]
        if df is not None:
            if cv is None:
                raise ValueError("smooth_spline(): 'cv' must not be None when 'df' is specified")
            if 1 < df <= nx:
                icrit, dofoff = 3, float(df)
            else:
                _warn(f"not using invalid df; must have 1 < df <= n := #{{unique x}} = {nx}")
        control = {"low": -1.5, "high": 1.5, "tol": 1e-4, "eps": 2e-8, "maxit": 500, **(opts["control_spar"] or {})}
        engine = _Sbart(xbar.tolist(), ybar.tolist(), wbar.tolist(), yssw, knot.tolist(), nk, penalty, dofoff, icrit)
        lam_given, spar = opts["lambda"] is not None, opts["spar"]
        self.iterations = None
        if lam_given:
            lam = float(opts["lambda"])
            out = engine.solve(lam)
            spar, crit = float("nan"), out["crit"]
        elif spar is not None:
            lam = engine.lam(float(spar), False)
            out = engine.solve(lam)
            crit = out["crit"]
        else:
            spar, crit, out, lam, self.iterations = engine.search(control["low"], control["high"], control["tol"],
                                                                  control["eps"], int(control["maxit"]))
        ty = out["sz"]
        if out["ier"] > 0:
            small = lam_given or spar < .5
            message = "smoothing parameter value too " + ("extreme" if lam_given else "small" if small else "large")
            if small:
                raise ValueError(f"smooth_spline(): {message}")
            _warn(message + "\nsetting df = 1  __use with care!__")
            ty = [float(np.mean(y))] * nx
        ty = np.asarray(ty)
        if cv is None:
            lev, dfit = None, float("nan")
        else:
            lev = np.asarray(out["lev"]) if out["lev"] is not None else np.full(nx, np.nan)
            dfit = 1.0 if out["ier"] > 0 else float(np.sum(lev))
        resid = y - ty[codes]
        ww = 1.0 if w is None else w
        if cv is None:
            cv_crit = float("nan")
        elif cv:
            denom = np.where(wbar == 0, 1, wbar)
            rr = resid / (1 - (lev[codes] * ww) / denom[codes])
            cv_crit = float(np.mean(rr**2) if w is None else np.sum(w * rr**2) / np.sum(w))
        else:
            mean = np.mean(resid**2) if w is None else np.sum(w * resid**2) / np.sum(w)
            cv_crit = float(mean / (1 - (float(opts["df_offset"]) + penalty * dfit) / n)**2)
        self.x, self.y, self.w, self.yin, self.lev = ux, ty, wbar, ybar, lev
        self.df, self.spar, self.lambda_, self.crit, self.cv_crit = dfit, float(spar), lam, crit, cv_crit
        self.pen_crit = float(np.sum(wbar * (ybar - ty)**2))
        self.ratio = float("nan") if lam_given else engine.ratio
        self.knot, self.nk, self.min, self.range, self.coef = knot, nk, ux[0], r_ux, np.asarray(out["coef"])
        self.cv, self.no_weights = cv, w is None
        self._codes, self._xin, self._yin = codes, x, y
        self.fitted, self.residuals = ty[codes], resid

    def _values(self, x, deriv):
        """predict.smooth.spline.fit: the spline (or a derivative) at ``x``."""
        if deriv < 0 or deriv != int(deriv):
            raise ValueError("predict(): 'deriv' must be a non-negative integer")
        deriv = int(deriv)
        xs = (np.asarray(x, dtype=float) - self.min) / self.range
        out = np.array(xs, dtype=float)
        inside = (xs >= 0) & (xs <= 1)
        out[inside] = _bvalues(np.asarray(self.knot, dtype=float), np.asarray(self.coef, dtype=float),
                               self.nk, xs[inside], deriv)
        left, right = xs < 0, xs > 1
        if left.any() or right.any():
            ends = np.array([self.min, self.min + self.range])
            if deriv == 0:
                value = self._values(ends, 0)
                slope = self._values(ends, 1) * self.range
                out[left] = value[0] + slope[0] * (xs[left] - 0)
                out[right] = value[1] + slope[1] * (xs[right] - 1)
            elif deriv == 1:
                slope = self._values(ends, 1) * self.range
                out[left], out[right] = slope[0], slope[1]
            else:
                out[left | right] = 0.0
        if deriv > 0:
            out = out / self.range**deriv
        return out

    def predict(self, newdata=None, *, deriv=0):
        """The spline at the fit's unique x (no ``newdata``), at a frame's x
        column, or at a sequence of x values: columns ``x`` and ``y``."""
        data = self._data
        if newdata is None:
            if deriv == 0:
                return self._with_key_write(pl.DataFrame({"x": self.x, "y": self.y}))
            x = self.x
        elif isinstance(newdata, (list, tuple, np.ndarray)) or (hasattr(newdata, "dtype") and not hasattr(newdata, "columns")):
            x = np.asarray(newdata, dtype=float)
        else:
            data = Data.read(newdata, None, "predict")
            label = self.formula.variables()[0]
            x = data._lf.select(variable_expr(label, "predict")[0].cast(pl.Float64)).collect().to_series().to_numpy()
        return data.write(pl.DataFrame({"x": x, "y": self._values(x, deriv)}))

    def _with_key_write(self, frame):
        return self._data.write(frame)

    def glance(self):
        return self._with_key(pl.DataFrame({
            "df": [self.df], "spar": [self.spar], "lambda": [self.lambda_], "cv_crit": [self.cv_crit],
            "pen_crit": [self.pen_crit], "crit": [self.crit], "nobs": [self.n]}).with_columns(
            pl.col(c).fill_nan(None) for c in ("df", "spar", "cv_crit", "crit")))

    def augment(self, newdata=None):
        if newdata is not None:
            raise TypeError("augment(): use predict(fit, newdata) for a smoothing spline")
        cols = {".fitted": self.fitted, ".resid": self.residuals}
        return pl.concat([self._rows(), self._padded(pl.DataFrame(cols))], how="horizontal")

    def tidy(self, **kwargs):
        raise TypeError("tidy(): smoothing splines have no coefficient table; use glance() or predict()")

    def vcov_table(self, vcov=None):
        raise TypeError("vcov(): smoothing splines have no coefficient covariance")

    def anova_table(self, test=None):
        raise TypeError("anova(): not available for smoothing splines")

    def anova_type(self, kind, test=None):
        raise TypeError("anova(): not available for smoothing splines")

    def __repr__(self):
        crit = "" if self.cv is None else f"\n{'PRESS(l.o.o. CV)' if self.cv else 'GCV'}: {self.cv_crit:.7g}"
        return (f"smooth_spline({self.formula.text})\n"
                f"Smoothing Parameter  spar= {self.spar:.7g}  lambda= {self.lambda_:.7g}\n"
                f"Equivalent Degrees of Freedom (Df): {self.df:.7g}\n"
                f"Penalized Criterion ({'' if self.no_weights else 'weighted '}RSS): {self.pen_crit:.7g}{crit}")


@step
def smooth_spline(x: str, y: str, *, w=None, df=None, spar=None, lambda_=None, cv=False, all_knots=False,
                  nknots=None, df_offset=0, penalty=1, control_spar=None, tol=None):
    """A cubic smoothing spline, as R's ``smooth.spline``.

    ``smooth_spline(cars, "speed", "dist")`` chooses the smoothing by GCV
    (``cv=True`` for leave-one-out CV, ``cv=None`` for none); ``df`` matches
    equivalent degrees of freedom, ``spar`` fixes R's scaled parameter
    (lambda = r * 256^(3 spar - 1)) and ``lambda_`` fixes lambda. Knots follow
    ``.nknots.smspl`` unless ``all_knots`` or ``nknots``. x values within
    ``tol`` (1e-6 of the IQR) are merged, as in R. ``w`` names a weight
    column. ``glance`` gives df, spar, lambda and the criteria;
    ``predict(fit, x, deriv=)`` evaluates the spline or its derivatives,
    extrapolating linearly; ``augment`` adds fitted values and residuals.
    """
    if spar is not None and lambda_ is not None:
        raise ValueError("smooth_spline(): must not specify both 'spar' and 'lambda_'")
    if not (isinstance(df_offset, (int, float)) and df_offset >= 0):
        raise ValueError("smooth_spline(): df_offset must be numeric and >= 0")
    if not (isinstance(penalty, (int, float)) and penalty > 0):
        raise ValueError("smooth_spline(): penalty must be numeric and > 0")
    for name, spec in (("x", x), ("y", y)):
        if not isinstance(spec, str):
            raise TypeError(f"smooth_spline(): {name} must name a column")
    opts = dict(df=df, spar=spar, cv=cv, all_knots=all_knots, nknots=nknots, df_offset=df_offset,
                penalty=penalty, control_spar=control_spar, tol=tol)
    opts["lambda"] = lambda_
    formula = f"{_backtick(y)} ~ {_backtick(x)}"
    return _model("smooth_spline", lambda f, fr, d: SmoothingSpline(f, fr, d, opts, w is not None),
                  formula, w, None, None, "omit")
