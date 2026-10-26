"""Penalized regression: glmnet's lasso and elastic-net paths, and cv.glmnet."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Data, is_frame, step
from stats3.core import _warn

# glmnet.control() defaults.
_FDEV, _DEVMAX, _EPS, _BIG, _MNLAM, _PMIN = 1e-5, .999, 1e-6, 9.9e35, 5, 1e-9
FAMILIES = ("gaussian", "binomial", "poisson", "multinomial", "cox")


# losses ------------------------------------------------------------------------------
#
# Each loss is glmnet's smooth objective with weights summing to one, as a
# function of the linear predictor. ``grad`` and ``hess`` are with respect to
# the parameters theta, where eta = z @ theta + offset (one column per class
# for the multinomial).


class _Single:
    classes = 1

    def __init__(self, y, w):
        self.y, self.w = y, w

    def grad(self, z, eta):
        return z.T @ self._d1(eta)

    def hess(self, z, eta):
        return z.T @ (self._d2(eta)[:, None] * z)


class _Gaussian(_Single):
    """Squared error, which glmnet fits in "covariance" mode: after one pass
    over the data for Z'WZ, Z'Wy and y'Wy, every lambda's value and
    gradient are O(p^2) in the parameters alone (``value_at`` and
    ``grad_at``), so the path never revisits the rows."""

    def _gram(self, z):
        cached = getattr(self, "_cache", None)
        if cached is None or cached[0] is not z:
            wz = self.w[:, None] * z
            self._cache = (z, z.T @ wz, z.T @ (self.w * self.y), float(self.w @ self.y**2))
        return self._cache[1:]

    def hess(self, z, eta=None):
        return self._gram(z)[0]

    def grad_at(self, z, theta):
        g, b, _ = self._gram(z)
        return g @ theta - b

    def value_at(self, z, theta):
        g, b, yy = self._gram(z)
        return .5 * float(yy - 2 * theta @ b + theta @ g @ theta)

    def value(self, eta):
        return .5 * float(self.w @ (self.y - eta)**2)

    def _d1(self, eta):
        return -self.w * (self.y - eta)

    def _d2(self, eta):
        return self.w

    saturated = 0.0


class _Binomial(_Single):
    def value(self, eta):
        return float(self.w @ (np.logaddexp(0, eta) - self.y * eta))

    def clipped(self, eta):
        # glmnet's dev2(): probabilities held inside [pmin, 1 - pmin].
        p = np.clip(_expit(eta), _PMIN, 1 - _PMIN)
        return -float(self.w @ (self.y * np.log(p) + (1 - self.y) * np.log(1 - p)))

    def _d1(self, eta):
        return -self.w * (self.y - _expit(eta))

    def _d2(self, eta):
        p = _expit(eta)
        return self.w * p * (1 - p)

    @property
    def saturated(self):
        y = self.y
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(y > 0, y * np.log(y), 0) + np.where(y < 1, (1 - y) * np.log(1 - y), 0)
        return -float(self.w @ terms)


class _Poisson(_Single):
    def value(self, eta):
        return float(self.w @ (np.exp(eta) - self.y * eta))

    def _d1(self, eta):
        return -self.w * (self.y - np.exp(eta))

    def _d2(self, eta):
        return self.w * np.exp(eta)

    @property
    def saturated(self):
        y = self.y
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(self.w @ (y - np.where(y > 0, y * np.log(y), 0)))


class _Multinomial:
    def __init__(self, y, w):
        self.y, self.w, self.classes = y, w, y.shape[1]

    def value(self, eta):
        from scipy.special import logsumexp

        return float(self.w @ (logsumexp(eta, axis=1) - (self.y * eta).sum(axis=1)))

    def _p(self, eta):
        from scipy.special import softmax

        return softmax(eta, axis=1)

    def grad(self, z, eta):
        return (z.T @ (-self.w[:, None] * (self.y - self._p(eta)))).ravel(order="F")

    def hess(self, z, eta):
        p, k, q = self._p(eta), self.classes, z.shape[1]
        h = np.zeros((q * k, q * k))
        for a in range(k):
            for b in range(a, k):
                v = self.w * p[:, a] * ((a == b) - p[:, b])
                block = z.T @ (v[:, None] * z)
                h[a*q:(a+1)*q, b*q:(b+1)*q] = block
                h[b*q:(b+1)*q, a*q:(a+1)*q] = block.T
        return h

    @property
    def saturated(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return -float(self.w @ np.where(self.y > 0, self.y * np.log(self.y), 0).sum(axis=1))


class _Cox:
    """The weighted Cox partial likelihood (Breslow or Efron ties), as coxdev."""

    classes = 1

    def __init__(self, time, status, w, efron):
        self.w, self.efron = w, efron
        order = np.argsort(-time, kind="stable")
        self.order, self.time, self.status = order, time, status
        # Distinct event times, each with its deaths and its risk set: the rows
        # (in decreasing time order) up to the last one at or after that time.
        t, d = time[order], status[order]
        events = np.unique(time[status > 0])[::-1]
        self.at_risk = np.searchsorted(-t, -events, side="right")
        self.deaths = [order[(t == e) & (d > 0)] for e in events]
        self.wc = np.array([w[i].sum() for i in self.deaths])
        self.kc = np.array([len(i) for i in self.deaths])

    def _sums(self, z, eta):
        e = self.w * np.exp(eta)
        ez = e[self.order, None] * z[self.order]
        s0 = np.cumsum(e[self.order])[self.at_risk - 1]
        s1 = np.cumsum(ez, axis=0)[self.at_risk - 1]
        return e, s0, s1, ez

    def _ties(self, i):
        """Efron's fractions m/K of the tied deaths' own risk (zero for Breslow)."""
        return np.arange(len(i)) / len(i) if self.efron else np.zeros(len(i))

    def value(self, eta):
        e = self.w * np.exp(eta)
        s0 = np.cumsum(e[self.order])[self.at_risk - 1]
        total = 0.0
        for c, i in enumerate(self.deaths):
            fractions = self._ties(i)
            total += self.wc[c] / self.kc[c] * float(np.log(s0[c] - fractions * e[i].sum()).sum())
        return total - float((self.w * eta) @ (self.status > 0))

    def grad(self, z, eta):
        e, s0, s1, _ = self._sums(z, eta)
        g = -z.T @ (self.w * (self.status > 0))
        for c, i in enumerate(self.deaths):
            fractions = self._ties(i)
            d0, d1 = e[i].sum(), e[i] @ z[i]
            a0 = s0[c] - fractions * d0
            a1 = s1[c] - fractions[:, None] * d1
            g += self.wc[c] / self.kc[c] * (a1 / a0[:, None]).sum(axis=0)
        return g

    def hess(self, z, eta):
        e, s0, s1, ez = self._sums(z, eta)
        zo = z[self.order]
        s2 = np.cumsum(ez[:, :, None] * zo[:, None, :], axis=0)[self.at_risk - 1]
        h = np.zeros((z.shape[1], z.shape[1]))
        for c, i in enumerate(self.deaths):
            fractions = self._ties(i)
            d0, d1 = e[i].sum(), e[i] @ z[i]
            d2 = (e[i, None] * z[i]).T @ z[i]
            for f in fractions:
                a0, a1, a2 = s0[c] - f * d0, s1[c] - f * d1, s2[c] - f * d2
                h += self.wc[c] / self.kc[c] * (a2 / a0 - np.outer(a1, a1) / a0**2)
        return h

    @property
    def saturated(self):
        total = 0.0
        for c in range(len(self.deaths)):
            wc, k = self.wc[c], self.kc[c]
            if wc > 0:
                total += wc * math.log(wc)
                if self.efron:
                    total += wc / k * float(np.log(1 - np.arange(k) / k).sum())
        return total


def _expit(eta):
    from scipy.special import expit

    return expit(eta)


# the solver ----------------------------------------------------------------------------


def _qp(q, c, l1, l2, lo, hi, z, free, maxit):
    """Minimize z'Qz/2 - c'z + sum(l1|z| + l2 z^2/2) within [lo, hi].

    Coordinate descent finds the active set; once the signs and bound
    pattern stop changing, the solution on that set comes from one linear
    solve and is kept if it satisfies the optimality conditions. Returns the
    solution and the number of passes.
    """
    z = z.copy()
    qz = q @ z
    diag = np.diag(q) + l2
    free = [j for j in free if diag[j] > 0]
    pattern, passes = None, 0
    scale = max(1.0, float(np.max(np.abs(c))) if len(c) else 1.0)
    while True:
        passes += 1
        change = 0.0
        for j in free:
            old = z[j]
            r = c[j] - qz[j] + q[j, j] * old
            new = (r - l1[j] if r > l1[j] else r + l1[j] if r < -l1[j] else 0.0) / diag[j]
            new = min(max(new, lo[j]), hi[j])
            if new != old:
                qz += q[:, j] * (new - old)
                z[j] = new
                change = max(change, diag[j] * (new - old)**2)
        now = tuple(int(np.sign(z[j])) * (2 if z[j] in (lo[j], hi[j]) else 1) for j in free)
        if now == pattern or change < 1e-30 * scale:
            polished = _polish(q, c, l1, l2, lo, hi, z, free, scale)
            if polished is not None:
                return polished, passes
        pattern = now
        if passes >= maxit:
            raise _MaxitError()


class _MaxitError(Exception):
    pass


def _polish(q, c, l1, l2, lo, hi, z, free, scale):
    """The exact solution with z's active set, if it is optimal."""
    inner = [j for j in free if z[j] != 0 and lo[j] < z[j] < hi[j]]
    out = z.copy()
    if inner:
        sign = np.sign(z[inner])
        a = q[np.ix_(inner, inner)] + np.diag(l2[inner])
        rest = [j for j in range(len(z)) if j not in set(inner)]
        b = c[inner] - q[np.ix_(inner, rest)] @ z[rest] - l1[inner] * sign
        try:
            out[inner] = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return None
        if np.any(np.sign(out[inner]) != sign) or np.any(out[inner] < lo[inner]) or np.any(out[inner] > hi[inner]):
            return None
    g = q @ out - c + l2 * out
    tol = 1e-9 * scale
    for j in free:
        v = out[j]
        if lo[j] < v < hi[j] and v != 0:
            continue
        up = g[j] + l1[j] if v >= 0 else g[j] - l1[j]   # derivative moving up
        down = -(g[j] - l1[j]) if v <= 0 else -(g[j] + l1[j])   # moving down
        if (v < hi[j] and up < -tol) or (v > lo[j] and down < -tol):
            return None
    return out


def _penalty(theta, l1, l2):
    return float(l1 @ np.abs(theta) + .5 * l2 @ theta**2)


def _solve(loss, z, offset, theta, l1, l2, lo, hi, free, maxit, quadratic):
    """Proximal Newton: exact quadratic subproblems, with a backtracking step."""
    k = loss.classes
    shape = (z.shape[1], k) if k > 1 else (z.shape[1],)

    def eta_of(t):
        return z @ t.reshape(shape, order="F") + offset

    def objective(t):
        return loss.value(eta_of(t)) + _penalty(t, l1, l2)

    passes = 0
    for _ in range(100):
        if quadratic:
            g, h = loss.grad_at(z, theta), loss.hess(z)
        else:
            eta = eta_of(theta)
            g, h = loss.grad(z, eta), loss.hess(z, eta)
        new, used = _qp(h, h @ theta - g, l1, l2, lo, hi, theta, free, maxit)
        passes += used
        d = new - theta
        size = float(np.max(np.abs(d))) if len(d) else 0.0
        if quadratic:
            return new, passes
        step = 1.0
        if size > 1e-8:
            before = objective(theta)
            decrease = float(g @ d) + _penalty(new, l1, l2) - _penalty(theta, l1, l2)
            while objective(theta + step * d) > before + 1e-4 * step * decrease and step > 1e-10:
                step /= 2
        theta = theta + step * d
        if size * step <= 1e-13 * (1 + float(np.max(np.abs(theta)))):
            return theta, passes
    _warn("glmnet(): Newton iterations did not converge")
    return theta, passes


# the path ------------------------------------------------------------------------------


class _Path:
    """glmnet's fit along a lambda path, on the original scale of x."""

    def __init__(self, x, y, family, weights, offset, alpha, nlambda, lambda_min_ratio, lambda_,
                 standardize, intercept, thresh, dfmax, exclude, penalty_factor, lower, upper,
                 maxit, ties):
        x = np.asarray(x, dtype=float)
        n, p = x.shape
        if p < 2:
            raise ValueError("glmnet(): x should be a matrix with 2 or more columns")
        if not np.all(np.isfinite(x)):
            raise ValueError("glmnet(): x has missing values")
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        if len(w) != n:
            raise ValueError("glmnet(): weights need one value per row of x")
        if alpha > 1:
            _warn("glmnet(): alpha >1; set to 1")
            alpha = 1.0
        if alpha < 0:
            _warn("glmnet(): alpha<0; set to 0")
            alpha = 0.0
        if lambda_ is None and lambda_min_ratio >= 1:
            raise ValueError("glmnet(): lambda_min_ratio should be less than 1")
        if lambda_ is not None and np.any(np.asarray(lambda_) < 0):
            raise ValueError("glmnet(): lambdas should be non-negative")
        vp = np.ones(p) if penalty_factor is None else np.asarray(penalty_factor, dtype=float).copy()
        if len(vp) != p:
            raise ValueError("glmnet(): the length of penalty_factor does not match the number of variables")
        if np.any(vp < 0):
            _warn("glmnet(): values for penalty_factor must be between 0 and Inf; negative values set to zero")
            vp = np.maximum(vp, 0)
        excluded = np.zeros(p, dtype=bool)
        if exclude is not None:
            excluded[np.asarray(exclude, dtype=int)] = True
        excluded |= np.isinf(vp)
        vp[excluded] = 1.0
        lo = np.broadcast_to(np.asarray(lower, dtype=float), (p,)).copy()
        hi = np.broadcast_to(np.asarray(upper, dtype=float), (p,)).copy()
        if np.any(lo > 0):
            raise ValueError("glmnet(): lower limits should be non-positive")
        if np.any(hi < 0):
            raise ValueError("glmnet(): upper limits should be non-negative")
        fdev = 0.0 if np.any(lo == 0) or np.any(hi == 0) else _FDEV
        # glmnet's Poisson and Cox paths scale the deviance-change rule up.
        fdev *= {"poisson": 10, "cox": 100}.get(family, 1)
        intercept = intercept and family != "cox"
        self.family, self.alpha, self.intercept, self.p = family, alpha, intercept, p
        self.offset_used = offset is not None
        off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
        keep = np.ones(n, dtype=bool)
        ys = 1.0
        if family == "gaussian":
            y = np.asarray(y, dtype=float) - (off if offset is not None else 0)
            ybar = np.average(y, weights=w) if intercept else 0.0
            self.nulldev = float(w @ (y - ybar)**2)
            if self.nulldev == 0:
                raise ValueError("glmnet(): y is constant; gaussian glmnet fails at standardization step")
            off = np.zeros(n)
            wn = w / w.sum()
            ym = float(wn @ y) if intercept else 0.0
            ys = math.sqrt(float(wn @ (y - ym)**2))
            loss, scale = _Gaussian(y / ys, wn), 1.0
        elif family in ("binomial", "multinomial"):
            counts = np.asarray(y, dtype=float) * w[:, None]
            rows = counts.sum(axis=1)
            keep = rows > 0
            counts, rows, x = counts[keep], rows[keep], x[keep]
            if off.ndim == 1:
                off = off[keep]
            else:
                off = off[keep]
            sw = rows.sum()
            wn = rows / sw
            prop = counts / rows[:, None]
            if family == "binomial":
                if off.ndim == 2:
                    off = off[:, 1] if off.shape[1] == 2 else off[:, 0]
                loss = _Binomial(prop[:, 1], wn)
            else:
                if offset is not None and (off.ndim != 2 or off.shape[1] != prop.shape[1]):
                    raise ValueError("glmnet(): offset should have same shape as y in multinomial call to glmnet")
                if off.ndim == 1:
                    off = np.zeros(prop.shape)
                loss = _Multinomial(prop, wn)
            scale = 2 * sw
        elif family == "poisson":
            y = np.asarray(y, dtype=float)
            if np.any(y < 0):
                raise ValueError("glmnet(): negative responses encountered; not permitted for Poisson family")
            ww = np.maximum(w, 0)
            sw = ww.sum()
            wn = ww / sw
            loss, scale = _Poisson(y, wn), 2 * sw
        else:
            time, status = np.asarray(y[:, 0], dtype=float), np.asarray(y[:, 1], dtype=float)
            if np.any(time <= 0):
                raise ValueError("glmnet(): non-positive event times encountered; not permitted for Cox family")
            ww = np.maximum(w, 0)
            sw = ww.sum()
            wn = ww / sw
            loss, scale = _Cox(time, status, wn, ties == "efron"), 2 * sw
        self.nobs = n
        m = x.shape[0]
        usable = ~excluded & (np.ptp(x, axis=0) > 0) if m else ~excluded
        if not usable.any():
            raise ValueError("glmnet(): all variables are excluded or constant")
        if vp.max() <= 0:
            raise ValueError("glmnet(): at least one penalty factor must be positive")
        vp = vp * p / vp.sum()
        if intercept:
            xm = wn @ x
            xs = np.sqrt(wn @ (x - xm)**2) if standardize else np.ones(p)
        else:
            xm = np.zeros(p)
            xs = np.sqrt(wn @ x**2 - (wn @ x)**2) if standardize else np.ones(p)
        xs = np.where(usable, xs, 1.0)
        zx = np.where(usable, (x - xm) / xs, 0.0)
        z = np.column_stack([np.ones(m), zx])
        k = loss.classes
        q = p + 1
        # Parameters theta: intercept then the p slopes, per class (column-major).
        free_one = ([0] if intercept else []) + [j + 1 for j in range(p) if usable[j]]
        free = [j + c * q for c in range(k) for j in free_one if not (k > 1 and j == 0 and c == k - 1)]
        lo_s = np.r_[-np.inf, np.where(lo <= -_BIG, -np.inf, lo * xs / ys)]
        hi_s = np.r_[np.inf, np.where(hi >= _BIG, np.inf, hi * xs / ys)]
        lo_t, hi_t = np.tile(lo_s, k), np.tile(hi_s, k)
        pen = np.tile(np.r_[0.0, vp], k)
        quadratic = family == "gaussian"
        # The null model: intercepts (and offsets) alone.
        theta = np.zeros(q * k)
        null_free = [j for j in free if j % q == 0]
        if null_free:
            theta, _ = _solve(loss, z, off, theta, np.zeros(q * k), np.zeros(q * k), lo_t, hi_t,
                              null_free, maxit, quadratic)
        shape = (q, k) if k > 1 else (q,)
        eta0 = z @ theta.reshape(shape, order="F") + off
        null_value = loss.value(eta0)
        null_dev = null_value - loss.saturated
        if family != "gaussian":
            self.nulldev = float(scale * null_dev)


        def lambda_max(theta):
            # glmnet starts from the gradient at the first solution (infinite
            # lambda), where unpenalized variables are already fitted.
            if quadratic:
                gradient = loss.grad_at(z, theta)
            else:
                eta = z @ theta.reshape(shape, order="F") + off
                gradient = loss.grad(z, eta)
            gradient = np.abs(gradient.reshape(shape, order="F"))
            if k > 1:
                gradient = gradient.max(axis=1)
            gradient = gradient[1:]
            screen = usable & (vp > 0)
            top = float(np.max(gradient[screen] / vp[screen])) if screen.any() else 0.0
            return top / max(alpha, 1e-3)

        alf = 1.0
        user = lambda_ is not None
        if user:
            ulam = np.sort(np.asarray(lambda_, dtype=float))[::-1] / ys
            nlam = len(ulam)
        else:
            nlam = int(nlambda)
            alf = max(_EPS, lambda_min_ratio)**(1.0 / (nlam - 1))
        ne = p + 1 if dfmax is None else int(dfmax)
        mnl = min(_MNLAM, nlam)
        lambdas, thetas, devs, npasses = [], [], [], 0
        lam = 0.0
        for step_ in range(nlam):
            if user:
                lam = ulam[step_]
            elif step_ == 0:
                lam = _BIG
            elif step_ == 1:
                lam = lambda_max(theta) * alf
            else:
                lam *= alf
            l1, l2 = lam * alpha * pen, lam * (1 - alpha) * pen
            try:
                theta, used = _solve(loss, z, off, theta, l1, l2, lo_t, hi_t, free, maxit, quadratic)
            except _MaxitError:
                _warn(f"glmnet(): convergence for {step_ + 1}th lambda value not reached after "
                      f"maxit={maxit} iterations; solutions for larger lambdas returned")
                break
            npasses += used
            if family == "gaussian":
                dev = 1 - 2 * loss.value_at(z, theta)
            else:
                eta = z @ theta.reshape(shape, order="F") + off
            if family == "gaussian":
                pass
            elif family == "binomial":
                dev = (null_value - loss.clipped(eta)) / null_dev
            else:
                dev = (null_value - loss.value(eta)) / null_dev
            if abs(dev) < 1e-13:
                dev = 0.0   # the null model itself, which glmnet reports as exactly 0
            lambdas.append(lam * ys)
            thetas.append(theta.copy())
            devs.append(dev)
            if len(lambdas) < mnl or user:
                continue
            slopes = theta.reshape(shape, order="F")[1:]
            me = int(np.count_nonzero(slopes.any(axis=1) if k > 1 else slopes))
            if family == "gaussian":
                change = math.inf if dev == 0 else (dev - (devs[-2] if len(devs) > 1 else 0)) / dev
            elif family in ("binomial", "multinomial"):
                change = dev - (devs[-2] if len(devs) > 1 else 0)
            else:
                change = (dev - devs[-mnl]) / dev
            if fdev == 0 and abs(change) < 1e-12:
                change = 0.0
            if me > ne or change < fdev or dev > _DEVMAX:
                break
            if family == "binomial":
                pr = _expit(eta)
                if float(loss.w @ (pr * (1 - pr))) <= (1 + _PMIN) * _PMIN * (1 - _PMIN):
                    break
        self.npasses = npasses
        lambdas = np.array(lambdas)
        if not user and len(lambdas) > 2:
            lambdas[0] = math.exp(2 * math.log(lambdas[1]) - math.log(lambdas[2]))
        self.lambda_ = lambdas
        self.dev_ratio = np.array(devs)
        th = np.array(thetas).reshape((len(thetas), q, k), order="F")   # (L, q, k)
        slopes = th[:, 1:, :] * (ys / xs)[None, :, None]
        a0 = th[:, 0, :] * ys - np.einsum("lpk,p->lk", slopes, xm)
        if not intercept:
            a0 = np.zeros_like(a0)
        if k > 1:
            a0 = a0 - a0.mean(axis=1, keepdims=True)
        self.a0 = a0                               # (L, k)
        self.beta = slopes                         # (L, p, k)
        self.classes = k


# predictions along the path ------------------------------------------------------------


def _interp(lam, s):
    """glmnet's lambda.interp: neighbours and weights for each s."""
    s = np.asarray(s, dtype=float)
    k = len(lam)
    if k == 1:
        ones = np.zeros(len(s), dtype=int)
        return ones, ones, np.ones(len(s))
    span = lam[0] - lam[-1]
    sfrac = (lam[0] - s) / span
    scaled = (lam[0] - lam) / span
    sfrac = np.clip(sfrac, scaled.min(), scaled.max())
    coord = np.interp(sfrac, scaled, np.arange(1, k + 1))
    left, right = np.floor(coord).astype(int) - 1, np.ceil(coord).astype(int) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (sfrac - scaled[right]) / (scaled[left] - scaled[right])
    frac[left == right] = 1
    frac[np.abs(scaled[left] - scaled[right]) < np.finfo(float).eps] = 1
    return left, right, frac


def _r_format(values):
    """R's format() of a numeric vector: 7 significant digits, a common layout."""
    values = [float(v) for v in values]
    digits = []
    for v in values:
        target = float(f"{v:.7g}")
        digits.append(next(d for d in range(1, 8) if float(f"{v:.{d}g}") == target))
    exps = [math.floor(math.log10(abs(v))) if v else 0 for v in values]
    decimals = max(max(0, d - 1 - e) for d, e in zip(digits, exps))
    fixed = [f"{v:.{decimals}f}" for v in values]
    mantissa = max(d - 1 for d in digits)
    sci = [f"{v:.{mantissa}e}" for v in values]
    out = fixed if max(map(len, fixed)) <= max(map(len, sci)) else sci
    width = max(map(len, out))
    return [o.rjust(width) for o in out]


def _s_names(s):
    if isinstance(s, dict):
        return list(s), np.array(list(s.values()), dtype=float)
    values = np.atleast_1d(np.asarray(s, dtype=float))
    return [f"s={v}" for v in _r_format(values)], values


# fits ----------------------------------------------------------------------------------


def _fit_base():
    from stats3.models import _Fit

    return _Fit


class GlmnetFit(_fit_base()):
    """A glmnet path; see ``glmnet``."""

    kind = "glmnet"

    def __init__(self, path, names, classnames, data, formula=None, design=None, frame=None):
        from types import SimpleNamespace

        self.path, self.names, self.classnames = path, names, classnames
        self.family = path.family
        self.formula, self.design, self.frame, self._data = formula, design, frame, data
        self.groups = list(data._groups) if frame is not None else []
        self.key = frame.key if frame is not None else {}
        self.verb = "glmnet"
        if design is not None:
            self.qr = SimpleNamespace(kept=list(range(len(design.names))))

    # -- R's components ----------------------------------------------------------------

    @property
    def lambda_(self):
        return self.path.lambda_

    @property
    def dev_ratio(self):
        return self.path.dev_ratio

    @property
    def nulldev(self):
        return self.path.nulldev

    @property
    def npasses(self):
        return self.path.npasses

    @property
    def df(self):
        b = self.path.beta
        return (b != 0).any(axis=2).sum(axis=1) if self.path.classes > 1 else (b[:, :, 0] != 0).sum(axis=1)

    def _coefs(self, s):
        """Intercepts (L', k) and slopes (L', p, k) at the path's lambdas or at s."""
        a0, beta = self.path.a0, self.path.beta
        if self.family == "cox":
            a0 = np.zeros_like(a0)
        if s is None:
            return a0, beta, [f"s{i}" for i in range(len(self.lambda_))]
        names, values = _s_names(s)
        left, right, frac = _interp(self.lambda_, values)
        a = a0[left] * frac[:, None] + a0[right] * (1 - frac[:, None])
        b = beta[left] * frac[:, None, None] + beta[right] * (1 - frac[:, None, None])
        if self.family == "cox":
            names = [str(i + 1) for i in range(len(values))]
        return a, b, names

    # -- tables -------------------------------------------------------------------------

    def _with_key(self, frame):
        if not self.groups:
            return frame
        keys = self.frame.rows.select(self.groups).head(1)
        return pl.concat([pl.concat([keys] * frame.height) if frame.height else keys.clear(), frame],
                         how="horizontal")

    def _write(self, frame, data=None):
        return (data or self._data).write(frame)

    def __tidy__(self, **kwargs):
        from stats3.models import tidy

        return tidy(self, **kwargs)

    def tidy(self, return_zeros=False, **_):
        a0, beta, _ = self._coefs(None)
        steps = np.arange(1, len(self.lambda_) + 1, dtype=float)
        terms = ["(Intercept)", *self.names]
        parts = []
        for c in range(self.path.classes):
            values = np.column_stack([a0[:, c], beta[:, :, c]])        # (L, 1 + p)
            frame = pl.DataFrame({"term": np.repeat(terms, len(steps)), "step": np.tile(steps, len(terms)),
                                  "estimate": values.T.ravel(),
                                  "lambda": np.tile(self.lambda_, len(terms)),
                                  "dev_ratio": np.tile(self.dev_ratio, len(terms))})
            if self.family == "cox":
                frame = frame.filter(pl.col("term") != "(Intercept)")
            if self.path.classes > 1:
                frame = frame.select(pl.lit(str(self.classnames[c])).alias("class"), pl.all())
            parts.append(frame)
        out = pl.concat(parts)
        if not return_zeros:
            out = out.filter(pl.col("estimate") != 0)
        return self._with_key(out)

    def glance(self, **_):
        return self._with_key(pl.DataFrame({"nulldev": [float(self.nulldev)], "npasses": [int(self.npasses)],
                                            "nobs": [int(self.path.nobs)]}))

    def coef(self, s=None):
        a0, beta, names = self._coefs(s)
        terms = ["(Intercept)", *self.names]
        parts = []
        for c in range(self.path.classes):
            values = np.column_stack([a0[:, c], beta[:, :, c]]).T             # (1 + p, L')
            frame = pl.DataFrame({"term": terms, **{n: values[:, i] for i, n in enumerate(names)}})
            if self.family == "cox":
                frame = frame.slice(1)
            if self.path.classes > 1:
                frame = frame.select(pl.lit(str(self.classnames[c])).alias("class"), pl.all())
            parts.append(frame)
        return self._with_key(pl.concat(parts))

    def _newx(self, newdata, newoffset):
        if newdata is None:
            raise ValueError("predict(): you need to supply new data (newx) for this type")
        if self.design is not None:
            from stats3.models import _Fit

            x, offset, _, data = _Fit._new_design(self, newdata)
            keep = [j for j, n in enumerate(self.design.names) if n != "(Intercept)"]
            x = x[:, keep]
            used = self.path.offset_used
            if newoffset is not None:
                offset, used = np.asarray(newoffset, dtype=float), True
            return x, (offset if used else None), data
        data = Data.read(newdata, None, "predict") if is_frame(newdata) else self._data
        x = _matrix(newdata, "predict")[0]
        if x.shape[1] != self.path.p:
            raise ValueError(f"predict(): the number of variables in newx must be {self.path.p}")
        if self.path.offset_used and newoffset is None:
            raise ValueError("predict(): no newoffset provided for prediction, yet offset used in fit of glmnet")
        return x, (None if newoffset is None else np.asarray(newoffset, dtype=float)), data

    def predict(self, newdata=None, *, s=None, type="link", newoffset=None, newx=None):  # noqa: A002
        types = ("link", "response", "coefficients", "nonzero", "class")
        if type not in types:
            raise ValueError(f"predict(): type must be one of {list(types)}")
        if type == "class" and self.family not in ("binomial", "multinomial"):
            raise ValueError("predict(): type='class' is for binomial and multinomial fits")
        newdata = newx if newdata is None else newdata
        if type == "coefficients":
            return self._write(self.coef(s))
        a0, beta, names = self._coefs(s)
        if type == "nonzero":
            rows = []
            for c in range(self.path.classes):
                for i, name in enumerate(names):
                    for j in np.flatnonzero(beta[i, :, c]):
                        rows.append(({"class": str(self.classnames[c])} if self.path.classes > 1 else {})
                                    | {"s": name, "term": self.names[j]})
            schema = ({"class": pl.String} if self.path.classes > 1 else {}) | {"s": pl.String, "term": pl.String}
            return self._write(self._with_key(pl.DataFrame(rows, schema=schema)))
        x, offset, data = self._newx(newdata, newoffset)
        eta = np.einsum("np,lpk->nlk", x, beta) + a0[None, :, :]
        if offset is not None:
            off = offset if offset.ndim == 2 else offset[:, None]
            if self.family == "binomial" and off.shape[1] == 2:
                off = off[:, 1:]
            eta = eta + off[:, None, :]
        out = {}
        if self.path.classes == 1:
            values = eta[:, :, 0]
            if type == "response":
                values = _expit(values) if self.family == "binomial" else values if self.family == "gaussian" else np.exp(values)
            if type == "class":
                labels = np.asarray([str(c) for c in self.classnames], dtype=object)
                out = {n: labels[(values[:, i] > 0).astype(int)].tolist() for i, n in enumerate(names)}
            else:
                out = {n: values[:, i] for i, n in enumerate(names)}
        else:
            from scipy.special import softmax

            labels = [str(c) for c in self.classnames]
            if type == "class":
                out = {n: [labels[j] for j in eta[:, i, :].argmax(axis=1)] for i, n in enumerate(names)}
            else:
                values = softmax(eta, axis=2) if type == "response" else eta
                for i, n in enumerate(names):
                    for c, label in enumerate(labels):
                        out[f"{label}.{n}" if len(names) > 1 else label] = values[:, i, c]
        return data.write(pl.DataFrame(out))

    def __repr__(self):
        table = pl.DataFrame({"Df": self.df, "%Dev": np.round(100 * self.dev_ratio, 2),
                              "Lambda": self.lambda_})
        head = "glmnet(" + (self.formula.text + ", " if self.formula is not None else "")
        return f"{head}family={self.family!r})\n{table}"


# inputs ---------------------------------------------------------------------------------


def _matrix(x, verb):
    if is_frame(x):
        frame = Data.read(x, None, verb)._lf.collect()
        return frame.cast(pl.Float64).to_numpy(), list(frame.columns)
    array = np.asarray(x, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{verb}(): x should be a matrix with 2 or more columns")
    return array, [f"V{j + 1}" for j in range(array.shape[1])]


def _response(y, family, levels, verb):
    """glmnet's response: counts per class, or (time, status), or numbers."""
    y = np.asarray(y)
    if family in ("binomial", "multinomial"):
        if y.ndim == 2:
            if family == "binomial":
                if y.shape[1] != 2:
                    raise ValueError(f"{verb}(): a binomial matrix response needs two columns")
                return y[:, ::-1].astype(float), (["failure", "success"] if levels is None else levels)
            return y.astype(float), (levels or [str(j + 1) for j in range(y.shape[1])])
        if levels is None:
            values = y.tolist()
            levels = sorted(set(values), key=lambda v: (str(type(v)), v))
            codes = np.array([levels.index(v) for v in values])
        else:
            codes = y.astype(int)
        counts = np.bincount(codes, minlength=len(levels))
        if counts.min() <= 1:
            raise ValueError(f"{verb}(): one multinomial or binomial class has 1 or 0 observations; not allowed")
        if counts.min() < 8:
            _warn(f"{verb}(): one multinomial or binomial class has fewer than 8 observations; dangerous ground")
        if family == "binomial" and len(levels) > 2:
            raise ValueError(f"{verb}(): more than two classes; use multinomial family instead")
        return np.eye(len(levels))[codes], [_label(v) for v in levels]
    if levels is not None:
        raise ValueError(f"{verb}(): a {family} response must be numeric")
    if family == "cox":
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"{verb}(): the Cox model needs a Surv(time, status) response")
        return y.astype(float), None
    return y.astype(float), None


def _label(level):
    from stats3.formula import _level_name

    return _level_name(level)


def _options(family, alpha, nlambda, lambda_min_ratio, lambda_, standardize, intercept, thresh,
             dfmax, exclude, penalty_factor, lower_limits, upper_limits, maxit, cox_ties, verb):
    if family not in FAMILIES:
        raise ValueError(f"{verb}(): family must be one of {list(FAMILIES)}")
    if cox_ties not in ("breslow", "efron"):
        raise ValueError(f"{verb}(): cox_ties must be 'breslow' or 'efron'")
    if isinstance(nlambda, bool) or not isinstance(nlambda, (int, np.integer)) or nlambda < 2:
        raise ValueError(f"{verb}(): nlambda must be an integer of at least 2")
    if not (isinstance(thresh, (int, float)) and thresh > 0):
        raise ValueError(f"{verb}(): thresh must be positive")
    return dict(alpha=float(alpha), nlambda=int(nlambda), lambda_=lambda_,
                standardize=bool(standardize), intercept=bool(intercept), thresh=float(thresh),
                dfmax=dfmax, exclude=exclude, penalty_factor=penalty_factor, lower=lower_limits,
                upper=upper_limits, maxit=int(maxit), ties=cox_ties, lambda_min_ratio=lambda_min_ratio)


def _fit_matrix(x, y, family, weights, offset, options, names, n, p):
    ratio = options["lambda_min_ratio"]
    if ratio is None:
        ratio = 1e-2 if n < p else 1e-4
    exclude = options["exclude"]
    if exclude is not None:
        exclude = [names.index(e) if isinstance(e, str) else int(e) for e in np.atleast_1d(exclude)]
    penalty = options["penalty_factor"]
    if isinstance(penalty, dict):
        unknown = set(penalty) - set(names)
        if unknown:
            raise ValueError(f"glmnet(): penalty_factor names unknown terms {sorted(unknown)}")
        penalty = [penalty.get(name, 1.0) for name in names]
    return _Path(x, y, family, weights, offset, options["alpha"], options["nlambda"], ratio,
                 options["lambda_"], options["standardize"], options["intercept"], options["thresh"],
                 options["dfmax"], exclude, penalty, options["lower"], options["upper"],
                 options["maxit"], options["ties"])


def _surv(formula):
    lhs, sep, rhs = formula.partition("~")
    text = lhs.strip()
    if text.startswith("Surv(") and text.endswith(")"):
        return f"cbind({text[5:-1]}) ~{rhs}", True
    return formula, False


def _frames(data, formula, weights, offset, verb, na_action, extra=None):
    """The model frames for each group, and the formula's design for each."""
    from stats3.formula import build_design, parse_formula
    from stats3.models import NA_ACTIONS, _model_frames

    if na_action not in NA_ACTIONS:
        raise ValueError(f"{verb}(): na_action must be one of {list(NA_ACTIONS)}, got {na_action!r}")
    if na_action == "exclude":
        raise ValueError(f"{verb}(): na_action='exclude' is not supported; use 'omit' or 'fail'")
    text, survival = _surv(formula)
    columns = [c for c in data.columns if c not in data._groups and c not in (extra or [])]
    parsed = parse_formula(text, verb, columns)
    parsed.offset_arg = offset
    frames = _model_frames(data, parsed, weights, offset, verb, na_action)
    out = []
    for frame in frames:
        design = build_design(parsed, frame.variables, len(frame.weights))
        # glmnet fits its own intercept: the design without that column (a
        # view, since the intercept is always the first column).
        keep = [j for j, name in enumerate(design.names) if name != "(Intercept)"]
        x = design.matrix[:, 1:] if keep and keep[0] == 1 and len(keep) == design.matrix.shape[1] - 1 else design.matrix[:, keep]
        out.append((frame, design, x, [design.names[j] for j in keep]))
    used_offset = offset is not None or bool(parsed.offsets)
    return parsed, survival, out, used_offset


def _glmnet_frame(formula, family, weights, offset, options, verb, na_action):
    from stats3.models import Models, _ModelStep

    def _apply(data):
        parsed, survival, pieces, used_offset = _frames(data, formula, weights, offset, verb, na_action)
        if survival and family != "cox":
            raise ValueError(f"{verb}(): a Surv() response needs family='cox'")
        fits = []
        for frame, design, x, names in pieces:
            y, classes = _response(frame.y, family, frame.response_levels, verb)
            path = _fit_matrix(x, y, family, frame.weights, frame.offset if used_offset else None,
                               options, names, *x.shape)
            fits.append(GlmnetFit(path, names, classes, data, parsed, design, frame))
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, verb)


_glmnet_step = step(_glmnet_frame)


def glmnet(*args: Any, x: Any = None, y: Any = None, family: str = "gaussian", weights: Any = None,
           offset: Any = None, alpha: float = 1.0, nlambda: int = 100, lambda_min_ratio: float | None = None,
           lambda_: Any = None, standardize: bool = True, intercept: bool = True, thresh: float = 1e-7,
           dfmax: int | None = None, exclude: Any = None, penalty_factor: Any = None,
           lower_limits: Any = -math.inf, upper_limits: Any = math.inf, maxit: int = 100000,
           cox_ties: str = "breslow", na_action: str = "omit", by: Any = None, as_: str | None = None) -> Any:
    """Lasso and elastic-net regularization paths, as R's glmnet.

    ``glmnet(data, "y ~ .", family="binomial", alpha=0.5)`` takes a formula,
    as glmnetUtils does (model-matrix columns without the intercept;
    ``Surv(time, status) ~ ...`` for ``family="cox"``); ``glmnet(x=X, y=y)``
    takes a matrix (or frame) and a response. Families: gaussian, binomial,
    poisson, multinomial (ungrouped) and cox (right-censored, Breslow or
    Efron ties). The lambda path, standardization, penalty factors, limits,
    weights, offsets and stopping rules follow glmnet 5.1; each solution is
    computed to full precision rather than to glmnet's ``thresh``, which is
    accepted for compatibility. ``tidy``, ``glance``, ``coef(fit, s=)`` and
    ``predict(fit, newdata, s=, type=)`` give R's tables; ``by=`` fits one
    path per group.
    """
    options = _options(family, alpha, nlambda, lambda_min_ratio, lambda_, standardize, intercept,
                       thresh, dfmax, exclude, penalty_factor, lower_limits, upper_limits, maxit,
                       cox_ties, "glmnet")
    if x is None and y is None:
        return _glmnet_step(*args, family=family, weights=weights, offset=offset, options=options,
                            verb="glmnet", na_action=na_action, by=by, as_=as_)
    if args or by is not None:
        raise TypeError("glmnet(): give either data and a formula, or x= and y=")
    return _glmnet_xy(x, y, family, weights, offset, options)


def _glmnet_xy(x, y, family, weights, offset, options, verb="glmnet"):
    matrix, names = _matrix(x, verb)
    levels = None
    if family in ("binomial", "multinomial") and np.asarray(y).ndim == 1:
        values = list(np.asarray(y).tolist())
        levels = sorted(set(values), key=lambda v: (str(type(v)), v))
        y = np.array([levels.index(v) for v in values])
    response, classes = _response(y, family, levels, verb)
    if len(response) != matrix.shape[0]:
        raise ValueError(f"{verb}(): number of observations in y not equal to the number of rows of x")
    data = Data.read(x, None, verb) if is_frame(x) else Data.read(pl.DataFrame(), None, verb)
    path = _fit_matrix(matrix, response, family, weights, offset, options, names, *matrix.shape)
    fit = GlmnetFit(path, names, classes, data)
    fit._xy = (matrix, response, weights, offset)
    return fit


def coef(fit: Any, s: Any = None) -> Any:
    """Coefficients along a glmnet path, or at ``s`` (interpolated, as R)."""
    from stats3.models import Models

    if isinstance(fit, Models):
        return fit._stack([f.coef(s) if not isinstance(f, CVGlmnet) else f.coef(s or "lambda.1se") for f in fit])
    if isinstance(fit, CVGlmnet):
        return fit._write(fit.coef("lambda.1se" if s is None else s))
    if isinstance(fit, GlmnetFit):
        return fit._write(fit.coef(s))
    if hasattr(fit, "coefficients"):
        return fit.coefficients
    raise TypeError("coef(): expected a model")


# cross-validation ---------------------------------------------------------------------

_MEASURES = {"gaussian": ("mse", "deviance", "mae"), "binomial": ("deviance", "class", "auc", "mse", "mae"),
             "poisson": ("deviance", "mse", "mae"), "cox": ("deviance", "C"),
             "multinomial": ("deviance", "class", "mse", "mae")}
_NAMES = {"mse": "Mean-Squared Error", "mae": "Mean Absolute Error", "auc": "AUC",
          "class": "Misclassification Error", "C": "C-index"}
_DEVIANCE = {"gaussian": "Mean-squared Error", "binomial": "Binomial Deviance", "poisson": "Poisson Deviance",
             "cox": "Partial Likelihood Deviance", "multinomial": "Multinomial Deviance"}


def _concordance(score, positive, negative, w_pos, w_neg):
    """Weighted concordance of pairs (positive, negative): ties count a half."""
    total = concordant = 0.0
    for i in positive:
        weight = w_pos[i] * w_neg[negative]
        total += weight.sum()
        concordant += weight @ ((score[i] > score[negative]) + .5 * (score[i] == score[negative]))
    return concordant / total


def _auc(y, prob, weights):
    """survival::concordance(y ~ prob) on the stacked 0/1 rows, as auc.mat."""
    rows = np.arange(len(prob))
    w0, w1 = weights * y[:, 0], weights * y[:, 1]
    return _concordance(prob, rows[w1 > 0], rows[w0 > 0], w1, w0)


def _cindex(pred, time, status, weights):
    """Harrell's C, as survival::concordance(Surv ~ -pred, weights)."""
    total = concordant = 0.0
    for i in np.flatnonzero(status > 0):
        later = np.flatnonzero((time > time[i]) | ((time == time[i]) & (status == 0)))
        weight = weights[i] * weights[later]
        total += weight.sum()
        concordant += weight @ ((pred[i] > pred[later]) + .5 * (pred[i] == pred[later]))
    return concordant / total


def _cox_deviance(x, y, offset, weights, a0, beta):
    """coxnet.deviance with raw weights and Breslow ties, per lambda."""
    loss = _Cox(y[:, 0], y[:, 1], weights, False)
    eta = x @ beta[:, :, 0].T + (0 if offset is None else offset[:, None])
    return np.array([2 * (loss.value(eta[:, l]) - loss.saturated) for l in range(eta.shape[1])])


class CVGlmnet(_fit_base()):
    """cv.glmnet's result: the cross-validated error along the path."""

    kind = "cv.glmnet"

    def __init__(self, fit, lambda_, cvm, cvsd, nzero, measure, name, foldid):
        self.glmnet_fit, self.lambda_, self.cvm, self.cvsd = fit, lambda_, cvm, cvsd
        self.cvup, self.cvlo, self.nzero = cvm + cvsd, cvm - cvsd, nzero
        self.type_measure, self.name, self.foldid = measure, name, foldid
        self.groups, self.key, self.frame, self._data = fit.groups, fit.key, fit.frame, fit._data
        self.formula, self.design, self.verb = fit.formula, fit.design, "cv_glmnet"
        sign = -1 if measure in ("auc", "C") else 1
        m = sign * cvm
        self.lambda_min = float(np.max(lambda_[m <= np.nanmin(m)]))
        index_min = int(np.flatnonzero(lambda_ == self.lambda_min)[0])
        self.lambda_1se = float(np.max(lambda_[m <= (m + cvsd)[index_min]]))
        self.index = (index_min + 1, int(np.flatnonzero(lambda_ == self.lambda_1se)[0]) + 1)

    _with_key = GlmnetFit._with_key
    _write = GlmnetFit._write
    __tidy__ = GlmnetFit.__tidy__

    def _s(self, s):
        if isinstance(s, str):
            if s not in ("lambda.1se", "lambda.min"):
                raise ValueError("cv_glmnet: s must be 'lambda.1se', 'lambda.min' or numbers")
            return {s: getattr(self, s.replace(".", "_"))}
        return s

    def tidy(self, **_):
        return self._with_key(pl.DataFrame({"lambda": self.lambda_, "estimate": self.cvm, "std_error": self.cvsd,
                                            "conf_low": self.cvlo, "conf_high": self.cvup,
                                            "nzero": self.nzero.astype(np.int64)}))

    def glance(self, **_):
        return self._with_key(pl.DataFrame({"lambda_min": [self.lambda_min], "lambda_1se": [self.lambda_1se],
                                            "nobs": [int(self.glmnet_fit.path.nobs)]}))

    def coef(self, s="lambda.1se"):
        return self.glmnet_fit.coef(self._s(s))

    def predict(self, newdata=None, *, s="lambda.1se", **kwargs):
        return self.glmnet_fit.predict(newdata, s=self._s(s), **kwargs)

    def __repr__(self):
        return (f"cv_glmnet(type_measure={self.type_measure!r})\n"
                f"lambda_min={self.lambda_min:.6g}, lambda_1se={self.lambda_1se:.6g}\n{self.tidy()}")


def _cv(fit, x, y, weights, offset, options, foldid, measure, grouped, names, verb):
    family = fit.family
    n = x.shape[0]
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    offset = None if offset is None else np.asarray(offset, dtype=float)
    foldid = np.asarray(foldid).astype(int)
    if len(foldid) != n:
        raise ValueError(f"{verb}(): foldid needs one fold per row used")
    nfolds = int(foldid.max())
    if nfolds < 3:
        raise ValueError(f"{verb}(): nfolds must be bigger than 3; nfolds=10 recommended")
    if measure == "default":
        measure = _MEASURES[family][0]
    if measure not in _MEASURES[family]:
        _warn(f"{verb}(): only {', '.join(_MEASURES[family])} available as type_measure for {family} models; "
              f"{_MEASURES[family][0]} used instead")
        measure = _MEASURES[family][0]
    lam = fit.lambda_
    nlam = len(lam)
    if family == "multinomial":
        per_class = (fit.path.beta != 0).sum(axis=1)
        nz = np.ceil(np.median(per_class, axis=1)).astype(int)
    else:
        nz = fit.df
    folds = []
    for i in range(1, nfolds + 1):
        train = foldid != i
        path = _fit_matrix(x[train], y[train], family, weights[train],
                           None if offset is None else offset[train], options, names, int(train.sum()), x.shape[1])
        folds.append(GlmnetFit(path, names, fit.classnames, fit._data))
    k = fit.path.classes
    if family == "cox":
        if n / nfolds < 10 and not grouped:
            grouped = True
        cvraw = np.full((nfolds, nlam), np.nan)
        predmat = np.full((n, nlam), np.nan)
        for i, f in enumerate(folds):
            held = foldid == i + 1
            a0, beta, _ = f._coefs(dict(zip(map(str, range(nlam)), lam)))
            if measure == "deviance":
                if grouped:
                    full = _cox_deviance(x, y, offset, weights, a0, beta)
                    rest = _cox_deviance(x[~held], y[~held], None if offset is None else offset[~held],
                                         weights[~held], a0, beta)
                    cvraw[i] = full - rest
                else:
                    cvraw[i] = _cox_deviance(x[held], y[held], None if offset is None else offset[held],
                                             weights[held], a0, beta)
            predmat[held] = x[held] @ beta[:, :, 0].T
        fold_weights = np.array([weights[foldid == i].sum() for i in range(1, nfolds + 1)])
        if measure == "deviance":
            cvraw = cvraw / fold_weights[:, None]
        else:
            for i in range(nfolds):
                held = foldid == i + 1
                cvraw[i] = [_cindex(predmat[held, j], y[held, 0], y[held, 1], weights[held]) for j in range(nlam)]
        return _cvstats(cvraw, fold_weights, np.full(nlam, nfolds), lam, nz, measure, family), measure
    predmat = np.full((n, nlam, k), np.nan)
    for i, f in enumerate(folds):
        held = foldid == i + 1
        a0, beta, _ = f._coefs(dict(zip(map(str, range(nlam)), lam)))
        eta = np.einsum("np,lpk->nlk", x[held], beta) + a0[None]
        if offset is not None:
            off = offset if offset.ndim == 2 else offset[:, None]
            eta = eta + off[held][:, None, :]
        predmat[held] = eta
    if family == "gaussian":
        diff = y[:, None] - predmat[:, :, 0]
        cvraw = np.abs(diff) if measure == "mae" else diff**2
    elif family == "poisson":
        eta = predmat[:, :, 0]
        if measure == "deviance":
            with np.errstate(divide="ignore", invalid="ignore"):
                devy = np.where(y > 0, y * np.log(y), 0) - y
            cvraw = 2 * (devy[:, None] - (y[:, None] * eta - np.exp(eta)))
        else:
            diff = y[:, None] - np.exp(eta)
            cvraw = np.abs(diff) if measure == "mae" else diff**2
    else:
        ywt = y.sum(axis=1)
        prop = y / ywt[:, None]
        if family == "binomial":
            pr = _expit(predmat[:, :, 0])
            probs = np.stack([1 - pr, pr], axis=2)
        else:
            from scipy.special import softmax

            probs = softmax(predmat, axis=2)
        if measure == "auc":
            if n / nfolds < 10:
                _warn(f"{verb}(): too few (< 10) observations per fold for type_measure='auc'; "
                      "changed to type_measure='deviance'")
                measure = "deviance"
            else:
                cvraw = np.array([[_auc(y[foldid == i], pr[foldid == i, j], weights[foldid == i])
                                   for j in range(nlam)] for i in range(1, nfolds + 1)])
                fold_weights = np.array([weights[foldid == i].sum() for i in range(1, nfolds + 1)])
                return _cvstats(cvraw, fold_weights, np.full(nlam, nfolds), lam, nz, measure, family), measure
        weights = weights * ywt
        big = prop[:, None, :]
        if measure == "deviance":
            clipped = np.clip(probs, 1e-5, 1 - 1e-5)
            with np.errstate(divide="ignore", invalid="ignore"):
                ly = np.where(big > 0, big * np.log(big), 0)
            cvraw = (2 * (ly - big * np.log(clipped))).sum(axis=2)
        elif measure == "mse":
            cvraw = ((big - probs)**2).sum(axis=2)
        elif measure == "mae":
            cvraw = np.abs(big - probs).sum(axis=2)
        elif family == "binomial":
            cvraw = prop[:, 0:1] * (pr > .5) + prop[:, 1:2] * (pr <= .5)
        else:
            choice = probs.argmax(axis=2)
            cvraw = 1 - np.take_along_axis(np.broadcast_to(big, probs.shape), choice[:, :, None], axis=2)[:, :, 0]
    if n / nfolds < 3 and grouped:
        _warn(f"{verb}(): option grouped=FALSE enforced, since < 3 observations per fold")
        grouped = False
    if grouped:
        cvraw = np.where(np.isinf(cvraw), np.nan, cvraw)
        fold_weights = np.array([weights[foldid == i].sum() for i in range(1, nfolds + 1)])
        cvraw = np.array([_weighted_mean(cvraw[foldid == i], weights[foldid == i]) for i in range(1, nfolds + 1)])
        return _cvstats(cvraw, fold_weights, np.full(nlam, nfolds), lam, nz, measure, family), measure
    count = n - np.isnan(cvraw).sum(axis=0)
    return _cvstats(cvraw, weights, count, lam, nz, measure, family), measure


def _weighted_mean(values, weights):
    ok = ~np.isnan(values)
    return (np.where(ok, values, 0) * weights[:, None]).sum(axis=0) / (ok * weights[:, None]).sum(axis=0)


def _cvstats(cvraw, weights, count, lam, nz, measure, family):
    cvm = _weighted_mean(cvraw, weights)
    cvsd = np.sqrt(_weighted_mean((cvraw - cvm)**2, weights) / (count - 1))
    keep = ~np.isnan(cvsd)
    return lam[keep], cvm[keep], cvsd[keep], np.asarray(nz)[keep]


def _cv_result(fit, stats, measure, foldid):
    lam, cvm, cvsd, nz = stats
    name = _DEVIANCE[fit.family] if measure == "deviance" else _NAMES[measure]
    return CVGlmnet(fit, lam, cvm, cvsd, nz, measure, name, foldid)


def _folds(foldid, nfolds, n, seed):
    if foldid is not None:
        return np.asarray(foldid)
    rng = np.random.default_rng(seed)
    return rng.permutation(np.resize(np.arange(1, nfolds + 1), n))


def _cv_frame(formula, family, weights, offset, options, verb, na_action, nfolds, foldid, measure,
              grouped, seed):
    from stats3.models import Models, _ModelStep

    def _apply(data):
        fold_column = data.column(foldid, "foldid", verb) if isinstance(foldid, str) else None
        parsed, survival, pieces, used_offset = _frames(data, formula, weights, offset, verb, na_action,
                                                        [fold_column] if fold_column else None)
        if survival and family != "cox":
            raise ValueError(f"{verb}(): a Surv() response needs family='cox'")
        if fold_column is not None:
            all_folds = data._lf.select(pl.col(fold_column)).collect().to_series().to_numpy()
        fits = []
        for frame, design, x, names in pieces:
            y, classes = _response(frame.y, family, frame.response_levels, verb)
            off = frame.offset if used_offset else None
            path = _fit_matrix(x, y, family, frame.weights, off, options, names, *x.shape)
            fit = GlmnetFit(path, names, classes, data, parsed, design, frame)
            if fold_column is not None:
                ids = all_folds[frame.index]
            elif foldid is not None and not data._groups:
                ids = np.asarray(foldid)
            elif foldid is not None:
                raise ValueError(f"{verb}(): with by=, give foldid as a column name")
            else:
                ids = _folds(None, nfolds, x.shape[0], seed)
            stats, used = _cv(fit, x, y, frame.weights, off, options, ids, measure, grouped, names, verb)
            fits.append(_cv_result(fit, stats, used, ids))
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, verb)


_cv_step = step(_cv_frame)


def cv_glmnet(*args: Any, x: Any = None, y: Any = None, family: str = "gaussian", weights: Any = None,
              offset: Any = None, type_measure: str = "default", nfolds: int = 10, foldid: Any = None,
              grouped: bool = True, seed: int | None = None, alpha: float = 1.0, nlambda: int = 100,
              lambda_min_ratio: float | None = None, lambda_: Any = None, standardize: bool = True,
              intercept: bool = True, thresh: float = 1e-7, dfmax: int | None = None, exclude: Any = None,
              penalty_factor: Any = None, lower_limits: Any = -math.inf, upper_limits: Any = math.inf,
              maxit: int = 100000, cox_ties: str = "breslow", na_action: str = "omit", by: Any = None,
              as_: str | None = None) -> Any:
    """Cross-validation for glmnet, as R's cv.glmnet.

    Takes glmnet's arguments, plus ``type_measure`` (``"default"``,
    ``"mse"``, ``"deviance"``, ``"class"``, ``"auc"``, ``"mae"`` or
    ``"C"``), ``nfolds`` and ``foldid`` (a column name, or one fold number
    per row used). With the same ``foldid`` the result is R's; without it,
    folds are drawn with NumPy (``seed=``), so they differ from R's
    ``sample()``. ``tidy`` gives cvm and cvsd per lambda (``estimate``,
    ``std_error``, ``conf_low``, ``conf_high``, ``nzero``); ``glance`` gives
    ``lambda_min`` and ``lambda_1se``. ``coef``/``predict`` use ``s=
    "lambda.1se"`` by default, or ``"lambda.min"`` or numbers.
    """
    options = _options(family, alpha, nlambda, lambda_min_ratio, lambda_, standardize, intercept,
                       thresh, dfmax, exclude, penalty_factor, lower_limits, upper_limits, maxit,
                       cox_ties, "cv_glmnet")
    if lambda_ is not None and len(np.atleast_1d(lambda_)) < 2:
        raise ValueError("cv_glmnet(): need more than one value of lambda")
    if x is None and y is None:
        return _cv_step(*args, family=family, weights=weights, offset=offset, options=options,
                        verb="cv_glmnet", na_action=na_action, nfolds=nfolds, foldid=foldid,
                        measure=type_measure, grouped=grouped, seed=seed, by=by, as_=as_)
    if args or by is not None:
        raise TypeError("cv_glmnet(): give either data and a formula, or x= and y=")
    fit = _glmnet_xy(x, y, family, weights, offset, options, "cv_glmnet")
    matrix, response, _, _ = fit._xy
    ids = _folds(foldid, nfolds, matrix.shape[0], seed)
    stats, used = _cv(fit, matrix, response, weights, offset, options, ids, type_measure, grouped,
                      fit.names, "cv_glmnet")
    return _cv_result(fit, stats, used, ids)
