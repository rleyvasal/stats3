"""quantreg's rank-inversion intervals (rq.fit.br's ci=TRUE) and xy bootstrap."""

from __future__ import annotations

import itertools
import math

import numpy as np

from stats3._rrng import RRandom
from stats3.core import _special, _warn


def _duals(z, w, basis, sign, tau):
    """Regression rank scores of rows with multiplicity w: w above zero
    residuals, zero below, and basic ones solving X_B'a_B = (1 - tau)X'w - X_N'a_N."""
    a = np.where(sign > 0, w, 0.0)
    a[basis] = 0
    rhs = (1 - tau) * (z.T @ w) - z.T @ a
    a[basis] = np.linalg.solve(z[basis].T, rhs)
    return a


def _line(z, xj, y, basis):
    """Residuals of y - b xj on z through the basis, as c - b g."""
    theta = np.linalg.solve(z[basis], np.column_stack((y[basis], xj[basis])))
    return y - z @ theta[:, 0], xj - z @ theta[:, 1]


def _next_basis(z, xj, y, w, tau, candidates, b, d, moving):
    """The candidate basis that stays optimal just beyond b in direction d.

    ``moving`` are observations whose residuals are zero at b but not in
    the new basis; their signs follow their slopes. Among dual-feasible
    candidates, the least infeasible one wins (rounding, or degenerate ties).
    """
    best = None
    for basis in candidates:
        try:
            c, g = _line(z, xj, y, basis)
        except np.linalg.LinAlgError:
            continue
        sign = np.sign(c - b * g)
        for i in moving:
            sign[i] = -d * np.sign(g[i])
        sign[basis] = 0
        a = _duals(z, w, basis, sign, tau)[basis]
        bad = max(0.0, float(np.max(-a / w[basis])), float(np.max(a / w[basis] - 1)))
        if best is None or bad < best[0]:
            best = (bad, basis, c, g, sign)
            if bad == 0:
                break
    if best is None:
        raise ValueError("rq(): singular basis in rank inversion")
    return best[1:]


def _rank_side(x, y, w, tau, coef, zero, j, d, qn, cutoff):
    """Walk the restricted fit of y - b x_j from b = coef[j] in direction d.

    Each vertex b_k pairs with the rank statistic on the interval beyond it,
    as rqbr reports them; the bound interpolates between the last vertex
    inside the cutoff and the first outside. Without one, it is infinite.
    """
    xj, z = x[:, j], np.delete(x, j, axis=1)
    n, p = x.shape
    scale = math.sqrt(qn * tau * (1 - tau))
    b = coef[j]
    candidates = [list(c) for c in itertools.islice(itertools.combinations(zero, p - 1), 5000)]
    basis, c, g, sign = _next_basis(z, xj, y, w, tau, candidates, b, d, zero)
    inner = (b, 0.0)
    for _ in range(50 * n):
        stat = float(xj @ (_duals(z, w, basis, sign, tau) - (1 - tau) * w)) / scale
        if abs(stat) > cutoff:
            (b2, t2), (b1, t1) = inner, (b, stat)
            return b2 + d * abs(b1 - b2) * (cutoff - abs(t2)) / abs(t1 - t2)
        inner = (b, stat)
        # Only residuals moving toward zero can meet it; signs, not rounded
        # residuals, decide that for those that have just left zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(sign * d * g > 0, np.abs(c - b * g) / np.abs(g), np.inf)
        hit = float(np.min(step))
        if not np.isfinite(hit):
            return d * math.inf
        # Rows meeting zero together (ties in the data) may each enter.
        hits = np.flatnonzero(step <= hit + 1e-10 * max(hit, abs(b), 1e-300)).tolist()
        b = b + d * hit
        candidates = [basis] + [[i for i in basis if i != left] + [k] for k in hits for left in basis]
        basis, c, g, sign = _next_basis(z, xj, y, w, tau, candidates, b, d, [*hits, *basis])
    raise ValueError("rq(): rank inversion did not terminate")


def rank_intervals(x, y, tau, coef, conf_level, iid, fit):
    """rq.fit.br(x, y, tau, alpha=1 - conf_level, ci=TRUE, iid=iid) bounds."""
    n, p = x.shape
    if p == 1:
        return np.full(1, np.nan), np.full(1, np.nan) # rq.fit.br gives no intervals for one coefficient.
    cutoff = float(_special().stdtrit(n - p, 1 - (1 - conf_level) / 2))
    if iid:
        qn = 1 / np.diag(np.linalg.inv(x.T @ x))
    else:
        from stats3.resistant import _bandwidth

        h = _bandwidth(tau, n)
        if tau - h <= 0 or tau + h >= 1:
            raise ValueError("rq(): tau +/- bandwidth leaves (0, 1); use iid=True rank intervals")
        eps = np.finfo(float).eps**(2 / 3)
        dyhat = x @ (fit(x, y, tau + h) - fit(x, y, tau - h))
        if np.any(dyhat <= 0):
            _warn(f"rq(): {100 * np.sum(dyhat <= 0) / n:g} percent fis <=0")
        f = np.maximum(eps, 2 * h / (dyhat - eps))
        sf = np.sqrt(f)
        qn = np.empty(p)
        for j in range(p):
            others = np.delete(x, j, axis=1)
            beta = np.linalg.lstsq(others * sf[:, None], x[:, j] * sf, rcond=None)[0]
            qn[j] = float(np.sum((x[:, j] - others @ beta)**2))
    # Zero-weight rows (all-zero x and y) never change the rank scores' sums,
    # and duplicate rows act as one row counted with their multiplicity.
    keep = np.any(x != 0, axis=1) | (y != 0)
    rows, w = np.unique(np.column_stack((x[keep], y[keep])), axis=0, return_counts=True)
    x, y, w = rows[:, :p], rows[:, p], w.astype(float)
    resid = np.abs(y - x @ coef)
    order = np.argsort(resid, kind="stable")
    tol = 1e-9 * max(1.0, float(np.max(np.abs(y))))
    zero = sorted(order[:max(p, int(np.sum(resid <= tol)))].tolist())
    lower = np.array([_rank_side(x, y, w, tau, coef, zero, j, -1, qn[j], cutoff) for j in range(p)])
    upper = np.array([_rank_side(x, y, w, tau, coef, zero, j, 1, qn[j], cutoff) for j in range(p)])
    return lower, upper


def boot_xy(x, y, tau, replicates, seed, fit):
    """boot.rq's xy-pair bootstrap: R's sample(n, n * R, replace=TRUE) by column."""
    n = len(y)
    draws = RRandom(seed).sample_replace(n, n * replicates).reshape(replicates, n) - 1
    return np.array([fit(x[rows], y[rows], tau) for rows in draws])
