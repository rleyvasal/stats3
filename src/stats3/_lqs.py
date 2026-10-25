"""MASS's lqs resampling search (lqs_fitlots), as rlm's MM and LTS starts use it."""

from __future__ import annotations

import itertools
import math

import numpy as np

from stats3._rrng import RRandom
from stats3.core import _special, _warn


def _chi(u, a):
    """lqs.c's Tukey biweight rho, scaled to one at |u| >= a."""
    x = (u / a)**2
    return np.where(x > 1, 1.0, x * (3 + x * (-3 + x)))


def _subsets(n, p, nsamp, seed):
    """Subsets in lqs_fitlots's order: all of them, or R's sample_noreplace."""
    exact = math.comb(n, p)
    if nsamp == "best":
        nsamp = "exact" if exact < 5000 else "sample"
    elif nsamp != "exact" and nsamp > exact:
        _warn(f"lqs(): only {exact} sets, so all sets will be tried")
        nsamp = "exact"
    if nsamp == "exact":
        return np.array(list(itertools.combinations(range(n), p)), dtype=np.int64).reshape(-1, p), True
    nsamp = min(500 * p, 3000) if nsamp == "sample" else int(nsamp)
    rng = RRandom(seed)
    which = np.empty((nsamp, p), dtype=np.int64)
    for t in range(nsamp):
        ind, nn = list(range(n)), n
        for i in range(p):
            j = int(nn * rng.unif_rand())
            which[t, i] = ind[j]
            nn -= 1
            ind[j] = ind[nn]
    return which, False


def _subset_fits(x, y, which):
    """Exact fits through each subset; None for subsets dqrdc2 finds singular."""
    xs, ys = x[which], y[which]
    q, r = np.linalg.qr(xs)
    norms = np.linalg.norm(xs, axis=1)
    diag = np.abs(np.diagonal(r, axis1=1, axis2=2))
    ok = np.all((norms > 0) & (diag >= 1e-7 * norms), axis=1)
    coef = np.full((len(which), x.shape[1]), np.nan)
    if ok.any():
        qty = np.einsum("kij,ki->kj", q[ok], ys[ok])
        coef[ok] = np.linalg.solve(r[ok], qty[..., None])[..., 0]
    return coef, ok


def lqs(x, y, method, nsamp="best", seed=None, k0=1.548):
    """lqs.default(x, y, intercept=FALSE, method=) for "lts" or "S".

    Returns the coefficients, residuals and scale lqs.default reports. For
    "S", residuals and scale are those of its final reweighting loop.
    """
    n, p = x.shape
    if method == "lts":
        quantile = n // 2 + (p + 1) // 2
    else:
        quantile = math.ceil(n / 2)
    if quantile > n - 1:
        raise ValueError(f"lqs(): 'quantile' must be at most {n - 1}")
    which, exact = _subsets(n, p, nsamp, seed)
    coefs, ok = _subset_fits(x, y, which)
    if not ok.any():
        raise ValueError("lqs(): all the samples were singular")
    resid_all = y[None, :] - coefs @ x.T
    best, bestcoef = math.inf, None
    target = (n - p) * .5
    first = True
    for t in np.flatnonzero(ok):
        res = resid_all[t]
        if method == "lts":
            crit = float(np.sum(np.sort(res**2)[:quantile]))
        else:
            if first:
                old = float(np.sort(np.abs(res))[n // 2]) / .6745
                first = False
            else:
                if float(np.sum(_chi(res, k0 * best))) > target:
                    continue
                old = best
            for _ in range(30):
                total = float(np.sum(_chi(res, k0 * old)))
                new = math.sqrt(total / target) * old
                if abs(total / target - 1) < 1e-4:
                    break
                old = new
            crit = new
        if crit < best:
            best, bestcoef = crit, coefs[t]
    best = max(best, 0.0)
    coef = bestcoef.copy()
    resid = y - x @ coef
    if method == "lts":
        c1 = 1 / float(_special().ndtri((n + quantile) / (2 * n)))
        scale = math.sqrt(best / quantile) / math.sqrt(1 - 2 * n * math.exp(-.5 / c1**2) / math.sqrt(2 * math.pi) / (quantile * c1))
        return coef, resid, scale
    from stats3.models import _qr

    scale = best
    for _ in range(30):
        w = (1 - np.minimum(1, np.abs(resid / scale / k0))**2)**2
        sw = np.sqrt(w)
        fit = _qr(x * sw[:, None], 1e-7, y * sw)
        resid = y - x[:, fit.kept] @ fit.coefficients()
        s2 = scale * math.sqrt(float(np.sum(_chi(resid / scale, k0))) / ((n - p) * .5))
        if abs(s2 / scale - 1) < 1e-5:
            break
        scale = s2
    return coef, resid, scale
