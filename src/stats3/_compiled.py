"""Compiled kernels for R's row-by-row algorithms, used when Numba is
installed (``stats3[fast]``); ``stats3._fast`` loads them.

Each kernel is the NumPy path's algorithm with the same operations in the
same order, so its results are the same bits. Sums of squares take
``fused``: whether R's compiled code on this platform fuses ``d * d + s``
into one rounding (``multivariate.R_FUSES_MULTIPLY_ADD``); the fused form
is LLVM's ``fma`` intrinsic, exact on every platform.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

_BIG = float(np.float32(1e30))


@intrinsic
def _fma(typingctx, x, y, z):
    """x * y + z rounded once: LLVM's fma intrinsic."""
    sig = types.float64(types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        fnty = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()] * 3)
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.fma.f64")
        return builder.call(fn, args)

    return sig, codegen


@nb.njit(cache=True, nogil=True)
def _sq(x, y, fused):  # pragma: no cover - compiled
    """Squared distance summed in column order, as R's kmns.f."""
    total = 0.0
    if fused:
        for j in range(x.shape[0]):
            d = x[j] - y[j]
            total = _fma(d, d, total)
    else:
        for j in range(x.shape[0]):
            d = x[j] - y[j]
            total = total + d * d
    return total


@nb.njit(cache=True, nogil=True)
def _sq_stop(x, y, stop, fused):  # pragma: no cover - compiled
    """``_sq``, or -1.0 once the partial sum reaches ``stop``."""
    total = 0.0
    if fused:
        for j in range(x.shape[0]):
            d = x[j] - y[j]
            total = _fma(d, d, total)
            if total >= stop:
                return -1.0
    else:
        for j in range(x.shape[0]):
            d = x[j] - y[j]
            total = total + d * d
            if total >= stop:
                return -1.0
    return total


# k-means: AS 136 as R's kmns.f ---------------------------------------------------------


@nb.njit(cache=True, nogil=True)
def _transfer(a, c, ic1, ic2, nc, an1, an2, i, l1, l2):  # pragma: no cover - compiled
    al1 = float(nc[l1])
    alw = al1 - 1.0
    al2 = float(nc[l2])
    alt = al2 + 1.0
    for j in range(a.shape[1]):
        c[l1, j] = (c[l1, j] * al1 - a[i, j]) / alw
        c[l2, j] = (c[l2, j] * al2 + a[i, j]) / alt
    nc[l1] -= 1
    nc[l2] += 1
    an2[l1] = alw / al1
    an1[l1] = alw / (alw - 1.0) if alw > 1.0 else _BIG
    an1[l2] = alt / al2
    an2[l2] = alt / (alt + 1.0)
    ic1[i] = l2
    ic2[i] = l1


@nb.njit(cache=True, nogil=True)
def _optra(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, live, indx, fused):  # pragma: no cover - compiled
    m, k = a.shape[0], c.shape[0]
    for l in range(k):
        if itran[l] == 1:
            live[l] = m + 1
    for i in range(m):
        pos = i + 1
        indx += 1
        l1 = ic1[i]
        l2 = ic2[i]
        ll = l2
        if nc[l1] != 1:
            if ncp[l1] != 0:
                d[i] = _sq(a[i], c[l1], fused) * an1[l1]
            r2 = _sq(a[i], c[l2], fused) * an2[l2]
            for l in range(k):
                if (pos >= live[l1] and pos >= live[l]) or l == l1 or l == ll:
                    continue
                rr = r2 / an2[l]
                dc = _sq_stop(a[i], c[l], rr, fused)
                if dc < 0.0:
                    continue
                r2 = dc * an2[l]
                l2 = l
            if r2 >= d[i]:
                ic2[i] = l2
            else:
                indx = 0
                live[l1] = m + pos
                live[l2] = m + pos
                ncp[l1] = pos
                ncp[l2] = pos
                _transfer(a, c, ic1, ic2, nc, an1, an2, i, l1, l2)
        if indx == m:
            return indx
    for l in range(k):
        itran[l] = 0
        live[l] -= m
    return indx


@nb.njit(cache=True, nogil=True)
def _qtran(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, indx, maxqtr, fused):  # pragma: no cover - compiled
    m = a.shape[0]
    icoun = 0
    istep = 0
    while True:
        for i in range(m):
            icoun += 1
            istep += 1
            if istep >= maxqtr:
                return indx, -1
            l1 = ic1[i]
            l2 = ic2[i]
            if nc[l1] != 1:
                if istep <= ncp[l1]:
                    d[i] = _sq(a[i], c[l1], fused) * an1[l1]
                if istep < ncp[l1] or istep < ncp[l2]:
                    r2 = d[i] / an2[l2]
                    if _sq_stop(a[i], c[l2], r2, fused) >= 0.0:
                        icoun = 0
                        indx = 0
                        itran[l1] = 1
                        itran[l2] = 1
                        ncp[l1] = istep + m
                        ncp[l2] = istep + m
                        _transfer(a, c, ic1, ic2, nc, an1, an2, i, l1, l2)
            if icoun == m:
                return indx, maxqtr


@nb.njit(cache=True, nogil=True)
def _means(a, ic1, k):  # pragma: no cover - compiled
    """Cluster means, each cluster's rows added in row order."""
    m, n = a.shape
    c = np.zeros((k, n))
    nc = np.zeros(k, dtype=np.int64)
    for i in range(m):
        nc[ic1[i]] += 1
        for j in range(n):
            c[ic1[i], j] += a[i, j]
    for l in range(k):
        for j in range(n):
            c[l, j] = c[l, j] / float(nc[l])
    return c, nc


@nb.njit(cache=True, nogil=True, error_model="numpy")
def hartigan_wong(a, c, iter_max, fused):  # pragma: no cover - compiled
    """``multivariate._hartigan_wong``: clusters, centres, within sums of
    squares, iterations and ifault (1 empty cluster, 2 not converged, 4 too
    many quick transfers)."""
    m, n = a.shape
    k = c.shape[0]
    c = c.copy()
    ic1 = np.zeros(m, dtype=np.int64)
    ic2 = np.zeros(m, dtype=np.int64)
    wss = np.zeros(k)
    for i in range(m):
        dt0 = _sq(a[i], c[0], fused)
        dt1 = _sq(a[i], c[1], fused)
        ic1[i] = 0
        ic2[i] = 1
        if dt0 > dt1:
            ic1[i] = 1
            ic2[i] = 0
            dt0, dt1 = dt1, dt0
        for l in range(2, k):
            db = _sq_stop(a[i], c[l], dt1, fused)
            if db < 0.0:
                continue
            if db >= dt0:
                dt1 = db
                ic2[i] = l
            else:
                dt1 = dt0
                ic2[i] = ic1[i]
                dt0 = db
                ic1[i] = l
    c, nc = _means(a, ic1, k)
    for l in range(k):
        if nc[l] == 0:
            return ic1, c, wss, 0, 1
    an2 = np.empty(k)
    an1 = np.empty(k)
    for l in range(k):
        an2[l] = nc[l] / (nc[l] + 1.0)
        an1[l] = nc[l] / max(nc[l] - 1.0, 1.0) if nc[l] > 1 else _BIG
    itran = np.ones(k, dtype=np.int64)
    ncp = np.full(k, -1, dtype=np.int64)
    live = np.zeros(k, dtype=np.int64)
    d = np.zeros(m)
    indx = 0
    maxqtr = min(2 ** 31 - 1, 50 * m)
    ifault = 2
    iterations = iter_max + 1
    for ij in range(1, iter_max + 1):
        indx = _optra(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, live, indx, fused)
        if indx == m:
            ifault, iterations = 0, ij
            break
        indx, maxqtr = _qtran(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, indx, maxqtr, fused)
        if maxqtr < 0:
            ifault, iterations = 4, ij
            break
        if k == 2:
            ifault, iterations = 0, ij
            break
        for l in range(k):
            ncp[l] = 0
    c, nc = _means(a, ic1, k)
    for j in range(n):
        for i in range(m):
            da = a[i, j] - c[ic1[i], j]
            if fused:
                wss[ic1[i]] = _fma(da, da, wss[ic1[i]])
            else:
                wss[ic1[i]] = da * da + wss[ic1[i]]
    return ic1, c, wss, iterations, ifault


@nb.njit(cache=True, nogil=True)
def _nearest(a, c, i, previous, fused):  # pragma: no cover - compiled
    best, choice = np.inf, previous
    for j in range(c.shape[0]):
        dd = _sq(a[i], c[j], fused)
        if dd < best:
            best, choice = dd, j
    return choice


@nb.njit(cache=True, nogil=True)
def _wss(a, c, cl, fused):  # pragma: no cover - compiled
    wss = np.zeros(c.shape[0])
    for i in range(a.shape[0]):
        wss[cl[i]] += _sq(a[i], c[cl[i]], fused)
    return wss


@nb.njit(cache=True, nogil=True, error_model="numpy")
def lloyd(a, c, iter_max, fused):  # pragma: no cover - compiled
    """``multivariate._lloyd``: R's kmeans_Lloyd."""
    m, n = a.shape
    k = c.shape[0]
    c = c.copy()
    cl = np.full(m, -1, dtype=np.int64)
    nc = np.zeros(k, dtype=np.int64)
    inew = 0
    iteration = 0
    converged = False
    for it in range(iter_max):
        iteration = it
        updated = False
        for i in range(m):
            inew = _nearest(a, c, i, inew, fused)
            if cl[i] != inew:
                updated = True
                cl[i] = inew
        if not updated:
            converged = True
            break
        nc[:] = 0
        sums = np.zeros((k, n))
        for i in range(m):
            nc[cl[i]] += 1
            for j in range(n):
                sums[cl[i], j] += a[i, j]
        for l in range(k):
            for j in range(n):
                c[l, j] = sums[l, j] / nc[l]
    if not converged:
        iteration = iter_max
    return cl, c, _wss(a, c, cl, fused), iteration + 1, nc


@nb.njit(cache=True, nogil=True, error_model="numpy")
def macqueen(a, c, iter_max, fused):  # pragma: no cover - compiled
    """``multivariate._macqueen``: R's kmeans_MacQueen."""
    m, n = a.shape
    k = c.shape[0]
    c = c.copy()
    cl = np.zeros(m, dtype=np.int64)
    inew = 0
    for i in range(m):
        inew = _nearest(a, c, i, inew, fused)
        cl[i] = inew
    nc = np.zeros(k, dtype=np.int64)
    sums = np.zeros((k, n))
    for i in range(m):
        nc[cl[i]] += 1
        for j in range(n):
            sums[cl[i], j] += a[i, j]
    for l in range(k):
        for j in range(n):
            c[l, j] = sums[l, j] / nc[l]
    iteration = 0
    converged = False
    for it in range(iter_max):
        iteration = it
        updated = False
        for i in range(m):
            inew = _nearest(a, c, i, inew, fused)
            old = cl[i]
            if old != inew:
                updated = True
                cl[i] = inew
                nc[old] -= 1
                nc[inew] += 1
                for j in range(n):
                    c[old, j] += (c[old, j] - a[i, j]) / nc[old]
                    c[inew, j] += (a[i, j] - c[inew, j]) / nc[inew]
        if not updated:
            converged = True
            break
    if not converged:
        iteration = iter_max
    return cl, c, _wss(a, c, cl, fused), iteration + 1, nc


# distances -----------------------------------------------------------------------------


@nb.njit(cache=True, nogil=True, parallel=True)
def euclidean(x, fused):  # pragma: no cover - compiled
    """R's distance.c for every pair, as ``multivariate._distances``:
    columns missing in either row are skipped and the sum scaled up by their
    share; the squares are added in column order."""
    n, nc = x.shape
    out = np.zeros((n, n))
    for i in nb.prange(n):
        for i2 in range(i + 1, n):
            total = 0.0
            count = 0
            for j in range(nc):
                u, v = x[i, j], x[i2, j]
                if np.isnan(u) or np.isnan(v):
                    continue
                dev = u - v
                if np.isnan(dev):
                    continue
                count += 1
                if fused:
                    total = _fma(dev, dev, total)
                else:
                    total = dev * dev + total
            if count == 0:
                d = np.nan
            else:
                if count != nc:
                    total = total / (count / nc)
                d = np.sqrt(total)
            out[i, i2] = d
            out[i2, i] = d
    return out


# hierarchical clustering: F. Murtagh's HCLUST as R's hclust.f ------------------------------


@nb.njit(cache=True, nogil=True)
def hclust(d, iopt, membr):  # pragma: no cover - compiled
    """``multivariate._hclust``'s nearest-neighbour chain: the merged pairs
    (0-based), the criterion at each merge, with R's strict-inequality ties."""
    n = d.shape[0]
    inf = 1e300
    diss = d.copy()
    if iopt == 8:
        for i in range(n):
            for j in range(n):
                diss[i, j] = diss[i, j] * diss[i, j]
    membr = membr.copy()
    flag = np.ones(n, dtype=np.bool_)
    nn = np.zeros(n, dtype=np.int64)
    disnn = np.zeros(n)
    ia = np.zeros(n - 1, dtype=np.int64)
    ib = np.zeros(n - 1, dtype=np.int64)
    crit = np.zeros(n - 1)
    jm = 0
    jj = 0
    for i in range(n - 1):
        dmin = inf
        for j in range(i + 1, n):
            if diss[i, j] < dmin:
                dmin = diss[i, j]
                jm = j
        nn[i] = jm
        disnn[i] = dmin
    for step in range(n - 1):
        dmin = inf
        im = 0
        for i in range(n - 1):
            if flag[i] and disnn[i] < dmin:
                dmin = disnn[i]
                im = i
        jm = nn[im]
        i2 = min(im, jm)
        j2 = max(im, jm)
        ia[step] = i2
        ib[step] = j2
        crit[step] = np.sqrt(dmin) if iopt == 8 else dmin
        flag[j2] = False
        d12 = diss[i2, j2]
        mi = membr[i2]
        mj = membr[j2]
        dmin = inf
        for kk in range(n):
            if not flag[kk] or kk == i2:
                continue
            d1 = diss[i2, kk]
            d2 = diss[j2, kk]
            mk = membr[kk]
            if iopt == 1 or iopt == 8:
                new = ((mi + mk) * d1 + (mj + mk) * d2 - mk * d12) / (mi + mj + mk)
            elif iopt == 2:
                new = min(d1, d2)
            elif iopt == 3:
                new = max(d1, d2)
            elif iopt == 4:
                new = (mi * d1 + mj * d2) / (mi + mj)
            elif iopt == 5:
                new = (d1 + d2) / 2
            elif iopt == 6:
                new = ((d1 + d2) - d12 / 2) / 2
            else:
                new = (mi * d1 + mj * d2 - mi * mj * d12 / (mi + mj)) / (mi + mj)
            diss[i2, kk] = new
            diss[kk, i2] = new
            if kk > i2:
                if new < dmin:
                    dmin = new
                    jj = kk
            elif new < disnn[kk]:
                disnn[kk] = new
                nn[kk] = i2
        membr[i2] = mi + mj
        disnn[i2] = dmin
        nn[i2] = jj
        for i in range(n - 1):
            if flag[i] and (nn[i] == i2 or nn[i] == j2):
                dmin = inf
                for j in range(i + 1, n):
                    if flag[j] and diss[i, j] < dmin:
                        dmin = diss[i, j]
                        jj = j
                nn[i] = jj
                disnn[i] = dmin
    return ia, ib, crit


# concordance: prefix sums over a sweep ---------------------------------------------------


@nb.njit(cache=True, nogil=True)
def _prefix(tree, k):  # pragma: no cover - compiled
    """Sum of the weights at dense keys 0..k (a Fenwick tree, 1-based)."""
    total = 0.0
    i = k + 1
    while i > 0:
        total += tree[i]
        i -= i & (-i)
    return total


@nb.njit(cache=True, nogil=True)
def sweep_sums(order, dense, weight, is_query, qindex, gstart, gend, size):  # pragma: no cover - compiled
    """``survival._sweep_sums``: for each query, the weight of the items
    earlier in the sweep whose dense key is below its own within its group,
    equal to it, and anywhere in its group, by a Fenwick tree over the keys."""
    tree = np.zeros(size + 1)
    nq = 0
    for idx in range(len(is_query)):
        if is_query[idx]:
            nq += 1
    lower = np.zeros(nq)
    equal = np.zeros(nq)
    total = np.zeros(nq)
    for pos in range(len(order)):
        idx = order[pos]
        k = dense[idx]
        if is_query[idx]:
            q = qindex[idx]
            below_group = _prefix(tree, gstart[idx] - 1)
            upto_k = _prefix(tree, k)
            before_k = _prefix(tree, k - 1)
            lower[q] = before_k - below_group
            equal[q] = upto_k - before_k
            total[q] = _prefix(tree, gend[idx] - 1) - below_group
        else:
            i = k + 1
            while i <= size:
                tree[i] += weight[idx]
                i += i & (-i)
    return lower, equal, total


# ARIMA: the Kalman filter loop ------------------------------------------------------------


@nb.njit(cache=True, nogil=True)
def arima_filter(y, missing, t, v, z, a, p, pnew, resid, raw, fast, after, warm, fixed_point):  # pragma: no cover
    """The loop of ``timeseries._arima_like`` on arrays of one dtype (real,
    or complex for the complex-step gradient): a, p and pnew are updated in
    place. Returns (ssq, sumlog, nu, stop, gain): ``stop`` is -1 when the
    series was filtered to its end, else the step at which the prediction
    covariance had converged (``fast``), where the caller takes over."""
    n = len(y)
    r = len(a)
    ssq = np.zeros_like(resid[:1])  # accumulators of the arrays' dtype
    sumlog = np.zeros_like(resid[:1])
    nu = 0
    previous = np.empty_like(p)
    have_previous = False
    gain = ssq[0]
    for i in range(n):
        anew = np.dot(t, a)
        if i > 0:
            pnew[:, :] = np.dot(np.dot(t, p), t.T) + v
            if fast and i >= after + warm and have_previous:
                biggest = 0.0
                change = 0.0
                for j in range(r):
                    for k in range(r):
                        biggest = max(biggest, abs(pnew[j, k].real))
                        change = max(change, abs((pnew[j, k] - previous[j, k]).real))
                if change <= fixed_point * biggest:
                    m = np.dot(pnew, z)
                    gain = np.dot(z, m)
                    if gain.real < 1e4:
                        return ssq[0], sumlog[0], nu, i, gain
            previous[:, :] = pnew
            have_previous = True
        if not missing[i]:
            resid0 = y[i] - np.dot(z, anew)
            m = np.dot(pnew, z)
            gain = np.dot(z, m)
            if gain.real < 1e4:
                nu += 1
                ssq[0] = ssq[0] + resid0 * resid0 / gain
                sumlog[0] = sumlog[0] + np.log(gain)
            resid[i] = resid0 / np.sqrt(gain)
            raw[i] = resid0
            a[:] = anew + m * (resid0 / gain)
            p[:, :] = pnew - np.outer(m, m) / gain
        else:
            a[:] = anew
            p[:, :] = pnew
    return ssq[0], sumlog[0], nu, -1, gain


# smooth.spline: LINPACK's banded Cholesky and the leverage inner products ---------------

from stats3 import smoothing as _smoothing  # noqa: E402

dpbfa = nb.njit(cache=True, nogil=True)(_smoothing._dpbfa)
dpbsl = nb.njit(cache=True, nogil=True)(_smoothing._dpbsl)
sinerp = nb.njit(cache=True, nogil=True)(_smoothing._sinerp)
