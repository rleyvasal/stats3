"""The studentized range distribution, as R computes it.

``ptukey`` and ``qtukey`` are ports of R's ``ptukey.c`` and ``qtukey.c``
(Copenhaver and Holland's Gauss-Legendre quadrature of Hartley's form of
the range, and the secant iteration started from an approximate quantile),
vectorized over the arguments: every element follows R's branches, loops and
early exits through masks, and the sums are added in R's order. The results
are R's to the last bits R's own ``pnorm``, ``exp`` and ``pow`` decide, so
Tukey intervals and adjusted p-values are R's, not an exact integral's; a
grouped ``tukey_hsd`` evaluates every group's quantile at once.
"""

from __future__ import annotations

import math

import numpy as np

# wprob: Hartley's form of the range, 12-point Legendre quadrature ------------------------

_C1, _C2, _C3 = -30.0, -50.0, 60.0
_BB, _WLAR, _WINCR1, _WINCR2 = 8.0, 3.0, 2.0, 3.0
_XLEG = np.array([
    0.981560634246719250690549090149, 0.904117256370474856678465866119,
    0.769902674194304687036893833213, 0.587317954286617447296702418941,
    0.367831498998180193752691536644, 0.125233408511468915472441369464,
])
_ALEG = np.array([
    0.047175336386511827194615961485, 0.106939325995318430960254718194,
    0.160078328543346226334652529543, 0.203167426723065921749064455810,
    0.233492536538354808760849898925, 0.249147045813402785000562436043,
])
_M_1_SQRT_2PI = 0.398942280401432677939946059934

# ptukey: 16-point Legendre quadrature over the chi distribution of the scale --------------

_EPS1, _EPS2 = -30.0, 1.0e-14
_DHAF, _DQUAR, _DEIGH, _DLARG = 100.0, 800.0, 5000.0, 25000.0
_XLEGQ = np.array([
    0.989400934991649932596154173450, 0.944575023073232576077988415535,
    0.865631202387831743880467897712, 0.755404408355003033895101194847,
    0.617876244402643748446671764049, 0.458016777657227386342419442984,
    0.281603550779258913230460501460, 0.950125098376374401853193354250e-1,
])
_ALEGQ = np.array([
    0.271524594117540948517805724560e-1, 0.622535239386478928628438369944e-1,
    0.951585116824927848099251076022e-1, 0.124628971255533872052476282192,
    0.149595988816576732081501730547, 0.169156519395002538189312079030,
    0.182603415044923588866763667969, 0.189450610455068496285396723208,
])


def _wprob(w: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """R's ``wprob``: the probability integral of the range of ``cc`` normal
    means over [0, w], raised to the ``rr`` groups, elementwise."""
    from scipy.special import ndtr

    qsqz = w * 0.5
    whole = qsqz >= _BB
    with np.errstate(all="ignore"):
        pr_w = 2 * ndtr(qsqz) - 1.0
        pr_w = np.where(pr_w >= np.exp(_C2 / cc), pr_w ** cc, 0.0)
        wincr = np.where(w > _WLAR, _WINCR1, _WINCR2)
        blb = qsqz
        binc = (_BB - qsqz) / wincr
        bub = blb + binc
        einsum = np.zeros_like(w)
        cc1 = cc - 1.0
        floor = np.exp(_C1 / cc1)
        for wi in (1.0, 2.0, 3.0):
            active = wi <= wincr
            elsum = np.zeros_like(w)
            a = 0.5 * (bub + blb)
            b = 0.5 * (bub - blb)
            for jj in range(1, 13):
                if jj > 6:
                    j = (12 - jj) + 1
                    xx = _XLEG[j - 1]
                else:
                    j = jj
                    xx = -_XLEG[j - 1]
                c = b * xx
                ac = a + c
                qexpo = ac * ac
                # ac grows with jj, so R's break at qexpo > C3 leaves out
                # this node and every later one: a mask on the node.
                pplus = 2 * ndtr(ac)
                pminus = 2 * ndtr(ac - w)
                rinsum = (pplus * 0.5) - (pminus * 0.5)
                use = (qexpo <= _C3) & (rinsum >= floor)
                term = (_ALEG[j - 1] * np.exp(-(0.5 * qexpo))) * np.where(use, rinsum, 1.0) ** cc1
                elsum = elsum + np.where(use, term, 0.0)
            elsum = elsum * (((2.0 * b) * cc) * _M_1_SQRT_2PI)
            einsum = einsum + np.where(active, elsum, 0.0)
            blb = bub
            bub = bub + binc
        pr_w = pr_w + einsum
        out = np.where(pr_w <= np.exp(_C1 / rr), 0.0, pr_w ** rr)
        out = np.where(out >= 1.0, 1.0, out)
    return np.where(whole, 1.0, out)


def ptukey(q, cc, df, rr=1.0, lower_tail: bool = True) -> np.ndarray:
    """R's ``ptukey(q, nmeans=cc, df, nranges=rr)``, elementwise over
    broadcast arrays; ``lower_tail=False`` gives R's ``1 - p``."""
    from scipy.special import gammaln

    q, cc, df, rr = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (q, cc, df, rr)))
    shape = q.shape
    q, cc, df, rr = (v.reshape(-1) for v in (q, cc, df, rr))
    out = np.full(q.shape, np.nan)
    bad = np.isnan(q) | np.isnan(cc) | np.isnan(df) | np.isnan(rr) | (df < 2) | (rr < 1) | (cc < 2)
    out[~bad & (q <= 0)] = 0.0
    out[~bad & np.isinf(q) & (q > 0)] = 1.0
    large = ~bad & (df > _DLARG) & (q > 0) & np.isfinite(q)
    if large.any():
        out[large] = _wprob(q[large], rr[large], cc[large])
    todo = np.flatnonzero(~bad & (q > 0) & np.isfinite(q) & (df <= _DLARG))
    if len(todo):
        q_, cc_, df_, rr_ = q[todo], cc[todo], df[todo], rr[todo]
        with np.errstate(all="ignore"):
            f2 = df_ * 0.5
            f2lf = ((f2 * np.log(df_)) - (df_ * math.log(2.0))) - gammaln(f2)
            f21 = f2 - 1.0
            ff4 = df_ * 0.25
            ulen = np.where(df_ <= _DHAF, 1.0, np.where(df_ <= _DQUAR, 0.5, np.where(df_ <= _DEIGH, 0.25, 0.125)))
            f2lf = f2lf + np.log(ulen)
            ans = np.zeros_like(q_)
            stopped = np.zeros(q_.shape, dtype=bool)
            otsum = np.zeros_like(q_)
            for i in range(1, 51):
                twa1 = (2 * i - 1) * ulen
                # The 16 nodes at once: t1 and the range argument per node.
                xu = _XLEGQ[:, None] * ulen[None, :]
                t1_hi = (f2lf + (f21 * np.log(twa1 + xu))) - ((xu + twa1) * ff4)  # nodes 9..16: j = 0..7
                t1_lo = (f2lf + (f21 * np.log(twa1 - xu))) + ((xu - twa1) * ff4)  # nodes 1..8: j = 0..7
                qs_hi = q_ * np.sqrt((xu + twa1) * 0.5)
                qs_lo = q_ * np.sqrt((-xu + twa1) * 0.5)
                t1 = np.concatenate([t1_lo, t1_hi])
                qsqz = np.concatenate([qs_lo, qs_hi])
                contributes = t1 >= _EPS1
                wprb = _wprob(qsqz, np.broadcast_to(rr_, qsqz.shape), np.broadcast_to(cc_, qsqz.shape))
                rotsum = (wprb * np.concatenate([_ALEGQ, _ALEGQ])[:, None]) * np.exp(t1)
                rotsum = np.where(contributes, rotsum, 0.0)
                new = np.zeros_like(q_)
                for jj in range(16):  # added in R's order
                    new = new + rotsum[jj]
                otsum = np.where(stopped, otsum, new)
                stop_now = ~stopped & (i * ulen >= 1.0) & (new <= _EPS2)
                stopped |= stop_now
                ans = np.where(stopped, ans, ans + new)
                if stopped.all():
                    break
            if (otsum > _EPS2).any():
                from stats3.core import _warn

                _warn("ptukey(): full precision may not have been achieved")
            ans = np.where(ans > 1.0, 1.0, ans)
        out[todo] = ans
    if not lower_tail:
        out = 1.0 - out  # R_DT_val for the upper tail: 1 - p, as R loses the tail's bits
    return out.reshape(shape)


def _qinv(p: np.ndarray, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R's ``qinv``: the starting quantile (Odeh and Evans' approximation)."""
    p0, q0 = 0.322232421088, 0.993484626060e-01
    p1, q1 = -1.0, 0.588581570495
    p2, q2 = -0.342242088547, 0.531103462366
    p3, q3 = -0.204231210125, 0.103537752850
    p4, q4 = -0.453642210148e-04, 0.38560700634e-02
    c1, c2, c3, c4, c5 = 0.8832, 0.2368, 1.214, 1.208, 1.4142
    vmax = 120.0
    ps = 0.5 - 0.5 * p
    yi = np.sqrt(np.log(1.0 / (ps * ps)))
    t = yi + ((((yi * p4 + p3) * yi + p2) * yi + p1) * yi + p0) / ((((yi * q4 + q3) * yi + q2) * yi + q1) * yi + q0)
    small = v < vmax
    t = np.where(small, t + (t * t * t + t) / v / 4.0, t)
    q = c1 - c2 * t
    q = np.where(small, q + (-c3 / v + c4 * t / v), q)
    return t * (q * np.log(c - 1.0) + c5)


def qtukey(p, cc, df, rr=1.0) -> np.ndarray:
    """R's ``qtukey(p, nmeans=cc, df, nranges=rr)``, elementwise over
    broadcast arrays: the secant iteration to R's ``eps = 1e-4``."""
    p, cc, df, rr = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (p, cc, df, rr)))
    shape = p.shape
    p, cc, df, rr = (v.reshape(-1).copy() for v in (p, cc, df, rr))
    out = np.full(p.shape, np.nan)
    bad = np.isnan(p) | np.isnan(cc) | np.isnan(df) | np.isnan(rr) | (df < 2) | (rr < 1) | (cc < 2) | (p < 0) | (p > 1)
    out[~bad & (p == 0)] = 0.0
    out[~bad & (p == 1)] = np.inf
    todo = np.flatnonzero(~bad & (p > 0) & (p < 1))
    if not len(todo):
        return out.reshape(shape)
    p_, cc_, df_, rr_ = p[todo], cc[todo], df[todo], rr[todo]
    with np.errstate(all="ignore"):
        x0 = _qinv(p_, cc_, df_)
        valx0 = ptukey(x0, cc_, df_, rr_) - p_
        x1 = np.where(valx0 > 0.0, np.maximum(0.0, x0 - 1.0), x0 + 1.0)
        valx1 = ptukey(x1, cc_, df_, rr_) - p_
        ans = np.zeros_like(p_)
        done = np.zeros(p_.shape, dtype=bool)
        for _ in range(1, 50):
            live = np.flatnonzero(~done)
            step = x1[live] - ((valx1[live] * (x1[live] - x0[live])) / (valx1[live] - valx0[live]))
            valx0[live] = valx1[live]
            x0[live] = x1[live]
            step = np.where(step < 0.0, 0.0, step)
            valx1[live] = ptukey(step, cc_[live], df_[live], rr_[live]) - p_[live]
            x1[live] = step
            ans[live] = step
            done[live] = np.abs(x1[live] - x0[live]) < 0.0001
            if done.all():
                break
        if not done.all():
            from stats3.core import _warn

            _warn("qtukey(): convergence failed")
    out[todo] = ans
    return out.reshape(shape)
