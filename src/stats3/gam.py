"""Generalized additive models: mgcv's ``gam``.

``gam(data, "y ~ s(x1) + s(x2, k=20) + f", family=...)`` fits penalized
regression smooths with smoothing parameters chosen as mgcv 1.9 chooses
them: GCV (unknown scale) or UBRE (known scale) for ``method="GCV.Cp"``, the
default, or Laplace-approximate REML for ``method="REML"``. The bases,
penalty scaling and identifiability constraints are ports of mgcv's (see
``gam_smooth``). Non-Gaussian fits iterate penalized IRLS to convergence for
every trial set of smoothing parameters, as mgcv's default outer iteration
does, with full Newton weights for non-canonical links.

mgcv stops its optimizers early (its Newton method at a relative gradient of
1e-7, and the Gaussian-additive GCV method, ``magic``, at 1e-6 in score and
1e-2 in gradient). stats3 optimizes the same criteria to full precision, so
results agree with mgcv's to the accuracy mgcv reaches, which the tests
state next to each tolerance.

``tidy`` gives ``summary.gam``'s parametric table (``parametric=False``: the
smooth terms with edf, Ref.df, F or Chi.sq and p-values, as broom's
``tidy.gam``), ``glance`` the fit statistics, ``augment`` and ``predict``
fitted values and standard errors (``type="link"``, ``"response"`` or
``"terms"``), ``anova`` ``anova.gam``, and ``vcov`` the Bayesian covariance
``Vp`` (``vcov="Ve"`` for the frequentist one, ``"Vc"`` for the one
corrected for smoothing-parameter uncertainty under REML).
"""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Data, step
from stats3.core import _check_conf_level, _special, _warn
from stats3.formula import Formula, Variable, build_design, parse_formula, variable_expr
from stats3.gam_smooth import SmoothSpec, smooth_con
from stats3.models import (
    NA_ACTIONS,
    GeneralizedLinearModel,
    Models,
    _as_family,
    _contrasts,
    _Fit,
    _glm_fit,
    _model_frames,
    _ModelStep,
    _table_text,
)

_EPS = np.finfo(float).eps
_METHODS = ("GCV.Cp", "REML")
_RHO_MAX = 25.0
_RHO_FAR = 15.0  # beyond this, a log smoothing parameter heading outward is taken to its limit


# formula --------------------------------------------------------------------


def _top_level_terms(rhs: str) -> list[tuple[str, str]]:
    """Split a right-hand side at top-level + and -, keeping signs."""
    out, depth, start, sign, quote = [], 0, 0, "+", None
    i = 0
    while i < len(rhs):
        c = rhs[i]
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'`":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c in "+-" and depth == 0:
            piece = rhs[start:i].strip()
            if piece:
                out.append((sign, piece))
            sign, start = c, i + 1
        i += 1
    piece = rhs[start:].strip()
    if piece:
        out.append((sign, piece))
    return out


def _r_value(node: ast.AST, text: str) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in ("TRUE", "T", "FALSE", "F", "NA"):
        return {"TRUE": True, "T": True, "FALSE": False, "F": False, "NA": None}[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_r_value(node.operand, text)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "c":
        return [_r_value(a, text) for a in node.args]
    raise ValueError(f"gam(): cannot read {ast.get_source_segment(text, node)!r} in a smooth term")


def _smooth_spec(text: str) -> SmoothSpec:
    tree = ast.parse(text.replace("`", ""), mode="eval").body
    kind = tree.func.id
    terms = [ast.get_source_segment(text, a).replace(" ", "") for a in tree.args]
    if not terms:
        raise ValueError(f"gam(): {text!r} names no variables")
    opts = {kw.arg: kw.value for kw in tree.keywords}
    known = {"k", "bs", "m", "by", "fx"} | ({"d", "np"} if kind == "te" else set())
    unknown = set(opts) - known
    if unknown:
        raise ValueError(f"gam(): {kind}() arguments {sorted(unknown)} are not supported")
    by = None
    if "by" in opts:
        node = opts["by"]
        by = None if isinstance(node, ast.Name) and node.id == "NA" else ast.get_source_segment(text, node).replace(" ", "")
    values = {name: _r_value(node, text) for name, node in opts.items() if name != "by"}
    fx = bool(values.get("fx", False))
    if kind == "s":
        bs = values.get("bs", "tp")
        k = values.get("k", -1)
        m = values.get("m")
        if isinstance(bs, list) or isinstance(k, list):
            raise ValueError("gam(): s() takes one bs and one k; use te() for tensor products")
        return SmoothSpec("s", terms, [bs], [int(k)], [m], by, fx, text)
    if values.get("d") is not None and any(int(v) != 1 for v in np.atleast_1d(values["d"])):
        raise ValueError("gam(): te() margins of more than one variable are not supported")
    if values.get("np", True) is False:
        raise ValueError("gam(): te(np=FALSE) is not supported")
    n = len(terms)

    def per_margin(value, default):
        if value is None:
            return [default] * n
        value = value if isinstance(value, list) else [value]
        if len(value) == 1:
            return value * n
        if len(value) != n:
            raise ValueError(f"gam(): te() needs one value per margin in {text!r}")
        return value

    bs = per_margin(values.get("bs"), "cr")
    k = [int(v) for v in per_margin(values.get("k"), 5)]
    if any(v < 3 for v in k):
        _warn("one or more supplied k too small - reset to default")
        k = [5] * n
    m = per_margin(values.get("m"), None)
    return SmoothSpec("te", terms, bs, k, m, by, fx, text)


class _GamFormula:
    """A gam formula: the parametric part (R's model formula) and the smooths."""

    def __init__(self, text: str, columns: list[str]):
        if not isinstance(text, str) or "~" not in text:
            raise ValueError(f"gam(): formula must look like 'y ~ s(x) + ...', got {text!r}")
        lhs, _, rhs = text.partition("~")
        self.text = " ".join(text.split())
        self.smooths: list[SmoothSpec] = []
        parametric = []
        for sign, piece in _top_level_terms(rhs):
            head = piece.split("(", 1)[0].strip()
            if head in ("s", "te") and piece.endswith(")"):
                if sign == "-":
                    raise ValueError(f"gam(): cannot remove a smooth term: {piece!r}")
                self.smooths.append(_smooth_spec(piece))
            elif head in ("ti", "t2"):
                raise ValueError(f"gam(): {head}() terms are not supported")
            else:
                parametric.append(("- " if sign == "-" else "+ ") + piece)
        sets = [(spec, set(spec.terms) | ({spec.by} if spec.by else set())) for spec in self.smooths
                if not (spec.kind == "s" and spec.bs[0] == "re")]
        for i, (a, va) in enumerate(sets):
            for b, vb in sets[i + 1:]:
                if va <= vb or vb <= va:
                    raise ValueError(f"gam(): {a.text!r} and {b.text!r} are nested; mgcv's gam.side "
                                     "identifiability constraints for nested smooths are not supported")
        prhs = " ".join(parametric).lstrip("+ ").strip() or "1"
        if prhs.startswith("- "):
            prhs = "1 " + prhs
        self.parametric = parse_formula(f"{lhs.strip()} ~ {prhs}", "gam", columns)
        self.parametric.text = self.text
        extra = []
        for spec in self.smooths:
            for name in [*spec.terms, *([spec.by] if spec.by else [])]:
                if name not in self.parametric.variables() and name not in extra:
                    extra.append(name)
        self.frame_formula = Formula(self.text, self.parametric.response,
                                     self.parametric.terms + [(v,) for v in extra],
                                     self.parametric.intercept, list(self.parametric.offsets))


# families: link and variance derivatives -------------------------------------------


def _link_derivatives(name: str, mu: np.ndarray, mu_eta: np.ndarray):
    """g'(mu), g''(mu) and g'''(mu), as mgcv's fix.family.link (g' = 1/mu.eta)."""
    g1 = 1 / mu_eta
    if name == "identity":
        return g1, np.zeros_like(mu), np.zeros_like(mu)
    if name == "log":
        return g1, -1 / mu**2, 2 / mu**3
    if name == "logit":
        return g1, -1 / mu**2 + 1 / (1 - mu) ** 2, 2 / mu**3 + 2 / (1 - mu) ** 3
    if name == "probit":
        special = _special()
        eta = special.ndtri(mu)
        dens = np.exp(-eta * eta / 2) / math.sqrt(2 * math.pi)
        return g1, eta / dens**2, (1 + 2 * eta**2) / dens**3
    if name == "cloglog":
        u = 1 - mu
        L = -np.log(u)
        return g1, (L - 1) / (u**2 * L**2), (L + 2 * (L - 1) ** 2) / (u**3 * L**3)
    if name == "inverse":
        return g1, 2 / mu**3, -6 / mu**4
    if name == "sqrt":
        return g1, -0.25 * mu**-1.5, 0.375 * mu**-2.5
    if name == "1/mu^2":
        return g1, 6 / mu**4, -24 / mu**5
    raise ValueError(f"gam(): link {name!r} is not supported")


def _variance_derivatives(family, mu: np.ndarray):
    name = family.family
    one, zero = np.ones_like(mu), np.zeros_like(mu)
    if family.theta is not None:
        th = family.theta
        return mu + mu**2 / th, 1 + 2 * mu / th, 2 / th * one
    if name == "gaussian":
        return one, zero, zero
    if name in ("binomial", "quasibinomial"):
        return mu * (1 - mu), 1 - 2 * mu, -2 * one
    if name in ("poisson", "quasipoisson"):
        return mu, one, zero
    if name == "Gamma":
        return mu**2, 2 * mu, 2 * one
    if name == "inverse.gaussian":
        return mu**3, 3 * mu**2, 6 * mu
    raise ValueError(f"gam(): family {name!r} is not supported")


def _log_saturated(family, y, w, n_trials, scale):
    """mgcv's fix.family.ls: the saturated log likelihood and its first two
    derivatives with respect to the scale."""
    from scipy import special, stats

    name = family.family
    ok = w > 0
    nobs = int(ok.sum())
    if name == "gaussian":
        return np.array([-nobs * math.log(2 * math.pi * scale) / 2 + np.sum(np.log(w[ok])) / 2,
                         -nobs / (2 * scale), nobs / (2 * scale**2)])
    if name == "poisson":
        return np.array([float(np.sum(stats.poisson.logpmf(y, y) * w)), 0, 0])
    if name == "binomial":
        return np.array([-family.aic(y, n_trials, y, w, 0) / 2, 0, 0])
    if name == "Gamma":
        yy, ww = y[ok], w[ok]
        s = scale / ww
        k = -special.gammaln(1 / s) - np.log(s) / s - 1 / s
        k2 = (special.digamma(1 / s) + np.log(s)) / s**2
        k3 = (-special.polygamma(1, 1 / s) / s + (1 - 2 * np.log(s) - 2 * special.digamma(1 / s))) / s**3
        return np.array([np.sum(k - np.log(yy)), np.sum(k2 / ww), np.sum(k3 / ww**2)])
    if name in ("quasipoisson", "quasibinomial"):
        return np.array([-nobs * math.log(scale) / 2 + np.sum(np.log(w[ok])) / 2,
                         -nobs / (2 * scale), nobs / (2 * scale**2)])
    if name == "inverse.gaussian":
        return np.array([-np.sum(np.log(2 * math.pi * scale * y[ok] ** 3)) / 2 + np.sum(np.log(w[ok])) / 2,
                         -nobs / (2 * scale), nobs / (2 * scale**2)])
    raise ValueError(f"gam(): method='REML' is not available for the {name} family")


def _mustart(family, y, w, n_trials):
    if family.family == "gaussian":  # mgcv's fix.family
        if family.link.name == "inverse":
            return y + (y == 0) * np.std(y, ddof=1) * 0.01
        if family.link.name == "log":
            return np.maximum(y, 0.01 * np.std(y, ddof=1))
    if family.family in ("binomial", "quasibinomial"):
        return (w * y + 0.5) / (w + 1) if np.all(n_trials == 1) else (n_trials * y + 0.5) / (n_trials + 1)
    if family.theta is not None:
        return y + (y == 0) / 6
    if family.family in ("poisson", "quasipoisson"):
        return y + 0.1
    return y.copy()


# Davies' algorithm (mgcv's psum.chisq) -----------------------------------------------


def _log1pmx(x: float) -> float:
    if x > 1 or x < -0.79149064:
        return math.log1p(x) - x
    r = x / (2 + x)
    y = r * r
    total, term, k = 0.0, 1.0, 0
    while True:
        add = term / (2 * k + 3)
        total += add
        if abs(add) < 1e-17 * abs(total):
            break
        term *= y
        k += 1
    return r * (2 * y * total - x)


def _ln1(x: float, first: bool) -> float:
    return math.log1p(x) if first else _log1pmx(x)


def _davies(lb, nc, n, sigma, c, lim, acc):
    """mgcv's davies.c: Pr(sum lb_j chi^2_{n_j}(nc_j) + sigma Z < c).
    Returns (probability, ifault)."""
    r = len(lb)
    pi = math.pi
    ln28 = math.log(2.0) / 8
    th = sorted(range(r), key=lambda j: -abs(lb[j]))

    def errbd(u, sigsq):
        cx = u * sigsq
        sum1 = u * cx
        u = u * 2
        for j in range(r - 1, -1, -1):
            x = u * lb[j]
            y = 1.0 - x
            cx += lb[j] * (nc[j] / y + n[j]) / y
            xy = x / y
            sum1 += nc[j] * xy * xy + n[j] * (x * xy + _ln1(-x, False))
        return math.exp(-0.5 * sum1), cx

    def ctff(accx, upn, mean, lmin, lmax, sigsq):
        u2, u1, c1 = upn, 0.0, mean
        rb = 2 * lmax if u2 > 0 else 2 * lmin
        while True:
            e, c2 = errbd(u2 / (1 + u2 * rb), sigsq)
            if e <= accx:
                break
            u1, c1, u2 = u2, c2, u2 * 2
        while (c1 - mean) / (c2 - mean) < 0.9:
            u = (u1 + u2) * 0.5
            e, cst = errbd(u / (1 + u * rb), sigsq)
            if e > accx:
                u1, c1 = u, cst
            else:
                u2, c2 = u, cst
        return c2, u2

    def truncation(u, tausq, sigsq):
        sum1 = prod2 = prod3 = 0.0
        s = 0
        sum2 = (sigsq + tausq) * u * u
        prod1 = 2 * sum2
        u = 2 * u
        for j in range(r):
            x = (u * lb[j]) ** 2
            sum1 += nc[j] * x / (1 + x)
            if x > 1:
                prod2 += n[j] * math.log(x)
                prod3 += n[j] * _ln1(x, True)
                s += n[j]
            else:
                prod1 += n[j] * _ln1(x, True)
        sum1 *= 0.5
        prod2 += prod1
        prod3 += prod1
        x = math.exp(-sum1 - 0.25 * prod2) / pi
        y = math.exp(-sum1 - 0.25 * prod3) / pi
        err1 = 1.0 if s == 0 else 2.0 * x / s
        err2 = 2.5 * y if prod3 > 1 else 1.0
        if err2 < err1:
            err1 = err2
        x = 0.5 * sum2
        err2 = 1.0 if x <= y else y / x
        return err1 if err1 < err2 else err2

    def findu(utx, accx, sigsq):
        ut = utx
        u = ut * 0.25
        if truncation(u, 0, sigsq) > accx:
            while truncation(ut, 0, sigsq) > accx:
                ut *= 4
        else:
            ut, u = u, u / 4
            while truncation(u, 0, sigsq) <= accx:
                ut, u = u, u / 4
        for a in (2.0, 1.4, 1.2, 1.1):
            u = ut / a
            if truncation(u, 0, sigsq) <= accx:
                ut = u
        return ut

    def integrate(nterm, interv, tausq, main, sigsq):
        inpi = interv / pi
        intl = ersm = 0.0
        for k in range(nterm, -1, -1):
            u = (k + 0.5) * interv
            sum1 = -2 * u * c
            sum2 = abs(sum1)
            sum3 = -0.5 * sigsq * u * u
            for j in range(r - 1, -1, -1):
                x = 2 * lb[j] * u
                y = x * x
                sum3 -= 0.25 * n[j] * _ln1(y, True)
                y = nc[j] * x / (1 + y)
                z = n[j] * math.atan(x) + y
                sum1 += z
                sum2 += abs(z)
                sum3 -= 0.5 * x * y
            x = inpi * math.exp(sum3) / u
            if not main:
                x *= 1 - math.exp(-0.5 * tausq * u * u)
            intl += math.sin(0.5 * sum1) * x
            ersm += 0.5 * sum2 * x
        return intl, ersm

    def cfe(x):
        axl = abs(x)
        sxl = -1 if x < 0 else 1
        sum1 = 0.0
        for j in range(r - 1, -1, -1):
            t = th[j]
            if lb[t] * sxl > 0.0:
                lj = abs(lb[t])
                axl1 = axl - lj * (n[t] + nc[t])
                axl2 = lj / ln28
                if axl1 > axl2:
                    axl = axl1
                else:
                    if axl > axl2:
                        axl = axl2
                    sum1 = (axl - axl1) / lj
                    for kk in range(j - 1, -1, -1):
                        sum1 += n[th[kk]] + nc[th[kk]]
                    break
        if sum1 > 100.0:
            return 1.0, True
        return 2.0 ** (sum1 * 0.25) / (pi * axl * axl), False

    intl = ersm = 0.0
    acc1 = acc
    sd = sigsq = sigma * sigma
    lmax = lmin = 0.0
    mean = 0.0
    for j in range(r):
        if n[j] < 0 or nc[j] < 0:
            return float("nan"), 3
        sd += lb[j] ** 2 * (2.0 * n[j] + 4.0 * nc[j])
        mean += lb[j] * (n[j] + nc[j])
        if lmax < lb[j]:
            lmax = lb[j]
        elif lmin > lb[j]:
            lmin = lb[j]
    if sd == 0.0:
        return (1.0 if c > 0 else 0.0), 0
    if lmin == 0.0 and lmax == 0.0 and sigma == 0.0:
        return float("nan"), 3
    sd = math.sqrt(sd)
    almx = -lmin if lmax < -lmin else lmax
    utx, up, un = 16.0 / sd, 4.5 / sd, -4.5 / sd
    utx = findu(utx, 0.5 * acc1, sigsq)
    if c != 0.0 and almx > 0.07 * sd:
        value, fail = cfe(c)
        tausq = 0.25 * acc1 / value
        if not fail and truncation(utx, tausq, sigsq) < 0.2 * acc1:
            sigsq += tausq
            utx = findu(utx, 0.25 * acc1, sigsq)
    acc1 *= 0.5
    while True:
        value, up = ctff(acc1, up, mean, lmin, lmax, sigsq)
        d1 = value - c
        if d1 < 0.0:
            return 1.0, 0
        value, un = ctff(acc1, un, mean, lmin, lmax, sigsq)
        d2 = c - value
        if d2 < 0.0:
            return 0.0, 0
        intv = 2 * pi / d1 if d1 > d2 else 2 * pi / d2
        nt = _round_half_up(utx / intv)
        ntm = _round_half_up(3.0 / math.sqrt(acc1))
        if nt > ntm * 1.5:
            intv1 = utx / ntm
            x = 2 * pi / intv1
            if x <= abs(c):
                break
            v1, fail1 = cfe(c - x)
            v2, fail2 = cfe(c + x)
            tausq = 0.33 * acc1 / (1.1 * (v1 + v2))
            if fail2:
                break
            acc1 *= 0.67
            if ntm > lim:
                return -1.0, 0
            a, b = integrate(ntm, intv1, tausq, False, sigsq)
            intl += a
            ersm += b
            lim -= ntm
            sigsq += tausq
            utx = findu(utx, 0.25 * acc1, sigsq)
            acc1 *= 0.75
        else:
            break
    if nt > lim:
        return -1.0, 1
    a, b = integrate(nt, intv, 0.0, True, sigsq)
    intl += a
    ersm += b
    out = 0.5 - intl
    x = ersm + acc / 10.0
    fault = 0
    j = 1
    for _ in range(4):
        if j * x == j * ersm:
            fault = 2
        j *= 2
    return out, fault


def _round_half_up(x: float) -> int:
    f = math.floor(x)
    return f + 1 if x - f > 0.5 else f


def _liu2(x: float, lam, h=None) -> float:
    from scipy import stats

    lam = np.asarray(lam, dtype=float)
    h = np.ones(len(lam)) if h is None else np.asarray(h, dtype=float)
    lh = lam * h
    muQ = lh.sum()
    lh = lh * lam
    c2 = lh.sum()
    lh = lh * lam
    c3 = lh.sum()
    if x <= 0 or c2 <= 0:
        return 1.0
    s1 = c3 / c2**1.5
    s2 = np.sum(lh * lam) / c2**2
    sigQ = math.sqrt(2 * c2)
    t = (x - muQ) / sigQ
    if s1**2 > s2:
        a = 1 / (s1 - math.sqrt(s1**2 - s2))
        delta = s1 * a**3 - a**2
        ll = a**2 - 2 * delta
    else:
        a, delta = 1 / s1, 0.0
        if c3 == 0:
            return 1.0
        ll = c2**3 / c3**2
    muX, sigX = ll + delta, math.sqrt(2) * a
    if delta == 0:
        return float(stats.chi2.sf(t * sigX + muX, ll))
    return float(stats.ncx2.sf(t * sigX + muX, ll, delta))


def _psum_chisq(q: float, lb, df=None) -> float:
    """mgcv's psum.chisq: Pr(q > sum lb_j chi^2_{df_j}), upper tail."""
    lb = [float(v) for v in lb]
    df = [1] * len(lb) if df is None else [int(round(v)) for v in df]
    value, fault = _davies(lb, [0.0] * len(lb), df, 0.0, float(q), 100000, 2e-5)
    if fault == 2:
        _warn("danger of round-off error")
    elif fault != 0:
        _warn("failure of Davies method, falling back on Liu et al approximtion")
        return _liu2(q, lb, df)
    return 1 - value


# the fit ------------------------------------------------------------------------------


class GeneralizedAdditiveModel(_Fit):
    """A fitted ``gam``."""

    kind = "gam"

    def __init__(self, gf: _GamFormula, frame, data: Data, family, method: str, gamma: float,
                 scale: float, maxit: int, formula_offset=None):
        formula = gf.parametric
        design = build_design(formula, frame.variables, len(frame.weights))
        super().__init__(formula, design, frame, data, "gam")
        self.gam_formula, self.family, self.method, self.gamma = gf, family, method, float(gamma)
        y, w, n_trials = GeneralizedLinearModel._prepare(self, frame, family)
        self.y, self.prior_weights, self.n_trials = y, w, n_trials
        n = len(y)
        self.n = n
        self.maxit = maxit
        smooths = []
        for spec in gf.smooths:
            smooths.extend(smooth_con(spec, frame.variables, n))
        Xp = design.matrix
        if np.linalg.matrix_rank(Xp) < Xp.shape[1]:
            raise ValueError("gam(): the parametric part of the model is rank deficient")
        self.nsdf = Xp.shape[1]
        start = self.nsdf
        blocks = [Xp]
        self.S, self.S_index, self.sp_names, self.S_rank = [], [], [], []
        for sm in smooths:
            q = sm.X.shape[1]
            sm.first, sm.last = start, start + q
            idx = np.arange(start, start + q)
            for j, Sj in enumerate(sm.S):
                self.S.append(Sj)
                self.S_rank.append(sm.rank[j])
                self.S_index.append(idx)
                self.sp_names.append(sm.label if len(sm.S) == 1 else f"{sm.label}{j + 1}")
            blocks.append(sm.X)
            start += q
        self.smooths = smooths
        self.X = np.hstack(blocks)
        self.p = self.X.shape[1]
        self.names = list(design.names) + [f"{sm.label}.{i + 1}" for sm in smooths for i in range(sm.X.shape[1])]
        self.offset = frame.offset
        # predict.gam leaves out an offset= argument (formula offsets stay)
        self.formula_offset = frame.offset if formula_offset is None else formula_offset
        self.am = family.family == "gaussian" and family.link.name == "identity" and method == "GCV.Cp"
        self.scale_known = scale > 0 or family.family in ("binomial", "poisson") or family.theta is not None
        self.fixed_scale = scale if scale > 0 else 1.0
        if method == "REML":
            _log_saturated(family, y, w, n_trials, 1.0)  # checks the family has one
        self._penalty_ranks()
        self._fit()

    # -- penalties ------------------------------------------------------------------------

    def _penalty_ranks(self):
        """Ranks of each smooth's total penalty, for log|S|+ and Mp."""
        self.blocks = []
        total = 0
        for sm in self.smooths:
            members = [i for i, idx in enumerate(self.S_index) if idx[0] == sm.first]
            if not members:
                continue
            Sb = sum(self.S[i] / np.linalg.norm(self.S[i]) for i in members)
            ev = np.linalg.eigvalsh(Sb)
            r = int(np.sum(ev > ev.max() * _EPS**0.66))
            self.blocks.append((np.arange(sm.first, sm.last), members, r))
            total += r
        self.Mp = self.p - total

    def _total_penalty(self, lam):
        St = np.zeros((self.p, self.p))
        for l, Sj, idx in zip(lam, self.S, self.S_index):
            St[np.ix_(idx, idx)] += l * Sj
        return St

    def _log_det_S(self, lam):
        """log|S_lambda|+ and its derivatives with respect to log lambda."""
        det, det1 = 0.0, np.zeros(len(lam))
        for idx, members, r in self.blocks:
            Sb = sum(lam[i] * self.S[i] for i in members)
            ev, vec = np.linalg.eigh((Sb + Sb.T) / 2)
            ev, vec = ev[::-1][:r], vec[:, ::-1][:, :r]
            det += float(np.sum(np.log(ev)))
            inv = (vec / ev) @ vec.T
            for i in members:
                det1[i] = lam[i] * float(np.sum(inv * self.S[i]))
        return det, det1

    # -- penalized IRLS -----------------------------------------------------------------------

    def _pirls(self, lam, start=None):
        fam, link = self.family, self.family.link
        X, y, w, off = self.X, self.y, self.prior_weights, self.offset
        St = self._total_penalty(lam)
        if fam.family == "gaussian" and link.name == "identity":
            H = X.T @ (w[:, None] * X) + St
            beta = _solve(H, X.T @ (w * (y - off)))
            return self._state(beta, St)
        if start is not None:
            beta = start
            eta = X @ beta + off
            mu = link.linkinv(eta)
            if not (link.valideta(eta) and fam.validmu(mu)):
                start = None
        if start is None:
            mu = _mustart(fam, y, w, self.n_trials)
            eta = link.linkfun(mu)
            beta = None
        dev = float(np.sum(fam.dev_resids(y, mu, w)))
        old = math.inf if beta is None else dev + float(beta @ St @ beta)
        coefold, etaold = beta, eta
        for _ in range(max(self.maxit, 200)):
            mevg = link.mu_eta(eta)
            V, V1, _ = _variance_derivatives(fam, mu)
            g1, g2, _ = _link_derivatives(link.name, mu, mevg)
            alpha = 1 + (y - mu) * (V1 / V + g2 / g1)
            alpha[alpha == 0] = _EPS
            wf = w * mevg**2 / V
            ww = wf * alpha
            z = eta - off + (y - mu) / (mevg * alpha)
            H = X.T @ (ww[:, None] * X) + St
            new = _solve(H, X.T @ (ww * z), fallback=True)
            if new is None:  # indefinite Newton Hessian: Fisher step
                z = eta - off + (y - mu) / mevg
                new = _solve(X.T @ (wf[:, None] * X) + St, X.T @ (wf * z))
            eta_new = X @ new + off
            mu_new = link.linkinv(eta_new)
            ok = link.valideta(eta_new) and fam.validmu(mu_new)
            dev = float(np.sum(fam.dev_resids(y, mu_new, w))) if ok else math.inf
            pdev = dev + float(new @ St @ new)
            halves = 0
            while (not math.isfinite(pdev) or pdev - old > 1e-12 * (0.1 + abs(old))) and coefold is not None and halves < 60:
                new = (new + coefold) / 2
                eta_new = X @ new + off
                mu_new = link.linkinv(eta_new)
                ok = link.valideta(eta_new) and fam.validmu(mu_new)
                dev = float(np.sum(fam.dev_resids(y, mu_new, w))) if ok else math.inf
                pdev = dev + float(new @ St @ new)
                halves += 1
            if not math.isfinite(pdev):
                raise ValueError("gam(): penalized IRLS failed: no valid coefficients found")
            change = 0.0 if coefold is None else float(np.max(np.abs(new - coefold)))
            size = float(np.max(np.abs(new))) + 1e-300
            beta, eta, mu = new, eta_new, mu_new
            converged = abs(pdev - old) <= 1e-14 * (abs(pdev) + 1) and change <= 1e-12 * size
            if halves and abs(pdev - old) <= 1e-14 * (abs(pdev) + 1):
                converged = True
            old, coefold = pdev, beta
            if converged:
                break
        else:
            _warn("gam(): penalized IRLS did not converge")
        return self._state(beta, St)

    def _state(self, beta, St):
        fam, link = self.family, self.family.link
        X, y, w, off = self.X, self.y, self.prior_weights, self.offset
        eta = X @ beta + off
        mu = link.linkinv(eta)
        mevg = link.mu_eta(eta)
        V, V1, V2 = _variance_derivatives(fam, mu)
        g1, g2, g3 = _link_derivatives(link.name, mu, mevg)
        V1r, V2r, g2r, g3r = V1 / V, V2 / V, g2 / g1, g3 / g1
        alpha = 1 + (y - mu) * (V1r + g2r)
        alpha[alpha == 0] = _EPS
        wf = w * mevg**2 / V
        wn = wf * alpha
        alpha1 = (-(V1r + g2r) + (y - mu) * (V2r - V1r**2 + g3r - g2r**2)) / alpha
        dev = float(np.sum(fam.dev_resids(y, mu, w)))
        return {
            "beta": beta, "eta": eta, "mu": mu, "dev": dev, "pen": float(beta @ St @ beta), "St": St,
            "wf": wf, "wn": wn, "a1": wn * (alpha1 - V1r - 2 * g2r) / g1, "af1": -wf * (V1r + 2 * g2r) / g1,
            "dev_grad": X.T @ (-2 * w * (y - mu) / (V * g1)),
        }

    # -- smoothness selection --------------------------------------------------------------

    def _criterion(self, theta, grad=True):
        # Line searches can try extreme smoothing parameters; a non-finite
        # score there just rejects the step.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            score, out, fit = self._criterion_at(theta, grad)
        return (score if math.isfinite(score) else math.inf), out, fit

    def _criterion_at(self, theta, grad):
        M = len(self.S)
        rho = theta[:M]
        lam = np.exp(rho)
        fit = self._pirls(lam, self._warm)
        self._warm = fit["beta"]
        X = self.X
        beta = fit["beta"]
        St = fit["St"]
        db = None
        if grad and M:
            Hn = X.T @ (fit["wn"][:, None] * X) + St
            Hn_inv = np.linalg.inv(Hn)
            Sb = np.column_stack([l * (self._embed(i) @ beta) for i, l in enumerate(lam)])
            db = -Hn_inv @ Sb
            deta = X @ db
        if self.method == "REML":
            phi = math.exp(theta[M]) if not self.scale_known else self.fixed_scale
            Hn = X.T @ (fit["wn"][:, None] * X) + St
            sign, logdet = np.linalg.slogdet(Hn)
            detS, detS1 = self._log_det_S(lam) if M else (0.0, np.zeros(0))
            Dp = fit["dev"] + fit["pen"]
            ls = _log_saturated(self.family, self.y, self.prior_weights, self.n_trials, phi)
            g = self.gamma
            score = (Dp / (2 * phi) - ls[0]) / g + logdet / 2 - detS / 2 - self.Mp / 2 * (math.log(2 * math.pi * phi) - math.log(g))
            if not grad:
                return score, None, fit
            out = np.zeros(len(theta))
            if M:
                h = np.sum((X @ Hn_inv) * X, axis=1)
                for k in range(M):
                    dDp = lam[k] * float(beta @ self._embed(k) @ beta)
                    dlog = float(np.sum(h * fit["a1"] * deta[:, k])) + lam[k] * float(np.sum(Hn_inv * self._embed(k)))
                    out[k] = dDp / (2 * phi * g) + dlog / 2 - detS1[k] / 2
            if not self.scale_known:
                out[M] = (-Dp / (2 * phi) - ls[1] * phi) / g - self.Mp / 2
            fit["db"] = db
            return score, out, fit
        # GCV / UBRE on the Fisher-weight influence matrix
        wf = fit["wf"]
        Hf = X.T @ (wf[:, None] * X) + St
        Hf_inv = np.linalg.inv(Hf)
        XWX = X.T @ (wf[:, None] * X)
        trA = float(np.sum(Hf_inv * XWX.T))
        nn = float(np.sum(self.prior_weights != 0)) if self.am else float(self.n)
        g = self.gamma
        dev = fit["dev"]
        delta = nn - g * trA
        ubre = self.scale_known
        phi = self.fixed_scale
        score = dev / nn - 2 * delta * phi / nn + phi if ubre else nn * dev / delta**2
        if not grad:
            return score, None, fit
        out = np.zeros(M)
        if M:
            P = Hf_inv @ St @ Hf_inv
            hP = np.sum((X @ P) * X, axis=1)
            for k in range(M):
                dw = fit["af1"] * deta[:, k]
                Sk = self._embed(k)
                dH = X.T @ (dw[:, None] * X) + lam[k] * Sk
                dtr = float(np.sum(P * dH)) - lam[k] * float(np.sum(Hf_inv * Sk))
                dD = float(fit["dev_grad"] @ db[:, k])
                out[k] = dD / nn + 2 * g * phi * dtr / nn if ubre else nn * dD / delta**2 + 2 * nn * dev * g * dtr / delta**3
        fit["db"] = db
        return score, out, fit

    def _embed(self, i):
        out = np.zeros((self.p, self.p))
        idx = self.S_index[i]
        out[np.ix_(idx, idx)] = self.S[i]
        return out

    def _hessian(self, theta, h=1e-4):
        k = len(theta)
        H = np.zeros((k, k))
        for i in range(k):
            e = np.zeros(k)
            e[i] = h
            H[:, i] = (self._criterion(theta + e)[1] - self._criterion(theta - e)[1]) / (2 * h)
        return (H + H.T) / 2

    def _initial_theta(self):
        M = len(self.S)
        if not M:
            theta = np.zeros(0)
        else:
            fam, link = self.family, self.family.link
            mu = _mustart(fam, self.y, self.prior_weights, self.n_trials)
            sw = np.sqrt(self.prior_weights * link.mu_eta(link.linkfun(mu)) ** 2 / fam.variance(mu))
            theta = np.log(_initial_sp(sw[:, None] * self.X, self.S, self.S_index))
        if self.method == "REML" and not self.scale_known:
            mean = np.mean(self.y)
            null_scale = float(np.sum(self.family.dev_resids(self.y, np.full(self.n, mean), self.prior_weights))) / self.n
            theta = np.r_[theta, math.log(null_scale / 10)]
        return theta

    def _fit(self):
        self._warm = None
        theta = self._initial_theta()
        if len(theta):
            theta = self._newton(theta)
        if len(theta) and self.am:
            theta = self._magic_edges(theta)
        score, grad, fit = self._criterion(theta, grad=True)
        self.theta, self.score, self._fitted = theta, score, fit
        self._post_process(theta, fit)

    def _magic_edges(self, theta):
        """magic's last step, which checks for optima at infinity: each log
        smoothing parameter in turn moves by 2, against the sign of its
        gradient at convergence, up to 5 times while the GCV score falls,
        and stays there, even past a local minimum.

        stats3 converges more tightly than mgcv, so a component of its
        gradient here can be rounding noise (a few units in the last place,
        about 1e-16 on a score near 1), whose sign differs between machines;
        following it sent one fit into a different local minimum on x86-64
        than on arm64. A gradient below that floor carries no direction, so
        the step then tries larger smoothing parameters (the smoother fit), as
        mgcv does in the cases checked. Gradients above it (2e-14 and up in
        those cases) are real, and mgcv follows them too. Every machine then
        makes the same choice."""
        score, grad, _ = self._criterion(theta)
        theta = theta.copy()
        noise = 1e-15 * (abs(score) + 1)
        for k in range(len(self.S)):
            sign = 1.0 if grad[k] < noise else -1.0
            for _ in range(5):
                trial = theta.copy()
                trial[k] += 2 * sign
                new = self._criterion(trial, grad=False)[0]
                if new >= score:
                    break
                theta, score = trial, new
        return theta

    def _newton(self, theta):
        """Newton's method on the criterion, with its Hessian from differences
        of the analytic gradient, eigenvalues made positive, steps capped at
        5 (mgcv's maxNstep) and halved until the score falls."""
        M = len(self.S)
        score, grad, _ = self._criterion(theta)
        stalls = 0
        for _ in range(200):
            H = self._hessian(theta)
            ev, vec = np.linalg.eigh(H)
            floor = max(np.max(np.abs(ev)) * 1e-8, 1e-12)
            ev = np.maximum(np.abs(ev), floor)
            step = -(vec / ev) @ vec.T @ grad
            biggest = np.max(np.abs(step))
            if biggest > 5:
                step *= 5 / biggest
            for _ in range(40):
                trial = theta + step
                # Smoothing parameters past e^25 are at their infinite limit
                # to rounding error; going further only ill-conditions X'WX + S.
                trial[:M] = np.clip(trial[:M], -_RHO_MAX, _RHO_MAX)
                try:
                    new_score, new_grad, _ = self._criterion(trial)
                except (ValueError, np.linalg.LinAlgError):
                    new_score = math.inf
                if new_score <= score + 1e-14 * abs(score):
                    break
                step /= 2
            else:
                break
            change = abs(score - new_score)
            moved = float(np.max(np.abs(trial - theta)))
            theta, score, grad = trial, new_score, new_grad
            # A smoothing parameter already far out (e^15: its smooth is in
            # its penalty's null space to about 1e-7) and still pulled outward
            # is at its infinite limit; creeping towards it, with the gradient
            # decaying like e^-rho, would take every remaining iteration.
            outward = (np.abs(theta[:M]) > _RHO_FAR) & (np.abs(theta[:M]) < _RHO_MAX) \
                & (np.sign(theta[:M]) * grad[:M] < 0)
            if outward.any():
                trial = theta.copy()
                trial[:M] = np.where(outward, np.sign(theta[:M]) * _RHO_MAX, theta[:M])
                try:
                    limit_score, limit_grad, _ = self._criterion(trial)
                except (ValueError, np.linalg.LinAlgError):
                    limit_score = math.inf
                if limit_score <= score + 1e-14 * abs(score):
                    theta, score, grad = trial, limit_score, limit_grad
            # Converge tightly: a smoothing parameter heading to infinity
            # (a smooth penalized to its null space) keeps a small gradient,
            # and following it brings results to their limit, as mgcv's are.
            free = np.ones(len(theta), dtype=bool)
            free[:M] = np.abs(theta[:M]) < _RHO_MAX
            if not free.any() or np.max(np.abs(grad[free])) < 1e-13 * (abs(score) + 1):
                break
            stalls = stalls + 1 if change <= 1e-15 * (abs(score) + 1) or moved < 1e-10 else 0
            if stalls >= 3:
                break
        else:
            _warn("gam(): smoothing parameter selection did not converge")
        return theta

    # -- results ----------------------------------------------------------------------------

    def _post_process(self, theta, fit):
        X, w = self.X, self.prior_weights
        M = len(self.S)
        self.sp = np.exp(theta[:M])
        self.coef = fit["beta"]
        self.linear_predictors, self.fitted = fit["eta"], fit["mu"]
        self.deviance = fit["dev"]
        wf = fit["wf"]
        self.weights = wf
        St = fit["St"]
        XWX = X.T @ (wf[:, None] * X)
        Hf_inv = np.linalg.inv(XWX + St)
        F = Hf_inv @ XWX
        self.edf = np.diag(F).copy()
        self.edf1 = 2 * self.edf - np.sum(F.T * F, axis=1)
        trA = float(self.edf.sum())
        self.reml_scale = math.exp(theta[M]) if self.method == "REML" and not self.scale_known else None
        if self.scale_known:
            scale = self.fixed_scale
        elif self.am:
            scale = self.deviance / (float(np.sum(w != 0)) - trA)
        else:
            # gam.control(scale.est = "fletcher"), mgcv's default: Pearson's
            # estimate with Fletcher's (2012) correction.
            mu = self.fitted
            V, V1, _ = _variance_derivatives(self.family, mu)
            scale = float(np.sum(w * (self.y - mu) ** 2 / V)) / (self.n - trA)
            s_bar = max(-0.9, float(np.mean(V1 * (self.y - mu) / V)))
            if math.isfinite(s_bar):
                scale /= 1 + s_bar
        self.scale = scale
        self.Vp = Hf_inv * scale
        self.Ve = F @ self.Vp
        self.hat = wf * np.sum((X @ Hf_inv) * X, axis=1)
        self.Vc = None
        self.edf2 = None
        if self.method == "REML" and M:
            self._uncertainty(theta, fit, XWX, scale)
        fam = self.family
        sumw = float(np.sum(w))
        if self.scale_known:
            dev1 = scale * sumw
        elif fam.family == "gaussian":
            dev1 = self.deviance
        else:
            dev1 = (scale if self.reml_scale is None else self.reml_scale) * sumw
        self.aic = fam.aic(self.y, self.n_trials, self.fitted, w, dev1) + 2 * trA
        self.df_residual = self.n - trA
        intercept = self.formula.intercept
        wtdmu = np.sum(w * self.y) / np.sum(w) if intercept else fam.link.linkinv(self.offset)
        self.null_deviance = float(np.sum(fam.dev_resids(self.y, wtdmu, w)))
        if intercept and np.any(self.offset != 0):
            self.null_deviance = _glm_fit(np.ones((self.n, 1)), self.y, w, self.offset, fam, self.n_trials,
                                          25, 1e-8, "gam")["deviance"]
        self._summary = None

    def _uncertainty(self, theta, fit, XWX, scale):
        """gam.fit3.post.proc: Vc and edf2 from the REML Hessian."""
        M = len(self.S)
        hess = self._hessian(theta)
        db = self._criterion(theta)[2]["db"]
        ev, vec = np.linalg.eigh(hess)
        ev, vec = ev[::-1], vec[:, ::-1]
        pos = ev > 0
        d = np.where(pos, 1 / np.sqrt(np.where(pos, ev, 1)), 0.0)
        rV = (d[:, None] * vec.T)[:, :M]
        Vc1 = (rV @ db.T).T @ (rV @ db.T)
        d2 = np.where(pos, ev, 0.0)
        Vr = (vec / (d2 + 0.1)) @ vec.T
        if not self.scale_known:
            Vr = Vr[:M, :M]
        lam = self.sp
        H = XWX + fit["St"]
        from scipy.linalg import cholesky, solve_triangular

        R = cholesky(H, lower=False)
        Rinv = solve_triangular(R, np.eye(self.p), lower=False)
        dRi = []
        for k in range(M):
            dH = lam[k] * self._embed(k)
            A = Rinv.T @ dH @ Rinv
            Phi = np.triu(A)
            Phi[np.diag_indices_from(Phi)] *= 0.5
            dR = Phi @ R
            dRi.append(-Rinv @ dR @ Rinv)
        Vc2 = np.zeros((self.p, self.p))
        for i in range(M):
            for j in range(M):
                Vc2 += Vr[i, j] * dRi[i] @ dRi[j].T
        Vc = self.Vp + Vc1 + scale * Vc2
        self.Vc = Vc
        edf2 = np.sum(Vc * XWX, axis=1) / scale
        self.edf2 = self.edf1 if edf2.sum() > self.edf1.sum() else edf2

    # -- summary.gam ----------------------------------------------------------------------------

    def _summary_tables(self):
        if self._summary is not None:
            return self._summary
        special = _special()
        est_disp = not self.scale_known
        rdf = self.n - float(self.edf.sum())
        se = np.sqrt(np.diag(self.Vp))
        coef = self.coef[:self.nsdf]
        t = coef / se[:self.nsdf]
        if est_disp:
            p = 2 * special.stdtr(rdf, -np.abs(t))
        else:
            p = 2 * special.ndtr(-np.abs(t))
        ptab = {"term": self.design.names, "estimate": coef, "std_error": se[:self.nsdf], "statistic": t, "p_value": p}
        # parametric terms as wholes (anova.gam's first table)
        terms = {"term": [], "df": [], "statistic": [], "p_value": []}
        assign = np.array(self.design.assign)
        for index, label in enumerate(self.design.term_labels, start=1):
            ii = np.flatnonzero(assign == index)
            b, V = self.coef[ii], self.Vp[np.ix_(ii, ii)]
            if len(ii) == 1:
                df, chi = 1, float(b[0] ** 2 / V[0, 0])
            else:
                Vi, df = _pinv(V, len(b), _EPS**0.5)
                chi = float(b @ Vi @ b)
            if est_disp:
                terms["statistic"].append(chi / df)
                terms["p_value"].append(float(special.fdtrc(df, rdf, chi / df)))
            else:
                terms["statistic"].append(chi)
                terms["p_value"].append(float(special.chdtrc(df, chi)))
            terms["term"].append(label)
            terms["df"].append(float(df))
        stab = {"term": [], "edf": [], "ref_df": [], "statistic": [], "p_value": []}
        sw = np.sqrt(self.weights)
        for i, sm in enumerate(self.smooths):
            idx = slice(sm.first, sm.last)
            V = self.Vp[idx, idx]
            b = self.coef[idx]
            edf = float(self.edf[idx].sum())
            edf1 = float(self.edf1[idx].sum())
            Xt = sw[:, None] * self.X[:, idx]
            if not sm.fixed and sm.null_space_dim == 0:
                stat, pval, rank = self._re_test(i)
            else:
                stat, pval, rank = _test_stat(b, Xt, V, min(Xt.shape[1], edf1), rdf if est_disp else -1)
            stab["term"].append(sm.label)
            stab["edf"].append(edf)
            stab["ref_df"].append(float(rank))
            stab["statistic"].append(stat / rank if est_disp else stat)
            stab["p_value"].append(pval)
        w = self.prior_weights
        mean_y = np.sum(w * self.y) / np.sum(w)
        sw0 = np.sqrt(w)
        r_sq = 1 - np.var(sw0 * (self.y - self.fitted), ddof=1) * (self.n - 1) / (np.var(sw0 * (self.y - mean_y), ddof=1) * rdf)
        dev_expl = (self.null_deviance - self.deviance) / self.null_deviance
        self._summary = (ptab, terms, stab, float(r_sq), float(dev_expl))
        return self._summary

    def _re_test(self, m):
        """mgcv's reTest, for random effects and fully penalized smooths."""
        rind = [i for i, sm in enumerate(self.smooths) if sm.random and i != m]
        p = self.p
        R = np.sqrt(self.weights)[:, None] * self.X
        lam_full = np.zeros((p, p))
        for l, Sj, idx in zip(self.sp, self.S, self.S_index):
            lam_full[np.ix_(idx, idx)] += l * Sj
        sm = self.smooths[m]
        ii = np.arange(sm.first, sm.last)
        if not rind:
            Ve = (self.Ve + self.Ve.T) / 2
            LR = np.vstack([R, _mroot(lam_full).T])
        else:
            rmask = np.zeros(p, dtype=bool)
            for i in rind:
                rmask[self.smooths[i].first:self.smooths[i].last] = True
            R1, R2 = R[:, ~rmask], R[:, rmask]
            S1 = lam_full[np.ix_(~rmask, ~rmask)]
            S2 = lam_full[np.ix_(rmask, rmask)]
            ev, vec = np.linalg.eigh((S2 + S2.T) / 2)
            keep = ev > ev.max() * _EPS**0.8
            inv = np.where(keep, 1 / np.where(keep, ev, 1), 0.0)
            S2r = np.sqrt(inv)[:, None] * vec.T
            n = R.shape[0]
            A = S2r @ R2.T
            L = np.linalg.cholesky(np.eye(n) + A.T @ A).T
            Ve = (L @ R @ self.Vp).T @ (L @ R @ self.Vp) / self.scale
            LR = np.vstack([L @ R1, _mroot(S1).T])
            ii_map = np.cumsum(~rmask) - 1
            ii = ii_map[ii]
        others = [j for j in range(LR.shape[1]) if j not in set(ii)]
        LR = LR[:, others + list(ii)]
        q = len(ii)
        Rm = np.linalg.qr(LR, mode="r")[-q:, -q:]
        idx = slice(sm.first, sm.last)
        Vm = Ve[np.ix_(np.arange(sm.first, sm.last), np.arange(sm.first, sm.last))]
        B = _mroot(Vm)
        d = Rm @ self.coef[idx]
        stat = float(d @ d) / self.scale
        ev = np.linalg.eigvalsh((Rm @ B).T @ (Rm @ B) / self.scale)
        ev = np.maximum(ev, 0)
        rank = int(np.sum(ev > ev.max() * _EPS**0.8))
        if not self.scale_known:
            k = max(1, _round_even(self.n - float(self.edf.sum())))
            pval = _psum_chisq(0, list(ev) + [-stat / k], [1] * len(ev) + [k])
        else:
            pval = _psum_chisq(stat, list(ev))
        return stat, pval, rank

    # -- tables ----------------------------------------------------------------------------------

    def tidy(self, conf_int=False, conf_level=0.95, exponentiate=False, parametric=True, vcov=None):
        if vcov is not None:
            raise ValueError("gam(): tidy() uses the fit's own covariance; vcov= is not available")
        ptab, _, stab, _, _ = self._summary_tables()
        if not parametric:
            if conf_int or exponentiate:
                raise ValueError("tidy(): conf_int and exponentiate are for parametric terms")
            return self._with_key(pl.DataFrame(stab, schema_overrides={"edf": pl.Float64, "ref_df": pl.Float64,
                                                                      "statistic": pl.Float64, "p_value": pl.Float64}))
        table = dict(ptab)
        if conf_int:
            _check_conf_level(conf_level)
            q = float(_special().ndtri((1 + conf_level) / 2))
            table["conf_low"] = table["estimate"] - q * table["std_error"]
            table["conf_high"] = table["estimate"] + q * table["std_error"]
        if exponentiate:
            for c in ("estimate", "conf_low", "conf_high"):
                if c in table:
                    table[c] = np.exp(table[c])
        return self._with_key(pl.DataFrame(table))

    def log_lik(self) -> float:
        sc = float(not self.scale_known)
        p = float(self.edf.sum()) + sc
        return p - self.aic / 2

    def _df(self) -> float:
        sc = float(not self.scale_known)
        p = float(self.edf.sum() if self.edf2 is None else self.edf2.sum()) + sc
        return min(p, self.p + sc)

    def glance(self) -> pl.DataFrame:
        _, _, _, r_sq, dev_expl = self._summary_tables()
        ll, df = self.log_lik(), self._df()
        nobs = int(np.sum(self.prior_weights != 0))
        criterion = {"GCV.Cp": "UBRE" if self.scale_known else "GCV", "REML": "REML"}[self.method]
        return self._with_key(pl.DataFrame({
            "df": [float(self.edf.sum())], "logLik": [ll], "AIC": [-2 * ll + 2 * df],
            "BIC": [-2 * ll + math.log(nobs) * df], "deviance": [self.deviance],
            "df_residual": [float(self.df_residual)], "nobs": [nobs], "adj_r_squared": [r_sq],
            "npar": [self.p], "dev_expl": [dev_expl], "method": [criterion], "sp_criterion": [float(self.score)],
            "scale": [self.scale],
        }).with_columns(pl.col(pl.Float64).fill_nan(None)))

    def vcov_table(self, vcov=None) -> pl.DataFrame:
        kind = "Vp" if vcov is None else vcov
        if kind not in ("Vp", "Ve", "Vc"):
            raise ValueError("gam(): vcov must be 'Vp' (Bayesian, the default), 'Ve' (frequentist) or 'Vc'")
        V = {"Vp": self.Vp, "Ve": self.Ve, "Vc": self.Vc}[kind]
        if V is None:  # vcov.gam(unconditional=TRUE) falls back to Vp
            V = self.Vp
        return self._with_key(pl.DataFrame({"term": self.names, **{n: V[:, i] for i, n in enumerate(self.names)}}))

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.names, self.coef.tolist()))

    @property
    def smoothing_parameters(self) -> dict[str, float]:
        return dict(zip(self.sp_names, self.sp.tolist()))

    # -- prediction ----------------------------------------------------------------------------

    def _new_variables(self, newdata):
        data = Data.read(newdata, None, "predict")
        labels = self.gam_formula.frame_formula.variables()
        exprs = {label: variable_expr(label, "predict")[0] for label in labels}
        for i, label in enumerate(self.formula.offsets):
            exprs[f"__offset{i}"] = variable_expr(label, "predict")[0].cast(pl.Float64)
        frame = data._lf.select(**exprs).collect() if exprs else data._lf.select(pl.len()).collect()
        n = frame.height if exprs else int(frame.item())
        variables = {}
        for label in labels:
            fit_var = self.frame.variables[label]
            if fit_var.is_factor:
                values = frame[label].to_list()
                index = {lv: i for i, lv in enumerate(fit_var.levels)}
                unknown = sorted({v for v in values if v not in index}, key=str)
                if unknown:
                    raise ValueError(f"predict(): factor {label!r} has new levels {unknown}")
                variables[label] = Variable(label, codes=np.array([index[v] for v in values]), levels=fit_var.levels)
            else:
                variables[label] = Variable(label, values=frame[label].cast(pl.Float64).to_numpy())
        offset = np.zeros(n)
        for i in range(len(self.formula.offsets)):
            offset = offset + frame[f"__offset{i}"].to_numpy()
        return variables, n, offset, data

    def _lpmatrix(self, newdata):
        if newdata is None:
            return self.X, self.formula_offset, self._data
        variables, n, offset, data = self._new_variables(newdata)
        blocks = [build_design(self.formula, variables, n).matrix]
        blocks += [sm.predict(variables) for sm in self.smooths]
        return np.hstack(blocks), offset, data

    def _prediction(self, newdata, type_, se_fit):
        if type_ not in ("link", "response", "terms"):
            raise ValueError("gam(): type must be 'link', 'response' or 'terms'")
        X, off, data = self._lpmatrix(newdata)
        if type_ == "terms":
            fits, ses = {}, {}
            assign = np.array(self.design.assign)
            for index, label in enumerate(self.design.term_labels, start=1):
                ii = np.flatnonzero(assign == index)
                fits[label] = X[:, ii] @ self.coef[ii]
                ses[label] = np.sqrt(np.maximum(0, np.sum((X[:, ii] @ self.Vp[np.ix_(ii, ii)]) * X[:, ii], axis=1)))
            for sm in self.smooths:
                ii = np.arange(sm.first, sm.last)
                fits[sm.label] = X[:, ii] @ self.coef[ii]
                ses[sm.label] = np.sqrt(np.maximum(0, np.sum((X[:, ii] @ self.Vp[np.ix_(ii, ii)]) * X[:, ii], axis=1)))
            cols = dict(fits)
            if se_fit:
                cols.update({f"se_{k}": v for k, v in ses.items()})
            return pl.DataFrame(cols), data
        eta = X @ self.coef + off
        se = np.sqrt(np.maximum(0, np.sum((X @ self.Vp) * X, axis=1)))
        fit = eta
        if type_ == "response":
            fit = self.family.link.linkinv(eta)
            se = se * np.abs(self.family.link.mu_eta(eta))
        cols = {"fit": fit}
        if se_fit:
            cols["se_fit"] = se
        return pl.DataFrame(cols), data

    def predict(self, newdata=None, *, type="link", se_fit=False):  # noqa: A002
        out, data = self._prediction(newdata, type, se_fit)
        if newdata is None:
            out = self._padded(out)
        return data.write(out)

    def augment(self, newdata=None, *, type_predict="link", se_fit=True):
        out, data = self._prediction(newdata, type_predict, True)
        out = out.rename({"fit": ".fitted", "se_fit": ".se_fit"})
        if not se_fit:
            out = out.drop(".se_fit")
        if newdata is not None:
            rows = data._lf.collect()
            return data.write(pl.concat([rows, out], how="horizontal"))
        fam = self.family
        d = fam.dev_resids(self.y, self.fitted, self.prior_weights)
        resid = np.sign(self.y - self.fitted) * np.sqrt(np.maximum(d, 0))
        pearson = (self.y - self.fitted) * np.sqrt(self.prior_weights) / np.sqrt(fam.variance(self.fitted))
        cooks = (pearson / (1 - self.hat)) ** 2 * self.hat / (self.scale * float(self.edf.sum()))
        out = out.with_columns(pl.Series(".resid", resid), pl.Series(".hat", self.hat), pl.Series(".cooksd", cooks))
        return pl.concat([self._rows(), self._padded(out)], how="horizontal")

    def anova_table(self, test=None) -> pl.DataFrame:
        """anova.gam for one model: Wald tests of parametric terms and smooths."""
        if test is not None:
            _warn("test argument ignored")
        _, terms, stab, _, _ = self._summary_tables()
        rows = {
            "term": terms["term"] + stab["term"],
            "type": ["parametric"] * len(terms["term"]) + ["smooth"] * len(stab["term"]),
            "df": terms["df"] + stab["edf"],
            "ref_df": [None] * len(terms["term"]) + stab["ref_df"],
            "statistic": terms["statistic"] + stab["statistic"],
            "p_value": terms["p_value"] + stab["p_value"],
        }
        return self._with_key(pl.DataFrame(rows, schema_overrides={"df": pl.Float64, "ref_df": pl.Float64,
                                                                  "statistic": pl.Float64, "p_value": pl.Float64}))

    def anova_type(self, kind, test=None):
        raise ValueError("anova(): type 2 and 3 tables are not available for gam fits")

    def __repr__(self) -> str:
        ptab, _, stab, r_sq, dev_expl = self._summary_tables()
        head = f"gam({self.gam_formula.text}, family = {self.family!r}, method = {self.method!r})"
        if self.key:
            head += "   [" + ", ".join(f"{k} = {v!r}" for k, v in self.key.items()) + "]"
        parts = [head, _table_text(pl.DataFrame(ptab))]
        if stab["term"]:
            parts.append(_table_text(pl.DataFrame(stab)))
        crit = {"GCV.Cp": "UBRE" if self.scale_known else "GCV", "REML": "-REML"}[self.method]
        parts.append(f"R-sq.(adj) = {r_sq:.3g} · deviance explained = {100 * dev_expl:.3g}% · "
                     f"{crit} = {self.score:.5g} · scale est. = {self.scale:.5g} · n = {self.n}")
        return "\n".join(parts)


# linear algebra helpers ------------------------------------------------------------------


def _solve(H, b, fallback=False):
    from scipy.linalg import LinAlgError, cho_factor, cho_solve

    try:
        return cho_solve(cho_factor(H), b)
    except LinAlgError:
        if fallback:
            return None
        return np.linalg.solve(H, b)


def _pinv(V, M, tol):
    ev, vec = np.linalg.eigh(V)
    ev, vec = ev[::-1], vec[:, ::-1]
    M1 = int(np.sum(ev > tol * ev[0]))
    M = min(M, M1)
    inv = np.zeros_like(ev)
    inv[:M] = 1 / ev[:M]
    return (vec * inv) @ vec.T, M


def _mroot(A):
    """A square root B with B B' = A (mgcv's mroot, by pivoted Cholesky rank)."""
    ev, vec = np.linalg.eigh((A + A.T) / 2)
    keep = ev > max(ev.max(), 0) * _EPS**0.8 * 10
    return vec[:, keep] * np.sqrt(ev[keep])


def _round_even(x: float) -> int:
    return int(round(x))


def _initial_sp(X, S, S_index):
    """mgcv's initial.sp (expensive=FALSE)."""
    ldxx = np.sum(X * X, axis=0)
    ldss = np.zeros_like(ldxx)
    pen = np.zeros(len(ldxx), dtype=bool)
    sp = np.zeros(len(S))
    for i, (Si, idx) in enumerate(zip(S, S_index)):
        maS = np.max(np.abs(Si))
        rsS, csS, dS = np.mean(np.abs(Si), axis=1), np.mean(np.abs(Si), axis=0), np.abs(np.diag(Si))
        thresh = _EPS**0.8 * maS
        ind = (rsS > thresh) & (csS > thresh) & (dS > thresh)
        ss = np.diag(Si)[ind]
        xx = ldxx[idx][ind]
        pen[idx] |= ind
        sp[i] = np.mean(xx) / np.mean(ss)
        ldss[idx] += sp[i] * np.diag(Si)
    ind = (ldss > 0) & pen & (ldxx > 0)
    a, b = ldxx[ind], ldss[ind]
    while np.mean(a / (a + b)) > 0.4:
        sp, b = sp * 10, b * 10
    while np.mean(a / (a + b)) < 0.4:
        sp, b = sp / 10, b / 10
    return sp


def _test_stat(p, X, V, rank, res_df):
    """mgcv's testStat (type 0): Wood (2013) Wald test of a smooth."""
    special = _special()
    R = np.linalg.qr(X, mode="r")
    V = R @ V @ R.T
    V = (V + V.T) / 2
    ev, vec = np.linalg.eigh(V)
    ev, vec = ev[::-1], vec[:, ::-1]
    siv = np.sign(vec[0])
    siv[siv == 0] = 1
    vec = vec * siv
    k = max(0, int(math.floor(rank)))
    nu = abs(rank - k)
    k1 = k + 1 if nu > 0 else k
    r_est = int(np.sum(ev > ev.max() * _EPS**0.9))
    if r_est < k1:
        k1 = k = r_est
        nu = 0
        rank = r_est
    if k1 < vec.shape[1]:
        vec = vec[:, :k1]
    if nu > 0 and k > 0:
        vec = vec.copy()
        if k > 1:
            vec[:, :k - 1] = vec[:, :k - 1] / np.sqrt(ev[:k - 1])
        b12 = math.sqrt(max(0.5 * nu * (1 - nu), 0))
        B = np.array([[1, b12], [b12, nu]])
        e = np.diag(ev[k - 1:k1] ** -0.5)
        B = e @ B @ e
        eb, vb = np.linalg.eigh(B)
        eb, vb = eb[::-1], vb[:, ::-1]
        rB = vb @ np.diag(np.sqrt(eb)) @ vb.T
        vec1 = vec.copy()
        vec1[:, k - 1:k1] = (rB @ np.diag([-1, 1]) @ vec[:, k - 1:k1].T).T
        vec[:, k - 1:k1] = (rB @ vec[:, k - 1:k1].T).T
    else:
        vec = vec * np.sqrt(1 / ev[0]) if k == 0 else vec / np.sqrt(ev[:k])
        vec1 = vec
        if k == 1:
            rank = 1
    Rp = R @ p
    d = float(np.sum((vec.T @ Rp) ** 2))
    d1 = float(np.sum((vec1.T @ Rp) ** 2))
    rank1 = rank
    pval = 2.0
    if nu > 0:
        if k1 == 1:
            rank1 = 1
            val = [1.0]
        else:
            val = [1.0] * k1
            rp = nu + 1
            val[k - 1] = (rp + math.sqrt(rp * (2 - rp))) / 2
            val[k1 - 1] = rp - val[k - 1]
        if res_df <= 0:
            pval = (_psum_chisq(d, val) + _psum_chisq(d1, val)) / 2
        else:
            k0 = max(1, _round_even(res_df))
            df = [1] * len(val) + [k0]
            pval = (_psum_chisq(0, val + [-d / k0], df) + _psum_chisq(0, val + [-d1 / k0], df)) / 2
    if pval > 1:
        if res_df <= 0:
            pval = (special.chdtrc(rank1, d) + special.chdtrc(rank1, d1)) / 2
        else:
            pval = (special.fdtrc(rank1, res_df, d / rank1) + special.fdtrc(rank1, res_df, d1 / rank1)) / 2
    return d, min(1.0, float(pval)), rank


# the verb --------------------------------------------------------------------------------


@step
def gam(formula: str, family: Any = "gaussian", *, method: str = "GCV.Cp", weights: Any = None,
        offset: Any = None, contrasts: Any = None, gamma: float = 1.0, scale: float = 0.0,
        maxit: int = 200, na_action: str = "omit"):
    """Fit a generalized additive model, as mgcv's ``gam``.

    ``gam(data, "y ~ s(x1) + s(x2, k=20, bs='cr') + te(x3, x4) + f")``, or
    ``data >> gam(...)``; ``by=`` fits each group. Smooths are ``s()`` with
    ``bs="tp"`` (thin plate regression spline, the default), ``"cr"``,
    ``"cs"``, ``"ps"`` or ``"re"`` (random effect), ``k=`` the basis
    dimension, ``m=`` the penalty order and ``by=`` a factor (one smooth per
    level) or numeric multiplier; ``te()`` builds tensor products of those
    marginals (``bs="cr"``, ``k=5`` each by default). ``family`` is any
    family ``glm`` takes. ``method`` is ``"GCV.Cp"`` (GCV, or UBRE for
    binomial and Poisson) or ``"REML"``. Weights, offsets, contrasts and
    ``na_action`` follow ``glm``; ``scale`` > 0 fixes the scale parameter.
    """
    if method not in _METHODS:
        raise ValueError(f"gam(): method must be one of {list(_METHODS)}, got {method!r}")
    if na_action not in NA_ACTIONS:
        raise ValueError(f"gam(): na_action must be one of {list(NA_ACTIONS)}, got {na_action!r}")
    fam = _as_family(family)
    if fam.theta is not None and method == "REML":
        raise ValueError("gam(): method='REML' is not available for negative_binomial(); use 'GCV.Cp'")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gam(): gamma must be positive")

    def _apply(data: Data) -> Any:
        gf = _GamFormula(formula, data.columns)
        gf.parametric.contrasts = _contrasts(contrasts, gf.frame_formula, "gam")
        gf.parametric.offset_arg = offset
        frames = _model_frames(data, gf.frame_formula, weights, offset, "gam", na_action)
        bare = [None] * len(frames)
        if offset is not None:
            plain = {tuple(fr.index): fr for fr in _model_frames(data, gf.frame_formula, weights, None, "gam", "omit")}
            index = np.concatenate([fr.index for fr in plain.values()])
            values = np.concatenate([fr.offset for fr in plain.values()])
            order = np.argsort(index)
            bare = [values[order][np.searchsorted(index[order], fr.index)] for fr in frames]
        fits = [GeneralizedAdditiveModel(gf, fr, data, fam, method, gamma, scale, maxit, off)
                for fr, off in zip(frames, bare)]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, "gam")


def _anova_gam(fits, test):
    """anova.gam for several fits: anova.glm's deviance table with the
    residual df n - sum(edf1) (less the edf2 correction under REML)."""
    special = _special()
    n = fits[0].n
    resdf = []
    for f in fits:
        dfc = 0.0 if f.edf2 is None else float(f.edf2.sum() - f.edf.sum())
        resdf.append(n - float(f.edf1.sum()) - dfc)
    resdev = [f.deviance for f in fits]
    big = int(np.argmin(resdf))
    if test is None:
        test = "Chisq" if fits[big].scale_known else "F"
    if test not in ("Chisq", "LRT", "F", False):
        raise ValueError("anova(): test must be 'Chisq', 'LRT', 'F' or False")
    df = [None] + [resdf[i - 1] - resdf[i] for i in range(1, len(fits))]
    dev = [None] + [resdev[i - 1] - resdev[i] for i in range(1, len(fits))]
    table = {"term": [f.gam_formula.text for f in fits], "df_residual": resdf, "residual_deviance": resdev,
             "df": df, "deviance": dev}
    scale = fits[big].scale
    if test in ("Chisq", "LRT"):
        table["p_value"] = [None if d is None else float(special.chdtrc(abs(d), abs(v) / scale)) if d != 0 else None
                            for d, v in zip(df, dev)]
    elif test == "F":
        dfs = resdf[big] if not fits[big].scale_known else math.inf
        fval = [None if d is None or d == 0 else (v / d) / scale for d, v in zip(df, dev)]
        table["statistic"] = fval
        table["p_value"] = [None if f is None else float(special.fdtrc(abs(d), dfs, f)) for d, f in zip(df, fval)]
    return pl.DataFrame(table, schema_overrides={"df": pl.Float64, "deviance": pl.Float64,
                                                 "statistic": pl.Float64, "p_value": pl.Float64})
