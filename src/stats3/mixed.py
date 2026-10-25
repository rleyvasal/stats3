"""Linear mixed models: lme4's ``lmer``, fitted by REML or maximum likelihood.

``lmer(data, "y ~ x + (1 | group)")`` fits the fixed effects of the formula
and the random effects of its ``(expr | group)`` terms. The fit follows
lme4: the profiled deviance (or REML criterion) is a function of the
relative covariance factor ``theta`` only, one element per random-effect
column (its standard deviation over the residual one), and the fixed
effects, conditional modes and residual scale come from the penalized
least-squares problem at the optimum, through the Cholesky factors ``L``
(of ``Lambda' Z' W Z Lambda + I``) and ``R_X``. The linear algebra is dense
in the number of random-effect columns ``q``, so groups with a few thousand
levels are fine; ``Z`` itself is kept sparse.

Random-effects terms: ``(1 | g)`` for intercepts, ``(0 + x | g)`` for a
slope on its own, ``(x || g)`` for independent intercepts and slopes,
``(1 | g1) + (1 | g2)`` for crossed factors, ``(1 | g1:g2)`` for an
interaction of factors and ``(1 | g1/g2)`` for nesting (``g1`` and
``g2:g1``, as lme4 expands it). A term whose columns would be correlated,
``(1 + x | g)``, is not implemented and raises with that message.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Data, expression, step
from stats3.core import _check_conf_level, _special, _warn
from stats3.formula import Formula, Variable, _level_name, build_design, encode, parse_formula, variable_expr
from stats3.models import (
    NA_ACTIONS, Models, _Fit, _ModelStep, _contrasts, _key_text, _levels_for, _model_frames, _qr, _table_text,
)

# random-effects terms --------------------------------------------------------------


def _segments(text: str) -> list[tuple[str, str]]:
    """Top-level pieces of a formula side, each with the sign before it."""
    out, depth, start, sign, i = [], 0, 0, "+", 0
    while i < len(text):
        ch = text[i]
        if ch == "`":
            i = text.index("`", i + 1) + 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in "+-" and depth == 0:
            out.append((sign, text[start:i]))
            sign, start = ch, i + 1
        i += 1
    out.append((sign, text[start:]))
    return [(sign, piece.strip()) for sign, piece in out if piece.strip()]


def _bar_split(inner: str) -> tuple[str, str, bool] | None:
    """``expr | group`` or ``expr || group`` at the top level, if there is one."""
    depth, i = 0, 0
    while i < len(inner):
        ch = inner[i]
        if ch == "`":
            i = inner.index("`", i + 1) + 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "|" and depth == 0:
            double = inner[i + 1: i + 2] == "|"
            return inner[:i].strip(), inner[i + 1 + double:].strip(), double
        i += 1
    return None


def _wrapped(piece: str) -> bool:
    """Whether the piece is one parenthesized group, ``( ... )``."""
    if not (piece.startswith("(") and piece.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(piece):
        depth += (ch == "(") - (ch == ")")
        if depth == 0:
            return i == len(piece) - 1
    return False


def _group_columns(spec: str, verb: str) -> list[list[str]]:
    """The grouping columns of ``g``, ``g1:g2`` or ``g1/g2`` (nesting gives
    ``g1`` and ``g2:g1``, as lme4 does)."""
    import re

    groups: list[list[str]] = []
    inner: list[str] = []
    for part in spec.split("/"):
        columns = []
        for name in part.split(":"):
            name = name.strip()
            if name.startswith("`") and name.endswith("`") and len(name) > 2:
                name = name[1:-1]
            elif not re.fullmatch(r"[A-Za-z_.][A-Za-z0-9_.]*", name):
                raise ValueError(f"{verb}(): random effects need grouping column names after '|', got {spec!r}")
            columns.append(name)
        inner = columns + inner
        groups.append(list(inner))
    return groups


def _double_bar(expr: str, verb: str) -> list[str]:
    """``(x || g)``: one term per column, as lme4's ``expandDoubleVerts``."""
    intercept, slopes = True, []
    for sign, piece in _segments(expr):
        if piece == "0":
            intercept = sign == "-"
        elif piece == "1":
            intercept = sign == "+"
        elif sign == "-":
            raise ValueError(f"{verb}(): cannot subtract {piece!r} inside a random-effects term")
        else:
            slopes.append(piece)
    return (["1"] if intercept else []) + [f"0 + {s}" for s in slopes]


def random_terms(text: str, verb: str) -> tuple[str, list[tuple[str, list[str]]]]:
    """Take lme4's random-effects terms out of a formula.

    Returns the fixed-effects formula and the random terms as (expression,
    grouping columns) pairs, with ``(x || g)`` expanded into independent
    terms and ``g1/g2`` into ``g1`` and ``g2:g1``.
    """
    if not isinstance(text, str) or "~" not in text:
        raise ValueError(f"{verb}(): formula must look like 'y ~ x + (1 | group)', got {text!r}")
    lhs, _, rhs = text.partition("~")
    fixed: list[str] = []
    random: list[tuple[str, list[str]]] = []
    for sign, piece in _segments(rhs):
        split = _bar_split(piece[1:-1]) if _wrapped(piece) else None
        if split is None:
            fixed.append(piece if not fixed and sign == "+" else f"{sign} {piece}")
            continue
        if sign == "-":
            raise ValueError(f"{verb}(): random-effects terms cannot be subtracted: {piece!r}")
        expr, group, double = split
        if "|" in expr or "|" in group:
            raise ValueError(f"{verb}(): cannot read the random-effects term {piece!r}")
        groups = _group_columns(group, verb)
        for e in (_double_bar(expr, verb) if double else [expr or "1"]):
            for g in groups:
                random.append((e, g))
    if not random:
        raise ValueError(f"{verb}(): the formula has no random effects; add a term like (1 | group), or use lm()")
    fixed_text = " ".join(fixed).strip() or "1"
    return f"{lhs.strip()} ~ {fixed_text}", random


def _unique_names(names: list[str]) -> list[str]:
    """R's ``make.unique``: repeats get ``.1``, ``.2``, ..."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        count = seen.get(name, 0)
        out.append(name if count == 0 else f"{name}.{count}")
        seen[name] = count + 1
    return out


@dataclass
class _RandomTerm:
    """One scalar random-effects term: a column of ``Z`` per group level."""

    expr: str
    columns: list[str]
    formula: Formula
    name: str  # the column's name, (Intercept) or the slope's
    group: str  # island, or island:species
    levels: list[str]
    codes: np.ndarray
    values: np.ndarray

    @property
    def text(self) -> str:
        return f"({self.expr} | {self.group})"


def _group_levels(rows: pl.DataFrame, columns: list[str], data: Data) -> tuple[list[str], list[str]]:
    """Each row's group label, and the levels in R's order (the first
    factor varying slowest, unused combinations dropped)."""
    per_column = [encode(rows, c, True, _levels_for(data, c)) for c in columns]
    labels = [":".join(_level_name(v.levels[v.codes[i]]) for v in per_column) for i in range(rows.height)]
    present = set(labels)
    levels = [":".join(_level_name(lv) for lv in combo)
              for combo in itertools.product(*[v.levels for v in per_column])]
    return labels, [lv for lv in levels if lv in present]


def _labels(frame: pl.DataFrame, columns: list[str]) -> list[str]:
    return [":".join(_level_name(v) for v in row) for row in frame.select(columns).iter_rows()]


# the penalized least-squares problem ------------------------------------------------


# Above this many random-effect columns the q x q system is factored as a
# sparse LU (SuperLU) instead of a dense Cholesky: for one grouping factor
# the system is block diagonal, so the sparse factor has almost no fill.
SPARSE_FROM = 1500


@dataclass
class _Parts:
    """The weighted problem's cross products, fixed through the optimization."""

    n: int
    p: int
    Xw: np.ndarray
    yw: np.ndarray
    Zw: Any  # sparse n x q
    ZtZ: Any  # q x q, dense below SPARSE_FROM columns, else sparse
    ZtX: np.ndarray
    Zty: np.ndarray
    XtX: np.ndarray
    Xty: np.ndarray
    block: np.ndarray  # theta index per column of Z
    log_weights: float  # sum of the log prior weights, in lme4's criteria


class _Factor:
    """A factorization of ``A = Lambda' Z' W Z Lambda + I``: its log
    determinant and solves, by dense Cholesky or sparse LU."""

    def __init__(self, ztz, lam: np.ndarray):
        from scipy import sparse
        from scipy.linalg import cho_factor

        if sparse.issparse(ztz):
            from scipy.sparse.linalg import splu

            scale = sparse.diags(lam)
            a = (scale @ ztz @ scale + sparse.identity(len(lam))).tocsc()
            self._lu = splu(a)
            self.logdet = float(np.sum(np.log(np.abs(self._lu.U.diagonal()))))
            self._dense = None
        else:
            a = ztz * np.outer(lam, lam)
            a[np.diag_indices_from(a)] += 1
            self._dense = cho_factor(a, lower=True)
            self.logdet = 2 * float(np.sum(np.log(np.diag(self._dense[0]))))

    def solve(self, b: np.ndarray) -> np.ndarray:
        from scipy.linalg import cho_solve

        if self._dense is not None:
            return cho_solve(self._dense, b)
        return self._lu.solve(np.asarray(b, dtype=float))


@dataclass
class _Solution:
    theta: np.ndarray
    lam: np.ndarray
    factor: _Factor
    LX: np.ndarray
    M: np.ndarray  # Lambda' Z' W X
    S: np.ndarray  # A^-1 M
    beta: np.ndarray
    u: np.ndarray
    b: np.ndarray
    wrss: float
    pwrss: float
    ldL2: float
    ldRX2: float
    deviance: float  # the ML deviance at these parameters
    reml: float  # the REML criterion at these parameters


def _solve(parts: _Parts, theta: np.ndarray) -> _Solution:
    """lme4's profiled fit for one theta: the factors, fixed effects,
    spherical and conditional modes, and both criteria. Written with solves
    against ``A`` rather than its Cholesky factor, so a sparse LU serves too:
    ``R_ZX' R_ZX = M' A^-1 M`` with ``M = Lambda' Z' W X``."""
    from scipy.linalg import cholesky, solve_triangular

    n, p = parts.n, parts.p
    lam = theta[parts.block]
    factor = _Factor(parts.ZtZ, lam)
    m = lam[:, None] * parts.ZtX
    v = lam * parts.Zty
    s_ = factor.solve(m) if p else np.zeros((len(lam), 0))
    av = factor.solve(v)
    if p:
        LX = cholesky(parts.XtX - m.T @ s_, lower=True)
        cb = solve_triangular(LX, parts.Xty - s_.T @ v, lower=True)
        beta = solve_triangular(LX.T, cb, lower=False)
    else:
        LX, beta = np.zeros((0, 0)), np.zeros(0)
    u = av - s_ @ beta
    b = lam * u
    resid = parts.yw - parts.Xw @ beta - parts.Zw @ b
    wrss = float(resid @ resid)
    pwrss = wrss + float(u @ u)
    ldL2 = factor.logdet
    ldRX2 = 2 * float(np.sum(np.log(np.diag(LX)))) if p else 0.0
    deviance = ldL2 + n * (1 + math.log(2 * math.pi * pwrss / n)) - parts.log_weights
    reml = ldL2 + ldRX2 + (n - p) * (1 + math.log(2 * math.pi * pwrss / (n - p))) - parts.log_weights
    return _Solution(theta, lam, factor, LX, m, s_, beta, u, b, wrss, pwrss, ldL2, ldRX2, deviance, reml)


def _derivatives(objective, theta: np.ndarray, value: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Gradient and Hessian of the criterion by central differences."""
    k = len(theta)
    h = 1e-4 * np.maximum(1.0, np.abs(theta))
    steps = np.eye(k) * h
    plus = [objective(theta + steps[i]) for i in range(k)]
    minus = [objective(theta - steps[i]) for i in range(k)]
    gradient, hessian, used = np.zeros(k), np.zeros((k, k)), 2 * k
    for i in range(k):
        gradient[i] = (plus[i] - minus[i]) / (2 * h[i])
        hessian[i, i] = (plus[i] - 2 * value + minus[i]) / h[i] ** 2
    for i in range(k):
        for j in range(i):
            pp = objective(theta + steps[i] + steps[j])
            mm = objective(theta - steps[i] - steps[j])
            used += 2
            hessian[i, j] = hessian[j, i] = (pp - 2 * value + mm - hessian[i, i] * h[i] ** 2
                                             - hessian[j, j] * h[j] ** 2) / (2 * h[i] * h[j])
    return gradient, hessian, used


def _optimize(objective, k: int) -> tuple[np.ndarray, int]:
    """Minimize the profiled criterion from lme4's start of ones (theta = 1).

    The criterion is an even function of each element of theta (changing a
    block's sign changes nothing in the Cholesky factors), so it is minimized
    without lme4's bounds and the absolute values are returned: at theta = 0
    with negative curvature, where a bounded search can stall, an unbounded
    one passes through. A quasi-Newton search first, then Newton steps on
    central differences until the step is below 1e-10. The criterion is flat
    near its optimum (a step of 1e-7 in theta changes it by less than its
    rounding), so the last steps are judged by the derivatives, not by the
    value; this finds the optimum more precisely than lme4's default
    optimizer, which stops on the value."""
    from scipy.optimize import minimize

    result = minimize(objective, np.ones(k), method="L-BFGS-B",
                      options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000, "maxfun": 20000})
    theta, value = np.asarray(result.x, dtype=float), float(result.fun)
    evaluations = int(result.nfev)
    noise = 1e-9 * max(1.0, abs(value))
    for _ in range(100):
        gradient, hessian, used = _derivatives(objective, theta, value)
        evaluations += used
        if np.all(np.linalg.eigvalsh(hessian) > 0):
            direction = -np.linalg.solve(hessian, gradient)
        else:
            scale = max(1.0, float(np.abs(theta).max())) / max(1.0, float(np.abs(gradient).max()))
            direction = -gradient * scale
            # A stationary point with negative curvature (theta = 0 on an
            # even function) is left along that axis.
            flat = (np.abs(gradient) < 1e-7) & (np.diag(hessian) < 0)
            direction[flat] = 1e-2 * np.maximum(1.0, np.abs(theta[flat]))
        accepted = False
        for _ in range(40):
            candidate = theta + direction
            new = objective(candidate)
            evaluations += 1
            if new <= value + noise:
                accepted = True
                break
            direction = direction / 2
        if not accepted:
            break
        change = float(np.abs(candidate - theta).max())
        theta, value = candidate, new
        if change < 1e-10:
            break
    return np.abs(theta), evaluations


# the model ----------------------------------------------------------------------------


class LinearMixedModel(_Fit):
    """A fitted ``lmer``: fixed effects, variance components and conditional modes."""

    kind = "mixed"

    def __init__(self, formula: Formula, frame, data: Data, random: list[tuple[str, list[str]]], reml: bool):
        design = build_design(formula, frame.variables, len(frame.weights))
        super().__init__(formula, design, frame, data, "lmer")
        if frame.y.ndim != 1 or frame.response_levels is not None:
            raise ValueError("lmer(): the response must be one numeric column")
        self.reml = reml
        self.y, self.prior_weights = frame.y, frame.weights
        n = self.n = len(self.y)
        w = frame.weights
        if np.any(w <= 0):
            raise ValueError("lmer(): weights must be positive")
        self.terms = self._random_terms(random, frame, data)
        self._check_levels()
        sw = np.sqrt(w)
        x = design.matrix
        qr = _qr(x * sw[:, None], 1e-7, np.zeros(n))
        if len(qr.kept) < x.shape[1]:
            dropped = x.shape[1] - len(qr.kept)
            _warn(f"fixed-effect model matrix is rank deficient so dropping {dropped} column{'s' if dropped > 1 else ''} / coefficient{'s' if dropped > 1 else ''}")
        self.kept = list(qr.kept)
        self.names = [design.names[j] for j in self.kept]
        self.x = x[:, self.kept]
        p = self.x.shape[1]
        if n <= p:
            raise ValueError("lmer(): more observations than fixed effects are needed")
        self.z = self._z_matrix(self.terms, n)
        self._parts = self._make_parts(self.z, self.x, self.y - frame.offset, w)
        self._fit()

    # -- building ----------------------------------------------------------------------

    @staticmethod
    def _random_terms(random, frame, data) -> list[_RandomTerm]:
        terms = []
        n = len(frame.weights)
        for expr, columns in random:
            parsed = parse_formula(f"~ {expr}", "lmer")
            design = build_design(parsed, frame.variables, n)
            group = ":".join(columns)
            if design.matrix.shape[1] != 1:
                hint = f"; use ({expr} || {group}) for independent effects" if parsed.intercept and design.matrix.shape[1] > 1 else ""
                raise ValueError(
                    f"lmer(): correlated random effects ({expr} | {group}) are not implemented yet "
                    f"({design.matrix.shape[1]} columns per level){hint}")
            labels, levels = _group_levels(frame.rows, columns, data)
            index = {lv: i for i, lv in enumerate(levels)}
            codes = np.array([index[lb] for lb in labels], dtype=np.int64)
            terms.append(_RandomTerm(expr, columns, parsed, design.names[0], group, levels, codes,
                                     design.matrix[:, 0].astype(float)))
        # lme4 orders the terms by decreasing number of levels, stably.
        terms.sort(key=lambda t: -len(t.levels))
        return terms

    def _check_levels(self) -> None:
        few = [t.group for t in self.terms if len(t.levels) < 2]
        if few:
            raise ValueError(f"lmer(): grouping factors must have > 1 sampled level (problems: {', '.join(few)})")
        many = [t.group for t in self.terms if len(t.levels) >= self.n]
        if many:
            raise ValueError("lmer(): number of levels of each grouping factor must be < number of observations "
                             f"(problems: {', '.join(many)})")

    @staticmethod
    def _z_matrix(terms: list[_RandomTerm], n: int):
        from scipy.sparse import csr_matrix

        rows, cols, values, offset = [], [], [], 0
        for term in terms:
            rows.append(np.arange(n))
            cols.append(term.codes + offset)
            values.append(term.values)
            offset += len(term.levels)
        return csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(n, offset))

    def _make_parts(self, z, x, y, w) -> _Parts:
        sw = np.sqrt(w)
        xw, yw = x * sw[:, None], y * sw
        zw = z.multiply(sw[:, None]).tocsr()
        block = np.concatenate([np.full(len(t.levels), i) for i, t in enumerate(self.terms)])
        ztz = (zw.T @ zw).tocsr()
        if ztz.shape[0] < SPARSE_FROM:
            ztz = ztz.toarray()
        return _Parts(len(y), x.shape[1], xw, yw, zw, ztz, np.asarray(zw.T @ xw), np.asarray(zw.T @ yw),
                      xw.T @ xw, xw.T @ yw, block, float(np.sum(np.log(w))))

    def _criterion(self, theta: np.ndarray) -> float:
        solution = _solve(self._parts, np.asarray(theta, dtype=float))
        return solution.reml if self.reml else solution.deviance

    def _fit(self) -> None:
        theta, self.evaluations = _optimize(self._criterion, len(self.terms))
        self._finish(theta)

    def _finish(self, theta: np.ndarray) -> None:
        from scipy.linalg import solve_triangular

        s = self._solution = _solve(self._parts, theta)
        n, p = self.n, self._parts.p
        self.theta = theta
        self.coef = s.beta
        self.b, self.u = s.b, s.u
        self.criterion = s.reml if self.reml else s.deviance
        self.sigma = math.sqrt(s.pwrss / (n - p)) if self.reml else math.sqrt(s.pwrss / n)
        if p:
            inverse = solve_triangular(s.LX, np.eye(p), lower=True)
            self.cov = self.sigma**2 * inverse.T @ inverse
        else:
            self.cov = np.zeros((0, 0))
        self.fixed = self.x @ self.coef + self.frame.offset
        self.fitted = self.fixed + self.z @ self.b
        self.residuals = self.y - self.fitted
        self.npar = p + len(self.terms) + 1
        self.df_residual = float(n - self.npar)
        self.singular = bool(np.any(np.abs(theta) < 1e-4))
        if self.singular:
            _warn("boundary (singular) fit: see lme4's isSingular")

    def refit_ml(self) -> "LinearMixedModel":
        """The same model fitted by maximum likelihood (itself, if it already was)."""
        if not self.reml:
            return self
        other = copy.copy(self)
        other.reml = False
        other._fit()
        return other

    # -- summaries ---------------------------------------------------------------------

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.names, self.coef.tolist()))

    @property
    def groups_unique(self) -> list[str]:
        return list(dict.fromkeys(t.group for t in self.terms))

    def var_corr(self) -> pl.DataFrame:
        """lme4's ``VarCorr``: one row per random-effect term, then the residual."""
        groups = _unique_names([t.group for t in self.terms]) + ["Residual"]
        names = [t.name for t in self.terms] + [None]
        sds = [abs(float(th)) * self.sigma for th in self.theta] + [self.sigma]
        return pl.DataFrame({"group": groups, "term": names, "variance": [sd**2 for sd in sds], "sd": sds})

    def log_lik(self) -> float:
        return -self.criterion / 2

    def _check_vcov(self, vcov) -> None:
        if vcov is not None:
            raise ValueError("lmer(): this fit supplies its own covariance; vcov= is for lm/glm")

    def tidy(self, conf_int: bool = False, conf_level: float = 0.95, effects: Any = ("fixed", "ran_pars"),
             vcov: Any = None) -> pl.DataFrame:
        """broom.mixed's table: fixed effects (t statistics, no p-values or
        degrees of freedom, as ``summary.merMod``) and the random-effect
        standard deviations as ``ran_pars`` rows; ``effects="ran_vals"``
        gives the conditional modes instead."""
        self._check_vcov(vcov)
        wanted = [effects] if isinstance(effects, str) else list(effects)
        unknown = set(wanted) - {"fixed", "ran_pars", "ran_vals"}
        if unknown or not wanted:
            raise ValueError("lmer(): effects must be chosen from 'fixed', 'ran_pars' and 'ran_vals'")
        if "ran_vals" in wanted:
            if len(wanted) > 1:
                raise ValueError("lmer(): effects='ran_vals' gives its own table; ask for it alone")
            table = self.ranef_table().select(pl.lit("ran_vals").alias("effect"), pl.all())
            return self._with_key(table)
        se = np.sqrt(np.diag(self.cov))
        fixed = pl.DataFrame({"effect": ["fixed"] * len(self.names), "group": [None] * len(self.names),
                              "term": self.names, "estimate": self.coef, "std_error": se,
                              "statistic": self.coef / se, "p_value": [None] * len(self.names)},
                             schema_overrides={"group": pl.String, "p_value": pl.Float64})
        vc = self.var_corr()
        random = pl.DataFrame({"effect": ["ran_pars"] * vc.height, "group": vc["group"],
                               "term": ["sd__" + (t or "Observation") for t in vc["term"]],
                               "estimate": vc["sd"], "std_error": [None] * vc.height,
                               "statistic": [None] * vc.height, "p_value": [None] * vc.height},
                              schema_overrides={"std_error": pl.Float64, "statistic": pl.Float64, "p_value": pl.Float64})
        if conf_int:
            # broom.mixed's default: Wald intervals on the normal quantile.
            _check_conf_level(conf_level)
            q = float(_special().ndtri((1 + conf_level) / 2))
            fixed = fixed.with_columns(conf_low=pl.col("estimate") - q * pl.col("std_error"),
                                       conf_high=pl.col("estimate") + q * pl.col("std_error"))
            random = random.with_columns(conf_low=pl.lit(None, pl.Float64), conf_high=pl.lit(None, pl.Float64))
        parts = [fixed] if "fixed" in wanted else []
        if "ran_pars" in wanted:
            parts.append(random)
        return self._with_key(pl.concat(parts, how="vertical"))

    def ranef_table(self) -> pl.DataFrame:
        """lme4's ``ranef(condVar=TRUE)`` as a table: group, term, level, the
        conditional mode and its conditional standard deviation
        (``sigma^2 Lambda A^-1 Lambda``, ignoring the fixed effects' uncertainty)."""
        s = self._solution
        q = len(s.lam)
        diagonal = np.empty(q)
        chunk = max(1, int(2e6 // q))
        for start in range(0, q, chunk):
            stop = min(q, start + chunk)
            columns = np.zeros((q, stop - start))
            columns[np.arange(start, stop), np.arange(stop - start)] = 1
            diagonal[start:stop] = s.factor.solve(columns)[np.arange(start, stop), np.arange(stop - start)]
        conditional = self.sigma * np.abs(s.lam) * np.sqrt(diagonal)
        offsets = np.cumsum([0] + [len(t.levels) for t in self.terms])
        rows = {"group": [], "term": [], "level": [], "estimate": [], "std_error": []}
        for group in self.groups_unique:
            for i, term in enumerate(self.terms):
                if term.group != group:
                    continue
                start, stop = offsets[i], offsets[i + 1]
                rows["group"].extend([group] * len(term.levels))
                rows["term"].extend([term.name] * len(term.levels))
                rows["level"].extend(term.levels)
                rows["estimate"].extend(self.b[start:stop].tolist())
                rows["std_error"].extend(conditional[start:stop].tolist())
        return pl.DataFrame(rows)

    def ranef(self) -> pl.DataFrame:
        return self._with_key(self.ranef_table())

    def glance(self) -> pl.DataFrame:
        ll = self.log_lik()
        s = self._solution
        return self._with_key(pl.DataFrame({
            "nobs": [self.n], "sigma": [self.sigma], "logLik": [ll],
            "AIC": [-2 * ll + 2 * self.npar], "BIC": [-2 * ll + math.log(self.n) * self.npar],
            "REMLcrit": [s.reml if self.reml else None], "deviance": [s.deviance],
            "df_residual": [self.df_residual],
        }, schema_overrides={"REMLcrit": pl.Float64}))

    def vcov_table(self, vcov: Any = None) -> pl.DataFrame:
        self._check_vcov(vcov)
        return self._with_key(pl.DataFrame({"term": self.names,
                                            **{name: self.cov[:, j] for j, name in enumerate(self.names)}}))

    # -- rows ------------------------------------------------------------------------------

    def hat_values(self) -> np.ndarray:
        """The diagonal of the hat matrix of the penalized problem, as lme4's
        ``hatvalues``; computed in chunks of rows so ``Z`` stays sparse."""
        from scipy.linalg import solve_triangular

        s, parts = self._solution, self._parts
        out = np.empty(self.n)
        chunk = max(1, int(2e6 // max(1, parts.Zw.shape[1])))
        for start in range(0, self.n, chunk):
            stop = min(self.n, start + chunk)
            zc = (parts.Zw[start:stop].toarray() * s.lam).T  # q x m
            t1 = s.factor.solve(zc)
            h = np.sum(zc * t1, axis=0)
            if parts.p:
                t2 = solve_triangular(s.LX, parts.Xw[start:stop].T - s.M.T @ t1, lower=True)
                h = h + np.sum(t2**2, axis=0)
            out[start:stop] = h
        return out

    def _influence(self) -> dict[str, np.ndarray]:
        h = self.hat_values()
        pearson = np.sqrt(self.prior_weights) * self.residuals
        p = max(1, len(self.names))
        with np.errstate(divide="ignore", invalid="ignore"):
            cooks = (pearson / (1 - h))**2 * h / (p * self.sigma**2)
        return {".fitted": self.fitted, ".resid": self.residuals, ".hat": h, ".cooksd": cooks, ".fixed": self.fixed}

    def augment(self, newdata: Any = None, *, re_form: Any = None, allow_new_levels: bool = False) -> Any:
        if newdata is None:
            cols = self._influence()
            frame = pl.concat([self._rows(), self._padded(pl.DataFrame(cols))], how="horizontal")
            return frame.with_columns(pl.col(c).fill_nan(None) for c in cols)
        fit, rows, data = self._predict(newdata, re_form, allow_new_levels)
        return data.write(pl.concat([rows, pl.DataFrame({".fitted": fit})], how="horizontal"))

    def predict(self, newdata: Any = None, *, re_form: Any = None, allow_new_levels: bool = False) -> Any:
        """Predictions as lme4's ``predict``: with the random effects
        (``re_form=None``), or at the population level (``re_form="population"``,
        lme4's ``re.form = NA`` or ``~0``). New levels of a grouping factor
        raise unless ``allow_new_levels=True``, which predicts them at zero."""
        population = _population(re_form)
        if newdata is None:
            fit = self._padded(pl.DataFrame({"fit": self.fixed if population else self.fitted}))
            return self._data.write(fit)
        fit, _, data = self._predict(newdata, re_form, allow_new_levels)
        return data.write(pl.DataFrame({"fit": fit}))

    def _predict(self, newdata: Any, re_form: Any, allow_new_levels: bool):
        population = _population(re_form)
        data = Data.read(newdata, None, "predict")
        labels = list(self.formula.variables())
        for term in self.terms:
            labels.extend(v for v in term.formula.variables() if v not in labels)
        exprs = {label: variable_expr(label, "predict")[0] for label in labels}
        for i, label in enumerate(self.formula.offsets):
            exprs[f"__offset{i}"] = variable_expr(label, "predict")[0].cast(pl.Float64)
        if self.formula.offset_arg is not None:
            exprs["__offset_arg"] = expression(self.formula.offset_arg).cast(pl.Float64)
        columns = [] if population else [c for t in self.terms for c in t.columns]
        frame = data._lf.select(*[pl.col(c) for c in dict.fromkeys(columns) if c not in exprs], **exprs).collect()
        n = frame.height
        variables = {}
        for label in labels:
            levels = self.design.levels.get(label)
            if levels is not None:
                values = frame[label].to_list()
                unknown = sorted({v for v in values if v not in set(levels)}, key=str)
                if unknown:
                    raise ValueError(f"predict(): factor {label!r} has new levels {unknown}")
                index = {lv: i for i, lv in enumerate(levels)}
                variables[label] = Variable(label, codes=np.array([index[v] for v in values]), levels=levels)
            else:
                variables[label] = Variable(label, values=frame[label].cast(pl.Float64).to_numpy())
        x = build_design(self.formula, variables, n).matrix[:, self.kept]
        fit = x @ self.coef
        for i in range(len(self.formula.offsets)):
            fit = fit + frame[f"__offset{i}"].to_numpy()
        if self.formula.offset_arg is not None:
            fit = fit + frame["__offset_arg"].to_numpy()
        if not population:
            offsets = np.cumsum([0] + [len(t.levels) for t in self.terms])
            for i, term in enumerate(self.terms):
                values = build_design(term.formula, variables, n).matrix[:, 0]
                index = {lv: j for j, lv in enumerate(term.levels)}
                effect = np.zeros(n)
                unknown = []
                for row, label in enumerate(_labels(frame, term.columns)):
                    j = index.get(label)
                    if j is None:
                        unknown.append(label)
                    else:
                        effect[row] = self.b[offsets[i] + j]
                if unknown and not allow_new_levels:
                    shown = ", ".join(sorted(set(unknown)))
                    raise ValueError(f"predict(): new levels detected in newdata for {term.group!r}: {shown}; "
                                     "use allow_new_levels=True or re_form='population'")
                fit = fit + values * effect
        return fit, data._lf.collect(), data

    # -- anova -----------------------------------------------------------------------------

    def anova_table(self, test: Any = None) -> pl.DataFrame:
        """lme4's single-model table: sequential sums of squares of the fixed
        terms from ``R_X``, F statistics on the residual variance, no p-values."""
        if test is not None:
            raise ValueError("lmer(): anova() of one model gives F statistics only; test= is not used")
        assign = [self.design.assign[j] for j in self.kept]
        ss = (self._solution.LX.T @ self.coef)**2
        rows: dict[str, list[Any]] = {"term": [], "df": [], "sumsq": [], "meansq": [], "statistic": []}
        for index, label in enumerate(self.design.term_labels, start=1):
            cols = [k for k, a in enumerate(assign) if a == index]
            if not cols:
                continue
            total, df = float(np.sum(ss[cols])), float(len(cols))
            rows["term"].append(label)
            rows["df"].append(df)
            rows["sumsq"].append(total)
            rows["meansq"].append(total / df)
            rows["statistic"].append(total / df / self.sigma**2)
        return self._with_key(pl.DataFrame(rows, schema_overrides={c: pl.Float64 for c in rows if c != "term"}))

    def anova_type(self, kind: int, test: Any = None) -> pl.DataFrame:
        raise TypeError("anova(): type II/III tables are not available for lmer fits")

    # -- printing ---------------------------------------------------------------------------

    def __repr__(self) -> str:
        fixed = self.tidy(effects="fixed").select("term", "estimate", "std_error", "statistic")
        head = (f"lmer({self.formula.text}{', reml=False' if not self.reml else ''})"
                + _key_text(self.key) + f"   n = {self.n}")
        crit = f"{'REML criterion' if self.reml else 'deviance'} {self.criterion:.4g}"
        return (f"{head}\n{_table_text(fixed)}\nrandom effects\n{_table_text(self.var_corr())}\n{crit}"
                + ("\nboundary (singular) fit" if self.singular else ""))


def _population(re_form: Any) -> bool:
    if re_form is None:
        return False
    if isinstance(re_form, str) and re_form.replace(" ", "") in ("population", "~0", "NA"):
        return True
    if isinstance(re_form, float) and math.isnan(re_form):
        return True
    raise ValueError("re_form must be None (all random effects) or 'population' (none)")


@step
def lmer(formula: str, *, reml: bool = True, weights: Any = None, offset: Any = None, contrasts: Any = None,
         na_action: str = "omit"):
    """Fit a linear mixed model, as lme4's ``lmer``.

    ``lmer(sleepstudy, "Reaction ~ Days + (Days || Subject)")`` or
    ``data >> lmer(...)``; ``by=`` fits one model per group. Random effects:
    ``(1 | g)``, ``(0 + x | g)``, ``(x || g)`` (independent intercept and
    slope), crossed ``(1 | g1) + (1 | g2)``, ``(1 | g1:g2)`` and nested
    ``(1 | g1/g2)``. Correlated random effects, ``(1 + x | g)``, are not
    implemented yet and raise. ``reml=False`` fits by maximum likelihood.
    Weights (positive), offsets, contrasts and ``na_action`` work as in
    ``lm``. Print the fit for a summary; ``tidy`` (fixed effects and
    variance components), ``glance``, ``augment``, ``ranef``, ``vcov``,
    ``predict(re_form=)`` and ``anova`` give tables, and
    ``anova(fit1, fit2)`` refits with ML for a likelihood ratio test.
    """
    if na_action not in NA_ACTIONS:
        raise ValueError(f"lmer(): na_action must be one of {list(NA_ACTIONS)}, got {na_action!r}")
    if not isinstance(reml, (bool, np.bool_)):
        raise ValueError("lmer(): reml must be True or False")
    fixed_text, random = random_terms(formula, "lmer")
    full = " ".join(formula.split())

    def apply(data: Data) -> Any:
        parsed = parse_formula(fixed_text, "lmer", data.columns)
        parsed.contrasts = _contrasts(contrasts, parsed, "lmer")
        parsed.offset_arg = offset
        parsed.text = full
        frame_formula = copy.deepcopy(parsed)
        extras = []
        for expr, columns in random:
            extras.extend(parse_formula(f"~ {expr}", "lmer").variables())
            extras.extend(columns)
        for label in extras:
            if label not in frame_formula.variables():
                frame_formula.terms.append((label,))
        frames = _model_frames(data, frame_formula, weights, offset, "lmer", na_action)
        fits = [LinearMixedModel(parsed, fr, data, random, bool(reml)) for fr in frames]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(apply, "lmer")


def ranef(x: Any) -> Any:
    """The conditional modes of the random effects, as lme4's ``ranef``:
    ``group``, ``term``, ``level``, ``estimate`` and the conditional
    ``std_error`` (``condVar=TRUE``), in the data's frame type."""
    from stats3.models import _table

    fits = x.fits if isinstance(x, Models) else [x]
    if not all(isinstance(f, LinearMixedModel) for f in fits):
        raise TypeError("ranef(): expected a model from lmer()")
    return _table(x, "ranef")


def _anova_lmer(fits: list[LinearMixedModel], test: Any) -> pl.DataFrame:
    """lme4's comparison of nested fits: refitted with ML, sorted by number
    of parameters, with likelihood ratio chi-squared tests."""
    if test not in (None, "Chisq"):
        raise ValueError("anova(): lmer comparisons use the likelihood ratio chi-squared test")
    first = fits[0]
    for f in fits[1:]:
        if (not np.array_equal(f.y, first.y) or not np.array_equal(f.frame.index, first.frame.index)
                or not np.array_equal(f.prior_weights, first.prior_weights)):
            raise ValueError("anova(): lmer fits must use the same response, rows and weights")
    if any(f.reml for f in fits):
        _warn("refitting model(s) with ML (instead of REML)")
    fits = sorted((f.refit_ml() for f in fits), key=lambda f: f.npar)
    ll = [f.log_lik() for f in fits]
    npar = [float(f.npar) for f in fits]
    df = [None] + [npar[i] - npar[i - 1] for i in range(1, len(fits))]
    stat = [None] + [-2 * (ll[i - 1] - ll[i]) for i in range(1, len(fits))]
    p = [None] + [float(_special().chdtrc(d, s)) if d > 0 else None for d, s in zip(df[1:], stat[1:])]
    return pl.DataFrame({
        "term": [f.formula.text for f in fits], "npar": npar,
        "AIC": [-2 * l + 2 * k for l, k in zip(ll, npar)],
        "BIC": [-2 * l + math.log(f.n) * k for l, k, f in zip(ll, npar, fits)],
        "logLik": ll, "deviance": [-2 * l for l in ll], "statistic": stat, "df": df, "p_value": p,
    }, schema_overrides={"statistic": pl.Float64, "df": pl.Float64, "p_value": pl.Float64})
