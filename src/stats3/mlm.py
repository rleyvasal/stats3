"""Multivariate linear models: R's ``mlm``, ``manova``, ``anova.mlm`` and
``mauchly.test``.

``lm(data, "cbind(y1, y2, y3) ~ g + x")`` fits one least-squares regression
per response on the same design (one QR, as R does). ``tidy`` stacks the
responses' coefficient tables with a ``response`` column, as broom's
``tidy.mlm``; ``predict`` and ``vcov`` follow ``predict.mlm`` and
``vcov.mlm``. ``anova`` gives ``anova.mlm``'s sequential multivariate tests
from the same sums of squares and cross products (SSCP) as ``manova``, and
compares nested mlm fits; ``test="Spherical"`` gives the univariate F tests
with Greenhouse-Geisser and Huynh-Feldt corrections. ``X=``, ``M=``,
``idata=`` (or ``T=``) transform the responses first, as R's within-subject
designs do; ``mauchly_test`` tests sphericity of the transformed covariance.

``manova(data, "cbind(y1, y2) ~ g")`` gives ``summary.manova``'s table
directly, as ``aov`` gives its ANOVA table.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Data, Step, step
from stats3.core import _special, _warn
from stats3.formula import Formula, build_design, encode, parse_formula, variable_expr
from stats3.models import (
    LinearModel,
    Models,
    NA_ACTIONS,
    _contrasts,
    _fits,
    _Fit,
    _is_model,
    _key_text,
    _model_frames,
    _ModelStep,
    _ModelVerb,
    _qr,
    _table_text,
)

TESTS = ("Pillai", "Wilks", "Hotelling-Lawley", "Roy")
_TEST_COLUMNS = {"Pillai": "pillai", "Wilks": "wilks", "Hotelling-Lawley": "hl", "Roy": "roy"}
_TRANSFORM_ARGS = ("X", "M", "idata", "T", "Sigma")


def _responses(response: str | None, verb: str) -> list[str]:
    """The labels in ``cbind(...)``; R needs at least two for an mlm."""
    if response is None or not response.startswith("cbind("):
        raise ValueError(f"{verb}(): need multiple responses: 'cbind(y1, y2, ...) ~ ...'")
    inner = response[len("cbind("):-1]
    parts, depth, start = [], 0, 0
    for i, c in enumerate(inner):
        depth += c == "("
        depth -= c == ")"
        if c == "," and depth == 0:
            parts.append(inner[start:i].strip())
            start = i + 1
    parts.append(inner[start:].strip())
    if len(parts) < 2 or not all(parts):
        raise ValueError(f"{verb}(): need multiple responses: {response!r}")
    return parts


def _deparse(label: str) -> str:
    """A response label as R deparses it: ``bill_depth_mm/10``, ``a + b``."""
    text = " ".join(label.split())
    text = re.sub(r"\s*([/^:])\s*", r"\1", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text


# fits ---------------------------------------------------------------------------------


class MultivariateLinearModel(_Fit):
    """A fitted ``lm`` with a matrix response (R's ``mlm``)."""

    kind = "mlm"

    def __init__(self, formula: Formula, frame, data: Data, responses: list[str], weighted: bool,
                 verb: str = "lm"):
        design = build_design(formula, frame.variables, len(frame.weights))
        super().__init__(formula, design, frame, data, verb)
        if frame.response_levels is not None:
            raise ValueError(f"{verb}(): the responses must be numeric")
        columns = [frame.y]
        for label in responses[1:]:
            variable = frame.variables[label]
            if variable.is_factor:
                raise ValueError(f"{verb}(): the responses must be numeric; {label!r} is not")
            columns.append(variable.values)
        y = np.column_stack(columns)
        self.responses = [_deparse(r) for r in responses]
        self.weighted = weighted
        x, w, off = design.matrix, frame.weights, frame.offset
        sw = np.sqrt(w)
        xw, yw = x * sw[:, None], (y - off[:, None]) * sw[:, None]
        self.qr = _qr(xw, 1e-7, yw[:, 0])
        kept = self.qr.kept
        self.rank = len(kept)
        coef = np.full((x.shape[1], y.shape[1]), np.nan)
        if kept:
            from scipy.linalg import qr, solve_triangular

            q, r = qr(xw[:, kept], mode="economic")
            # Q'Y for the kept columns: R's effects, whose rows give the
            # sequential SSCP matrices term by term.
            self.effects = q.T @ yw
            coef[kept] = solve_triangular(r, self.effects)
        else:
            self.effects = np.zeros((0, y.shape[1]))
        self.coef = coef
        self.y, self.weights = y, w
        self.n = int(np.sum(w > 0))
        self.df_residual = self.n - self.rank
        self.fitted = x[:, kept] @ coef[kept] + off[:, None]
        self.residuals = y - self.fitted
        wres = self.residuals * sw[:, None]
        self.sscp = wres.T @ wres
        self.cov_unscaled = self.qr.cov_unscaled()
        # One univariate lm per response, for the coefficient tables.
        self.fits = [
            LinearModel(dataclasses.replace(formula, response=label),
                        dataclasses.replace(frame, y=y[:, j]), data, verb)
            for j, label in enumerate(responses)
        ]

    @property
    def coefficients(self) -> dict[str, dict[str, float]]:
        return {r: dict(zip(self.design.names, self.coef[:, j].tolist())) for j, r in enumerate(self.responses)}

    # -- broom ----------------------------------------------------------------------------

    def tidy(self, conf_int: bool = False, conf_level: float = 0.95, vcov: Any = None) -> pl.DataFrame:
        if vcov is not None:
            raise TypeError("tidy(): vcov= is not available for multivariate lm fits")
        frames = [
            fit._coef_table(conf_int, conf_level, False).select(pl.lit(name).alias("response"), pl.all())
            for name, fit in zip(self.responses, self.fits)
        ]
        return self._with_key(pl.concat(frames))

    def glance(self, **kwargs: Any) -> pl.DataFrame:
        raise TypeError("glance(): not available for multivariate lm fits (broom has no glance.mlm)")

    def augment(self, newdata: Any = None, **kwargs: Any) -> pl.DataFrame:
        raise TypeError("augment(): not available for multivariate lm fits (broom has no augment.mlm)")

    def vcov_table(self, vcov: Any = None) -> pl.DataFrame:
        """R's ``vcov.mlm``: the residual covariance (x) the unscaled
        covariance, named ``response:term``; aliased terms are missing."""
        if vcov is not None:
            raise TypeError("vcov(): vcov= is not available for multivariate lm fits")
        k = len(self.design.names)
        unscaled = np.full((k, k), np.nan)
        kept = self.qr.kept
        unscaled[np.ix_(kept, kept)] = self.cov_unscaled
        full = np.kron(self.sscp / self.df_residual, unscaled)
        names = [f"{r}:{t}" for r in self.responses for t in self.design.names]
        frame = pl.DataFrame({"term": names, **{n: full[:, i] for i, n in enumerate(names)}})
        return self._with_key(frame.with_columns(pl.col(n).fill_nan(None) for n in names))

    def predict(self, newdata: Any = None, *, se_fit: bool = False) -> Any:
        """R's ``predict.mlm``: one column of predictions per response."""
        if se_fit:
            raise ValueError("predict(): the 'se_fit' argument is not yet implemented for \"mlm\" objects")
        x, off, _, data = self._design_for(newdata)
        kept = self.qr.kept
        pred = x[:, kept] @ self.coef[kept] + off[:, None]
        out = pl.DataFrame({r: pred[:, j] for j, r in enumerate(self.responses)})
        if newdata is None:
            out = self._padded(out)
        return data.write(out)

    # -- multivariate tests ------------------------------------------------------------

    def _term_sscp(self) -> tuple[list[str], list[float], list[np.ndarray]]:
        """Sequential SSCP matrices by term, from the effects, as R sums them."""
        assign = [self.design.assign[j] for j in self.qr.kept]
        labels = ["(Intercept)", *self.design.term_labels]
        names, dfs, ss = [], [], []
        for a in dict.fromkeys(assign):
            rows = [i for i, b in enumerate(assign) if b == a]
            names.append(labels[a])
            dfs.append(float(len(rows)))
            ss.append(self.effects[rows].T @ self.effects[rows])
        return names, dfs, ss

    def summary_manova(self, test: str = "Pillai", intercept: bool = False) -> pl.DataFrame:
        """R's ``summary.manova`` table: one row per term and ``Residuals``."""
        test = _match_test(test, TESTS, "manova")
        names, dfs, ss = self._term_sscp()
        if not intercept and "(Intercept)" in names:
            i = names.index("(Intercept)")
            del names[i], dfs[i], ss[i]
        if self.df_residual <= 0:
            raise ValueError(f"{self.verb}(): no residual degrees of freedom for the tests")
        stats = _multivariate(self.sscp, ss, dfs, float(self.df_residual), test, len(self.responses))
        return self._with_key(_test_frame(names, dfs, float(self.df_residual), stats, test))

    def _ssd(self, verb: str) -> tuple[np.ndarray, float]:
        """R's ``SSD``: residual SSCP and df; unweighted fits only, as R."""
        if self.weighted:
            raise ValueError(f"{verb}(): 'mlm' objects with weights are not supported")
        return self.sscp, float(self.df_residual)

    def anova_table(self, test: str | None = None, *, X: Any = None, M: Any = None, idata: Any = None,
                    T: Any = None, Sigma: Any = None) -> pl.DataFrame:  # noqa: N803 - R's names
        """R's ``anova.mlm`` for one fit: every term, the intercept included."""
        test = _match_test(test or "Pillai", (*TESTS, "Spherical"), "anova")
        ssd, df_res = self._ssd("anova")
        p = ssd.shape[0]
        t = _transform(p, X, M, idata, T, "anova")
        sigma = _sigma(Sigma, p)
        if self.rank == 0:
            raise ValueError("anova(): the model has no coefficients")
        names, dfs, ss = self._term_sscp()
        if test == "Spherical":
            gg, hf, s = _sphericity(ssd, df_res, t, sigma)
            stats = [_spherical_row(h, d, df_res, t, sigma, s, gg, hf) for h, d in zip(ss, dfs)]
            return self._with_key(_spherical_frame(names, dfs, df_res, stats, gg, hf))
        e, hs = t @ ssd @ t.T, [t @ h @ t.T for h in ss]
        stats = _multivariate(e, hs, dfs, df_res, test, t.shape[0], verb="anova")
        return self._with_key(_test_frame(names, dfs, df_res, stats, test))

    def mauchly(self, X: Any = None, M: Any = None, idata: Any = None, T: Any = None,
                Sigma: Any = None) -> pl.DataFrame:  # noqa: N803
        """R's ``mauchly.test`` on the fit's SSD matrix."""
        ssd, n = self._ssd("mauchly_test")
        p = ssd.shape[0]
        t = _transform(p, X, M, idata, T, "mauchly_test")
        psi = t @ _sigma(Sigma, p) @ t.T
        u = np.linalg.solve(psi, t @ ssd @ t.T)
        pp = t.shape[0]
        sign, logdet = np.linalg.slogdet(u)
        log_w = (logdet if sign > 0 else math.nan) - pp * math.log(np.trace(u) / pp)
        rho = 1 - (2 * pp**2 + pp + 2) / (6 * pp * n)
        # R's mauchly.test.SSD has 3 * p (the untransformed dimension) here.
        w2 = (pp + 2) * (pp - 1) * (pp - 2) * (2 * pp**3 + 6 * pp**2 + 3 * p + 2) / (288 * (n * pp * rho) ** 2)
        z = -n * rho * log_w
        f = pp * (pp + 1) / 2 - 1
        special = _special()
        # With one contrast f is 0: R's pchisq on 0 df is a point mass at 0.
        pr1 = (1.0 if z <= 0 else 0.0) if f == 0 else special.chdtrc(f, z)
        pr2 = special.chdtrc(f + 4, z)
        return self._with_key(pl.DataFrame({
            "statistic": [math.exp(log_w)], "p_value": [float(pr1 + w2 * (pr2 - pr1))],
            "method": ["Mauchly's test of sphericity"],
        }))

    # -- printing -----------------------------------------------------------------------

    def __repr__(self) -> str:
        table = pl.DataFrame({"term": self.design.names, **{r: self.coef[:, j] for j, r in enumerate(self.responses)}})
        head = f"{self.verb}({self.formula.text})" + _key_text(self.key) + f"   n = {self.n}"
        return f"{head}\n{_table_text(table.with_columns(pl.col(pl.Float64).fill_nan(None)))}{self._aliased_note()}"


# the tests ----------------------------------------------------------------------------


def _match_test(test: Any, choices: tuple[str, ...], verb: str) -> str:
    """R's ``match.arg``: a choice or a unique start of one."""
    if isinstance(test, str):
        if test in choices:
            return test
        hits = [c for c in choices if c.startswith(test)] if test else []
        if len(hits) == 1:
            return hits[0]
    raise ValueError(f"{verb}(): test must be one of {', '.join(repr(c) for c in choices)}, got {test!r}")


def _pillai(eig, q, df_res):
    test = float(np.sum(eig / (1 + eig)))
    p = len(eig)
    s = min(p, q)
    n = 0.5 * (df_res - p - 1)
    m = 0.5 * (abs(p - q) - 1)
    tmp1, tmp2 = 2 * m + s + 1, 2 * n + s + 1
    return test, (tmp2 / tmp1 * test) / (s - test), s * tmp1, s * tmp2


def _wilks(eig, q, df_res):
    test = float(np.prod(1 / (1 + eig)))
    p = len(eig)
    tmp1 = df_res - 0.5 * (p - q + 1)
    tmp2 = (p * q - 2) / 4
    tmp3 = p**2 + q**2 - 5
    tmp3 = math.sqrt(((p * q) ** 2 - 4) / tmp3) if tmp3 > 0 else 1
    return test, ((test ** (-1 / tmp3) - 1) * (tmp1 * tmp3 - 2 * tmp2)) / p / q, p * q, tmp1 * tmp3 - 2 * tmp2


def _hotelling_lawley(eig, q, df_res):
    test = float(np.sum(eig))
    p = len(eig)
    m = 0.5 * (abs(p - q) - 1)
    n = 0.5 * (df_res - p - 1)
    s = min(p, q)
    tmp1, tmp2 = 2 * m + s + 1, 2 * (s * n + 1)
    return test, (tmp2 * test) / s / s / tmp1, s * tmp1, tmp2


def _roy(eig, q, df_res):
    p = len(eig)
    test = float(np.max(eig))
    tmp1 = max(p, q)
    tmp2 = df_res - tmp1 + q
    return test, (tmp2 * test) / tmp1, tmp1, tmp2


_STATISTICS = {"Pillai": _pillai, "Wilks": _wilks, "Hotelling-Lawley": _hotelling_lawley, "Roy": _roy}


def _scaled_rank_check(e: np.ndarray, hs: list[np.ndarray], verb: str, use_max: bool = False) -> np.ndarray:
    """R's rescaling of the residual SSCP before its QR, and the rank check:
    the scale factors (1 where a response is negligible)."""
    sc = np.sqrt(np.diag(e))
    extra = [np.abs(np.diag(h)) if use_max else np.diag(h) for h in hs]
    added = (np.max(extra, axis=0) if use_max else np.sum(extra, axis=0)) if extra else 0
    d = np.sqrt(sc**2 + added)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(sc / d < 1e-6, 1.0, 1 / sc)
    scaled = e * np.outer(scale, scale)
    rank = len(_qr(scaled, 1e-7, np.zeros(len(scaled))).kept)
    if rank < len(scaled):
        raise ValueError(f"{verb}(): residuals have rank {rank} < {len(scaled)}")
    return scale


def _eigenvalues(e: np.ndarray, h: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Eigenvalues of E^-1 H (as R's ``eigen(qr.coef(E, H))``), from the
    symmetric-definite problem on the rescaled matrices."""
    from scipy.linalg import eigh

    s = np.outer(scale, scale)
    return eigh(h * s, e * s, eigvals_only=True)[::-1]


def _multivariate(e, hs, dfs, df_res, test, p, verb="manova", use_max=False):
    """One (statistic, F, num df, den df, p) per hypothesis SSCP matrix."""
    scale = _scaled_rank_check(e, hs, verb, use_max)
    special = _special()
    rows = []
    for h, q in zip(hs, dfs):
        if q == 0:
            rows.append((None,) * 5)
            continue
        eig = _eigenvalues(e, h, scale)
        stat, f, num, den = _STATISTICS[test](eig, abs(q), df_res)
        ok = f >= 0 and num > 0 and den > 0
        rows.append((stat, f, num, den, float(special.fdtrc(num, den, f)) if ok else None))
    return rows


def _nan_none(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _test_frame(names, dfs, df_res, stats, test) -> pl.DataFrame:
    cols = list(zip(*stats)) if stats else [()] * 5
    column = _TEST_COLUMNS[test]
    return pl.DataFrame({
        "term": [*names, "Residuals"], "df": [*dfs, df_res],
        **{name: [_nan_none(v) for v in values] + [None]
           for name, values in zip((column, "statistic", "num_df", "den_df", "p_value"), cols)},
    }, schema_overrides={c: pl.Float64 for c in ("df", column, "statistic", "num_df", "den_df", "p_value")})


def _sphericity(ssd, df_res, t, sigma):
    """R's ``sphericity``: Greenhouse-Geisser and Huynh-Feldt epsilons and
    the pooled within variance."""
    pp = t.shape[0]
    u = np.linalg.solve(t @ sigma @ t.T, t @ ssd @ t.T)
    s = float(np.trace(u)) / pp / df_res
    # sum(lambda)^2 / sum(lambda^2), from traces rather than eigenvalues.
    gg = float(np.trace(u)) ** 2 / float(np.trace(u @ u)) / pp
    hf = ((df_res + 1) * pp * gg - 2) / (pp * (df_res - pp * gg))
    return gg, hf, s


def _spherical_row(h, q, df_res, t, sigma, s, gg, hf):
    pp = t.shape[0]
    psi = t @ sigma @ t.T
    s2 = float(np.trace(np.linalg.solve(psi, t @ h @ t.T))) / pp / q
    f, num, den = abs(s2 / s), abs(q * pp), abs(df_res * pp)
    special = _special()
    hf1 = min(1.0, hf)
    return (f, num, den, float(special.fdtrc(num, den, f)), float(special.fdtrc(num * gg, den * gg, f)),
            float(special.fdtrc(num * hf1, den * hf1, f)))


_SPHERICAL = ("statistic", "num_df", "den_df", "p_value", "gg_p_value", "hf_p_value")


def _spherical_frame(names, dfs, df_res, stats, gg, hf) -> pl.DataFrame:
    cols = list(zip(*stats))
    n = len(names) + 1
    return pl.DataFrame({
        "term": [*names, "Residuals"], "df": [*dfs, df_res],
        **{name: [*values, None] for name, values in zip(_SPHERICAL, cols)},
        "gg_epsilon": [gg] * n, "hf_epsilon": [hf] * n,
    }, schema_overrides={c: pl.Float64 for c in ("df", *_SPHERICAL)})


# within-subject transformations ---------------------------------------------------------


def _zapsmall(x: np.ndarray, digits: int = 12) -> np.ndarray:
    mx = float(np.max(np.abs(x))) if x.size else 0.0
    if mx <= 0:
        return x
    return np.round(x, max(0, digits - math.ceil(math.log10(mx))))


def _rank(x: np.ndarray) -> int:
    """R's ``Rank``: the rank of the limited-pivoting QR, tolerance 1e-7."""
    if x.shape[1] == 0:
        return 0
    return len(_qr(_zapsmall(x), 1e-7, np.zeros(x.shape[0])).kept)


def _proj(x: np.ndarray) -> np.ndarray:
    """R's ``Proj``: projection onto the columns' span."""
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], x.shape[0]))
    x = x[:, _qr(x, 1e-7, np.zeros(x.shape[0])).kept]
    return x @ np.linalg.solve(x.T @ x, x.T)


def _thin_row(x: np.ndarray) -> np.ndarray:
    """R's ``Thin.row``: the linearly independent rows, in order."""
    x = _zapsmall(x)
    return x[_qr(x.T, 1e-7, np.zeros(x.shape[1])).kept]


def _idata_matrix(spec: Any, idata: pl.DataFrame, p: int, role: str, verb: str) -> np.ndarray:
    """``X=``/``M=`` as a matrix: a formula on ``idata`` (``"~ 1"``,
    ``"~ part + dim"``) through R's model.matrix, or a p-row matrix."""
    if isinstance(spec, str):
        text = spec.strip()
        if not text.startswith("~"):
            raise ValueError(f"{verb}(): {role} must be a one-sided formula like '~ 1', got {spec!r}")
        formula = parse_formula(text, verb)
        exprs = {}
        factors = set()
        for label in formula.variables():
            expr, as_factor = variable_expr(label, verb)
            exprs[label] = expr
            if as_factor:
                factors.add(label)
        frame = idata.select(**exprs) if exprs else idata
        variables = {label: encode(frame, label, label in factors, None) for label in formula.variables()}
        return build_design(formula, variables, p).matrix
    matrix = np.asarray(spec, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != p:
        raise ValueError(f"{verb}(): {role} must have one row per response ({p})")
    return matrix


def _transform(p: int, X: Any, M: Any, idata: Any, T: Any, verb: str) -> np.ndarray:  # noqa: N803
    """R's default ``T = Thin.row(Proj(M) - Proj(X))``, or ``T`` as given."""
    if T is not None:
        t = np.atleast_2d(np.asarray(T, dtype=float))
        if t.shape[1] != p:
            raise ValueError(f"{verb}(): T must have one column per response ({p})")
        return t
    if idata is None:
        frame = pl.DataFrame({"index": np.arange(1, p + 1)})
    else:
        frame = Data.read(idata, None, verb)._lf.collect()
        if frame.height != p:
            raise ValueError(f"{verb}(): idata must have one row per response ({p}), not {frame.height}")
    m = np.eye(p) if M is None else _idata_matrix(M, frame, p, "M", verb)
    x = np.zeros((p, 0)) if X is None else _idata_matrix(X, frame, p, "X", verb)
    if _rank(np.hstack([m, x])) != _rank(m):
        raise ValueError(f"{verb}(): X does not define a subspace of M")
    t = _thin_row(_proj(m) - _proj(x))
    if t.shape[0] == 0:
        raise ValueError(f"{verb}(): X spans M; no contrasts are left to test")
    return t


def _sigma(sigma: Any, p: int) -> np.ndarray:
    if sigma is None:
        return np.eye(p)
    s = np.asarray(sigma, dtype=float)
    if s.shape != (p, p):
        raise ValueError(f"Sigma must be {p} x {p}")
    return s


# model comparison --------------------------------------------------------------------------


def _anova_mlmlist(fits: list[MultivariateLinearModel], test: str | None, X=None, M=None, idata=None,  # noqa: N803
                   T=None, Sigma=None) -> pl.DataFrame:
    """R's ``anova.mlmlist``: nested mlm fits, in the order given."""
    test = _match_test(test or "Pillai", (*TESTS, "Spherical"), "anova")
    first = fits[0].responses
    same = [f.responses == first for f in fits]
    if not all(same):
        dropped = [f.formula.response for f, s in zip(fits, same) if not s]
        _warn(f"models with response {dropped} removed because response differs from model 1")
        fits = [f for f, s in zip(fits, same) if s]
    if len({len(f.y) for f in fits}) > 1:
        raise ValueError("anova(): models were not all fitted to the same size of dataset")
    if len(fits) == 1:
        return fits[0].anova_table(test)
    ssds = [f._ssd("anova") for f in fits]
    p = ssds[0][0].shape[0]
    t = _transform(p, X, M, idata, T, "anova")
    sigma = _sigma(Sigma, p)
    pp = t.shape[0]
    resdf = [d for _, d in ssds]
    df = [None] + [resdf[i] - resdf[i - 1] for i in range(1, len(fits))]
    deltas = [ssds[i][0] - ssds[i - 1][0] for i in range(1, len(fits))]
    gen_var = [float(np.linalg.det(t @ (s / d) @ t.T)) ** (1 / pp) for s, d in ssds]
    big = int(np.argmin(resdf))
    table: dict[str, Any] = {"term": [f.formula.text for f in fits], "df_residual": resdf, "df": df,
                             "gen_var": gen_var}
    if test == "Spherical":
        gg, hf, s = _sphericity(ssds[big][0], resdf[big], t, sigma)
        rows = [(None,) * 6] + [_spherical_row(delta, q, resdf[big], t, sigma, s, gg, hf)
                                for delta, q in zip(deltas, df[1:])]
        table.update({name: list(values) for name, values in zip(_SPHERICAL, zip(*rows))})
        table.update(gg_epsilon=[gg] * len(fits), hf_epsilon=[hf] * len(fits))
        types = _SPHERICAL
    else:
        signed = [(1 if q > 0 else -1 if q < 0 else 0) * (t @ delta @ t.T) for delta, q in zip(deltas, df[1:])]
        e = t @ ssds[big][0] @ t.T
        rows = [(None,) * 5] + _multivariate(e, signed, [abs(q) for q in df[1:]], resdf[big], test, pp,
                                             verb="anova", use_max=True)
        column = _TEST_COLUMNS[test]
        table.update({name: [_nan_none(v) for v in values]
                      for name, values in zip((column, "statistic", "num_df", "den_df", "p_value"), zip(*rows))})
        types = (column, "statistic", "num_df", "den_df", "p_value")
    return pl.DataFrame(table, schema_overrides={c: pl.Float64 for c in ("df_residual", "df", *types)})


def _anova_hook(models: tuple, test: Any, kind: int, kwargs: dict[str, Any]) -> Any:
    """``anova()`` for mlm fits, or None to leave the call to ``anova``.
    The transformation arguments (``X=``, ``M=``, ``idata=``, ``T=``,
    ``Sigma=``) are for mlm fits only."""
    from stats3.models import anova

    unknown = set(kwargs) - set(_TRANSFORM_ARGS)
    if unknown:
        raise TypeError(f"anova(): unexpected arguments {sorted(unknown)}")
    if not models:
        return _ModelVerb(lambda m: anova(m, test=test, type=kind, **kwargs), "anova") if kwargs else None
    multivariate = [isinstance(f, MultivariateLinearModel) for m in models if _is_model(m) for f in _fits(m)]
    if not any(multivariate):
        if kwargs:
            raise TypeError(f"anova(): {', '.join(sorted(kwargs))} are for multivariate lm fits")
        return None
    if not all(multivariate) or not all(_is_model(m) for m in models):
        raise TypeError("anova(): compare multivariate lm fits with multivariate lm fits")
    if kind != 1:
        raise ValueError("anova(): type 2 and 3 tables are not available for multivariate lm fits")
    if len(models) == 1:
        model = models[0]
        frames = [f.anova_table(test, **kwargs) for f in _fits(model)]
        return model._stack(frames) if isinstance(model, Models) else model._write(frames[0])
    if any(isinstance(m, Models) for m in models):
        raise TypeError("anova(): compare ungrouped multivariate lm fits")
    return models[0]._write(_anova_mlmlist(list(models), test, **kwargs))


# verbs ---------------------------------------------------------------------------------------


def _mlm_frames(formula: str, weights: Any, offset: Any, contrasts: Any, na_action: str, data: Data, verb: str):
    """The parsed formula, response labels and model frames for an mlm.
    The responses after the first ride along as extra variables, so rows
    missing any response are dropped with the rest (R's na.omit)."""
    if na_action not in NA_ACTIONS:
        raise ValueError(f"{verb}(): na_action must be one of {list(NA_ACTIONS)}, got {na_action!r}")
    parsed = parse_formula(formula, verb, data.columns)
    responses = _responses(parsed.response, verb)
    parsed.contrasts = _contrasts(contrasts, parsed, verb)
    parsed.offset_arg = offset
    variables = parsed.variables()
    extra = [(label,) for label in dict.fromkeys(responses[1:]) if label not in variables]
    for g in data._groups:
        if g in responses:
            raise ValueError(f"{verb}(): {g!r} is a grouping column; it cannot be in the formula")
    carrier = dataclasses.replace(parsed, response=responses[0], terms=[*parsed.terms, *extra])
    return parsed, responses, _model_frames(data, carrier, weights, offset, verb, na_action)


def _mlm(formula: str, weights: Any, offset: Any, contrasts: Any, na_action: str, verb: str = "lm") -> Step:
    """``lm`` with a ``cbind(...)`` response: a multivariate fit."""
    def _apply(data: Data) -> Any:
        parsed, responses, frames = _mlm_frames(formula, weights, offset, contrasts, na_action, data, verb)
        fits = [MultivariateLinearModel(parsed, frame, data, responses, weights is not None, verb)
                for frame in frames]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, verb)


@step
def manova(formula: str, *, test: str = "Pillai", intercept: bool = False, weights: Any = None,
           contrasts: Any = None) -> Step:
    """Multivariate analysis of variance, as R's ``summary(manova(...))``.

    ``manova(iris, "cbind(sepal_length, petal_length) ~ species")`` gives
    one row per term (sequential SSCP, as ``summary.manova``) and one for
    ``Residuals``: ``df``, the test statistic (a column named ``pillai``,
    ``wilks``, ``hl`` or ``roy``, as broom names them), its approximate F
    (``statistic``), ``num_df``, ``den_df`` and ``p_value``. ``test`` is
    ``"Pillai"`` (default), ``"Wilks"``, ``"Hotelling-Lawley"`` or
    ``"Roy"``; ``intercept=True`` adds the intercept's row. ``by=`` gives a
    table per group. For the fit itself, use ``lm`` with the same formula
    and ``anova(fit)`` (R's ``anova.mlm``).
    """
    test = _match_test(test, TESTS, "manova")

    def _apply(data: Data) -> pl.DataFrame:
        parsed, responses, frames = _mlm_frames(formula, weights, None, contrasts, "omit", data, "manova")
        tables = [MultivariateLinearModel(parsed, frame, data, responses, weights is not None, "manova")
                  .summary_manova(test, intercept) for frame in frames]
        return pl.concat(tables, how="diagonal_relaxed")

    return Step(_apply, "manova")


def mauchly_test(x: Any = None, *, X: Any = None, M: Any = None, idata: Any = None, T: Any = None,  # noqa: N803
                 Sigma: Any = None) -> Any:
    """Mauchly's test of sphericity for a multivariate ``lm``, as R's
    ``mauchly.test``: ``statistic`` (W), ``p_value`` and ``method``.

    The test is of the covariance of ``T @ y``, with ``T`` the contrasts in
    the span of ``M`` orthogonal to ``X``: ``X="~ 1"`` tests the
    differences between responses (repeated measures); ``M=`` and ``X=``
    may be formulas on ``idata`` (one row per response) or matrices.
    ``fit >> mauchly_test(X="~ 1")`` works too; a grouped fit gives a row
    per group.
    """
    def run(model):
        if not _is_model(model) or not all(isinstance(f, MultivariateLinearModel) for f in _fits(model)):
            raise TypeError("mauchly_test(): expected a multivariate fit, lm(data, 'cbind(y1, y2, ...) ~ ...')")
        frames = [f.mauchly(X, M, idata, T, Sigma) for f in _fits(model)]
        return model._stack(frames) if isinstance(model, Models) else model._write(frames[0])

    if x is None:
        return _ModelVerb(run, "mauchly_test")
    return run(x)


__all__ = ["MultivariateLinearModel", "manova", "mauchly_test"]
