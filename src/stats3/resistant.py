"""Robust M-estimation (MASS::rlm) and quantile regression (quantreg::rq)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from stats3._data import step
from stats3._lqs import lqs
from stats3._rqinference import boot_xy, rank_intervals
from stats3.core import _check_conf_level, _special, _warn
from stats3.formula import build_design
from stats3.models import _Fit, _model, _qr, _table_text


class _RegressionFit(_Fit):
    """Shared formula, table and prediction behavior without OLS inference."""

    def __init__(self, formula, frame, data, verb):
        design = build_design(formula, frame.variables, len(frame.weights))
        super().__init__(formula, design, frame, data, verb)
        if frame.y.ndim != 1 or frame.response_levels is not None:
            raise ValueError(f"{verb}(): the response must be one numeric column")
        self.x, self.y, self.weights = design.matrix, frame.y, frame.weights
        self.prior_weights = self.weights
        self.n, self.rank = self.x.shape
        self.qr = _qr(self.x, 1e-7, np.zeros(self.n))
        if len(self.qr.kept) != self.rank:
            raise ValueError(f"{verb}(): singular design matrix")
        if self.n <= self.rank:
            raise ValueError(f"{verb}(): more rows than coefficients are needed for inference")
        self.df_residual = self.n - self.rank
        self.scale = 1.0

    def _check_vcov(self, vcov):
        if vcov is not None:
            raise ValueError(f"{self.verb}(): this fit supplies its own covariance; vcov= is for lm/glm")

    def tidy(self, conf_int=False, conf_level=.95, vcov=None):
        self._check_vcov(vcov)
        return self._with_key(self._coef_table(conf_int, conf_level, False))

    def vcov_table(self, vcov=None):
        self._check_vcov(vcov)
        return super().vcov_table()

    def predict(self, newdata=None, *, interval="none", level=.95, se_fit=False):
        return self._predict(newdata, interval, level, se_fit, False)

    def _predict(self, newdata, interval, level, se_fit, augment):
        if interval not in ("none", "confidence"):
            raise ValueError(f"{self.verb}(): interval must be 'none' or 'confidence'")
        _check_conf_level(level)
        x, offset, rows, data = self._design_for(newdata)
        fitted = x @ self.coef + offset
        cols = {".fitted" if augment else "fit": fitted}
        if interval != "none" or se_fit:
            se = np.sqrt(np.einsum("ij,jk,ik->i", x, self._prediction_cov(), x))
            if interval != "none":
                width = self._quantile((1 + level) / 2) * se
                cols[".lower" if augment else "lwr"] = fitted - width
                cols[".upper" if augment else "upr"] = fitted + width
            if se_fit:
                cols[".se_fit" if augment else "se_fit"] = se
        out = pl.DataFrame(cols)
        if newdata is None:
            out, rows = self._padded(out), self._rows()
        return data.write(pl.concat([rows, out], how="horizontal") if augment else out)

    def augment(self, newdata=None, *, interval="none", conf_level=.95, se_fit=False):
        if newdata is not None:
            return self._predict(newdata, interval, conf_level, se_fit, True)
        cols = {".fitted": self.fitted, ".resid": self.residuals}
        return pl.concat([self._rows(), self._padded(pl.DataFrame(cols))], how="horizontal")

    def anova_table(self, test=None):
        raise TypeError(f"anova(): single-model tables are not available for {self.verb} fits")

    def anova_type(self, kind, test=None):
        raise TypeError(f"anova(): type II/III tables are not available for {self.verb} fits")

    def __repr__(self):
        return f"{self.verb}({self.formula.text})\n{_table_text(self.tidy())}"


def _psi(u, name, tuning, derivative=False):
    a = np.abs(u)
    if name == "huber":
        if derivative:
            return np.where(np.isnan(u), np.nan, (a <= tuning).astype(float))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.minimum(1, tuning / a)
    if name == "bisquare":
        t = (u / tuning)**2
        return np.where(np.isnan(t), np.nan, np.where(t < 1, (1 - t) * (1 - 5 * t), 0)) if derivative else (1 - np.minimum(1, a / tuning)**2)**2
    low, middle, high = tuning
    v = np.minimum(a + 1e-50, high)
    if derivative:
        return np.where(np.isnan(u), np.nan, np.where(a <= high, np.where(v <= low, 1, np.where(v <= middle, 0, -low / (high - middle))), 0))
    return np.where(v <= low, v, np.where(v <= middle, low, low * (high - v) / (high - middle))) / v


def _weighted_mad(resid, weights):
    order = np.argsort(np.abs(resid), kind="stable")
    values, w = np.abs(resid)[order], weights[order]
    cumulative = np.cumsum(w) / w.sum()
    i = int(np.sum(cumulative < .5))
    return (values[i] if cumulative[i] > .5 else (values[i] + values[i + 1]) / 2) / .6745


class RobustLinearModel(_RegressionFit):
    """M-estimation with MASS's IRLS stopping rule and corrected covariance."""

    def __init__(self, formula, frame, data, psi, tuning, scale_est, k2, wt_method, cov_method, maxit, acc,
                 method="M", init="ls", nsamp=None, seed=None):
        super().__init__(formula, frame, data, "rlm")
        self.psi, self.tuning, self.wt_method, self.cov_method, self.method = psi, tuning, wt_method, cov_method, method
        case = wt_method == "case"
        sw = np.ones(self.n) if case else np.sqrt(self.weights)
        x, y = self.x * sw[:, None], (self.y - frame.offset) * sw
        if method == "MM":
            # rlm's S-estimate start; its scale stays fixed in the bisquare M-steps.
            coef, resid, scale = lqs(x, y, "S", "best" if nsamp is None else nsamp, seed)
            scale_est = "MM"
        elif isinstance(init, str) and init == "lts":
            coef, resid, _ = lqs(x, y, "lts", 200 if nsamp is None else nsamp, seed)
        elif isinstance(init, str):
            w = self.weights.copy() if case else np.ones(self.n)
            fit = _qr(x * np.sqrt(w)[:, None], 1e-7, y * np.sqrt(w))
            if len(fit.kept) != self.rank:
                raise ValueError("rlm(): singular weighted design matrix")
            coef = fit.coefficients()
            resid = y - x @ coef
        else:
            coef = np.asarray(init, dtype=float)
            if coef.shape != (self.rank,):
                raise ValueError(f"rlm(): init must have {self.rank} coefficients")
            resid = y - x @ coef
        if scale_est != "MM":
            scale = _weighted_mad(resid, self.weights) if case else float(np.median(np.abs(resid)) * 1.4826)
        rdf = (self.weights.sum() if case else self.n) - self.rank
        normal = _special().ndtr(k2)
        gamma = (2 * normal - 1) + k2**2 * (2 - 2 * normal) - 2 * k2 * math.exp(-k2**2 / 2) / math.sqrt(2 * math.pi)
        self.converged, self.iterations = False, 0
        for _ in range(maxit):
            old = resid.copy()
            if scale_est != "MM":
                if scale_est == "MAD":
                    scale = _weighted_mad(resid, self.weights) if case else float(np.median(np.abs(resid)) / .6745)
                else:
                    scale = math.sqrt(float(np.sum((self.weights if case else 1) * np.minimum(resid**2, (k2 * scale)**2))) / (rdf * gamma))
                if scale == 0:
                    self.converged = True
                    break
            w = _psi(resid / scale, psi, tuning) * (self.weights if case else 1)
            fit = _qr(x * np.sqrt(w)[:, None], 1e-7, y * np.sqrt(w))
            if len(fit.kept) != self.rank:
                raise ValueError("rlm(): robust weights produced a singular design matrix")
            coef = fit.coefficients()
            resid = y - x @ coef
            self.iterations += 1
            if math.sqrt(float(np.sum((old - resid)**2)) / max(1e-20, float(np.sum(old**2)))) <= acc:
                self.converged = True
                break
        if not self.converged:
            _warn(f"rlm failed to converge in {maxit} steps")
        self.coef, self.sigma, self.robust_weights, self.wresid = coef, scale, w, resid
        self.fitted = self.x @ coef + frame.offset
        self.residuals = self.y - self.fitted
        self.df_residual = float("nan") # MASS deliberately leaves the model's residual df undefined.
        self._cov = None

    @property
    def cov_unscaled(self):
        if self._cov is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                w = _psi(self.wresid / self.sigma, self.psi, self.tuning)
                prime = _psi(self.wresid / self.sigma, self.psi, self.tuning, True)
                if self.wt_method == "case":
                    nn = self.weights.sum()
                    mean = float(self.weights @ prime / nn)
                    variance = float((self.weights @ prime**2 - (self.weights @ prime)**2 / nn) / (nn - 1))
                    S = float(self.weights @ (self.wresid * w)**2 / (nn - self.rank))
                else:
                    nn = self.n
                    mean, variance = float(prime.mean()), float(prime.var(ddof=1))
                    S = float(np.sum((self.wresid * w)**2) / (self.n - self.rank))
                kappa = 1 + self.rank * variance / (nn * mean**2)
                stddev = np.sqrt(S) * kappa / mean
                x = self.x * np.sqrt(self.weights)[:, None]
                if self.cov_method == "XtWX":
                    average = self.weights @ w / self.weights.sum()
                    x = x * np.sqrt(w / average)[:, None]
                self._cov = _qr(x, 1e-7, np.zeros(self.n)).cov_unscaled() * stddev**2
        return self._cov

    def _p_values(self, stat):
        return np.full(len(stat), np.nan) # summary.rlm supplies t statistics, not p-values.

    def _quantile(self, p):
        return float(_special().ndtri(p))

    def _prediction_cov(self):
        x = self.x * np.sqrt(self.weights)[:, None]
        return _qr(x, 1e-7, np.zeros(self.n)).cov_unscaled() * self.sigma**2

    def glance(self):
        return self._with_key(pl.DataFrame({"sigma": [self.sigma], "df_residual": pl.Series([None], dtype=pl.Float64),
            "nobs": [int(np.sum(self.weights > 0))], "converged": [self.converged], "iterations": [self.iterations]}))

    def __repr__(self):
        return f"rlm({self.formula.text}, psi={self.psi!r})\n{_table_text(self.tidy())}\nrobust scale {self.sigma:.5g} · converged {self.converged}"


@step
def rlm(formula: str, *, psi="huber", tuning=None, scale_est="MAD", k2=1.345, method="M", init="ls",
        wt_method="inv.var", cov_method="XtX", maxit=20, acc=1e-4, lqs_control=None, seed=None,
        weights=None, offset=None, contrasts=None, na_action="omit"):
    """MASS's rlm: M-estimation (Huber, bisquare or Hampel psi and IRLS) or MM.

    ``scale_est`` is MAD or Huber's proposal 2 (``"Huber"`` / ``"proposal 2"``).
    ``init`` starts the M iterations from least squares (``"ls"``), from
    ``lqs``'s least trimmed squares (``"lts"``, 200 random subsets) or from
    given coefficients. ``method="MM"`` starts from lqs's S-estimate and runs
    bisquare M-steps at its fixed scale; as in MASS, ``psi`` and
    ``scale_est`` are then ignored and ``tuning`` is the bisquare constant
    (default 4.685, at least 1.548). ``lqs_control={"nsamp": ...}`` sets the
    number of subsets (``"best"``: all of them below 5000, else random ones;
    ``"exact"``; or a count). Random subsets come from R's generator
    started as ``set.seed(seed)``, so a seed reproduces R's fit.
    ``wt_method`` is ``"inv.var"`` or ``"case"``. ``cov_method`` selects
    summary.rlm's XtX or XtWX covariance. The coefficient table leaves
    p_value missing, as summary.rlm supplies none.
    """
    if method not in ("M", "MM"):
        raise ValueError("rlm(): method must be 'M' or 'MM'")
    if method == "MM":
        psi, values = "bisquare", 4.685
        if tuning is not None:
            if np.ndim(tuning) or not math.isfinite(tuning):
                raise ValueError("rlm(): tuning for method='MM' must be one number")
            if tuning > 1.548:
                values = float(tuning)
            else:
                _warn("rlm(): 'c' must be at least 1.548 and has been ignored")
        tuning = values
    elif psi not in ("huber", "bisquare", "hampel"):
        raise ValueError("rlm(): psi must be 'huber', 'bisquare' or 'hampel'")
    tuning = {"huber": 1.345, "bisquare": 4.685, "hampel": (2, 4, 8)}[psi] if tuning is None else tuning
    values = np.asarray(tuning, dtype=float)
    if psi == "hampel":
        valid = values.shape == (3,) and np.all(np.isfinite(values)) and 0 < values[0] < values[1] < values[2]
    else:
        valid = values.ndim == 0 and np.isfinite(values) and values > 0
    if not valid:
        raise ValueError("rlm(): tuning must be positive (three increasing values for Hampel)")
    tuning = tuple(values.tolist()) if psi == "hampel" else float(values)
    if scale_est not in ("MAD", "Huber", "proposal 2") or wt_method not in ("inv.var", "case") or cov_method not in ("XtX", "XtWX"):
        raise ValueError("rlm(): invalid scale_est, wt_method or cov_method")
    if maxit < 1 or not math.isfinite(acc) or acc <= 0 or not math.isfinite(k2) or k2 <= 0:
        raise ValueError("rlm(): maxit, acc and k2 must be positive")
    if isinstance(init, str) and init not in ("ls", "lts"):
        raise ValueError("rlm(): 'init' method is unknown")
    control = dict(lqs_control or {})
    if set(control) - {"nsamp"}:
        raise ValueError("rlm(): lqs_control supports 'nsamp'")
    nsamp = None if lqs_control is None else control.get("nsamp", "best")
    if nsamp is not None and nsamp not in ("best", "exact") and (isinstance(nsamp, (str, bool)) or not nsamp >= 1):
        raise ValueError("rlm(): lqs_control nsamp must be 'best', 'exact' or a positive count")
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31)) # as R's unseeded stream
    return _model("rlm", lambda f, fr, d: RobustLinearModel(f, fr, d, psi, tuning, scale_est, k2, wt_method, cov_method,
                                                            maxit, acc, method, init, nsamp, seed),
                  formula, weights, offset, contrasts, na_action)


def _rq_fit(x, y, tau):
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, eye, hstack

    n, p = x.shape
    constraints = hstack((csr_matrix(x), eye(n), -eye(n)), format="csc")
    objective = np.r_[np.zeros(p), np.full(n, tau), np.full(n, 1 - tau)]
    fit = linprog(objective, A_eq=constraints, b_eq=y, bounds=[(None, None)] * p + [(0, None)] * (2 * n),
                  method="highs", options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9})
    if not fit.success:
        raise ValueError(f"rq(): linear program failed: {fit.message}")
    return fit.x[:p]


def _bandwidth(tau, n):
    z = float(_special().ndtri(tau))
    density = math.exp(-z**2 / 2) / math.sqrt(2 * math.pi)
    return n**(-1 / 3) * float(_special().ndtri(.975))**(2 / 3) * (1.5 * density**2 / (2 * z**2 + 1))**(1 / 3)


class QuantileModel(_RegressionFit):
    """Pinball-loss LP fit with quantreg's rank, nid, iid or bootstrap inference."""

    def __init__(self, formula, frame, data, tau, se, iid, replicates, seed):
        super().__init__(formula, frame, data, "rq")
        self.tau, self.iid, self.replicates, self.seed = tau, iid, replicates, seed
        # summary.rq's default: rank inversion below 1001 rows, else nid.
        self.se_method = se if se is not None else ("rank" if self.n < 1001 else "nid")
        # quantreg fits the response less its offsets; fitted values add them back.
        self._wx, self._wy = self.x * self.weights[:, None], (self.y - frame.offset) * self.weights
        if len(_qr(self._wx, 1e-7, np.zeros(self.n)).kept) != self.rank:
            raise ValueError("rq(): singular weighted design matrix")
        self.coef = _rq_fit(self._wx, self._wy, tau)
        self.fitted = self.x @ self.coef + frame.offset
        self.residuals = self.y - self.fitted
        loss = self.residuals * (tau - (self.residuals < 0))
        self.rho, self.objective = float(loss.sum()), float(self.weights @ loss)
        self._cov, self._boot, self._ranks = None, None, {}

    @property
    def boot_coefficients(self):
        """The bootstrap replicates (``se="boot"``), one row per replication."""
        if self.se_method != "boot":
            return None
        if self._boot is None:
            self._boot = boot_xy(self._wx, self._wy, self.tau, self.replicates, self.seed, _rq_fit)
        return self._boot

    @property
    def cov_unscaled(self):
        # Rank inversion gives intervals only; like summary.rq(covariance=TRUE),
        # its covariance (for vcov, predictions and marginal means) is nid's.
        if self._cov is None:
            x, y, tau = self._wx, self._wy, self.tau
            eps = np.finfo(float).eps**.5
            if self.se_method == "boot":
                self._cov = np.atleast_2d(np.cov(self.boot_coefficients, rowvar=False))
            elif self.se_method in ("nid", "rank"):
                h = _bandwidth(tau, self.n)
                while tau - h < 0 or tau + h > 1:
                    h /= 2
                delta = x @ (_rq_fit(x, y, tau + h) - _rq_fit(x, y, tau - h))
                if np.any(delta <= 0):
                    _warn(f"rq(): {np.sum(delta <= 0)} non-positive density estimates")
                density = np.maximum(0, 2 * h / (delta - eps))
                qr = _qr(x * np.sqrt(density)[:, None], 1e-7, np.zeros(self.n))
                if len(qr.kept) != self.rank:
                    raise ValueError("rq(): singular density-weighted design for nid inference")
                inv = qr.cov_unscaled()
                self._cov = tau * (1 - tau) * inv @ (x.T @ x) @ inv
            else:
                resid = self.residuals * self.weights
                zero = int(np.sum(np.abs(resid) < eps))
                h = max(self.rank + 1, math.ceil(self.n * _bandwidth(tau, self.n)))
                ir = np.arange(zero + 1, h + zero + 2)
                if ir[-1] > self.n:
                    raise ValueError("rq(): too few residuals for iid sparsity estimation")
                ordered = np.sort(resid[np.argsort(np.abs(resid), kind="stable")][ir - 1])
                sparsity = _rq_fit(np.column_stack((np.ones(len(ir)), ir / self.df_residual)), ordered, .5)[1]
                qr = _qr(x, 1e-7, np.zeros(self.n))
                if len(qr.kept) != self.rank:
                    raise ValueError("rq(): singular weighted design for iid inference")
                self._cov = sparsity**2 * qr.cov_unscaled() * tau * (1 - tau)
        return self._cov

    def rank_intervals(self, conf_level=.95):
        """rq.fit.br's rank-inversion bounds at ``conf_level``, interpolated
        between simplex vertices; unbounded sides are infinite."""
        _check_conf_level(conf_level)
        if conf_level not in self._ranks:
            self._ranks[conf_level] = rank_intervals(self._wx, self._wy, self.tau, self.coef, conf_level, self.iid, _rq_fit)
        return self._ranks[conf_level]

    def tidy(self, conf_int=False, conf_level=.95, vcov=None):
        if self.se_method != "rank":
            return super().tidy(conf_int, conf_level, vcov)
        self._check_vcov(vcov)
        # summary.rq's rank table has estimates and intervals, no standard errors or tests.
        missing = np.full(self.rank, np.nan)
        table = {"term": self.design.names, "estimate": self.coef, "std_error": missing, "statistic": missing, "p_value": missing}
        if conf_int:
            table["conf_low"], table["conf_high"] = self.rank_intervals(conf_level)
        frame = pl.DataFrame(table)
        return self._with_key(frame.with_columns(pl.col(c).fill_nan(None) for c in frame.columns if c != "term"))

    def _prediction_cov(self):
        return self.cov_unscaled

    def _p_values(self, stat):
        return 2 * _special().stdtr(self.df_residual, -np.abs(stat))

    def _quantile(self, p):
        return float(_special().stdtrit(self.df_residual, p))

    def glance(self):
        # quantreg's logLik.rq uses unweighted rho, even for weighted fits.
        with np.errstate(divide="ignore"):
            ll = float(self.n * (math.log(self.tau * (1 - self.tau)) - 1 - np.log(self.rho / self.n)))
        return self._with_key(pl.DataFrame({"tau": [self.tau], "rho": [self.rho], "logLik": [ll],
            "AIC": [-2 * ll + 2 * self.rank], "BIC": pl.Series([None], dtype=pl.Float64),
            "df_residual": [float(self.df_residual)], "nobs": [self.n]}))

    def __repr__(self):
        return f"rq({self.formula.text}, tau={self.tau:g}, se={self.se_method!r})\n{_table_text(self.tidy(conf_int=self.se_method == 'rank'))}"


@step
def rq(formula: str, *, tau=.5, se=None, iid=True, r=200, seed=None, weights=None, offset=None,
       contrasts=None, na_action="omit"):
    """A conditional quantile, minimizing weighted pinball loss with SciPy's LP.

    ``tau`` is one quantile strictly between zero and one. Inference follows
    summary.rq: ``se="rank"`` inverts rank tests for intervals (quantreg's
    default below 1001 rows; ``iid=False`` for its nid rank variant),
    ``"nid"`` (default from 1001 rows) and ``"iid"`` give asymptotic standard
    errors, and ``"boot"`` the xy-pair bootstrap with ``r`` replications,
    drawn from R's generator started as ``set.seed(seed)``. Rank fits use
    nid covariance for ``vcov``, predictions and marginal means, as quantreg
    does. Offsets are subtracted from the response. Nonunique LP optima may
    differ from R's.
    """
    if not np.isscalar(tau) or not math.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("rq(): tau must be one number strictly between zero and one")
    if se not in (None, "rank", "nid", "iid", "boot"):
        raise ValueError("rq(): se must be 'rank', 'nid', 'iid' or 'boot'")
    if not isinstance(iid, bool):
        raise ValueError("rq(): iid must be True or False")
    if isinstance(r, bool) or not isinstance(r, (int, np.integer)) or r < 2:
        raise ValueError("rq(): r must be a whole number of bootstrap replications, at least 2")
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31)) # as R's unseeded stream
    return _model("rq", lambda f, fr, d: QuantileModel(f, fr, d, float(tau), se, iid, int(r), seed),
                  formula, weights, offset, contrasts, na_action)


def _anova_rq(fits, test):
    if test not in (None, "Wald"):
        raise ValueError("anova(): rq comparisons support test='Wald'")
    fits = sorted(fits, key=lambda f: -f.rank)
    big, *others = fits
    rows = []
    for small in others:
        if small.tau != big.tau or not np.array_equal(small.y, big.y) or not np.array_equal(small.weights, big.weights):
            raise ValueError("anova(): rq models must use the same tau, response rows and weights")
        if not set(small.design.names) < set(big.design.names):
            raise ValueError("anova(): rq models must be nested with different numbers of coefficients")
        idx = [i for i, name in enumerate(big.design.names) if name not in small.design.names]
        b, cov = big.coef[idx], big.cov_unscaled[np.ix_(idx, idx)]
        stat = float(b @ np.linalg.solve(cov, b)) / len(idx)
        rows.append({"term": small.formula.text, "df": float(len(idx)), "df_residual": float(big.df_residual),
                     "statistic": stat, "p_value": float(_special().fdtrc(len(idx), big.df_residual, stat))})
    return pl.DataFrame(rows)
