"""Cumulative-link ordinal regression, following MASS::polr conventions."""

import numpy as np
import polars as pl

from stats3._data import step
from stats3.core import _check_conf_level, _warn
from stats3.formula import build_design
from stats3.models import _Fit, _model, _qr, _table_text


def _distribution(method):
    from scipy.stats import logistic, norm, gumbel_r, gumbel_l

    return {"logistic": logistic, "probit": norm, "loglog": gumbel_r, "cloglog": gumbel_l}[method]


def _probabilities(eta, thresholds, cdf):
    cumulative = cdf(np.asarray(thresholds) - np.asarray(eta)[:, None])
    endpoints = np.column_stack((np.zeros(len(eta)), cumulative, np.ones(len(eta))))
    return np.diff(endpoints, axis=1)


def _negative_log_likelihood(eta, thresholds, y, weights, cdf):
    """Weighted ordinal likelihood, with zero-based response categories."""
    used = np.asarray(weights) > 0
    probabilities = _probabilities(np.asarray(eta)[used], thresholds, cdf)
    observed = probabilities[np.arange(used.sum()), np.asarray(y)[used]]
    with np.errstate(divide="ignore"):
        return -float(np.asarray(weights)[used] @ np.log(observed))


def _thresholds(theta):
    return np.cumsum(np.r_[theta[0], np.exp(theta[1:])])


def _objective(parameters, x, y, weights, offset, cdf):
    p = x.shape[1]
    eta = x @ parameters[:p] + offset
    return _negative_log_likelihood(eta, _thresholds(parameters[p:]), y, weights, cdf)


def _score(parameters, x, y, weights, offset, cdf, pdf):
    used = weights > 0
    x, y, weights, offset = x[used], y[used], weights[used], offset[used]
    p = x.shape[1]
    theta = parameters[p:]
    bounds = np.r_[-np.inf, _thresholds(theta), np.inf]
    eta = x @ parameters[:p] + offset
    upper, lower = bounds[y + 1] - eta, bounds[y] - eta
    probability = cdf(upper) - cdf(lower)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = weights / probability
        high, low = np.zeros_like(upper), np.zeros_like(lower)
        finite_high, finite_low = np.isfinite(upper), np.isfinite(lower)
        high[finite_high], low[finite_low] = pdf(upper[finite_high]), pdf(lower[finite_low])
        beta = x.T @ (ratio * (high - low))
        cuts = np.zeros(len(theta))
        top, bottom = y < len(theta), y > 0
        np.add.at(cuts, y[top], -ratio[top] * high[top])
        np.add.at(cuts, y[bottom] - 1, ratio[bottom] * low[bottom])
    jacobian = np.tril(np.ones((len(theta), len(theta)))) * np.r_[1, np.exp(theta[1:])]
    return np.r_[beta, jacobian.T @ cuts]


def _optimize(start, x, y, weights, offset, cdf, pdf, maxit, epsilon):
    from scipy.optimize import minimize

    return minimize(_objective, np.asarray(start, dtype=float),
                    args=(x, y, weights, offset, cdf), method="BFGS",
                    jac=lambda par, *args: _score(par, x, y, weights, offset, cdf, pdf),
                    options={"maxiter": maxit, "gtol": epsilon})


def _covariance(parameters, x, y, weights, offset, cdf, pdf):
    # R's optimHess uses central differences of the analytic score, step .001.
    steps = np.eye(len(parameters)) * .001
    information = np.column_stack([(_score(parameters + delta, x, y, weights, offset, cdf, pdf)
        - _score(parameters - delta, x, y, weights, offset, cdf, pdf)) / .002 for delta in steps])
    p, n = x.shape[1], len(parameters)
    jacobian = np.eye(n)
    jacobian[p:, p:] = np.tril(np.ones((n - p, n - p))) * np.r_[1, np.exp(parameters[p + 1:])]
    inverse = np.linalg.pinv((information + information.T) / 2,
                             rcond=np.sqrt(np.finfo(float).eps), hermitian=True)
    return jacobian @ inverse @ jacobian.T


class OrdinalModel(_Fit):
    """Proportional cumulative-link model with slopes and ordered thresholds."""

    kind = "ordinal"

    def __init__(self, formula, frame, data, method, start, maxit, epsilon):
        design = build_design(formula, frame.variables, len(frame.weights))
        super().__init__(formula, design, frame, data, "polr")
        if frame.response_levels is None or len(frame.response_levels) < 3:
            raise ValueError("polr(): the response must be categorical with at least three levels")
        self.response_levels, self.method = frame.response_levels, method
        self.y, self.prior_weights = frame.y.astype(int), frame.weights
        self.n, self.nobs = len(self.y), float(frame.weights.sum())
        q = len(self.response_levels) - 1
        counts = np.bincount(self.y, weights=frame.weights, minlength=q + 1)
        if np.any(counts <= 0):
            raise ValueError("polr(): each response category needs positive total weight")
        if not formula.intercept:
            _warn("polr(): an intercept is needed and assumed through the thresholds")
        candidates = [j for j, name in enumerate(design.names) if name != "(Intercept)"]
        basis = np.column_stack([np.ones(self.n), design.matrix[:, candidates]])
        qr = _qr(basis * np.sqrt(frame.weights)[:, None], 1e-7, np.zeros(self.n))
        self.slope_indices = [candidates[j-1] for j in qr.kept if j > 0]
        if len(self.slope_indices) < len(candidates):
            _warn("polr(): design is rank-deficient; dropping aliased coefficients")
        self.slope_names = [design.names[j] for j in self.slope_indices]
        self.x = design.matrix[:, self.slope_indices]
        self.qr = _qr(design.matrix, 1e-7, np.zeros(self.n))
        self.qr.kept = sorted(self.slope_indices + [j for j, name in enumerate(design.names) if name == "(Intercept)"])
        self.rank = len(self.slope_indices) + q
        self.df_residual = self.nobs - self.rank
        distribution = _distribution(method)
        self._cdf, self._pdf = distribution.cdf, distribution.pdf
        if start is None:
            cuts = distribution.ppf(np.cumsum(counts)[:-1] / counts.sum()) + np.average(frame.offset, weights=frame.weights)
            start = np.r_[np.zeros(self.x.shape[1]), cuts]
        start = np.asarray(start, dtype=float)
        if start.shape != (self.rank,) or not np.all(np.isfinite(start)) or np.any(np.diff(start[self.x.shape[1]:]) <= 0):
            raise ValueError("polr(): start needs one finite value per retained slope and strictly increasing thresholds")
        p = self.x.shape[1]
        initial = np.r_[start[:p+1], np.log(np.diff(start[p:]))]
        result = _optimize(initial, self.x, self.y, frame.weights, frame.offset, self._cdf, self._pdf, maxit, epsilon)
        if not np.isfinite(result.fun):
            raise ValueError("polr(): initial likelihood is not finite; choose different start values")
        self.converged, self.iterations = bool(result.success), int(result.nit)
        if not self.converged:
            _warn(f"polr(): optimization did not converge: {result.message}")
        self.coef, self.thresholds = result.x[:p], _thresholds(result.x[p:])
        self.threshold_names = [f"{a}|{b}" for a, b in zip(self.response_levels[:-1], self.response_levels[1:])]
        self.parameter_names = [*self.slope_names, *self.threshold_names]
        self._cov = _covariance(result.x, self.x, self.y, frame.weights, frame.offset, self._cdf, self._pdf)
        if not np.all(np.isfinite(self._cov)) or np.any(np.diag(self._cov) < 0):
            raise ValueError("polr(): covariance is not finite and positive; check identifiability")
        self.deviance = 2 * float(result.fun)
        self.linear_predictor = self.x @ self.coef + frame.offset
        self.fitted = _probabilities(self.linear_predictor, self.thresholds, self._cdf)

    @property
    def coefficients(self):
        return dict(zip(self.slope_names, self.coef.tolist()))

    @property
    def zeta(self):
        return dict(zip(self.threshold_names, self.thresholds.tolist()))

    def tidy(self, conf_int=False, conf_level=.95, exponentiate=False, vcov=None):
        from scipy.special import ndtri

        self._check_vcov(vcov)
        estimate = np.r_[self.coef, self.thresholds]
        se = np.sqrt(np.diag(self._cov))
        cols = {"term": self.parameter_names, "estimate": estimate, "std_error": se,
                "statistic": estimate/se, "p_value": pl.Series([None]*self.rank, dtype=pl.Float64)}
        if conf_int:
            _check_conf_level(conf_level)
            width = ndtri((1+conf_level)/2) * se
            cols.update(conf_low=estimate-width, conf_high=estimate+width)
        if exponentiate:
            for name in ("estimate", "conf_low", "conf_high"):
                if name in cols:
                    cols[name] = np.exp(cols[name])
        return self._with_key(pl.DataFrame(cols))

    def _check_vcov(self, vcov):
        if vcov is not None:
            raise ValueError("polr(): vcov= is for lm/glm; this model supplies its own covariance")

    def vcov_table(self, vcov=None):
        self._check_vcov(vcov)
        return self._with_key(pl.DataFrame({"term": self.parameter_names,
            **{name: self._cov[:, j] for j, name in enumerate(self.parameter_names)}}))

    def log_lik(self):
        return -.5*self.deviance

    def glance(self):
        return self._with_key(pl.DataFrame({"logLik": [self.log_lik()], "AIC": [self.deviance+2*self.rank],
            "BIC": [self.deviance+np.log(self.nobs)*self.rank], "deviance": [self.deviance],
            "df_residual": [self.df_residual], "nobs": [self.nobs], "converged": [self.converged]}))

    def _prediction(self, newdata, type):
        x, off, rows, data = self._design_for(newdata)
        eta = x[:, self.slope_indices] @ self.coef + off
        probabilities = _probabilities(eta, self.thresholds, self._cdf)
        if type == "probs":
            out = pl.DataFrame({str(level): probabilities[:, j] for j, level in enumerate(self.response_levels)})
        elif type == "class":
            out = pl.DataFrame({"fit": [self.response_levels[j] for j in probabilities.argmax(axis=1)]})
        elif type == "link":
            out = pl.DataFrame({"fit": eta})
        else:
            raise ValueError("polr(): prediction type must be 'class', 'probs', or 'link'")
        latent = pl.DataFrame({".linear_predictor": eta})
        if newdata is None:
            out, latent, rows = self._padded(out), self._padded(latent), self._rows()
        return out, latent, rows, data

    def predict(self, newdata=None, *, type="class"):
        out, _, _, data = self._prediction(newdata, type)
        return data.write(out)

    def augment(self, newdata=None):
        out, latent, rows, data = self._prediction(newdata, "class")
        out = pl.concat([rows, out.rename({"fit": ".fitted"}), latent], how="horizontal")
        return out if newdata is None else data.write(out)

    def anova_table(self, test=None):
        raise TypeError("anova(): a single polr model has no ANOVA table; compare nested models")

    def anova_type(self, kind, test=None):
        return self.anova_table(test)

    def __repr__(self):
        return f"polr({self.formula.text}, method={self.method!r})\n{_table_text(self.tidy())}"


@step
def polr(formula, *, method="logistic", weights=None, offset=None, contrasts=None,
         na_action="omit", start=None, maxit=500, epsilon=1e-5):
    """Cumulative-link ordinal regression, with MASS::polr's sign convention.

    P(Y<=k)=F(threshold[k]-X beta-offset). The categorical response needs
    three or more levels; set their order with Enum/category/factor metadata.
    Methods: logistic, probit, loglog and cloglog. Cauchit is deferred.
    ``tidy(conf_int=True)`` gives normal Wald intervals for slopes and
    thresholds; p-values stay missing, as in summary.polr. Grouped models,
    weights, offsets and na_action follow the other model functions.
    """
    if method not in ("logistic", "probit", "loglog", "cloglog"):
        raise ValueError("polr(): method must be logistic, probit, loglog or cloglog")
    if isinstance(maxit, bool) or not isinstance(maxit, (int, np.integer)) or maxit < 1:
        raise ValueError("polr(): maxit must be a positive integer")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("polr(): epsilon must be positive and finite")
    return _model("polr", lambda f, fr, d: OrdinalModel(f, fr, d, method, start, maxit, epsilon),
                  formula, weights, offset, contrasts, na_action)


def _anova_polr(fits, test):
    from scipy.stats import chi2

    if test not in (None, "Chisq", "none", False):
        raise ValueError("polr comparisons use test='Chisq' or 'none'")
    first = fits[0]
    if any(f.method != first.method or f.response_levels != first.response_levels
           or not np.array_equal(f.y, first.y) or not np.array_equal(f.frame.index, first.frame.index)
           or not np.array_equal(f.prior_weights, first.prior_weights) for f in fits[1:]):
        raise ValueError("anova(): polr fits must use the same response, levels, rows, weights and link")
    fits = sorted(fits, key=lambda f: f.rank)
    df = [None] + [b.rank-a.rank for a, b in zip(fits[:-1], fits[1:])]
    statistic = [None] + [a.deviance-b.deviance for a, b in zip(fits[:-1], fits[1:])]
    cols = {"term": [f.formula.text for f in fits], "df_residual": [f.df_residual for f in fits],
            "residual_deviance": [f.deviance for f in fits], "df": df, "statistic": statistic}
    if test not in ("none", False):
        cols["p_value"] = [None] + [chi2.sf(s, d) if d > 0 else None for s, d in zip(statistic[1:], df[1:])]
    return pl.DataFrame(cols, schema_overrides={"df": pl.Float64, "statistic": pl.Float64, "p_value": pl.Float64})
