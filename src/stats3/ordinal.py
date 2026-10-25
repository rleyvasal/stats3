"""Cumulative-link ordinal regression, following MASS::polr conventions."""

import numpy as np
import polars as pl

from stats3._data import step
from stats3.core import _check_conf_level, _warn
from stats3.formula import build_design
from stats3.models import _Fit, _model, _qr, _table_text


METHODS = ("logistic", "probit", "loglog", "cloglog", "cauchit")
EMM_MODES = ("latent", "prob", "mean.class")


def _distribution(method):
    from scipy.stats import cauchy, logistic, norm, gumbel_r, gumbel_l

    return {"logistic": logistic, "probit": norm, "loglog": gumbel_r, "cloglog": gumbel_l,
            "cauchit": cauchy}[method]


def _probabilities(eta, thresholds, cdf):
    cumulative = cdf(np.asarray(thresholds) - np.asarray(eta)[:, None])
    endpoints = np.column_stack((np.zeros(len(eta)), cumulative, np.ones(len(eta))))
    return np.diff(endpoints, axis=1)


def _bounds(eta, thresholds, y):
    """Each row's latent interval, clamped to [-100, 100] as MASS's polr.fit
    (which matters only for the heavy-tailed cauchit)."""
    bounds = np.r_[-np.inf, thresholds, np.inf]
    return np.minimum(100, bounds[y+1] - eta), np.maximum(-100, bounds[y] - eta)


def _negative_log_likelihood(eta, thresholds, y, weights, cdf):
    """Weighted ordinal likelihood, with zero-based response categories."""
    used = np.asarray(weights) > 0
    upper, lower = _bounds(np.asarray(eta)[used], thresholds, np.asarray(y)[used])
    observed = cdf(upper) - cdf(lower)
    if np.any(observed <= 0):
        return np.inf
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
    eta = x @ parameters[:p] + offset
    upper, lower = _bounds(eta, _thresholds(theta), y)
    probability = cdf(upper) - cdf(lower)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = weights / probability
        # As MASS's gmin, the slopes see the density at the clamped ends too.
        high, low = pdf(upper), pdf(lower)
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
        self.y, self.prior_weights, self.offset = frame.y.astype(int), frame.weights, frame.offset
        self.maxit, self.epsilon = maxit, epsilon
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

    def tidy(self, conf_int=False, conf_level=.95, exponentiate=False, conf_method=None, vcov=None):
        from scipy.special import ndtri, stdtr, stdtrit

        if conf_method is None:
            conf_method = "profile" if vcov is None else "wald"
        if conf_method not in ("profile", "wald"):
            raise ValueError("conf_method must be 'profile' or 'wald'")
        if conf_method == "profile" and vcov is not None:
            raise ValueError("profile intervals don't use vcov; use conf_method='wald'")
        estimate = np.r_[self.coef, self.thresholds]
        with np.errstate(invalid="ignore"):
            # Multi-way clustering can give negative variances; R reports NaN.
            se = np.sqrt(np.diag(self._covariance(vcov)))
        statistic = estimate/se
        if vcov is None:
            p = pl.Series([None]*self.rank, dtype=pl.Float64)
        else:
            # lmtest's coeftest: t tests on df.residual, applied to every row.
            p = 2*stdtr(self.df_residual, -np.abs(statistic))
        cols = {"term": self.parameter_names, "estimate": estimate, "std_error": se,
                "statistic": statistic, "p_value": p}
        if conf_int:
            _check_conf_level(conf_level)
            if conf_method == "profile":
                low, high = np.full(self.rank, np.nan), np.full(self.rank, np.nan)
                low[:len(self.coef)], high[:len(self.coef)] = self._profile_intervals(conf_level)
                cols.update(conf_low=pl.Series(low).fill_nan(None), conf_high=pl.Series(high).fill_nan(None))
            else:
                q = ndtri((1+conf_level)/2) if vcov is None else stdtrit(self.df_residual, (1+conf_level)/2)
                cols.update(conf_low=estimate-q*se, conf_high=estimate+q*se)
        if exponentiate:
            for name in ("estimate", "conf_low", "conf_high"):
                if name in cols:
                    cols[name] = np.exp(cols[name])
        frame = pl.DataFrame(cols)
        return self._with_key(frame.with_columns(pl.col(c).fill_nan(None) for c in frame.columns if c != "term"))

    def _profile_intervals(self, conf_level):
        """MASS's ``confint.polr``: ``profile.polr`` then ``confint.profile.polr``.

        Each slope is fixed at up to 9 points either side of its estimate
        (steps of zmax/5 standard errors, zmax from alpha = (1 - level)/4)
        and the model refitted with it as an offset; the signed root deviance
        profile is splined (FMM) and inverted at the normal quantiles. MASS
        refits at optim's default tolerance; these refits converge fully.
        """
        from scipy.special import ndtri
        from scipy.stats import chi2
        from stats3.models import _approx, _spline

        p = len(self.coef)
        zmax = np.sqrt(chi2.isf((1-conf_level)/4, 1))
        delta = zmax/5
        se = np.sqrt(np.diag(self._cov))[:p]
        cut = ndtri(np.array([(1-conf_level)/2, (1+conf_level)/2]))
        start = np.r_[self.coef, self.thresholds]
        low, high = np.full(p, np.nan), np.full(p, np.nan)
        for i in range(p):
            xi = np.delete(self.x, i, axis=1)
            others = np.delete(start, i)
            initial = np.r_[others[:p], np.log(np.diff(others[p-1:]))]
            zs, values = [0.], [self.coef[i]]
            for sign in (-1, 1):
                step, z = 0, 0.
                while (step := step+1) < 10 and abs(z) < zmax:
                    bi = self.coef[i] + sign*step*delta*se[i]
                    offset = self.offset + self.x[:, i]*bi
                    result = _optimize(initial, xi, self.y, self.prior_weights, offset,
                                       self._cdf, self._pdf, self.maxit, self.epsilon)
                    zz = 2*float(result.fun) - self.deviance
                    if zz <= -.001:
                        raise ValueError("profiling has found a better solution, so original fit had not converged")
                    z = sign*np.sqrt(max(zz, 0.))
                    zs.append(z)
                    values.append(bi)
            order = np.argsort(zs, kind="stable")
            sx, sy = _spline(np.array(values)[order], np.array(zs)[order])
            low[i], high[i] = _approx(sy, sx, cut)
        return low, high

    def _estfun(self):
        """sandwich's ``estfun.polr``: each row's weighted score in the slopes
        and thresholds. As sandwich, the densities leave out any offset."""
        if self.method == "loglog":
            raise ValueError("polr(): sandwich covariances need make.link(method), "
                             "and R stops with \u2018loglog\u2019 link not recognised")
        n, p, q = self.n, len(self.coef), len(self.thresholds)
        bounds = np.column_stack((np.zeros(n), self._pdf(self.thresholds[None, :] - (self.x @ self.coef)[:, None]),
                                  np.zeros(n)))
        upper, lower = bounds[np.arange(n), self.y+1], bounds[np.arange(n), self.y]
        scores = np.zeros((n, p+q+2))
        scores[:, :p] = -self.x*(upper-lower)[:, None]
        scores[np.arange(n), p+self.y] = -lower
        scores[np.arange(n), p+self.y+1] = upper
        scores = scores[:, np.r_[0:p, p+1:p+q+1]]
        probability = self.fitted[np.arange(n), self.y]
        return (self.prior_weights/probability)[:, None]*scores

    def _covariance(self, vcov=None):
        """The (slopes, thresholds) covariance: the Hessian's, or sandwich's
        ``vcovCL`` with ``bread.polr`` (the covariance times the total weight,
        as R) and the meat averaged over the rows."""
        from stats3.robust import _codes, _column_values, _meat_cl, as_vcov

        spec = as_vcov(vcov)
        if spec is None:
            return self._cov
        if spec.kind == "HC":
            raise ValueError("polr(): sandwich's vcovHC fails on polr fits in R (cannot match dimension of "
                             "model.matrix and estfun to obtain working residuals); use vcov_cl()")
        if spec.kind == "NW":
            raise ValueError("polr(): sandwich's NeweyWest fails on polr fits in R ('x' must be an array "
                             "of at least two dimensions); use vcov_cl()")
        if spec.type in ("HC2", "HC3"):
            raise ValueError("polr(): clustered HC2/HC3 are only applicable to (generalized) linear "
                             "regression models; R stops for polr fits. Use type='HC0' or 'HC1'")
        ef = self._estfun()
        clusters = [_codes(_column_values(self, c, "vcov_cl()")) for c in spec.cluster]
        meat = _meat_cl({"ef": ef, "x": ef, "res": None, "glm": True}, clusters, spec.type, spec.cadjust)
        bread = self._cov*self.nobs
        return bread @ meat @ bread/len(ef)

    def vcov_table(self, vcov=None):
        cov = self._covariance(vcov)
        return self._with_key(pl.DataFrame({"term": self.parameter_names,
            **{name: cov[:, j] for j, name in enumerate(self.parameter_names)}}))

    def _emm_parameters(self, vcov):
        return np.r_[self.coef, self.thresholds], self._covariance(vcov)

    def _emm_cells(self, grid, x, offset, weights, mode):
        """emmeans' polr basis on the reference grid: each cell's value and
        its gradient in (slopes, thresholds), written as a linear function
        (gradient rows, and value minus gradient times parameters as offset)
        so emmeans' averaging and contrasts apply. ``mode="prob"`` adds the
        response as a grid factor, one row per level, as emmeans does."""
        xs = x[:, self.slope_indices]
        n, p, q = len(xs), len(self.coef), len(self.thresholds)
        eta = xs @ self.coef + offset
        if mode == "latent":
            values = (eta - self.thresholds.mean())[:, None]
            gradient = np.column_stack((xs, np.full((n, q), -1/q)))[:, None, :]
        else:
            z = self.thresholds[None, :] - eta[:, None]
            density = self._pdf(z)
            cumulative = np.zeros((n, q+2, p+q))
            cumulative[:, 1:q+1, :p] = -density[:, :, None]*xs[:, None, :]
            cumulative[:, 1+np.arange(q), p+np.arange(q)] = density
            gradient = np.diff(cumulative, axis=1)
            values = np.diff(np.column_stack((np.zeros(n), self._cdf(z), np.ones(n))), axis=1)
            if mode == "mean.class":
                classes = np.arange(1., q+2)
                values = (values @ classes)[:, None]
                gradient = np.einsum("nkp,k->np", gradient, classes)[:, None, :]
        theta = np.r_[self.coef, self.thresholds]
        m = values.shape[1]
        rows = gradient.reshape(n*m, p+q)
        constant = values.reshape(-1) - rows @ theta
        if mode == "prob":
            response = self.formula.response
            if response in grid.columns:
                raise ValueError(f"emmeans(): the response {response!r} is also a predictor")
            levels = [str(level) for level in self.response_levels]
            grid = grid.select(pl.all().gather(np.repeat(np.arange(n), m))).with_columns(
                pl.Series(response, levels*n, dtype=pl.String))
        return grid, rows, constant, np.repeat(weights, m)

    def log_lik(self):
        return -.5*self.deviance

    def glance(self, vcov=None):
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
    Methods: logistic, probit, loglog, cloglog and cauchit (MASS's default
    start values often fail for cauchit; stats3's start from the marginal
    proportions does not). ``tidy(conf_int=True)`` gives MASS's profile
    likelihood intervals for the slopes (``confint.polr``) and none for the
    thresholds; ``conf_method="wald"`` gives normal Wald intervals for both.
    p-values stay missing, as in summary.polr. ``tidy(fit, vcov=vcov_cl(g))``
    uses sandwich's clustered covariance (HC0 or HC1), tested as lmtest's
    ``coeftest`` (t on the residual df); as in R, its bread scales by the
    total weight while its meat averages over rows, and its scores leave out
    offsets. vcovHC, NeweyWest and clustered HC2/HC3 stop with R's errors, as
    does any robust covariance for loglog. ``emmeans(fit, "x")`` gives the
    latent mean (``mode="latent"``), class probabilities (``mode="prob"``, with
    the response as a factor) or the mean class (``mode="mean.class"``).
    Grouped models, weights, offsets and na_action follow the other models.
    """
    if method not in METHODS:
        raise ValueError("polr(): method must be logistic, probit, loglog, cloglog or cauchit")
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
