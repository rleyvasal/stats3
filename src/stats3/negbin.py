"""Negative binomial models: MASS's negative.binomial and glm.nb."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from stats3._data import step
from stats3.core import _special, _warn
from stats3.models import Family, GeneralizedLinearModel, _glm_fit, _link, _model, poisson


def _log_lik(theta, y, mu, w):
    s = _special()
    return float(np.sum(w * (s.gammaln(theta + y) - s.gammaln(theta) - s.gammaln(y + 1)
                            + theta * math.log(theta) + y * np.log(mu + (y == 0))
                            - (theta + y) * np.log(theta + mu))))


def negative_binomial(theta: float, link: str = "log") -> Family:
    """MASS's negative.binomial(theta): Var(Y) = mu + mu²/theta.

    ``glm(data, "y ~ x", negative_binomial(2))`` fixes theta. As in R's
    ``glm``, its coefficient table estimates the Pearson dispersion and
    uses t tests. ``glm_nb`` estimates theta and fixes dispersion at one.
    """
    theta = float(theta)
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("negative_binomial(): theta must be finite and positive")
    if link not in ("log", "identity", "sqrt"):
        raise ValueError("negative_binomial(): link must be 'log', 'identity' or 'sqrt'")

    def dev(y, mu, wt):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2 * wt * (y * np.log(np.maximum(1, y) / mu)
                             - (y + theta) * np.log((y + theta) / (mu + theta)))

    return Family("negative_binomial", _link(link), lambda mu: mu + mu**2 / theta,
                  dev, lambda y, n, mu, wt, d: -2 * _log_lik(theta, y, mu, wt),
                  lambda mu: bool(np.all(np.isfinite(mu)) and np.all(mu > 0)), None, theta)


def _theta_ml(y, mu, w, limit):
    """MASS's theta.ml: Newton score/information steps, including its SE."""
    s = _special()
    theta = float(np.sum(w) / np.sum(w * (y / mu - 1)**2))
    delta, iteration = 1.0, 0
    while (iteration := iteration + 1) < limit and abs(delta) > np.finfo(float).eps**0.25:
        theta = abs(theta)
        score = np.sum(w * (s.digamma(theta + y) - s.digamma(theta) + math.log(theta) + 1
                            - np.log(theta + mu) - (y + theta) / (mu + theta)))
        info = np.sum(w * (-s.polygamma(1, theta + y) + s.polygamma(1, theta) - 1 / theta
                           + 2 / (mu + theta) - (y + theta) / (mu + theta)**2))
        delta = float(score / info)
        theta += delta
    if theta < 0:
        theta = 0.0
        _warn("theta.ml: estimate truncated at zero")
    if iteration == limit:
        _warn("theta.ml: iteration limit reached")
    with np.errstate(invalid="ignore"):
        se = float(np.sqrt(1 / info))
    return theta, se, iteration < limit and theta > 0


class NegativeBinomialModel(GeneralizedLinearModel):
    """A glm.nb fit, with theta and se_theta in addition to GLM results."""

    _extra_parameters = 1

    def __init__(self, formula, frame, data, link, init_theta, maxit, epsilon):
        initial = poisson(link) if init_theta is None else negative_binomial(init_theta, link)
        if init_theta is not None:
            initial.dispersion = 1.0
        super().__init__(formula, frame, data, initial, maxit, epsilon, "glm_nb")
        mu = self.mu
        theta, se, theta_ok = _theta_ml(self.y, mu, self.prior_weights, maxit)
        family = negative_binomial(theta, link)
        family.dispersion = 1.0
        d1 = math.sqrt(2 * max(1, self.df_residual))
        delta = 1.0
        ll = _log_lik(theta, self.y, mu, self.prior_weights)
        old_ll = ll + 2 * d1
        iteration = 0
        while (iteration := iteration + 1) <= maxit and abs(old_ll - ll) / d1 + abs(delta) > epsilon:
            fit = _glm_fit(self.design.matrix, self.y, self.prior_weights, frame.offset, family,
                           self.n_trials, maxit, epsilon, "glm_nb", etastart=family.link.linkfun(mu))
            self.family = family
            self.__dict__.update(fit)
            previous = theta
            # MASS updates theta from the previous means, before replacing mu.
            theta, se, theta_ok = _theta_ml(self.y, mu, self.prior_weights, maxit)
            family = negative_binomial(theta, link)
            family.dispersion = 1.0
            mu = fit["mu"]
            delta = previous - theta
            old_ll, ll = ll, _log_lik(theta, self.y, mu, self.prior_weights)
        if iteration > maxit:
            _warn("glm_nb: alternation limit reached")
        self.theta, self.se_theta = theta, se
        self.theta_converged = iteration <= maxit and theta_ok
        self.cov_unscaled = self.qr.cov_unscaled()
        self.scale = 1.0
        self.aic = -2 * ll + 2 * (self.rank + 1)
        if formula.intercept and np.any(frame.offset != 0):
            null = _glm_fit(np.ones((len(mu), 1)), self.y, self.prior_weights, frame.offset,
                            family, self.n_trials, maxit, epsilon, "glm_nb")
            self.null_deviance = null["deviance"]

    def glance(self):
        # MASS's logLik.negbin counts zero-weight rows for BIC, unlike nobs.glm.
        return super().glance().with_columns(BIC=pl.lit(-2 * self.log_lik() + math.log(len(self.y)) * (self.rank + 1)))

    def anova_table(self, test=None):
        _warn("tests made without re-estimating theta")
        return super().anova_table(test)

    def __repr__(self):
        return super().__repr__() + f"\ntheta {self.theta:.5g} (SE {self.se_theta:.5g})"


@step
def glm_nb(formula: str, *, link: str = "log", init_theta: float | None = None,
           weights=None, offset=None, contrasts=None, maxit: int = 25,
           epsilon: float = 1e-8, na_action: str = "omit"):
    """Fit MASS's glm.nb: alternate coefficient and maximum-likelihood theta fits.

    ``glm_nb(data, "count ~ x + group")`` or ``data >> glm_nb(...)``;
    ``by=`` fits each group. ``theta`` is the fitted size (smaller means
    more overdispersion); ``se_theta`` is its observed-information SE.
    Links are log, identity and sqrt. Weights, offsets, contrasts and
    ``na_action`` follow ``glm``. Coefficient inference fixes theta, as R
    does; comparing several glm_nb models uses their fitted likelihoods.
    """
    if maxit < 2:
        raise ValueError("glm_nb(): maxit must be at least 2")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("glm_nb(): epsilon must be finite and positive")
    negative_binomial(1 if init_theta is None else init_theta, link)
    return _model("glm_nb", lambda f, fr, d: NegativeBinomialModel(f, fr, d, link, init_theta, maxit, epsilon),
                  formula, weights, offset, contrasts, na_action)


def _anova_nb(fits, test):
    """MASS's anova.negbin: likelihood ratios with each theta re-estimated."""
    if test not in (None, "Chisq"):
        _warn("only Chi-squared LR tests are implemented for glm_nb comparisons")
    fits = sorted(fits, key=lambda f: -f.df_residual)
    dfs = [float(f.df_residual) for f in fits]
    ll = [f.log_lik() for f in fits]
    df = [None] + [dfs[i - 1] - dfs[i] for i in range(1, len(fits))]
    stat = [None] + [2 * (ll[i] - ll[i - 1]) for i in range(1, len(fits))]
    p = [None] + [float(_special().chdtrc(d, s)) for d, s in zip(df[1:], stat[1:])]
    return pl.DataFrame({"term": [f.formula.text for f in fits], "theta": [f.theta for f in fits],
                         "df_residual": dfs, "logLik": ll, "df": df, "statistic": stat, "p_value": p},
                        schema_overrides={"df": pl.Float64, "statistic": pl.Float64, "p_value": pl.Float64})
