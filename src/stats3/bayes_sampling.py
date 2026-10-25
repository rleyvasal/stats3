"""Optional PyMC backend for Bayesian generalized and hierarchical models."""

from __future__ import annotations

import copy
import re

import numpy as np
import polars as pl

from stats3._data import step
from stats3.bayes import BayesianModel, Prior, _sampling_options, prior
from stats3.core import _warn
from stats3.formula import parse_formula
from stats3.models import Models, NA_ACTIONS, _ModelStep, _contrasts, _model_frames


FAMILIES = {"gaussian": "identity", "student_t": "identity", "binomial": "logit", "poisson": "log",
            "negative_binomial": "log", "gamma": "log", "lognormal": "log", "exponential": "log",
            "ordinal": "logit", "multinomial": "logit"}


def _pymc():
    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError("Install stats3[bayes] for Bayesian GLMs and hierarchical sampling") from exc
    return pm


def _random_formula(formula):
    random = []

    def remove(match):
        lhs, group = (s.strip() for s in match.groups())
        if not re.fullmatch(r"[A-Za-z_]\w*", group):
            raise ValueError("bayes_glm(): random effects need a single grouping column")
        labels = [s.strip() for s in lhs.split("+")]
        if not all(re.fullmatch(r"[A-Za-z_]\w*|[01]", s) for s in labels):
            raise ValueError("bayes_glm(): random slopes must name numeric columns, e.g. (1 + x | group)")
        intercept = "0" not in labels
        slopes = [s for s in labels if s not in ("0", "1")]
        if not intercept and not slopes:
            raise ValueError("bayes_glm(): empty random-effects term")
        random.append((group, intercept, slopes))
        return ""

    fixed = re.sub(r"\(([^()|]+)\|([^()|]+)\)", remove, formula)
    if "|" in fixed:
        raise ValueError("bayes_glm(): unsupported random-effects syntax")
    lhs, rhs = fixed.split("~", 1)
    rhs = re.sub(r"\+(?:\s*\+)+", "+", rhs).strip().strip("+").strip() or "1"
    if len({g for g, _, _ in random}) != len(random):
        raise ValueError("bayes_glm(): combine slopes for the same group in one random-effects term")
    return f"{lhs.strip()} ~ {rhs}", random


def _rv(pm, name, spec, **kwargs):
    if not isinstance(spec, Prior):
        raise TypeError("priors must be declared with prior()")
    if (name in ("sigma", "alpha", "nu_minus_two") or name.endswith("_sd")) and spec.distribution not in (
        "half_normal", "half_student_t", "exponential", "gamma", "inverse_gamma", "beta"
    ):
        raise ValueError(f"{name} needs a prior supported on positive values")
    distributions = {"normal": pm.Normal, "student_t": pm.StudentT, "half_normal": pm.HalfNormal,
        "half_student_t": pm.HalfStudentT, "exponential": pm.Exponential, "gamma": pm.Gamma,
        "inverse_gamma": pm.InverseGamma, "beta": pm.Beta}
    return distributions[spec.distribution](name, **spec.parameters, **kwargs)


class SampledModel(BayesianModel):
    engine = "pymc"

    def __init__(self, formula, frame, data, family, link, random, priors, draws, tune, chains,
                 target_accept, max_treedepth, seed, progressbar, cores):
        self._initialize(formula, frame, data, "bayes_glm", draws, chains, seed)
        self.family_name, self.link_name = family, link
        self.max_treedepth = max_treedepth
        self.priors, self.random = dict(priors or {}), []
        if not all(isinstance(spec, Prior) for spec in self.priors.values()):
            raise TypeError("priors must be declared with prior()")
        self._used_priors = set()
        self.response_levels = frame.response_levels
        self.n_trials = np.ones(self.n, dtype=np.int64)
        self._prepare_response(frame)
        self._ordinal_intercept = family == "ordinal" and "(Intercept)" in self.design.names
        if self._ordinal_intercept:
            self.x = self.x[:, 1:]
            self.parameter_names = self.parameter_names[1:]
        p = self.x.shape[1]
        if family == "multinomial":
            if random:
                raise ValueError("bayes_glm(): multinomial random effects are not included in this first implementation")
            self.parameter_names = [f"{lv}:{n}" for lv in self.response_levels[1:] for n in self.design.names]
        for j, (group, intercept, slopes) in enumerate(random):
            levels = sorted(frame.rows[group].unique().to_list(), key=str)
            columns = [np.ones(self.n)] if intercept else []
            for slope in slopes:
                if frame.variables[slope].levels is not None:
                    raise ValueError("bayes_glm(): random slopes must be numeric")
                columns.append(frame.variables[slope].values)
            self.random.append({"group": group, "slopes": slopes, "intercept": intercept,
                "levels": levels, "x": np.column_stack(columns),
                "codes": np.array([levels.index(v) for v in frame.rows[group].to_list()]),
                "name": f"group_{j}"})
        pm = _pymc()
        import pytensor.tensor as pt

        center, scale = self._response_scale()
        defaults = self._default_coefficient_priors(center, scale)
        # Sample on predictors centred at their means, as rstanarm does: the
        # intercept (or the ordinal cutpoints) is then the linear predictor at
        # the means, nearly uncorrelated with the slopes. Reported coefficients
        # are for the uncentred predictors.
        columns = list(self.design.names) if family == "multinomial" else self.parameter_names
        intercept = np.array([name == "(Intercept)" for name in columns], dtype=float)
        centers = self.x.mean(axis=0) if self.n and (intercept.any() or family == "ordinal") else np.zeros(p)
        centers[intercept > 0] = 0
        coords = {"coefficient": self.parameter_names, "observation": np.arange(self.n)}
        with pm.Model(coords=coords) as self.pymc_model:
            beta_rvs = []
            for j, name in enumerate(self.parameter_names):
                spec = self._prior(name, self._prior("beta", defaults[j]))
                beta_rvs.append(_rv(pm, f"coefficient_{j}", spec))
            theta = pt.stack(beta_rvs) if beta_rvs else pt.zeros(0)
            if family == "multinomial":
                theta = theta.reshape((len(self.response_levels)-1, p))
                eta = (self.x - centers) @ theta.T + frame.offset[:, None]
                beta = (theta - (theta @ centers)[:, None] * intercept).reshape((-1,))
            else:
                eta = (self.x - centers) @ theta + frame.offset
                beta = theta - (theta @ centers) * intercept
            pm.Deterministic("beta", beta, dims="coefficient")
            for effect in self.random:
                name = effect["name"]
                spread = effect["x"].std(axis=0, ddof=1) if self.n > 1 else np.zeros(effect["x"].shape[1])
                spread[~(spread > 0)] = 1  # the intercept column, and constant slopes
                sd = _rv(pm, f"{name}_sd", self._prior("group_sd", prior("half_normal", sigma=2.5 * scale / spread)),
                         shape=effect["x"].shape[1])
                z = pm.Normal(f"z_{name}", 0, 1, shape=(len(effect["levels"]), effect["x"].shape[1]))
                b = pm.Deterministic(name, z * sd)
                eta = eta + pt.sum(b[effect["codes"]] * effect["x"], axis=1)
            if family in ("gaussian", "student_t", "lognormal"):
                sigma = _rv(pm, "sigma", self._prior("sigma", prior("exponential", lam=1 / scale)))
                sd = sigma / np.sqrt(frame.weights) if family in ("gaussian", "student_t") else sigma
            if family == "gaussian":
                pm.Normal("observed", mu=eta, sigma=sd, observed=self.y, dims="observation")
            elif family == "student_t":
                nu = pm.Deterministic("nu", 2 + _rv(pm, "nu_minus_two", self._prior("nu_minus_two", prior("exponential", lam=1/30))))
                pm.StudentT("observed", mu=eta, sigma=sd, nu=nu, observed=self.y, dims="observation")
            elif family == "binomial":
                prob = pm.math.sigmoid(eta) if link == "logit" else (
                    .5 * (1 + pm.math.erf(eta / np.sqrt(2))) if link == "probit" else -pt.expm1(-pm.math.exp(eta)))
                pm.Binomial("observed", n=self.n_trials, p=prob, observed=self.y, dims="observation")
            elif family == "poisson":
                pm.Poisson("observed", mu=pm.math.exp(eta), observed=self.y, dims="observation")
            elif family in ("negative_binomial", "gamma"):
                alpha = _rv(pm, "alpha", self._prior("alpha", prior("exponential", lam=1)))
                if family == "negative_binomial":
                    pm.NegativeBinomial("observed", mu=pm.math.exp(eta), alpha=alpha, observed=self.y, dims="observation")
                else:
                    pm.Gamma("observed", alpha=alpha, beta=alpha/pm.math.exp(eta), observed=self.y, dims="observation")
            elif family == "exponential":
                pm.Exponential("observed", lam=pm.math.exp(-eta), observed=self.y, dims="observation")
            elif family == "lognormal":
                pm.LogNormal("observed", mu=eta, sigma=sigma, observed=self.y, dims="observation")
            elif family == "ordinal":
                cutpoint_prior = self._prior("cutpoints", prior("normal", mu=0, sigma=5))
                if any(np.ndim(v) != 0 for v in cutpoint_prior.parameters.values()):
                    raise ValueError("cutpoint priors must be iid with scalar parameters")
                centered = _rv(pm, "cutpoints_centered", cutpoint_prior,
                    shape=len(self.response_levels)-1, transform=pm.distributions.transforms.ordered)
                pm.Deterministic("cutpoints", centered + theta @ centers)
                distribution = pm.OrderedLogistic if link == "logit" else pm.OrderedProbit
                distribution("observed", eta=eta, cutpoints=centered, observed=self.y, compute_p=False, dims="observation")
            else:
                logits = pt.concatenate([pt.zeros((self.n, 1)), eta], axis=1)
                pm.Categorical("observed", p=pm.math.softmax(logits, axis=1), observed=self.y, dims="observation")
            unknown = set(self.priors) - self._used_priors
            if unknown:
                raise ValueError(f"bayes_glm(): unused prior names {sorted(unknown)}")
            self.idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                random_seed=int(self.rng.integers(1, 2**30)), target_accept=target_accept,
                nuts={"max_treedepth": max_treedepth}, progressbar=progressbar,
                initvals={"cutpoints_centered": np.linspace(-1, 1, len(self.response_levels)-1)} if family == "ordinal" else None,
                return_inferencedata=True)
            pm.compute_log_likelihood(self.idata, progressbar=False)
        self.beta_draws = self.idata.posterior["beta"].values
        self.extra_draws = {name: self.idata.posterior[name].values for name in self.idata.posterior.data_vars
                            if name != "beta" and not name.startswith("coefficient_") and not name.endswith("_centered")}
        self.coef = self.beta_draws.mean(axis=(0, 1))
        self.posterior_covariance = np.atleast_2d(np.cov(self.beta_draws.reshape(chains*draws, -1), rowvar=False)) if self.parameter_names else np.empty((0, 0))
        self.qr.kept = list(range(len(self.design.names)))

    def _response_scale(self):
        """Location and scale of the response on the linear-predictor scale,
        for families whose coefficients are in the response's units."""
        if self.family_name not in ("gaussian", "student_t", "lognormal"):
            return 0.0, 1.0
        y = (np.log(self.y) if self.family_name == "lognormal" else self.y) - self.frame.offset
        scale = float(np.std(y, ddof=1)) if self.n > 1 else np.nan
        return float(np.mean(y)), scale if np.isfinite(scale) and scale > 0 else 1.0

    def _default_coefficient_priors(self, center, scale):
        """rstanarm-style autoscaled priors: Normal(0, 2.5 s_y / s_x) slopes, and
        Normal(center, 2.5 s_y) for the intercept at the predictors' means."""
        names = list(self.design.names) if self.family_name == "multinomial" else self.parameter_names
        spreads = self.x.std(axis=0, ddof=1) if self.n > 1 else np.zeros(len(names))
        column = [prior("normal", mu=center, sigma=2.5 * scale) if name == "(Intercept)" else
                  prior("normal", mu=0, sigma=2.5 * scale / (spreads[j] if spreads[j] > 0 else 1))
                  for j, name in enumerate(names)]
        # Multinomial coefficients repeat per non-reference category.
        return column * (len(self.response_levels) - 1) if self.family_name == "multinomial" else column

    def _prior(self, name, default):
        if name in self.priors:
            self._used_priors.add(name)
        return self.priors.get(name, default)

    def _prepare_response(self, frame):
        family, y = self.family_name, self.y
        if family == "binomial":
            if y.ndim == 2:
                if np.any(frame.weights != 1):
                    raise ValueError("bayes_glm(): do not combine cbind counts and weights")
                self.n_trials, y = y.sum(axis=1), y[:, 0]
            else:
                if frame.response_levels is not None and len(frame.response_levels) == 1 and isinstance(frame.response_levels[0], (bool, np.bool_)):
                    y = np.full(self.n, int(frame.response_levels[0]))
                    self.response_levels = [False, True]
                elif frame.response_levels is not None and len(frame.response_levels) != 2:
                    raise ValueError("binomial responses must have exactly two categories")
                self.n_trials = frame.weights.copy()
                y = y * self.n_trials
            if np.any(self.n_trials < 0) or np.any(y < 0) or np.any(y > self.n_trials):
                raise ValueError("binomial successes must lie between zero and the number of trials")
            if np.any(np.abs(y-np.round(y)) > 1e-7) or np.any(np.abs(self.n_trials-np.round(self.n_trials)) > 1e-7):
                raise ValueError("binomial successes and trials must be integers")
            self.y, self.n_trials = np.round(y).astype(int), np.round(self.n_trials).astype(int)
            if np.any(self.n_trials == 0):
                raise ValueError("binomial rows need at least one trial")
        elif family in ("ordinal", "multinomial"):
            if frame.response_levels is None or len(frame.response_levels) < (3 if family == "ordinal" else 2):
                raise ValueError(f"{family} responses need categorical levels (at least {'three' if family == 'ordinal' else 'two'})")
            self.y = y.astype(int)
        elif y.ndim != 1 or frame.response_levels is not None:
            raise ValueError("bayes_glm(): this family needs one numeric response")
        if family not in ("gaussian", "student_t", "binomial") and np.any(frame.weights != 1):
            raise ValueError("weights are supported as Gaussian/Student-t precisions or binomial trials")
        if family in ("gaussian", "student_t") and np.any(frame.weights <= 0):
            raise ValueError("precision weights must be strictly positive")
        if family in ("poisson", "negative_binomial") and (np.any(y < 0) or np.any(np.abs(y-np.round(y)) > 1e-7)):
            raise ValueError("count responses must be nonnegative integers")
        if family in ("gamma", "lognormal") and np.any(y <= 0):
            raise ValueError("gamma/lognormal responses must be strictly positive")
        if family == "exponential" and np.any(y < 0):
            raise ValueError("exponential responses must be nonnegative")

    def _new_design(self, newdata):
        x, offset, rows, data = super()._new_design(newdata)
        return (x[:, 1:] if self._ordinal_intercept else x), offset, rows, data

    def _design_for(self, newdata):
        if newdata is None:
            return self.x, self.frame.offset, self.frame.rows, self._data
        return self._new_design(newdata)

    def _eta(self, samples, x, offset, rows, *, re_formula=None, allow_new_levels=False, rng=None):
        if re_formula not in (None, "population"):
            raise ValueError("re_formula must be None or 'population'")
        if self.family_name == "multinomial":
            beta = samples["beta"].reshape(*samples["beta"].shape[:2], len(self.response_levels)-1, x.shape[1])
            return np.einsum("cdkp,np->cdnk", beta, x) + offset[:, None]
        eta = np.einsum("cdp,np->cdn", samples["beta"], x) + offset
        if re_formula == "population":
            return eta
        rng = np.random.default_rng() if rng is None else rng
        for effect in self.random:
            group = effect["group"]
            if group not in rows.columns:
                raise ValueError(f"prediction needs grouping column {group!r}, or re_formula='population'")
            columns = [np.ones(rows.height)] if effect["intercept"] else []
            columns.extend(rows[s].cast(pl.Float64).to_numpy() for s in effect["slopes"])
            z = np.column_stack(columns)
            values = samples[effect["name"]]
            new = {}
            for i, level in enumerate(rows[group].to_list()):
                if level in effect["levels"]:
                    b = values[:, :, effect["levels"].index(level)]
                elif allow_new_levels:
                    if level not in new:
                        new[level] = rng.normal(size=values.shape[:2]+(z.shape[1],)) * samples[effect["name"]+"_sd"]
                    b = new[level]
                else:
                    raise ValueError(f"new level {level!r} in {group!r}; use allow_new_levels=True or re_formula='population'")
                eta[:, :, i] += np.einsum("cdp,p->cd", b, z[i])
        return eta

    def _mean(self, samples, eta):
        from scipy.special import expit, ndtr, softmax

        family = self.family_name
        if family in ("gaussian", "student_t"):
            return eta
        if family == "binomial":
            return expit(eta) if self.link_name == "logit" else (ndtr(eta) if self.link_name == "probit" else -np.expm1(-np.exp(eta)))
        if family == "lognormal":
            return np.exp(eta + samples["sigma"][..., None]**2 / 2)
        if family == "ordinal":
            cdf = expit if self.link_name == "logit" else ndtr
            cumulative = cdf(samples["cutpoints"][:, :, None, :] - eta[..., None])
            return np.diff(np.concatenate([np.zeros((*eta.shape, 1)), cumulative, np.ones((*eta.shape, 1))], axis=-1), axis=-1)
        if family == "multinomial":
            return softmax(np.concatenate([np.zeros((*eta.shape[:-1], 1)), eta], axis=-1), axis=-1)
        return np.exp(eta)

    def _simulate(self, samples, mean, rng, trials=1):
        family = self.family_name
        if family in ("gaussian", "student_t"):
            noise = rng.normal(size=mean.shape) if family == "gaussian" else rng.standard_t(samples["nu"][..., None], size=mean.shape)
            return mean + noise * samples["sigma"][..., None]
        if family == "binomial":
            return rng.binomial(trials, mean)
        if family == "poisson":
            return rng.poisson(mean)
        if family == "negative_binomial":
            alpha = samples["alpha"][..., None]
            return rng.negative_binomial(alpha, alpha/(alpha+mean))
        if family == "gamma":
            alpha = samples["alpha"][..., None]
            return rng.gamma(alpha, mean/alpha)
        if family == "exponential":
            return rng.exponential(mean)
        if family == "lognormal":
            sd = samples["sigma"][..., None]
            return rng.lognormal(np.log(mean)-sd**2/2, sd)
        return np.sum(rng.random(mean.shape[:-1])[..., None] > np.cumsum(mean, axis=-1), axis=-1).clip(max=len(self.response_levels)-1)

    def _trials(self, newdata, trials):
        if self.family_name != "binomial":
            return 1
        value = self.n_trials if newdata is None and trials is None else (1 if trials is None else np.asarray(trials))
        if np.any(np.asarray(value) < 1) or np.any(np.asarray(value) != np.round(value)):
            raise ValueError("trials must be positive integers")
        return np.asarray(value, dtype=int)

    def _prior_samples(self, draws, rng):
        _sampling_options(draws, 1, None)
        pm = _pymc()
        names = ["beta", *self.extra_draws]
        with self.pymc_model:
            result = pm.sample_prior_predictive(draws=draws, var_names=names,
                random_seed=int(rng.integers(1, 2**30)))
        samples = {name: result.prior[name].values for name in names}
        if "cutpoints" in samples:
            # Ordered iid cutpoint priors are order statistics; PyMC ignores
            # transforms when drawing from the prior.
            samples["cutpoints"] = np.sort(samples["cutpoints"], axis=-1)
        return samples

    def glance(self):
        out = super().glance()
        stats = self.idata.sample_stats
        return out.with_columns(pl.lit(self.family_name).alias("family"),
            pl.lit(int(stats.diverging.sum())).alias("divergences"))


@step
def bayes_glm(formula, family="gaussian", *, link=None, priors=None, draws=1000, tune=1000,
              chains=4, seed=None, target_accept=.9, max_treedepth=10, cores=1,
              progressbar=False, weights=None, offset=None, contrasts=None, na_action="omit"):
    """Bayesian GLMs and independent noncentered random intercepts/slopes.

    Requires ``stats3[bayes]`` (PyMC 6 and ArviZ 1).
    Families: gaussian, student_t, binomial, poisson, negative_binomial,
    gamma, lognormal, exponential, ordinal and multinomial. Defaults are
    identity for continuous symmetric outcomes, logit for binomial/ordinal,
    and log for positive/count outcomes; binomial also has probit/cloglog,
    ordinal has probit. Lognormal eta is the mean log outcome.

    ``(1 + x | subject)`` adds independent varying intercepts/slopes.
    ``by=`` instead fits separate, unpooled models. Multinomial random
    effects and correlated varying coefficients are not included.
    Sampling uses predictors centred at their means, as rstanarm does, so
    intercept and cutpoint priors apply at those means; reported
    coefficients are for the uncentred predictors.
    Defaults are autoscaled, as rstanarm's: slopes Normal(0, 2.5 s_y/s_x),
    the intercept Normal(mean response, 2.5 s_y) (Gaussian-type families),
    sigma Exponential(1/s_y), group SDs HalfNormal(2.5 s_y/s_x), where s_y
    is the response SD for gaussian/student_t/lognormal and 1 otherwise;
    alpha Exponential(1), nu=2+Exponential(1/30), ordered cutpoints Normal(0,5).
    ``priors`` overrides beta, individual coefficient names, sigma,
    alpha, group_sd, nu_minus_two and cutpoints with fixed ``prior()`` objects.
    ``family=Gamma()`` warns and uses the log link.
    """
    if callable(family):
        family = family()
    if hasattr(family, "family"):
        if getattr(family, "theta", None) is not None:
            raise ValueError("bayes_glm(): negative-binomial dispersion is estimated; use family='negative_binomial'")
        if link is None and family.family == "Gamma" and family.link.name == "inverse":
            # R's Gamma() default; the samplers here keep the mean positive with a log link.
            _warn("bayes_glm(): Gamma()'s inverse link is not available; using the log link")
            link = "log"
        link = link or family.link.name
        family = family.family
    if not isinstance(family, str):
        raise TypeError("bayes_glm(): family must be a name or a stats3 family")
    family = family.lower()
    if family not in FAMILIES:
        raise ValueError(f"bayes_glm(): family must be one of {sorted(FAMILIES)}")
    link = link or FAMILIES[family]
    allowed = {"binomial": {"logit", "probit", "cloglog"}, "ordinal": {"logit", "probit"}}
    if link not in allowed.get(family, {FAMILIES[family]}):
        raise ValueError(f"bayes_glm(): unsupported {family} link {link!r}")
    _sampling_options(draws, chains, seed)
    for name, value, minimum in (("tune", tune, 0), ("cores", cores, 1), ("max_treedepth", max_treedepth, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(target_accept) or not 0 < target_accept < 1:
        raise ValueError("target_accept must lie between zero and one")
    if na_action not in NA_ACTIONS:
        raise ValueError(f"na_action must be one of {NA_ACTIONS}")
    fixed, random = _random_formula(formula)

    def apply(data):
        # A fresh seed sequence per application, so a reused step repeats its draws.
        sequence = np.random.SeedSequence(seed)
        parsed = parse_formula(fixed, "bayes_glm", data.columns)
        parsed.contrasts, parsed.offset_arg = _contrasts(contrasts, parsed, "bayes_glm"), offset
        frame_formula = copy.deepcopy(parsed)
        extras = [v for group, _, slopes in random for v in [group, *slopes]]
        for label in extras:
            if label not in frame_formula.variables():
                frame_formula.terms.append((label,))
        frames = _model_frames(data, frame_formula, weights, offset, "bayes_glm", na_action)
        fits = [SampledModel(parsed, fr, data, family, link, random, priors, draws, tune, chains,
            target_accept, max_treedepth, sequence.spawn(1)[0], progressbar, cores) for fr in frames]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(apply, "bayes_glm")
