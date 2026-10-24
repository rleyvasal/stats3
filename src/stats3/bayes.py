"""Bayesian regression: exact conjugate Gaussian inference and optional PyMC.

The exact engine uses beta | sigma² ~ N(m, sigma² V) and
sigma² ~ InverseGamma(shape, rate). Sampled models use independent priors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import polars as pl

from stats3._data import step
from stats3.core import _check_conf_level
from stats3.formula import build_design
from stats3.models import NA_ACTIONS, Models, _Fit, _ModelStep, _model, _qr, _table_text


@dataclass(frozen=True)
class Prior:
    """A proper PyMC prior, created with ``prior('normal', mu=0, sigma=2)``."""

    distribution: str
    parameters: dict


def prior(distribution="normal", **parameters):
    """Declare normal, student_t, half_normal, half_student_t, exponential,
    gamma, inverse_gamma or beta priors using PyMC's parameter names.
    """
    allowed = {"normal", "student_t", "half_normal", "half_student_t", "exponential",
               "gamma", "inverse_gamma", "beta"}
    if distribution not in allowed:
        raise ValueError(f"prior(): distribution must be one of {sorted(allowed)}")
    return Prior(distribution, dict(parameters))


def _sampling_options(draws, chains, seed):
    for name, value in (("draws", draws), ("chains", chains)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return np.random.default_rng(seed)


def _vector(value, names, default, label):
    if isinstance(value, Mapping):
        unknown = set(value) - set(names)
        if unknown:
            raise ValueError(f"{label}: unknown coefficients {sorted(unknown)}")
        value = [value.get(n, default) for n in names]
    result = np.broadcast_to(np.asarray(value, dtype=float), (len(names),)).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def _draw_summary(values, names, level=.95):
    _check_conf_level(level)
    flat = values.reshape(values.shape[0]*values.shape[1], values.shape[-1])
    lo, hi = np.quantile(flat, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return pl.DataFrame({"term": pl.Series(names, dtype=pl.String), "estimate": flat.mean(axis=0),
                         "std_error": flat.std(axis=0, ddof=1) if len(flat) > 1 else np.full(len(names), np.nan),
                         "conf_low": lo, "conf_high": hi,
                         "prob_positive": (flat > 0).mean(axis=0)}).with_columns(pl.col("std_error").fill_nan(None))


class BayesianModel(_Fit):
    """Common posterior interfaces; ``std_error`` is the posterior SD.

    Coefficients are posterior means. No frequentist p-values or test
    statistics are assigned to posterior summaries.
    """

    kind = "bayesian"
    engine = "exact"
    family_name = "gaussian"

    def _initialize(self, formula, frame, data, verb, draws, chains, seed):
        design = build_design(formula, frame.variables, len(frame.weights))
        super().__init__(formula, design, frame, data, verb)
        self.rng = _sampling_options(draws, chains, seed)
        self.draws, self.chains = draws, chains
        self.x, self.y, self.prior_weights = design.matrix, frame.y, frame.weights
        self.n, self.rank = self.x.shape
        if not self.rank:
            raise ValueError(f"{verb}(): the model needs at least one coefficient")
        self.qr = _qr(self.x, 1e-7, np.zeros(self.n))
        self.df_residual = math.inf
        self.parameter_names = list(design.names)
        self.extra_draws = {}
        self.idata = None

    @property
    def coefficients(self):
        return dict(zip(self.parameter_names, self.coef.tolist()))

    def _check_vcov(self, vcov):
        if vcov is not None:
            raise ValueError("Bayesian fits use posterior covariance; vcov= is for lm/glm")

    def tidy(self, conf_int=False, conf_level=.95, exponentiate=False, vcov=None):
        self._check_vcov(vcov)
        out = self._summary(conf_level)
        if exponentiate:
            if self.engine == "exact":
                raise ValueError("exponentiated Student-t posteriors have no finite mean; transform posterior draws explicitly")
            # Summarize transformed draws, rather than exponentiating their mean;
            # prob_positive stays P(beta > 0), i.e. P(exp(beta) > 1).
            out = _draw_summary(np.exp(self.beta_draws), self.parameter_names, conf_level).with_columns(
                out["prob_positive"])
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("statistic"),
                               pl.lit(None, dtype=pl.Float64).alias("p_value"))
        if not conf_int:
            out = out.drop("conf_low", "conf_high")
        return self._with_key(out)

    def _summary(self, level):
        return _draw_summary(self.beta_draws, self.parameter_names, level)

    def vcov_table(self, vcov=None):
        self._check_vcov(vcov)
        matrix = self.posterior_covariance
        return self._with_key(pl.DataFrame({"term": self.parameter_names,
            **{name: matrix[:, i] for i, name in enumerate(self.parameter_names)}}))

    def glance(self):
        return self._with_key(pl.DataFrame({"nobs": [self.n], "parameters": [len(self.parameter_names)],
            "draws": [self.draws], "chains": [self.chains], "engine": [self.engine],
            "log_marginal_likelihood": [getattr(self, "log_evidence", None)]}))

    def _eta(self, samples, x, offset, rows, *, re_formula=None, allow_new_levels=False, rng=None):
        if re_formula not in (None, "population"):
            raise ValueError("re_formula must be None or 'population'")
        return np.einsum("cdp,np->cdn", samples["beta"], x) + offset

    def _mean(self, samples, eta):
        return eta

    def _simulate(self, samples, mean, rng, trials=1):
        return mean + rng.normal(size=mean.shape) * samples["sigma"][..., None]

    def _posterior_samples(self):
        return {"beta": self.beta_draws, **self.extra_draws}

    def _prediction_draws(self, newdata=None, *, predictive=False, seed=None, prior=False,
                          draws=1000, re_formula=None, allow_new_levels=False, trials=None):
        x, offset, rows, data = self._design_for(newdata)
        rng = np.random.default_rng(seed)
        samples = self._prior_samples(draws, rng) if prior else self._posterior_samples()
        eta = self._eta(samples, x, offset, rows, re_formula=re_formula, allow_new_levels=allow_new_levels, rng=rng)
        mean = self._mean(samples, eta)
        values = self._simulate(samples, mean, rng, self._trials(newdata, trials)) if predictive else mean
        if predictive and newdata is None and self.family_name in ("gaussian", "student_t"):
            values = mean + (values - mean) / np.sqrt(self.frame.weights)
        return values, rows, data

    def _trials(self, newdata, trials):
        return 1

    def predict(self, newdata=None, *, interval="credible", level=.95, type="response", seed=None,
                re_formula=None, allow_new_levels=False, trials=None):
        _check_conf_level(level)
        if interval not in ("none", "credible", "prediction"):
            raise ValueError("Bayesian interval must be 'none', 'credible', or 'prediction'")
        if type not in ("response", "link"):
            raise ValueError("Bayesian prediction type must be 'response' or 'link'")
        if type == "link" and interval == "prediction":
            raise ValueError("prediction intervals are on the response scale")
        if self.family_name in ("ordinal", "multinomial") and interval == "prediction":
            raise ValueError("use posterior_predict() for categorical outcome draws; credible intervals summarize probabilities")
        if type == "link":
            x, off, rows, data = self._design_for(newdata)
            values = self._eta(self._posterior_samples(), x, off, rows,
                               re_formula=re_formula, allow_new_levels=allow_new_levels, rng=np.random.default_rng(seed))
            if self.family_name == "multinomial":
                values = np.concatenate([np.zeros((*values.shape[:-1], 1)), values], axis=-1)
        else:
            values, rows, data = self._prediction_draws(newdata, predictive=interval == "prediction", seed=seed,
                re_formula=re_formula, allow_new_levels=allow_new_levels, trials=trials)
            if interval == "prediction" and self.family_name == "binomial":
                # Success counts as proportions, the scale of the fitted probabilities.
                values = values / self._trials(newdata, trials)
        out = self._prediction_table(values, level, interval != "none")
        if interval == "prediction":
            expected, _, _ = self._prediction_draws(newdata, seed=seed, re_formula=re_formula,
                allow_new_levels=allow_new_levels)
            out = out.with_columns(pl.Series("fit", expected.mean(axis=(0, 1)).reshape(-1)))
        if newdata is None:
            if values.ndim == 4:
                # Multi-category probabilities are one row per observation and category.
                out = self._pad_categories(out)
            else:
                out = self._padded(out)
        return data.write(out)

    def _prediction_table(self, values, level, intervals):
        flat = values.reshape((-1, *values.shape[2:]))
        cols = {"fit": flat.mean(axis=0).reshape(-1)}
        if intervals:
            lo, hi = np.quantile(flat, [(1-level)/2, (1+level)/2], axis=0)
            cols.update(lwr=lo.reshape(-1), upr=hi.reshape(-1))
        if values.ndim == 4:
            n, k = values.shape[2:]
            cols = {".row": np.repeat(np.arange(1, n+1), k),
                    "category": list(self.response_levels) * n, **cols}
        return pl.DataFrame(cols)

    def _pad_categories(self, out):
        if self.frame.all_rows is None:
            return out
        parts = []
        for j, category in enumerate(self.response_levels):
            part = self._padded(out.filter(pl.col("category") == category).drop(".row", "category"))
            parts.append(part.with_row_index(".row", offset=1).with_columns(
                pl.lit(category).alias("category"), pl.lit(j).alias("__category_order")))
        return pl.concat(parts).sort(".row", "__category_order").drop("__category_order")

    def augment(self, newdata=None, *, interval="credible", conf_level=.95, seed=None, **kwargs):
        from stats3._data import Data

        out = self.predict(newdata, interval=interval, level=conf_level, seed=seed, **kwargs)
        table = Data.read(out, None, "augment")._lf.collect()
        if "category" in table.columns:
            raise ValueError("augment(): use predict() for multi-category Bayesian fits")
        table = table.rename({n: {"fit": ".fitted", "lwr": ".lower", "upr": ".upper"}[n] for n in table.columns})
        rows = self._rows() if newdata is None else Data.read(newdata, None, "augment")._lf.collect()
        if newdata is None:
            observed = self.y / self.n_trials if self.family_name == "binomial" else self.y
            observed = self._padded(pl.DataFrame({"observed": observed}))["observed"]
            table = table.with_columns((observed-table[".fitted"]).alias(".resid"))
        out = pl.concat([rows, table], how="horizontal")
        return out if newdata is None else Data.read(newdata, None, "augment").write(out)

    def anova_table(self, test=None):
        raise TypeError("anova(): use bayes_compare() for Bayesian models")

    def anova_type(self, kind, test=None):
        return self.anova_table(test)

    def __repr__(self):
        return f"{self.verb}({self.formula.text}, engine={self.engine!r})\n{_table_text(self.tidy(conf_int=True))}"


class ConjugateLinearModel(BayesianModel):
    """Normal-inverse-gamma regression with exact posterior moments."""

    def __init__(self, formula, frame, data, prior_mean, prior_scale, prior_shape, prior_rate,
                 draws, chains, seed):
        from scipy.special import gammaln

        self._initialize(formula, frame, data, "bayes_lm", draws, chains, seed)
        if self.y.ndim != 1 or frame.response_levels is not None:
            raise ValueError("bayes_lm(): the response must be numeric")
        if np.any(frame.weights <= 0):
            raise ValueError("bayes_lm(): precision weights must be strictly positive")
        if not np.isfinite(prior_shape) or prior_shape <= 1 or not np.isfinite(prior_rate) or prior_rate <= 0:
            raise ValueError("bayes_lm(): prior_shape must exceed 1 and prior_rate must be positive")
        self.prior_mean = _vector(prior_mean, self.parameter_names, 0, "prior_mean")
        self.prior_scale = _vector(prior_scale, self.parameter_names, 10, "prior_scale")
        if np.any(self.prior_scale <= 0):
            raise ValueError("bayes_lm(): prior_scale must be positive")
        self.prior_shape, self.prior_rate = float(prior_shape), float(prior_rate)
        precision = np.diag(1 / self.prior_scale**2)
        self.posterior_precision = precision + self.x.T @ (frame.weights[:, None] * self.x)
        self.posterior_scale = np.linalg.solve(self.posterior_precision, np.eye(self.rank))
        self.coef = np.linalg.solve(self.posterior_precision,
            precision @ self.prior_mean + self.x.T @ (frame.weights * (self.y - frame.offset)))
        self.posterior_shape = prior_shape + self.n / 2
        resid = self.y - frame.offset - self.x @ self.coef
        diff = self.coef - self.prior_mean
        self.posterior_rate = float(prior_rate + (frame.weights @ resid**2 + diff @ precision @ diff) / 2)
        self.posterior_covariance = self.posterior_scale * self.posterior_rate / (self.posterior_shape - 1)
        self.log_evidence = float(-self.n / 2 * math.log(2 * math.pi) + .5 * np.log(frame.weights).sum()
            - np.log(self.prior_scale).sum() - .5 * np.linalg.slogdet(self.posterior_precision)[1]
            + prior_shape * math.log(prior_rate) - self.posterior_shape * math.log(self.posterior_rate)
            + gammaln(self.posterior_shape) - gammaln(prior_shape))
        samples = self._sample(self.coef, self.posterior_scale, self.posterior_shape, self.posterior_rate,
                               (chains, draws), self.rng)
        self.beta_draws, self.extra_draws = samples["beta"], {"sigma": samples["sigma"]}
        # A proper prior identifies even rank-deficient likelihoods.
        self.qr.kept = list(range(self.rank))

    @staticmethod
    def _sample(mean, scale, shape, rate, size, rng):
        variance = 1 / rng.gamma(shape, 1 / rate, size=size)
        beta = mean + (rng.normal(size=(*size, len(mean))) @ np.linalg.cholesky(scale).T) * np.sqrt(variance)[..., None]
        return {"beta": beta, "sigma": np.sqrt(variance)}

    def _prior_samples(self, draws, rng):
        _sampling_options(draws, 1, None)
        return self._sample(self.prior_mean, np.diag(self.prior_scale**2), self.prior_shape,
                            self.prior_rate, (1, draws), rng)

    def _summary(self, level):
        from scipy.stats import t

        _check_conf_level(level)
        scale = np.sqrt(np.diag(self.posterior_scale) * self.posterior_rate / self.posterior_shape)
        df = 2 * self.posterior_shape
        lo, hi = t.ppf([(1-level)/2, (1+level)/2], df)
        return pl.DataFrame({"term": self.parameter_names, "estimate": self.coef,
            "std_error": np.sqrt(np.diag(self.posterior_covariance)),
            "conf_low": self.coef + lo * scale, "conf_high": self.coef + hi * scale,
            "prob_positive": t.cdf(self.coef / scale, df)})

    def predict(self, newdata=None, *, interval="credible", level=.95, type="response", seed=None,
                re_formula=None, allow_new_levels=False, trials=None):
        from scipy.stats import t

        _check_conf_level(level)
        if interval not in ("none", "credible", "prediction") or type not in ("response", "link"):
            raise ValueError("bayes_lm(): use interval='none'/'credible'/'prediction' and type='response'/'link'")
        if re_formula not in (None, "population"):
            raise ValueError("re_formula must be None or 'population'")
        x, offset, rows, data = self._design_for(newdata)
        out = {"fit": x @ self.coef + offset}
        if interval != "none":
            variance = np.einsum("ij,jk,ik->i", x, self.posterior_scale, x)
            if interval == "prediction":
                variance += 1 / self.frame.weights if newdata is None else 1
            width = t.ppf((1+level)/2, 2*self.posterior_shape) * np.sqrt(variance * self.posterior_rate / self.posterior_shape)
            out.update(lwr=out["fit"]-width, upr=out["fit"]+width)
        out = pl.DataFrame(out)
        return data.write(self._padded(out) if newdata is None else out)

@step
def bayes_lm(formula, *, prior_mean=0, prior_scale=10, prior_shape=2, prior_rate=1,
             draws=1000, chains=4, seed=None, weights=None, offset=None, contrasts=None, na_action="omit"):
    """Exact Bayesian linear regression, needing only the base dependencies.

    beta | sigma² ~ N(prior_mean, sigma² diag(prior_scale²));
    sigma² ~ InverseGamma(prior_shape, prior_rate), with shape > 1.
    Means and scales may be scalars, vectors or dictionaries of coefficient
    names. Weights are known observation precisions, not replicated cases.
    Intervals and moments are exact; stored draws are independent samples.
    ``by=`` fits independent models per group (not partial pooling).
    """
    if na_action not in NA_ACTIONS:
        raise ValueError(f"bayes_lm(): na_action must be one of {list(NA_ACTIONS)}, got {na_action!r}")

    def apply(data):
        # A fresh seed sequence per application, so a reused step repeats its draws.
        sequence = np.random.SeedSequence(seed)
        return _model("bayes_lm", lambda f, fr, d: ConjugateLinearModel(f, fr, d, prior_mean, prior_scale,
            prior_shape, prior_rate, draws, chains, sequence.spawn(1)[0]),
            formula, weights, offset, contrasts, na_action)._fn(data)

    return _ModelStep(apply, "bayes_lm")


def _bayesian_fits(model):
    fits = model.fits if isinstance(model, Models) else [model]
    if not all(isinstance(f, BayesianModel) for f in fits):
        raise TypeError("expected a fit from bayes_lm() or bayes_glm()")
    return fits


def _single_fit(model):
    if isinstance(model, Models):
        raise ValueError("select one group's model, e.g. models[0], for predictive draws")
    return _bayesian_fits(model)[0]


def posterior_summary(model, *, level=.95, parameters="all"):
    """Posterior means, SDs, equal-tail credible intervals and P(parameter>0).
    ``parameters='all'`` includes scales, cutpoints and random effects.
    """
    frames = []
    for fit in _bayesian_fits(model):
        table = fit._summary(level)
        if parameters == "all":
            for name, values in fit.extra_draws.items():
                if name.startswith("z_"):
                    continue
                shape = values.shape[2:]
                name = "aux:"+name if name in fit.parameter_names else name
                labels = [name] if not shape else [f"{name}[{','.join(map(str,i))}]" for i in np.ndindex(shape)]
                table = pl.concat([table, _draw_summary(values.reshape(fit.chains, fit.draws, -1), labels, level)])
        elif parameters != "coefficients":
            raise ValueError("parameters must be 'all' or 'coefficients'")
        frames.append(fit._with_key(table))
    return model._stack(frames) if isinstance(model, Models) else model._write(frames[0])


def posterior_draws(model, *, parameters="all"):
    """One row per chain and draw, with coefficient/auxiliary columns."""
    frames = []
    for fit in _bayesian_fits(model):
        cols = {".chain": np.repeat(np.arange(1, fit.chains+1), fit.draws),
                ".draw": np.tile(np.arange(1, fit.draws+1), fit.chains)}
        flat = fit.beta_draws.reshape(fit.chains*fit.draws, len(fit.parameter_names))
        cols.update({name: flat[:, j] for j, name in enumerate(fit.parameter_names)})
        if parameters == "all":
            for name, values in fit.extra_draws.items():
                if name.startswith("z_"):
                    continue
                name = "aux:"+name if name in fit.parameter_names else name
                labels = [name] if values.ndim == 2 else [f"{name}[{','.join(map(str,i))}]" for i in np.ndindex(values.shape[2:])]
                cols.update({label: values.reshape(len(flat), -1)[:, j] for j, label in enumerate(labels)})
        elif parameters != "coefficients":
            raise ValueError("parameters must be 'all' or 'coefficients'")
        frames.append(fit._with_key(pl.DataFrame(cols)))
    return model._stack(frames) if isinstance(model, Models) else model._write(frames[0])


def _prediction_frame(fit, values, row_ids=None):
    chains, draws, n = values.shape[:3]
    k = values.shape[3] if values.ndim == 4 else 1
    cols = {".chain": np.repeat(np.arange(1, chains+1), draws*n*k),
            ".draw": np.tile(np.repeat(np.arange(1, draws+1), n*k), chains),
            ".row": np.tile(np.repeat(np.arange(1, n+1) if row_ids is None else row_ids, k), chains*draws),
            ".prediction": values.reshape(-1)}
    if k > 1:
        cols["category"] = list(fit.response_levels) * (chains*draws*n)
    return pl.DataFrame(cols)


def posterior_predict(model, newdata=None, *, seed=None, re_formula=None, allow_new_levels=False, trials=None):
    """Posterior predictive draws, including observation noise.
    Binomial draws are success counts; with new data ``trials`` defaults to 1.
    """
    fit = _single_fit(model)
    values, rows, data = fit._prediction_draws(newdata, predictive=True, seed=seed,
        re_formula=re_formula, allow_new_levels=allow_new_levels, trials=trials)
    return data.write(_prediction_frame(fit, values, fit.frame.index+1 if newdata is None else None))


def posterior_epred(model, newdata=None, *, seed=None, re_formula=None, allow_new_levels=False):
    """Draws of the conditional response mean (category probabilities for categorical models)."""
    fit = _single_fit(model)
    values, rows, data = fit._prediction_draws(newdata, seed=seed, re_formula=re_formula, allow_new_levels=allow_new_levels)
    return data.write(_prediction_frame(fit, values, fit.frame.index+1 if newdata is None else None))


def prior_predict(model, newdata=None, *, draws=1000, seed=None, re_formula=None, allow_new_levels=False, trials=None):
    """Prior predictive draws, before conditioning on the observed response."""
    fit = _single_fit(model)
    values, rows, data = fit._prediction_draws(newdata, predictive=True, prior=True, draws=draws, seed=seed,
        re_formula=re_formula, allow_new_levels=allow_new_levels, trials=trials)
    return data.write(_prediction_frame(fit, values, fit.frame.index+1 if newdata is None else None))


def hypothesis(model, coefficients, *, value=0, rope=None, level=.95):
    """Posterior inference for a named linear contrast of the coefficients.
    ``coefficients={'x': 1, 'z': -1}`` evaluates x-z-value. ROPE is an
    optional (lower, upper) interval; no point-null Bayes factor is inferred.
    """
    if not isinstance(coefficients, Mapping) or not coefficients:
        raise ValueError("hypothesis(): coefficients must be a nonempty mapping")
    frames = []
    for fit in _bayesian_fits(model):
        unknown = set(coefficients) - set(fit.parameter_names)
        if unknown:
            raise ValueError(f"hypothesis(): unknown coefficients {sorted(unknown)}")
        weights = np.array([coefficients.get(n, 0) for n in fit.parameter_names], dtype=float)
        if not np.all(np.isfinite(weights)) or not np.isfinite(value):
            raise ValueError("hypothesis(): coefficients and value must be finite")
        values = (fit.beta_draws @ weights - value)[..., None]
        out = _draw_summary(values, ["contrast"], level)
        if rope is not None:
            if len(rope) != 2 or not np.all(np.isfinite(rope)) or rope[0] >= rope[1]:
                raise ValueError("hypothesis(): rope must be a finite increasing pair")
            out = out.with_columns(pl.lit(float(((values >= rope[0]) & (values <= rope[1])).mean())).alias("rope_probability"))
        frames.append(fit._with_key(out))
    return model._stack(frames) if isinstance(model, Models) else model._write(frames[0])


def _arviz():
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError("Install stats3[bayes] for PyMC sampling and ArviZ diagnostics") from exc
    return az


def _inference_data(fit):
    if fit.idata is not None:
        return fit.idata
    from scipy.stats import norm

    samples = fit._posterior_samples()
    eta = fit._eta(samples, fit.x, fit.frame.offset, fit.frame.rows)
    loglik = norm.logpdf(fit.y, eta, samples["sigma"][..., None] / np.sqrt(fit.frame.weights))
    return _arviz().from_dict(posterior=samples, log_likelihood={"observed": loglik}, observed_data={"observed": fit.y})


def bayes_diagnostics(model):
    """Rank-normalized R-hat, bulk/tail ESS and MCSE; exact draws are IID.
    NUTS diagnostics include divergences, maximum tree-depth hits and BFMI.
    """
    frames = []
    for fit in _bayesian_fits(model):
        if fit.engine == "exact":
            out = pl.DataFrame({"term": fit.parameter_names, "r_hat": [None]*len(fit.parameter_names),
                "ess_bulk": [float(fit.chains*fit.draws)]*len(fit.parameter_names),
                "ess_tail": [float(fit.chains*fit.draws)]*len(fit.parameter_names),
                "mcse_mean": np.sqrt(np.diag(fit.posterior_covariance)/(fit.chains*fit.draws)),
                "divergences": [0]*len(fit.parameter_names), "engine": ["exact_iid"]*len(fit.parameter_names)},
                schema_overrides={"r_hat": pl.Float64})
        else:
            az = _arviz()
            variables = [n for n in fit.idata.posterior.data_vars
                         if not n.startswith(("coefficient_", "z_")) and not n.endswith("_centered")]
            summary = az.summary(fit.idata, var_names=variables, kind="diagnostics", round_to="none")
            stats = fit.idata.sample_stats
            terms = list(summary.index)
            beta_rows = [i for i, name in enumerate(terms) if name.startswith("beta[")]
            for i, name in zip(beta_rows, fit.parameter_names):
                terms[i] = name
            count = len(terms)
            out = pl.DataFrame({"term": terms,
                **{name: summary[name].to_numpy() for name in ("r_hat", "ess_bulk", "ess_tail", "mcse_mean")},
                "divergences": [int(stats.diverging.sum())]*count,
                "max_treedepth_hits": [int((stats.tree_depth >= fit.max_treedepth).sum())]*count,
                "min_bfmi": [float(np.min(az.bfmi(fit.idata)))]*count,
                "engine": ["pymc"]*count})
        frames.append(fit._with_key(out))
    return model._stack(frames) if isinstance(model, Models) else model._write(frames[0])


def bayes_compare(models, *, method="loo"):
    """Compare named models fitted to identical responses and row identities.
    LOO/WAIC use ArviZ stacking weights. ``method='evidence'`` gives exact
    Gaussian log Bayes factors and weights under equal prior model odds.
    """
    if not isinstance(models, Mapping) or len(models) < 2:
        raise ValueError("bayes_compare(): supply at least two named models")
    fits = list(models.values())
    for fit in fits:
        _bayesian_fits(fit)
        if isinstance(fit, Models):
            raise ValueError("bayes_compare(): compare individual groups' models")
    first = fits[0]
    if any(f.family_name != first.family_name or f.y.shape != first.y.shape
           or not np.array_equal(f.y, first.y) or not np.array_equal(f.frame.index, first.frame.index)
           or not np.array_equal(f.frame.weights, first.frame.weights)
           or getattr(f, "response_levels", None) != getattr(first, "response_levels", None)
           or not np.array_equal(getattr(f, "n_trials", None), getattr(first, "n_trials", None)) for f in fits[1:]):
        raise ValueError("bayes_compare(): models must use identical responses, rows and weights")
    if method == "evidence":
        if not all(isinstance(f, ConjugateLinearModel) for f in fits):
            raise ValueError("evidence comparison is available only for exact bayes_lm fits")
        from scipy.special import softmax

        evidence = np.array([f.log_evidence for f in fits])
        table = pl.DataFrame({"model": list(models), "log_marginal_likelihood": evidence,
                              "log_bayes_factor": evidence-evidence[0], "weight": softmax(evidence)})
    elif method in ("loo", "waic"):
        comparison = _arviz().compare({n: _inference_data(f) for n, f in models.items()}, ic=method)
        table = pl.DataFrame(comparison.rename_axis("model").reset_index().to_dict("list"))
    else:
        raise ValueError("bayes_compare(): method must be 'loo', 'waic' or 'evidence'")
    return first._write(table)


def pp_check(model, *, statistic="mean", level=.95, seed=None):
    """Posterior predictive check for mean, SD, maximum or zero fraction.
    ``prob_greater`` is P(T(y_rep) >= T(y_observed) | y), a posterior
    predictive tail area, not a model-comparison Bayes factor.
    """
    _check_conf_level(level)
    fit = _single_fit(model)
    if fit.family_name in ("ordinal", "multinomial"):
        raise ValueError("use posterior_predict() for categorical predictive checks")
    functions = {"mean": lambda a: np.mean(a, axis=-1), "sd": lambda a: np.std(a, axis=-1, ddof=1),
                 "max": lambda a: np.max(a, axis=-1), "zero_fraction": lambda a: np.mean(a == 0, axis=-1)}
    if statistic not in functions:
        raise ValueError(f"statistic must be one of {list(functions)}")
    if statistic == "sd" and fit.n < 2:
        raise ValueError("SD predictive checks need at least two observations")
    values, _, _ = fit._prediction_draws(predictive=True, seed=seed)
    observed = float(functions[statistic](fit.y))
    replicated = functions[statistic](values)
    low, high = np.quantile(replicated, [(1-level)/2, (1+level)/2])
    return fit._write(pl.DataFrame({"statistic": [statistic], "observed": [observed],
        "estimate": [float(replicated.mean())], "conf_low": [low], "conf_high": [high],
        "prob_greater": [float(np.mean(replicated >= observed))]}))


def bayes_binom_test(x, n, *, alpha=1, beta=1, null=.5, level=.95):
    """Exact beta-binomial inference for scalar successes/trials.
    The log Bayes factor compares the Beta(alpha,beta) alternative with
    the point null, including their specified prior normalizations.
    """
    from scipy.special import betaln, xlogy, xlog1py
    from scipy.stats import beta as beta_dist

    _check_conf_level(level)
    if not np.all(np.isfinite([x, n, alpha, beta, null])) or n < 0 or x < 0 or x > n or x != round(x) or n != round(n):
        raise ValueError("successes/trials must be finite integers with 0 <= x <= n")
    if alpha <= 0 or beta <= 0 or not 0 <= null <= 1:
        raise ValueError("alpha/beta must be positive and null must be in [0,1]")
    a, b = alpha+x, beta+n-x
    distribution = beta_dist(a, b)
    lo, hi = distribution.ppf([(1-level)/2, (1+level)/2])
    logbf = betaln(a, b)-betaln(alpha, beta)-xlogy(x, null)-xlog1py(n-x, -null)
    return pl.DataFrame({"estimate": [distribution.mean()], "std_error": [distribution.std()],
        "conf_low": [lo], "conf_high": [hi], "prob_greater": [distribution.sf(null)],
        "alpha": [a], "beta": [b], "log_bayes_factor": [float(logbf)]})


def bayes_poisson_test(x, exposure=1, *, shape=1, rate=1, null=1, level=.95):
    """Exact gamma-Poisson inference for a total count and exposure.
    The Gamma prior is parameterized by shape and rate; the log Bayes
    factor compares it with a point null rate.
    """
    from scipy.special import gammaln, xlogy
    from scipy.stats import gamma

    _check_conf_level(level)
    if not np.all(np.isfinite([x, exposure, shape, rate, null])) or x < 0 or x != round(x):
        raise ValueError("count must be a finite nonnegative integer; arguments must be finite")
    if exposure <= 0 or shape <= 0 or rate <= 0 or null < 0:
        raise ValueError("exposure/shape/rate must be positive and null nonnegative")
    a, b = shape+x, rate+exposure
    distribution = gamma(a, scale=1/b)
    lo, hi = distribution.ppf([(1-level)/2, (1+level)/2])
    logbf = gammaln(a)-gammaln(shape)+shape*math.log(rate)-a*math.log(b)-xlogy(x, null)+exposure*null
    return pl.DataFrame({"estimate": [distribution.mean()], "std_error": [distribution.std()],
        "conf_low": [lo], "conf_high": [hi], "prob_greater": [distribution.sf(null)],
        "shape": [a], "rate": [b], "log_bayes_factor": [float(logbf)]})
