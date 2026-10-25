"""Baseline-category multinomial logit, following nnet::multinom conventions."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from stats3._data import Data, expression, step
from stats3.core import _check_conf_level, _warn
from stats3.formula import build_design
from stats3.models import _Fit, _levels_for, _model, _qr, _table_text


def _log_probabilities(eta):
    """Row-wise log softmax of an (n, K) predictor whose first column is the baseline."""
    top = eta.max(axis=1, keepdims=True)
    return eta - top - np.log(np.exp(eta - top).sum(axis=1, keepdims=True))


def _predictor(x, coef, offset):
    return np.column_stack((np.zeros(len(x)), x @ coef.T)) + offset


def _deviance(x, coef, y, weights, offset):
    used = weights > 0
    logp = _log_probabilities(_predictor(x[used], coef, offset[used]))
    return -2 * float(weights[used] @ logp[np.arange(used.sum()), y[used]])


def _score(x, coef, y, weights, offset):
    """Gradient of the log-likelihood, class-major: level 2's terms, level 3's, ..."""
    p = np.exp(_log_probabilities(_predictor(x, coef, offset)))
    indicator = np.eye(p.shape[1])[y]
    return ((weights[:, None] * (indicator - p))[:, 1:].T @ x).ravel()


def _information(x, probabilities, weights):
    """The exact Hessian of minus the log-likelihood, as nnet's multinomHess."""
    n, p = x.shape
    q = probabilities.shape[1] - 1
    info = np.empty((q * p, q * p))
    for j in range(q):
        for k in range(j, q):
            pj, pk = probabilities[:, j + 1], probabilities[:, k + 1]
            w = weights * pj * ((j == k) - pk)
            block = x.T @ (w[:, None] * x)
            info[j*p:(j+1)*p, k*p:(k+1)*p] = block
            info[k*p:(k+1)*p, j*p:(j+1)*p] = block.T
    return info


def _ginv(a):
    """nnet's ginv: the Moore-Penrose inverse, dropping singular values below sqrt(eps) * largest."""
    u, d, vt = np.linalg.svd(a)
    positive = d > max(np.sqrt(np.finfo(float).eps) * d[0], 0) if len(d) else d > 0
    if not positive.any():
        return np.zeros(a.T.shape)
    return vt[positive].T @ (u[:, positive] / d[positive]).T


def _newton(x, y, weights, offset, q, maxit):
    """Maximum likelihood by Newton-Raphson with step halving, from zero (nnet's start)."""
    coef = np.zeros((q, x.shape[1]))
    deviance = _deviance(x, coef, y, weights, offset)
    for iteration in range(1, maxit + 1):
        probabilities = np.exp(_log_probabilities(_predictor(x, coef, offset)))
        info = _information(x, probabilities, weights)
        step = np.linalg.lstsq(info, _score(x, coef, y, weights, offset), rcond=None)[0].reshape(coef.shape)
        for _ in range(60):
            proposal = _deviance(x, coef + step, y, weights, offset)
            if np.isfinite(proposal) and proposal <= deviance + 1e-12 * abs(deviance):
                break
            step /= 2
        coef, deviance = coef + step, proposal
        if np.max(np.abs(step) / (np.abs(coef) + 1)) < 1e-10:
            return coef, deviance, True, iteration
    return coef, deviance, False, maxit


def _level_label(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class MultinomialModel(_Fit):
    """Baseline-category logit: log(P(level k) / P(first level)) = X beta_k + offset."""

    kind = "multinomial"

    def __init__(self, formula, frame, data, offset_columns, maxit):
        design = build_design(formula, frame.variables, len(frame.weights))
        super().__init__(formula, design, frame, data, "multinom")
        if frame.response_levels is None:
            values = np.unique(frame.y)  # R's multinom treats any response as a factor
            levels, codes = [_level_label(float(v)) for v in values], np.searchsorted(values, frame.y)
        else:
            # The model frame keeps only the levels present; R warns about the others.
            declared = _levels_for(data, formula.response)
            present = list(frame.response_levels)
            levels = declared if declared and set(present) <= set(declared) else present
            codes = np.array([levels.index(lv) for lv in present])[frame.y.astype(int)]
        counts = np.bincount(codes, minlength=len(levels))
        if np.any(counts == 0):
            empty = [f"'{level}'" for level, count in zip(levels, counts) if count == 0]
            _warn(f"multinom(): group{'s' if len(empty) > 1 else ''} {' '.join(empty)} "
                  f"{'are' if len(empty) > 1 else 'is'} empty")
            codes = np.cumsum(counts > 0)[codes] - 1
            levels = [level for level, count in zip(levels, counts) if count > 0]
        if len(levels) < 2:
            raise ValueError("multinom(): need two or more classes to fit a multinom model")
        self.response_levels, self.y = levels, codes
        self.prior_weights, self.n = frame.weights, len(codes)
        q = len(levels) - 1
        self._offset_columns = offset_columns
        self.offset = self._offset_matrix(frame.offset, self._matrix_offset(data, frame.index), "multinom")
        self.x = design.matrix
        p = self.x.shape[1]
        # nnet keeps every column; its BFGS from zero stays in the row space of X,
        # so aliased columns share the minimum-norm solution. Fit the identified
        # columns, then map to that solution.
        kept = _qr(self.x, 1e-7, np.zeros(self.n)).kept
        self.rank = len(kept)
        self.qr = SimpleNamespace(kept=list(range(p)))
        fitted, self.deviance, self.converged, self.iterations = _newton(
            self.x[:, kept], self.y, self.prior_weights, self.offset, q, maxit)
        if not self.converged:
            _warn(f"multinom(): Newton iterations did not converge in {maxit} steps; "
                  "fitted probabilities may be 0 or 1")
        if len(kept) < p:
            used = self.prior_weights > 0
            self.coef = (np.linalg.pinv(self.x[used]) @ (self.x[used][:, kept] @ fitted.T)).T
        else:
            self.coef = fitted
        self.fitted = np.exp(_log_probabilities(_predictor(self.x, self.coef, self.offset)))
        self._cov = _ginv(_information(self.x, self.fitted, self.prior_weights))
        self.edf = q * self.rank
        self.aic = self.deviance + 2 * self.edf
        self.nobs = self.n
        self.df_residual = self.n * q - self.edf
        terms = self.design.names
        self.parameter_names = terms if q == 1 else [f"{lv}:{t}" for lv in levels[1:] for t in terms]

    # offsets: one column (two levels) or one column per response level, as R's offset(cbind(...))

    def _matrix_offset(self, data, index):
        if self._offset_columns is None:
            return None
        values = data._lf.select(expression(c).cast(pl.Float64).alias(f"o{i}")
                                 for i, c in enumerate(self._offset_columns)).collect().to_numpy()
        return values if index is None else values[index]

    def _offset_matrix(self, vector, matrix, verb):
        k = len(self.response_levels)
        if matrix is not None:
            if matrix.shape[1] != k:
                raise ValueError(f"{verb}(): offset needs one column per response level ({k}), "
                                 f"got {matrix.shape[1]}")
            if np.any(vector != 0):
                raise ValueError(f"{verb}(): give the offset either as one column per level or in the formula, not both")
            return matrix
        if np.any(vector != 0) or self.formula.offsets or self.formula.offset_arg is not None:
            if k > 2:
                raise ValueError(f"{verb}(): with {k} response levels the offset needs one column per "
                                 "level, offset=[...], as R's offset(cbind(...))")
            return np.column_stack((np.zeros(len(vector)), vector))
        return np.zeros((len(vector), k))

    @property
    def coefficients(self):
        return dict(zip(self.parameter_names, self.coef.ravel().tolist()))

    def tidy(self, conf_int=False, conf_level=.95, exponentiate=False, vcov=None):
        from scipy.special import ndtr, ndtri

        self._check_vcov(vcov)
        terms = self.design.names
        estimate = self.coef.ravel()
        se = np.sqrt(np.diag(self._cov))
        statistic = estimate / se
        cols = {"y_level": [str(lv) for lv in self.response_levels[1:] for _ in terms],
                "term": terms * (len(self.response_levels) - 1), "estimate": estimate,
                "std_error": se, "statistic": statistic, "p_value": 2 * ndtr(-np.abs(statistic))}
        if conf_int:
            _check_conf_level(conf_level)
            width = ndtri((1 + conf_level) / 2) * se
            cols.update(conf_low=estimate - width, conf_high=estimate + width)
        if exponentiate:
            for name in ("estimate", "conf_low", "conf_high"):
                if name in cols:
                    cols[name] = np.exp(cols[name])
        return self._with_key(pl.DataFrame(cols))

    def _check_vcov(self, vcov):
        if vcov is not None:
            raise ValueError("multinom(): vcov= is for lm/glm; this model supplies its own covariance")

    def vcov_table(self, vcov=None):
        self._check_vcov(vcov)
        return self._with_key(pl.DataFrame({"term": self.parameter_names,
            **{name: self._cov[:, j] for j, name in enumerate(self.parameter_names)}}))

    def log_lik(self):
        return -.5 * self.deviance

    def glance(self):
        return self._with_key(pl.DataFrame({"edf": [float(self.edf)], "deviance": [self.deviance],
            "AIC": [self.aic], "nobs": [self.nobs]}))

    def _prediction(self, newdata, type):
        if type not in ("class", "probs"):
            raise ValueError("multinom(): prediction type must be 'class' or 'probs'")
        if newdata is None:
            probabilities = self.fitted
        else:
            x, vector, _, data = self._new_design(newdata)
            offset = self._offset_matrix(vector, self._matrix_offset(data, None), "predict")
            probabilities = np.exp(_log_probabilities(_predictor(x, self.coef, offset)))
        labels = [str(lv) for lv in self.response_levels]
        classes = pl.DataFrame({"fit": [labels[j] for j in probabilities.argmax(axis=1)]})
        probs = pl.DataFrame({label: probabilities[:, j] for j, label in enumerate(labels)})
        if newdata is None:
            return self._padded(classes), self._padded(probs), self._rows(), self._data
        return classes, probs, data._lf.collect(), data

    def predict(self, newdata=None, *, type="class"):
        classes, probs, _, data = self._prediction(newdata, type)
        return data.write(classes if type == "class" else probs)

    def augment(self, newdata=None):
        classes, probs, rows, data = self._prediction(newdata, "class")
        probs = probs.rename({c: f".prob_{c}" for c in probs.columns})
        out = pl.concat([rows, classes.rename({"fit": ".fitted"}), probs], how="horizontal")
        return out if newdata is None else data.write(out)

    def anova_table(self, test=None):
        raise TypeError("anova(): anova is not implemented for a single multinom model; compare nested models")

    def anova_type(self, kind, test=None):
        return self.anova_table(test)

    def __repr__(self):
        return (f"multinom({self.formula.text})\n{_table_text(self.tidy())}\n"
                f"Residual Deviance: {self.deviance:.6g}  AIC: {self.aic:.6g}")


@step
def multinom(formula, *, weights=None, offset=None, contrasts=None, na_action="omit", maxit=100):
    """Multinomial (baseline-category) logit regression, as nnet::multinom.

    log(P(Y = level k) / P(Y = first level)) = X beta_k, one coefficient
    vector per non-reference level. Set the level order (and so the
    reference) with Enum/category/factor metadata; other responses become
    factors with sorted levels, as in R. The MLE is computed by Newton's
    method on the exact Hessian; standard errors come from that Hessian, as
    ``summary.multinom``. ``tidy`` gives one row per (y_level, term) with
    Wald z tests, as broom. ``offset`` is one column for a two-level
    response, or a list of one column per level (R's ``offset(cbind(...))``).
    Weights, ``contrasts``, ``na_action`` and ``by=`` follow the other
    model functions; aliased columns share nnet's minimum-norm solution.
    """
    if isinstance(maxit, bool) or not isinstance(maxit, (int, np.integer)) or maxit < 1:
        raise ValueError("multinom(): maxit must be a positive integer")
    columns = None
    if isinstance(offset, (list, tuple)):
        columns, offset = list(offset), sum((expression(c).cast(pl.Float64) * 0 for c in offset), pl.lit(0.0))
        if len(columns) < 2:
            raise ValueError("multinom(): a list offset needs one column per response level")
    return _model("multinom", lambda f, fr, d: MultinomialModel(f, fr, d, columns, maxit),
                  formula, weights, offset, contrasts, na_action)


def _anova_multinom(fits, test):
    """nnet's anova.multinom: likelihood-ratio tests of nested fits, ordered by edf."""
    from scipy.stats import chi2

    if test not in (None, "Chisq", "none", False):
        raise ValueError("multinom comparisons use test='Chisq' or 'none'")
    first = fits[0]
    if any(f.response_levels != first.response_levels or f.n != first.n for f in fits[1:]):
        raise ValueError("anova(): models were not all fitted to the same size of dataset")
    fits = sorted(fits, key=lambda f: f.edf)
    df = [None] + [float(b.edf - a.edf) for a, b in zip(fits[:-1], fits[1:])]
    statistic = [None] + [a.deviance - b.deviance for a, b in zip(fits[:-1], fits[1:])]
    cols = {"term": [f.formula.text for f in fits], "df_residual": [float(f.df_residual) for f in fits],
            "residual_deviance": [f.deviance for f in fits], "df": df, "statistic": statistic}
    if test not in ("none", False):
        cols["p_value"] = [None] + [float(chi2.sf(s, d)) for s, d in zip(statistic[1:], df[1:])]
    return pl.DataFrame(cols, schema_overrides={"df": pl.Float64, "statistic": pl.Float64, "p_value": pl.Float64})
