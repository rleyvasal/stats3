"""Exploratory multivariate methods, following base R's stats package.

``prcomp``, ``princomp`` and ``factanal`` reduce numeric columns;
``dist``, ``hclust``, ``cutree`` and ``cmdscale`` work from distances;
``kmeans`` partitions rows. The ports keep R's algorithms step for step
(hclust.f, kmns.f, kmeans.c, distance.c), so ties break as in R.
Eigen and singular vectors come from the same LAPACK routines R calls
(dsyevr, dgesdd); their signs are LAPACK's, which R's reference LAPACK
and the one NumPy links usually share but need not.
"""

from __future__ import annotations

import platform
from typing import Any

import numpy as np
import polars as pl

from stats3._data import Data, Step, is_frame, step
from stats3.core import _group_label, _sorted_keys, _warn
from stats3.formula import variable_expr
from stats3.models import Models, _Fit, _ModelStep, _table_text


# R's compiled sums of squares --------------------------------------------------------------

try:
    from math import fma as _fma
except ImportError:  # Python 3.12
    from fractions import Fraction

    def _fma(x: float, y: float, z: float) -> float:
        """x * y + z rounded once."""
        if not all(np.isfinite((x, y, z))):
            return x * y + z
        return float(Fraction(x) * Fraction(y) + Fraction(z))

# R's C and Fortran accumulate squared differences as ``sum += d * d``.
# Compilers for 64-bit ARM fuse that into one rounding by default; R's
# x86-64 builds target the baseline instruction set, which has no fused
# multiply-add, and round twice. The last bit decides exact ties in hclust
# and kmeans, so stats3 rounds as R's build for this machine does.
R_FUSES_MULTIPLY_ADD = platform.machine().lower() in ("arm64", "aarch64")

if not R_FUSES_MULTIPLY_ADD:
    def _fma(x: float, y: float, z: float) -> float:  # noqa: F811
        """x * y + z rounded twice, as R's x86-64 builds compute it."""
        return x * y + z

_fma_vector = np.frompyfunc(_fma, 3, 1)


# reading columns ------------------------------------------------------------------------


def _terms(text: str) -> list[str]:
    """The top-level ``+`` terms of a formula's right-hand side."""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        depth += c == "("
        depth -= c == ")"
        if c == "+" and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _variables(spec: Any, data: Data, verb: str, skip: list[str]) -> dict[str, pl.Expr]:
    """Column labels and their expressions: a list of columns or ``"~ a + b"``."""
    if isinstance(spec, str):
        lhs, tilde, rhs = spec.partition("~")
        if not tilde:
            labels = [spec.strip()]
        elif lhs.strip():
            raise ValueError(f"{verb}(): response not allowed in formula")
        else:
            labels = _terms(rhs)
    elif isinstance(spec, (list, tuple)):
        labels = list(spec)
    else:
        raise TypeError(f"{verb}(): give the columns as a list or a formula '~ a + b'")
    if labels == ["."]:
        labels = [c for c in data.columns if c not in skip]
    if not labels:
        raise ValueError(f"{verb}(): no columns given")
    out = {}
    for label in labels:
        if isinstance(label, str):
            if label in data.columns:
                out[label] = pl.col(label)
            else:
                expr = variable_expr(label, verb)[0]
                for name in expr.meta.root_names():
                    data.column(name, "column", verb)
                out[label] = expr
        else:
            name = data.column(label, "column", verb)
            out[name] = pl.col(name)
    for g in data._groups:
        if g in out:
            raise ValueError(f"{verb}(): {g!r} is a grouping column; it cannot be analysed")
    return out


class _Part:
    """One group's rows: the numeric matrix, row labels and source rows."""

    def __init__(self, key: dict[str, Any], keyrow: pl.DataFrame, x: np.ndarray, names: list[str],
                 labels: list[Any], rows: pl.DataFrame, label_named: bool):
        self.key, self.keyrow, self.x, self.names = key, keyrow, x, names
        self.labels, self.rows, self.label_named = labels, rows, label_named


def _split(frame: pl.DataFrame, data: Data, verb: str) -> list[tuple[dict[str, Any], pl.DataFrame]]:
    groups = data._groups
    if not groups:
        return [({}, frame)]
    keys = _sorted_keys(data, frame.select(groups).unique(maintain_order=True), groups)
    parts = frame.partition_by(groups, as_dict=True, maintain_order=True)
    return [(dict(zip(groups, key)), parts[tuple(key)]) for key in keys.iter_rows()]


def _read(data: Data, spec: Any, verb: str, labels: Any = None, drop: bool = True,
          kind: str = "the analysis") -> list[_Part]:
    """Each group's numeric matrix. Rows missing any column go, as R's
    ``na.omit`` in the formula methods; ``labels=`` names the rows (R's
    row names), otherwise they are numbered by position in the input."""
    label_col = None if labels is None else data.column(labels, "labels", verb)
    variables = _variables(spec, data, verb, [*data._groups, *([label_col] if label_col else [])])
    names = list(variables)
    schema = data._lf.select(**{f"__v{i}": e for i, e in enumerate(variables.values())}).collect_schema()
    for i, name in enumerate(names):
        dtype = schema[f"__v{i}"]
        if not (dtype.is_numeric() or dtype == pl.Boolean):
            raise ValueError(f"{verb}(): {kind} applies only to numerical variables, not {name!r}")
    frame = data._lf.select(
        pl.int_range(1, pl.len() + 1, dtype=pl.Int64).alias("__row"), pl.all(),
        *[e.cast(pl.Float64).alias(f"__v{i}") for i, e in enumerate(variables.values())],
    ).collect()
    values = [f"__v{i}" for i in range(len(names))]
    if drop:
        frame = frame.filter(pl.all_horizontal(pl.col(v).is_not_null() & pl.col(v).is_not_nan() for v in values))
        if frame.select(pl.any_horizontal(pl.col(v).is_infinite() for v in values).any()).item():
            raise ValueError(f"{verb}(): infinite values in the columns")
    out = []
    for key, part in _split(frame, data, verb):
        x = np.column_stack([part[v].fill_null(np.nan).to_numpy().astype(float) for v in values])
        lab = part[label_col].to_list() if label_col else part["__row"].to_list()
        keyrow = part.select(data._groups).head(1)
        out.append(_Part(key, keyrow, x, names, lab, part.select(data.columns), label_col is not None))
    return out


def _check_flag(value: Any, name: str, verb: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{verb}(): {name} must be TRUE or FALSE")
    return bool(value)


def _eigen(a: np.ndarray, only_values: bool = False):
    """R's ``eigen(symmetric=TRUE)``: LAPACK dsyevr, values decreasing."""
    from scipy.linalg import eigh

    if only_values:
        return eigh(a, lower=True, eigvals_only=True, driver="evr")[::-1]
    values, vectors = eigh(a, lower=True, driver="evr")
    return values[::-1], vectors[:, ::-1]


def _scale(x: np.ndarray, center: Any, scale: Any, verb: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """R's ``scale``: centre by the means (or given values), then divide by
    the root mean square with an n - 1 divisor (or given values)."""
    p = x.shape[1]
    cen = sc = None
    if center is True:
        cen = x.mean(axis=0)
    elif center is not False:
        cen = np.asarray(center, dtype=float)
        if cen.shape != (p,):
            raise ValueError(f"{verb}(): length of 'center' must equal the number of columns of 'x'")
    if cen is not None:
        x = x - cen
    if scale is True:
        sc = np.sqrt((x ** 2).sum(axis=0) / max(1, x.shape[0] - 1))
    elif scale is not False:
        sc = np.asarray(scale, dtype=float)
        if sc.shape != (p,):
            raise ValueError(f"{verb}(): length of 'scale' must equal the number of columns of 'x'")
    if sc is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            x = x / sc
    return x, cen, sc


# fits -----------------------------------------------------------------------------------


class _Multivariate(_Fit):
    """A multivariate fit: its own ``tidy`` matrices, ``glance`` and ``augment``."""

    kind = "multivariate"
    own_tidy = True

    def __init__(self, verb: str, data: Data, part: _Part):
        self.verb, self._data, self.key = verb, data, part.key
        self.groups = list(data._groups)
        self._keyrow, self._part = part.keyrow, part
        self.names, self.labels = part.names, part.labels

    def _with_key(self, frame: pl.DataFrame) -> pl.DataFrame:
        if not self.groups:
            return frame
        keys = self._keyrow
        return pl.concat([pl.concat([keys] * frame.height) if frame.height else keys.clear(), frame],
                         how="horizontal")

    def _label_series(self, name: str, values: list[Any]) -> pl.Series:
        return pl.Series(name, values)

    def _unsupported(self, what: str):
        raise TypeError(f"{what}(): not available for {self.verb}()")

    def glance(self, **kwargs: Any):
        self._unsupported("glance")

    def augment(self, newdata: Any = None, **kwargs: Any):
        self._unsupported("augment")

    def predict(self, newdata: Any = None, **kwargs: Any):
        self._unsupported("predict")

    def vcov_table(self, vcov: Any = None):
        self._unsupported("vcov")

    def anova_table(self, test: Any = None):
        self._unsupported("anova")

    def anova_type(self, kind: int, test: Any = None):
        self._unsupported("anova")

    @property
    def coefficients(self):
        self._unsupported("coefficients")

    def _augmented(self, columns: dict[str, np.ndarray]) -> pl.DataFrame:
        rows = self._part.rows
        return pl.concat([rows, pl.DataFrame(columns)], how="horizontal")

    def _new_matrix(self, newdata: Any) -> tuple[np.ndarray, pl.DataFrame, Data]:
        data = Data.read(newdata, None, "predict")
        exprs = _variables(self.names, data, "predict", [])
        frame = data._lf.collect()
        x = data._lf.select(*[e.cast(pl.Float64).alias(f"__v{i}") for i, e in enumerate(exprs.values())]).collect()
        return np.column_stack([x[c].fill_null(np.nan).to_numpy() for c in x.columns]), frame, data


def _matrix_name(matrix: str, choices: dict[str, tuple[str, ...]], verb: str) -> str:
    for kind, names in choices.items():
        if matrix in names:
            return kind
    allowed = ", ".join(repr(n) for names in choices.values() for n in names)
    raise ValueError(f"tidy(): matrix for {verb}() must be one of {allowed}")


_PCA_MATRICES = {"x": ("x", "samples", "u", "scores"),
                 "rotation": ("rotation", "variables", "v", "loadings"),
                 "pcs": ("pcs", "d", "eigenvalues")}


class _Components(_Multivariate):
    """What prcomp and princomp share: broom's three tidy matrices."""

    def _long(self, label: str, labels: list[Any], matrix: np.ndarray) -> pl.DataFrame:
        n, k = matrix.shape
        return pl.DataFrame({label: [labels[i] for i in range(n) for _ in range(k)],
                             "PC": np.tile(np.arange(1, k + 1, dtype=float), n),
                             "value": matrix.reshape(-1)})

    def tidy(self, matrix: str = "u") -> pl.DataFrame:
        kind = _matrix_name(matrix, _PCA_MATRICES, self.verb)
        if kind == "x":
            if self.scores is None:
                raise ValueError(f"tidy(): this {self.verb}() fit kept no scores")
            out = self._long("row", self.labels, self.scores)
        elif kind == "rotation":
            out = self._long("column", self.names, self.loadings)
        else:
            # summary.prcomp rounds the proportions to five decimals; broom reports them so.
            share = self.sdev ** 2 / np.sum(self.sdev ** 2)
            k = self.loadings.shape[1]
            out = pl.DataFrame({"PC": np.arange(1, k + 1, dtype=float), "std_dev": self.sdev[:k],
                                "percent": [round(float(v), 5) for v in share[:k]],
                                "cumulative": [round(float(v), 5) for v in np.cumsum(share)[:k]]})
        return self._with_key(out)

    def _project(self, x: np.ndarray) -> np.ndarray:
        if self.center is not None:
            x = x - self.center
        if self.scale is not None:
            x = x / self.scale
        return x @ self.loadings

    def predict(self, newdata: Any = None) -> Any:
        prefix = self._component
        if newdata is None:
            if self.scores is None:
                raise ValueError(f"predict(): this {self.verb}() fit kept no scores; give newdata")
            return self._data.write(self._with_key(pl.DataFrame(
                {f"{prefix}{j + 1}": self.scores[:, j] for j in range(self.scores.shape[1])})))
        x, _, data = self._new_matrix(newdata)
        scores = self._project(x)
        return data.write(pl.DataFrame({f"{prefix}{j + 1}": scores[:, j] for j in range(scores.shape[1])}))

    def augment(self, newdata: Any = None) -> pl.DataFrame:
        prefix = f".fitted{self._component}"
        if newdata is not None:
            x, frame, data = self._new_matrix(newdata)
            scores = self._project(x)
            return data.write(pl.concat([frame, pl.DataFrame(
                {f"{prefix}{j + 1}": scores[:, j] for j in range(scores.shape[1])})], how="horizontal"))
        if self.scores is None:
            raise ValueError(f"augment(): this {self.verb}() fit kept no scores")
        return self._augmented({f"{prefix}{j + 1}": self.scores[:, j] for j in range(self.scores.shape[1])})


class PrincipalComponents(_Components):
    """``prcomp``: principal components by the singular value decomposition."""

    _component = "PC"

    def __init__(self, data: Data, part: _Part, center: Any, scale: Any, tol: Any, rank: Any, retx: bool):
        from scipy.linalg import svd

        super().__init__("prcomp", data, part)
        x, self.center, self.scale = _scale(part.x, center, scale, "prcomp")
        if self.scale is not None and np.any(self.scale == 0):
            raise ValueError("prcomp(): cannot rescale a constant/zero column to unit variance")
        n, p = x.shape
        if not np.all(np.isfinite(x)):
            raise ValueError("prcomp(): infinite or missing values in 'x'")
        k = min(n, p) if rank is None else min(rank, n, p)
        _, d, vt = svd(x, full_matrices=False, lapack_driver="gesdd")
        d = d / np.sqrt(max(1, n - 1))
        if tol is not None:
            k = min(k, int(np.sum(d > d[0] * tol)))
        self.sdev, self.loadings = d, vt[:k].T
        self.rotation = self.loadings
        self.scores = x @ self.loadings if retx else None
        self.x = self.scores
        self.n_obs = n

    def __repr__(self) -> str:
        head = f"prcomp(): standard deviations (1, .., p={len(self.sdev)}){_key(self.key)}"
        return f"{head}\n{np.array2string(self.sdev)}\n{_table_text(self.tidy('pcs'))}"


class PrincomponentsML(_Components):
    """``princomp``: eigen decomposition of the covariance with divisor n."""

    _component = "Comp."

    def __init__(self, data: Data, part: _Part, cor: bool, scores: bool, fix_sign: bool):
        super().__init__("princomp", data, part)
        z = part.x
        n, p = z.shape
        if n < p:
            raise ValueError("princomp(): 'princomp' can only be used with more units than variables")
        center = z.mean(axis=0)
        dev = z - center
        cv = dev.T @ dev / (n - 1) * (1 - 1 / n)
        if cor:
            sds = np.sqrt(np.diag(cv))
            if np.any(sds == 0):
                raise ValueError("princomp(): cannot use 'cor = TRUE' with a constant variable")
            cv = cv / np.outer(sds, sds)
        values, vectors = _eigen(cv)
        negative = values < 0
        if np.any(negative):
            if np.any(values[negative] < -9 * np.finfo(float).eps * values[0]):
                raise ValueError("princomp(): covariance matrix is not non-negative definite")
            values[negative] = 0
        if fix_sign:
            vectors = vectors * np.where(vectors[0] < 0, -1.0, 1.0)
        self.sdev, self.loadings, self.center = np.sqrt(values), vectors, center
        self.scale = sds if cor else np.ones(p)
        self.scores = (z - center) / self.scale @ vectors if scores else None
        self.n_obs, self.cor = n, cor

    def __repr__(self) -> str:
        return f"princomp(cor={self.cor}){_key(self.key)}\n{_table_text(self.tidy('pcs'))}"


def _key(key: dict[str, Any]) -> str:
    return "" if not key else "   [" + ", ".join(f"{k} = {v!r}" for k, v in key.items()) + "]"


def _components(verb: str, build, columns: Any, labels: Any) -> _ModelStep:
    def _apply(data: Data) -> Any:
        fits = [build(data, part) for part in _read(data, columns, verb, labels, kind="PCA")]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, verb)


@step
def prcomp(columns: Any = "~ .", *, center: Any = True, scale_: Any = False, tol: float | None = None,
           rank_: int | None = None, retx: bool = True, labels: Any = None) -> _ModelStep:
    """Principal components, as R's ``prcomp``: the SVD of the centred (and
    with ``scale_=True``, scaled) columns, standard deviations with an n - 1
    divisor.

    ``columns`` is a list or ``"~ a + b + c"`` (``"~ ."`` takes every column
    but the groups and ``labels``). ``rank_`` keeps at most that many
    components; ``tol`` drops those with sdev at most ``tol`` times the first.
    ``tidy(fit, matrix=)`` gives broom's tables: ``"x"`` (scores, the
    default ``"u"``), ``"rotation"`` (loadings) and ``"pcs"`` (variance
    explained, proportions rounded to five decimals as ``summary.prcomp``
    does). ``augment`` adds ``.fittedPC1``, ...; ``predict`` projects new rows.
    Rows with missing values are dropped, as R's formula method does.

    A component's sign is whatever LAPACK's ``dgesdd`` returns, as in R;
    a different LAPACK can flip a component (scores and loadings together).
    """
    if rank_ is not None and (isinstance(rank_, bool) or not isinstance(rank_, (int, np.integer)) or rank_ < 1):
        raise ValueError("prcomp(): rank_ must be a positive integer")
    if tol is not None and not np.isfinite(tol):
        raise ValueError("prcomp(): tol must be a finite number")
    _check_flag(retx, "retx", "prcomp")
    return _components("prcomp", lambda d, part: PrincipalComponents(d, part, center, scale_, tol, rank_, retx),
                       columns, labels)


@step
def princomp(columns: Any = "~ .", *, cor: bool = False, scores: bool = True, fix_sign: bool = True,
             labels: Any = None) -> _ModelStep:
    """Principal components, as R's ``princomp``: the eigen decomposition of
    the covariance matrix with divisor n (or with ``cor=True``, the
    correlation matrix). With ``fix_sign=True`` (R's default) each loading's
    first element is non-negative, so the signs are R's. Tables as ``prcomp``;
    ``augment`` adds ``.fittedComp.1``, ...
    """
    for name, value in (("cor", cor), ("scores", scores), ("fix_sign", fix_sign)):
        _check_flag(value, name, "princomp")
    return _components("princomp", lambda d, part: PrincomponentsML(d, part, cor, scores, fix_sign),
                       columns, labels)


# factor analysis ------------------------------------------------------------------------


class Rotation(dict):
    """A rotation's result, as R's list: ``loadings`` and ``rotmat``."""

    @property
    def loadings(self) -> np.ndarray:
        return self["loadings"]

    @property
    def rotmat(self) -> np.ndarray:
        return self["rotmat"]


def _as_loadings(x: Any) -> np.ndarray:
    if is_frame(x):
        frame = Data.read(x, None, "varimax")._lf.collect()
        return frame.select(pl.col(pl.NUMERIC_DTYPES)).cast(pl.Float64).to_numpy()
    return np.array(x, dtype=float, ndmin=2)


def varimax(x: Any, normalize: bool = True, eps: float = 1e-5) -> Rotation:
    """Varimax rotation of a loadings matrix, as R's ``varimax`` (Kaiser
    normalization, stopping when the criterion grows by less than ``eps``)."""
    x = _as_loadings(x)
    nc = x.shape[1]
    if nc < 2:
        return Rotation(loadings=x, rotmat=np.eye(nc))
    if normalize:
        sc = np.sqrt((x ** 2).sum(axis=1))
        x = x / sc[:, None]
    p = x.shape[0]
    tt = np.eye(nc)
    d = 0.0
    for _ in range(1000):
        z = x @ tt
        b = x.T @ (z ** 3 - z @ np.diag((z ** 2).sum(axis=0)) / p)
        u, s, vt = np.linalg.svd(b)
        tt = u @ vt
        dpast, d = d, float(np.sum(s))
        if d < dpast * (1 + eps):
            break
    z = x @ tt
    if normalize:
        z = z * sc[:, None]
    return Rotation(loadings=z, rotmat=tt)


def promax(x: Any, m: float = 4) -> Rotation:
    """Promax rotation, as R's ``promax``: varimax, then a least-squares fit
    to the loadings raised to the power ``m``."""
    x = _as_loadings(x)
    if x.shape[1] < 2:
        return Rotation(loadings=x, rotmat=np.eye(x.shape[1]))
    xx = varimax(x)
    x = xx.loadings
    q = x * np.abs(x) ** (m - 1)
    u = np.linalg.lstsq(x, q, rcond=None)[0]
    d = np.diag(np.linalg.inv(u.T @ u))
    u = u @ np.diag(np.sqrt(d))
    return Rotation(loadings=x @ u, rotmat=xx.rotmat @ u)


def _fa_loadings(psi: np.ndarray, s: np.ndarray, q: int) -> np.ndarray:
    sc = 1 / np.sqrt(psi)
    values, vectors = _eigen(s * np.outer(sc, sc))
    load = vectors[:, :q] * np.sqrt(np.maximum(values[:q] - 1, 0))
    return np.sqrt(psi)[:, None] * load


def _fa_objective(psi: np.ndarray, s: np.ndarray, q: int) -> float:
    sc = 1 / np.sqrt(psi)
    e = _eigen(s * np.outer(sc, sc), only_values=True)[q:]
    return -(float(np.sum(np.log(e) - e)) - q + s.shape[0])


def _fa_gradient(psi: np.ndarray, s: np.ndarray, q: int) -> np.ndarray:
    load = _fa_loadings(psi, s, q)
    g = load @ load.T + np.diag(psi) - s
    return np.diag(g) / psi ** 2


def _fa_fit(cmat: np.ndarray, factors: int, start: np.ndarray, lower: float):
    """R's ``factanal.fit.mle``: L-BFGS-B over the uniquenesses in [lower, 1],
    parameters scaled by 0.01. The optimizer runs to full precision, beyond
    R's default ``factr``; the optimum is the same."""
    from scipy.optimize import minimize

    scale = 0.01
    result = minimize(lambda u: _fa_objective(u * scale, cmat, factors), start / scale,
                      jac=lambda u: _fa_gradient(u * scale, cmat, factors) * scale,
                      method="L-BFGS-B", bounds=[(lower / scale, 1 / scale)] * len(start),
                      options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000, "maxcor": 5})
    psi = result.x * scale
    g = _fa_gradient(psi, cmat, factors)
    free = ~(((psi <= lower) & (g > 0)) | ((psi >= 1) & (g < 0)))
    converged = bool(result.success) or bool(np.max(np.abs(g[free]), initial=0) < 1e-6)
    return psi, float(_fa_objective(psi, cmat, factors)), converged


def _sort_loadings(lam: np.ndarray, rotm: np.ndarray | None = None):
    order = np.argsort(-(lam ** 2).sum(axis=0), kind="stable")
    lam = lam[:, order]
    neg = lam.sum(axis=0) < 0
    unit = np.where(neg, -1.0, 1.0)
    lam = lam * unit
    if rotm is not None:
        rotm = rotm[:, order] * unit
    return lam, rotm


class FactorAnalysis(_Multivariate):
    """``factanal``: maximum-likelihood factor analysis of the correlations."""

    def __init__(self, data: Data, part: _Part, factors: int, scores: str, rotation: Any,
                 start: Any, lower: float):
        from scipy.stats import chi2

        super().__init__("factanal", data, part)
        z = part.x
        n, p = z.shape
        if p < 3:
            raise ValueError("factanal(): factor analysis requires at least three variables")
        dof = 0.5 * ((p - factors) ** 2 - p - factors)
        if dof < 0:
            raise ValueError(f"factanal(): {factors} factor{'s are' if factors > 1 else ' is'} too many "
                             f"for {p} variables")
        dev = z - z.mean(axis=0)
        cv = dev.T @ dev / (n - 1)
        sds = np.sqrt(np.diag(cv))
        cv = cv / np.outer(sds, sds)
        if start is None:
            start = (1 - 0.5 * factors / p) / np.diag(np.linalg.inv(cv))
        start = np.asarray(start, dtype=float)
        if start.shape != (p,):
            raise ValueError(f"factanal(): 'start' must have {p} rows")
        psi, objective, converged = _fa_fit(cv, factors, start, max(lower, 0))
        if not converged:
            raise ValueError("factanal(): unable to optimize from this starting value")
        load, _ = _sort_loadings(_fa_loadings(psi, cv, factors))
        rotmat = None
        if rotation != "none":
            rot = {"varimax": varimax, "promax": promax}.get(rotation, rotation)(load)
            if isinstance(rot, dict):
                load, rotmat = np.asarray(rot["loadings"]), np.asarray(rot["rotmat"])
            else:
                load = np.asarray(rot)
        load, rotmat = _sort_loadings(load, rotmat)
        self.loadings, self.rotmat, self.uniquenesses = load, rotmat, psi
        self.correlation, self.factors, self.dof, self.n_obs = cv, factors, dof, n
        self.criteria = {"objective": objective}
        self.converged, self.method, self.rotation = converged, "mle", rotation
        self.statistic = self.p_value = None
        if dof > 0:
            self.statistic = (n - 1 - (2 * p + 5) / 6 - (2 * factors) / 3) * objective
            self.p_value = float(chi2.sf(self.statistic, dof))
        self.scores = None
        if scores != "none":
            zz = dev / np.sqrt((dev ** 2).sum(axis=0) / (n - 1))
            if scores == "regression":
                self.scores = zz @ np.linalg.solve(cv, load)
            else:
                tmp = (load / psi[:, None]).T
                self.scores = np.linalg.solve(tmp @ load, tmp @ zz.T).T

    def tidy(self) -> pl.DataFrame:
        cols = {"variable": self.names, "uniqueness": self.uniquenesses}
        cols.update({f"fl{j + 1}": self.loadings[:, j] for j in range(self.factors)})
        return self._with_key(pl.DataFrame(cols))

    def glance(self) -> pl.DataFrame:
        total = float(np.sum((self.loadings ** 2).sum(axis=0) / self.loadings.shape[0]))
        return self._with_key(pl.DataFrame({
            "n_factors": [self.factors], "total_variance": [total], "statistic": [self.statistic],
            "p_value": [self.p_value], "df": [self.dof], "n": [self.n_obs], "method": [self.method],
            "converged": [self.converged], "nobs": [self.n_obs]},
            schema_overrides={"statistic": pl.Float64, "p_value": pl.Float64, "df": pl.Float64}))

    def augment(self, newdata: Any = None) -> pl.DataFrame:
        if newdata is not None:
            raise TypeError("augment(): factanal() gives scores for its own rows only")
        if self.scores is None:
            raise ValueError('augment(): fit factanal() with scores="regression" or "Bartlett"')
        return self._augmented({f".fs{j + 1}": self.scores[:, j] for j in range(self.factors)})

    def __repr__(self) -> str:
        test = "" if self.statistic is None else (
            f"\nchi square statistic {self.statistic:.6g} on {self.dof:g} df, p = {self.p_value:.4g}")
        return f"factanal(factors={self.factors}, rotation={self.rotation!r}){_key(self.key)}\n" \
               f"{_table_text(self.tidy())}{test}"


@step
def factanal(columns: Any = "~ .", factors: int = 1, *, scores: str = "none", rotation: Any = "varimax",
             start: Any = None, lower: float = 0.005, labels: Any = None) -> _ModelStep:
    """Maximum-likelihood factor analysis, as R's ``factanal``.

    R's fit: L-BFGS-B over the uniquenesses (bounded by ``lower`` and 1)
    from R's start, loadings ordered by their sums of squares with
    positive column sums, then ``rotation`` (``"varimax"``, ``"promax"``,
    ``"none"`` or a function returning a matrix or ``{"loadings", "rotmat"}``).
    ``tidy`` gives broom's ``variable``, ``uniqueness``, ``fl1``, ...;
    ``glance`` the chi-squared test that the factors suffice
    (Bartlett's correction, as R); ``augment`` the ``scores=`` (``"regression"``
    or ``"Bartlett"``) as ``.fs1``, .... Rows with missing values are dropped.
    """
    if isinstance(factors, bool) or not isinstance(factors, (int, np.integer)) or factors < 1:
        raise ValueError("factanal(): factors must be a positive integer")
    if scores not in ("none", "regression", "Bartlett"):
        raise ValueError("factanal(): scores must be 'none', 'regression' or 'Bartlett'")
    if not callable(rotation) and rotation not in ("varimax", "promax", "none"):
        raise ValueError("factanal(): rotation must be 'varimax', 'promax', 'none' or a function")

    def _apply(data: Data) -> Any:
        fits = [FactorAnalysis(data, part, factors, scores, rotation, start, lower)
                for part in _read(data, columns, "factanal", labels, kind="factor analysis")]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, "factanal")


# distances ------------------------------------------------------------------------------


_DIST_METHODS = ("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski")


def _match(value: str, choices: tuple[str, ...], what: str, verb: str) -> str:
    hits = [c for c in choices if c.startswith(value)] if isinstance(value, str) else []
    if value in choices:
        return value
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"{verb}(): {'ambiguous' if hits else 'invalid'} {what} {value!r}")


def _distances(x: np.ndarray, method: str, p: float) -> np.ndarray:
    """R's distance.c for every pair: columns missing in either row are
    skipped and the sum scaled up by their share."""
    n, nc = x.shape
    out = np.zeros((n, n))
    for i in range(n):
        a = x[i]
        b = x[i + 1:]
        ok = ~(np.isnan(a) | np.isnan(b))
        if method == "binary":
            finite = ok & np.isfinite(a) & np.isfinite(b)
            if np.any(ok & ~finite):
                _warn("dist(): treating non-finite values as NA")
            nonzero = finite & ((a != 0) | (b != 0))
            count = nonzero.sum(axis=1)
            differ = (nonzero & ~((a != 0) & (b != 0))).sum(axis=1)
            total = finite.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                d = np.where(count == 0, 0.0, differ / np.maximum(count, 1))
            d = np.where(total == 0, np.nan, d)
        else:
            with np.errstate(invalid="ignore"):
                dev = np.where(ok, a - b, np.nan)
            if method == "canberra":
                with np.errstate(invalid="ignore", divide="ignore"):
                    total = np.where(ok, np.abs(a) + np.abs(b), np.nan)
                    diff = np.abs(dev)
                    term = diff / total
                    term = np.where(~np.isfinite(diff) & (diff == total), 1.0, term)
                tiny = np.finfo(float).tiny
                ok = ok & ((total > tiny) | (diff > tiny)) & ~np.isnan(term)
                vals = np.where(ok, term, 0.0)
            else:
                ok = ok & ~np.isnan(dev)
                vals = np.where(ok, dev, 0.0)
            count = ok.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                if method == "euclidean":
                    terms = vals * vals
                elif method == "manhattan" or method == "canberra":
                    terms = np.abs(vals)
                elif method == "minkowski":
                    terms = np.where(ok, np.abs(vals) ** p, 0.0)
                if method == "maximum":
                    d = np.where(ok, np.abs(vals), -np.inf).max(axis=1, initial=-np.inf)
                else:
                    # Column by column, as distance.c adds them.
                    d = np.zeros(len(b))
                    for j in range(nc):
                        if method == "euclidean":
                            d = _fma_vector(vals[:, j], vals[:, j], d).astype(float)
                        else:
                            d = d + terms[:, j]
                if method != "maximum":
                    d = np.where(count != nc, d / (count / nc), d)
                if method == "euclidean":
                    d = np.sqrt(d)
                elif method == "minkowski":
                    d = d ** (1.0 / p)
            d = np.where(count == 0, np.nan, d)
        out[i, i + 1:] = d
        out[i + 1:, i] = d
    return out


def _dist_table(labels: list[Any], d: np.ndarray) -> pl.DataFrame:
    """broom's ``tidy.dist`` layout: the lower triangle, row by row."""
    n = len(labels)
    i, j = np.tril_indices(n, -1)
    return pl.DataFrame({"item1": [labels[a] for a in i], "item2": [labels[b] for b in j],
                         "distance": d[i, j]}, schema_overrides={"distance": pl.Float64})


@step
def dist(columns: Any = "~ .", *, method: str = "euclidean", p: float = 2, labels: Any = None) -> Any:
    """Distances between rows, as R's ``dist``, as a table: ``item1``,
    ``item2``, ``distance`` for each pair, in broom's ``tidy.dist`` order.

    Methods: ``euclidean``, ``maximum``, ``manhattan``, ``canberra``,
    ``binary`` and ``minkowski`` (with power ``p``). Items are the
    ``labels=`` column, else the rows' positions in the input (R's row
    names). As in R, a missing value leaves out that column for the pairs
    it is in, scaling up the rest. Pipe the table into ``hclust`` or
    ``cmdscale``; grouping columns travel with it.
    """
    method = _match(method, _DIST_METHODS, "distance method", "dist")
    if method == "minkowski" and (not np.isfinite(p) or p <= 0):
        raise ValueError("dist(): distance(): invalid p")

    def _apply(data: Data) -> pl.DataFrame:
        frames = []
        for part in _read(data, columns, "dist", labels, drop=False, kind="dist"):
            table = _dist_table(part.labels, _distances(part.x, method, p))
            if data._groups:
                keys = part.keyrow
                table = pl.concat([pl.concat([keys] * table.height) if table.height else keys.clear(), table],
                                  how="horizontal")
            frames.append(table)
        return pl.concat(frames, how="diagonal_relaxed")

    return Step(_apply, "dist")


def _read_dist(data: Data, verb: str) -> list[tuple[_Part, np.ndarray]]:
    """Each group's distance matrix from a ``dist`` table. Columns other
    than item1, item2 and distance group the table when no ``by=`` is given."""
    for name in ("item1", "item2", "distance"):
        if name not in data.columns:
            raise ValueError(f"{verb}(): expected a distance table with item1, item2 and distance "
                             f"columns, as dist() returns")
    if not data._groups:
        data._groups = [c for c in data.columns if c not in ("item1", "item2", "distance")]
    frame = data._lf.collect()
    out = []
    for key, part in _split(frame, data, verb):
        item1, item2 = part["item1"].to_list(), part["item2"].to_list()
        # broom's order: item 1 heads the first row's item2, items 2..n follow as item1.
        labels = list(dict.fromkeys([*item2[:1], *item1]))
        n = len(labels)
        if part.height != n * (n - 1) // 2:
            raise ValueError(f"{verb}(): dissimilarities of improper length{_group_label(key, data._groups)}")
        at = {v: i for i, v in enumerate(labels)}
        if any(v not in at for v in item2) or len(set(zip(item1, item2))) != part.height:
            raise ValueError(f"{verb}(): the distance table must hold each pair of items once")
        d = np.zeros((n, n))
        a = np.array([at[v] for v in item1], dtype=int)
        b = np.array([at[v] for v in item2], dtype=int)
        values = part["distance"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        d[a, b] = values
        d[b, a] = values
        keyrow = part.select(data._groups).head(1)
        out.append((_Part(key, keyrow, d, [], labels, part, True), d))
    return out


# hierarchical clustering ----------------------------------------------------------------


_HCLUST_METHODS = ("ward.D", "single", "complete", "average", "mcquitty", "median", "centroid", "ward.D2")


def _hclust(d: np.ndarray, iopt: int, members: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F. Murtagh's HCLUST and HCASS2, as R's hclust.f: nearest neighbours
    chained by the Lance-Williams update, with R's strict-inequality ties."""
    n = d.shape[0]
    inf = 1e300
    diss = d.copy()
    if iopt == 8:
        diss = diss * diss
    membr = members.astype(float).copy()
    flag = np.ones(n, dtype=bool)
    nn = np.zeros(n, dtype=int)
    disnn = np.zeros(n)
    ia = np.zeros(n - 1, dtype=int)
    ib = np.zeros(n - 1, dtype=int)
    crit = np.zeros(n - 1)
    jm = jj = 0
    for i in range(n - 1):
        row = diss[i, i + 1:]
        j = int(np.argmin(row))
        if row[j] < inf:
            jm = i + 1 + j
        nn[i], disnn[i] = jm, min(inf, row[j])
    ward = iopt in (1, 8)
    for step_ in range(n - 1):
        live = np.where(flag[:n - 1], disnn[:n - 1], np.inf)
        im = int(np.argmin(live))
        jm = nn[im]
        i2, j2 = min(im, jm), max(im, jm)
        ia[step_], ib[step_] = i2, j2
        crit[step_] = np.sqrt(live[im]) if iopt == 8 else live[im]
        flag[j2] = False
        ks = np.flatnonzero(flag)
        ks = ks[ks != i2]
        d1, d2, d12 = diss[i2, ks], diss[j2, ks], diss[i2, j2]
        mi, mj, mk = membr[i2], membr[j2], membr[ks]
        if ward:
            new = ((mi + mk) * d1 + (mj + mk) * d2 - mk * d12) / (mi + mj + mk)
        elif iopt == 2:
            new = np.minimum(d1, d2)
        elif iopt == 3:
            new = np.maximum(d1, d2)
        elif iopt == 4:
            new = (mi * d1 + mj * d2) / (mi + mj)
        elif iopt == 5:
            new = (d1 + d2) / 2
        elif iopt == 6:
            new = ((d1 + d2) - d12 / 2) / 2
        else:
            new = (mi * d1 + mj * d2 - mi * mj * d12 / (mi + mj)) / (mi + mj)
        diss[i2, ks] = new
        diss[ks, i2] = new
        after = ks > i2
        dmin = inf
        if np.any(after):
            values = new[after]
            j = int(np.argmin(values))
            if values[j] < dmin:
                dmin, jj = values[j], int(ks[after][j])
        before = ks[~after]
        closer = new[~after] < disnn[before]
        disnn[before[closer]] = new[~after][closer]
        nn[before[closer]] = i2
        membr[i2] = mi + mj
        disnn[i2], nn[i2] = dmin, jj
        for i in range(n - 1):
            if flag[i] and (nn[i] == i2 or nn[i] == j2):
                dmin = inf
                rest = np.flatnonzero(flag[i + 1:]) + i + 1
                if len(rest):
                    values = diss[i, rest]
                    j = int(np.argmin(values))
                    if values[j] < dmin:
                        dmin, jj = values[j], int(rest[j])
                nn[i], disnn[i] = jj, dmin
    return _hcass2(n, ia + 1, ib + 1) + (crit,)


def _hcass2(n: int, ia: np.ndarray, ib: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R's merge matrix (negative singletons, positive earlier merges) and
    the dendrogram's leaf order."""
    iia, iib = ia.astype(int).tolist(), ib.astype(int).tolist()
    ia, ib = list(iia), list(iib)
    for i in range(n - 2):
        k = min(ia[i], ib[i])
        for j in range(i + 1, n - 1):
            if ia[j] == k:
                iia[j] = -(i + 1)
            if ib[j] == k:
                iib[j] = -(i + 1)
    iia = [-v for v in iia]
    iib = [-v for v in iib]
    for i in range(n - 1):
        if iia[i] > 0 and iib[i] < 0:
            iia[i], iib[i] = iib[i], iia[i]
        if iia[i] > 0 and iib[i] > 0:
            iia[i], iib[i] = min(iia[i], iib[i]), max(iia[i], iib[i])
    order = [iia[n - 2], iib[n - 2]]
    for i in range(n - 2, 0, -1):
        for j in range(len(order)):
            if order[j] == i:
                order[j] = iia[i - 1]
                order.insert(j + 1, iib[i - 1])
                break
    return np.column_stack([iia, iib]), -np.array(order, dtype=int)


class Hierarchical(_Multivariate):
    """``hclust``: R's merge matrix, heights and leaf order."""

    def __init__(self, data: Data, part: _Part, d: np.ndarray, method: str, members: Any):
        super().__init__("hclust", data, part)
        n = d.shape[0]
        if n < 2:
            raise ValueError("hclust(): must have n >= 2 objects to cluster")
        if not np.all(np.isfinite(d)):
            raise ValueError("hclust(): NA/NaN/Inf in the dissimilarities")
        members = np.ones(n) if members is None else np.asarray(members, dtype=float)
        if members.shape != (n,):
            raise ValueError("hclust(): invalid length of members")
        self.method = method
        self.merge, self.order, self.height = _hclust(d, _HCLUST_METHODS.index(method) + 1, members)

    def tidy(self, matrix: str = "merge") -> pl.DataFrame:
        kind = _matrix_name(matrix, {"merge": ("merge",), "order": ("order",)}, "hclust")
        if kind == "merge":
            out = pl.DataFrame({"step": np.arange(1, len(self.height) + 1), "merge1": self.merge[:, 0],
                                "merge2": self.merge[:, 1], "height": self.height})
        else:
            out = pl.DataFrame({"position": np.arange(1, len(self.order) + 1), "order": self.order,
                                "label": [self.labels[i - 1] for i in self.order]})
        return self._with_key(out)

    def cutree(self, k: Any = None, h: Any = None) -> pl.DataFrame:
        n = len(self.labels)
        if k is None:
            if np.any(np.diff(self.height) < 0):
                raise ValueError("cutree(): the 'height' component of 'tree' is not sorted (increasingly)")
            k = n + 1 - (int(np.argmax(np.r_[self.height, np.inf] > h)) + 1)
        elif isinstance(k, bool) or int(k) != k or not 1 <= k <= n:
            raise ValueError(f"cutree(): elements of 'k' must be between 1 and {n}")
        k = int(k)
        cluster = list(range(n))
        made: list[list[int]] = []
        for a, b in self.merge[:n - k]:
            members = (made[a - 1] if a > 0 else [-a - 1]) + (made[b - 1] if b > 0 else [-b - 1])
            made.append(members)
            for m in members:
                cluster[m] = members[0]
        number: dict[int, int] = {}
        out = [number.setdefault(c, len(number) + 1) for c in cluster]
        return self._with_key(pl.DataFrame({"item": self.labels, "cluster": out}))

    def __repr__(self) -> str:
        return f"hclust(method={self.method!r}): {len(self.labels)} objects{_key(self.key)}"


@step
def hclust(method: str = "complete", *, members: Any = None) -> _ModelStep:
    """Hierarchical clustering of a ``dist`` table, as R's ``hclust``:
    ``complete``, ``single``, ``average``, ``mcquitty``, ``ward.D``,
    ``ward.D2``, ``centroid`` and ``median`` (Lance-Williams updates on the
    dissimilarities, as R; centroid and median expect squared Euclidean
    distances). Gives R's merge steps, heights and leaf order exactly,
    ties included: ``tidy(fit)`` is the merge table (``merge1``/``merge2``
    negative for single items, positive for earlier steps), and
    ``tidy(fit, matrix="order")`` the order. ``cutree(fit, k=)`` or ``h=``
    cuts it.
    """
    if method == "ward":
        _warn('hclust(): The "ward" method has been renamed to "ward.D"; note new "ward.D2"')
        method = "ward.D"
    method = _match(method, _HCLUST_METHODS, "clustering method", "hclust")

    def _apply(data: Data) -> Any:
        fits = [Hierarchical(data, part, d, method, members) for part, d in _read_dist(data, "hclust")]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, "hclust")


def cutree(tree: Any, k: int | None = None, h: float | None = None) -> Any:
    """Cluster memberships from an ``hclust`` fit, as R's ``cutree``: ``k``
    groups, or the cut at height ``h``. Clusters are numbered by the order
    of their first item, as in R."""
    if k is None and h is None:
        raise ValueError("cutree(): either 'k' or 'h' must be specified")
    fits = tree.fits if isinstance(tree, Models) else [tree]
    if not all(isinstance(f, Hierarchical) for f in fits):
        raise TypeError("cutree(): expected an hclust() fit")
    frames = [f.cutree(k, h) for f in fits]
    return tree._stack(frames) if isinstance(tree, Models) else fits[0]._write(frames[0])


# classical scaling ----------------------------------------------------------------------


class Scaling(_Multivariate):
    """``cmdscale``: classical (Torgerson-Gower) multidimensional scaling."""

    def __init__(self, data: Data, part: _Part, d: np.ndarray, k: int, eig: bool):
        super().__init__("cmdscale", data, part)
        n = d.shape[0]
        if np.any(np.isnan(d)):
            raise ValueError("cmdscale(): NA values not allowed in 'd'")
        if k > n - 1 or k < 1:
            raise ValueError("cmdscale(): 'k' must be in {1, 2, ..  n - 1}")
        x = d ** 2
        x = x - x.mean(axis=1, keepdims=True)
        x = x - x.mean(axis=0, keepdims=True)
        values, vectors = _eigen(-x / 2)
        ev, evec = values[:k], vectors[:, :k]
        positive = ev > 0
        if positive.sum() < k:
            _warn(f"cmdscale(): only {positive.sum()} of the first {k} eigenvalues are > 0")
            ev, evec = ev[positive], evec[:, positive]
        self.points = evec * np.sqrt(ev)
        self.eig = values if eig else None
        self.ac = 0.0
        self.gof = (float(np.sum(ev) / np.sum(np.abs(values))), float(np.sum(ev) / np.sum(np.maximum(values, 0))))

    def tidy(self, matrix: str = "points") -> pl.DataFrame:
        kind = _matrix_name(matrix, {"points": ("points",), "eig": ("eig",)}, "cmdscale")
        if kind == "points":
            out = pl.DataFrame({"item": self.labels,
                                **{f"dim{j + 1}": self.points[:, j] for j in range(self.points.shape[1])}})
        else:
            if self.eig is None:
                raise ValueError("tidy(): fit cmdscale() with eig=True for the eigenvalues")
            out = pl.DataFrame({"dim": np.arange(1, len(self.eig) + 1), "eigenvalue": self.eig})
        return self._with_key(out)

    def glance(self) -> pl.DataFrame:
        return self._with_key(pl.DataFrame({"k": [self.points.shape[1]], "gof1": [self.gof[0]],
                                            "gof2": [self.gof[1]], "ac": [self.ac]}))

    def __repr__(self) -> str:
        return f"cmdscale(k={self.points.shape[1]}){_key(self.key)}\n{_table_text(self.tidy())}"


@step
def cmdscale(k: int = 2, *, eig: bool = True) -> _ModelStep:
    """Classical multidimensional scaling of a ``dist`` table, as R's
    ``cmdscale``: the leading eigenvectors of the doubly centred squared
    distances, scaled by the root eigenvalues. ``tidy(fit)`` gives the
    points (``item``, ``dim1``, ...), ``tidy(fit, matrix="eig")`` every
    eigenvalue, and ``glance`` R's ``GOF`` pair. An axis's sign is LAPACK's,
    as in R. The additive constant (R's ``add=TRUE``) is not offered.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("cmdscale(): k must be an integer")
    _check_flag(eig, "eig", "cmdscale")

    def _apply(data: Data) -> Any:
        fits = [Scaling(data, part, d, int(k), eig) for part, d in _read_dist(data, "cmdscale")]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, "cmdscale")


# k-means --------------------------------------------------------------------------------


_BIG = float(np.float32(1e30))  # kmns.f's REAL constant BIG /1.E30/


def _hartigan_wong(a: np.ndarray, c: np.ndarray, iter_max: int):
    """AS 136 as R's kmns.f: optimal-transfer and quick-transfer stages."""
    m, n = a.shape
    k = c.shape[0]
    c = c.copy()
    if k <= 1 or k >= m:
        return None, None, None, None, 3
    ic1 = np.zeros(m, dtype=int)
    ic2 = np.zeros(m, dtype=int)
    for i in range(m):
        dt = [_sq(a[i], c[0]), _sq(a[i], c[1])]
        ic1[i], ic2[i] = 0, 1
        if dt[0] > dt[1]:
            ic1[i], ic2[i] = 1, 0
            dt = [dt[1], dt[0]]
        for l_ in range(2, k):
            db = _sq(a[i], c[l_], dt[1])
            if db is None:
                continue
            if db >= dt[0]:
                dt[1], ic2[i] = db, l_
            else:
                dt[1], ic2[i] = dt[0], ic1[i]
                dt[0], ic1[i] = db, l_
    nc = np.bincount(ic1, minlength=k)
    if np.any(nc == 0):
        return None, None, None, None, 1
    c = np.zeros((k, n))
    for i in range(m):
        c[ic1[i]] += a[i]
    c = c / nc[:, None].astype(float)
    nc = nc.astype(int)
    an2 = nc / (nc + 1.0)
    an1 = np.where(nc > 1, nc / np.maximum(nc - 1.0, 1), _BIG)
    itran = np.ones(k, dtype=int)
    ncp = np.full(k, -1, dtype=int)
    live = np.zeros(k, dtype=int)
    d = np.zeros(m)
    state = {"indx": 0, "maxqtr": int(min(2 ** 31 - 1, 50 * m))}
    ifault = 0
    iterations = iter_max + 1
    for ij in range(1, iter_max + 1):
        _optra(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, live, state)
        if state["indx"] == m:
            iterations = ij
            break
        _qtran(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, state)
        if state["maxqtr"] < 0:
            ifault, iterations = 4, ij
            break
        if k == 2:
            iterations = ij
            break
        ncp[:] = 0
    else:
        ifault = 2
    c = np.zeros((k, n))
    for i in range(m):
        c[ic1[i]] += a[i]
    c = c / nc[:, None].astype(float)
    wss = np.zeros(k)
    for j in range(n):
        for i in range(m):
            da = a[i, j] - c[ic1[i], j]
            wss[ic1[i]] = _fma(da, da, wss[ic1[i]])
    return ic1, c, wss, iterations, ifault


def _sq(x: np.ndarray, y: np.ndarray, stop: float | None = None) -> float | None:
    """Squared distance summed in column order, fused as R's compiled code;
    None once it reaches ``stop``."""
    total = 0.0
    for u, v in zip(x.tolist(), y.tolist()):
        dv = u - v
        total = _fma(dv, dv, total)
        if stop is not None and total >= stop:
            return None
    return total


def _transfer(a, c, ic1, ic2, nc, an1, an2, i, l1, l2):
    al1 = float(nc[l1])
    alw = al1 - 1.0
    al2 = float(nc[l2])
    alt = al2 + 1.0
    c[l1] = (c[l1] * al1 - a[i]) / alw
    c[l2] = (c[l2] * al2 + a[i]) / alt
    nc[l1] -= 1
    nc[l2] += 1
    an2[l1] = alw / al1
    an1[l1] = alw / (alw - 1.0) if alw > 1.0 else _BIG
    an1[l2] = alt / al2
    an2[l2] = alt / (alt + 1.0)
    ic1[i], ic2[i] = l2, l1


def _optra(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, live, state):
    m, k = a.shape[0], c.shape[0]
    for l_ in range(k):
        if itran[l_] == 1:
            live[l_] = m + 1
    for i in range(m):
        pos = i + 1
        state["indx"] += 1
        l1, l2 = ic1[i], ic2[i]
        ll = l2
        if nc[l1] != 1:
            if ncp[l1] != 0:
                d[i] = _sq(a[i], c[l1]) * an1[l1]
            r2 = _sq(a[i], c[l2]) * an2[l2]
            for l_ in range(k):
                if (pos >= live[l1] and pos >= live[l_]) or l_ == l1 or l_ == ll:
                    continue
                rr = r2 / an2[l_]
                dc = _sq(a[i], c[l_], rr)
                if dc is None:
                    continue
                r2, l2 = dc * an2[l_], l_
            if r2 >= d[i]:
                ic2[i] = l2
            else:
                state["indx"] = 0
                live[l1] = m + pos
                live[l2] = m + pos
                ncp[l1] = pos
                ncp[l2] = pos
                _transfer(a, c, ic1, ic2, nc, an1, an2, i, l1, l2)
        if state["indx"] == m:
            return
    for l_ in range(k):
        itran[l_] = 0
        live[l_] -= m


def _qtran(a, c, ic1, ic2, nc, an1, an2, ncp, d, itran, state):
    m = a.shape[0]
    icoun = istep = 0
    while True:
        for i in range(m):
            icoun += 1
            istep += 1
            if istep >= state["maxqtr"]:
                state["maxqtr"] = -1
                return
            l1, l2 = ic1[i], ic2[i]
            if nc[l1] != 1:
                if istep <= ncp[l1]:
                    d[i] = _sq(a[i], c[l1]) * an1[l1]
                if istep < ncp[l1] or istep < ncp[l2]:
                    r2 = d[i] / an2[l2]
                    if _sq(a[i], c[l2], r2) is not None:
                        icoun = 0
                        state["indx"] = 0
                        itran[l1] = itran[l2] = 1
                        ncp[l1] = ncp[l2] = istep + m
                        _transfer(a, c, ic1, ic2, nc, an1, an2, i, l1, l2)
            if icoun == m:
                return


def _nearest(a: np.ndarray, c: np.ndarray, i: int, previous: int) -> int:
    best, choice = np.inf, previous
    for j in range(c.shape[0]):
        dd = _sq(a[i], c[j])
        if dd < best:
            best, choice = dd, j
    return choice


def _wss(a: np.ndarray, c: np.ndarray, cl: np.ndarray) -> np.ndarray:
    wss = np.zeros(c.shape[0])
    for i in range(a.shape[0]):
        wss[cl[i]] += _sq(a[i], c[cl[i]])
    return wss


def _lloyd(a: np.ndarray, c: np.ndarray, iter_max: int):
    """R's kmeans_Lloyd: reassign every point, then move every centre."""
    m, k = a.shape[0], c.shape[0]
    c = c.copy()
    cl = np.full(m, -1)
    nc = np.zeros(k, dtype=int)
    inew = 0
    iteration = 0
    for iteration in range(iter_max):
        updated = False
        for i in range(m):
            inew = _nearest(a, c, i, inew)
            if cl[i] != inew:
                updated, cl[i] = True, inew
        if not updated:
            break
        nc = np.bincount(cl, minlength=k)
        sums = np.zeros_like(c)
        for i in range(m):
            sums[cl[i]] += a[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            c = sums / nc[:, None]
    else:
        iteration = iter_max
    return cl, c, _wss(a, c, cl), iteration + 1, nc


def _macqueen(a: np.ndarray, c: np.ndarray, iter_max: int):
    """R's kmeans_MacQueen: move the two centres after each reassignment."""
    m, k = a.shape[0], c.shape[0]
    c = c.copy()
    cl = np.zeros(m, dtype=int)
    inew = 0
    for i in range(m):
        inew = _nearest(a, c, i, inew)
        cl[i] = inew
    nc = np.bincount(cl, minlength=k)
    sums = np.zeros_like(c)
    for i in range(m):
        sums[cl[i]] += a[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = sums / nc[:, None]
    iteration = 0
    for iteration in range(iter_max):
        updated = False
        for i in range(m):
            inew = _nearest(a, c, i, inew)
            old = cl[i]
            if old != inew:
                updated = True
                cl[i] = inew
                nc[old] -= 1
                nc[inew] += 1
                c[old] += (c[old] - a[i]) / nc[old]
                c[inew] += (a[i] - c[inew]) / nc[inew]
        if not updated:
            break
    else:
        iteration = iter_max
    return cl, c, _wss(a, c, cl), iteration + 1, nc


_KMEANS_ALGORITHMS = ("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen")


def _kmeans_once(a: np.ndarray, centers: np.ndarray, iter_max: int, method: int):
    if method == 1:
        cl, c, wss, iterations, ifault = _hartigan_wong(a, centers, iter_max)
        if ifault == 1:
            raise ValueError("kmeans(): empty cluster: try a better set of initial centers")
        if ifault == 3:
            raise ValueError("kmeans(): number of cluster centres must lie between 1 and nrow(x)")
        if ifault == 2:
            iterations = max(iterations, iter_max + 1)
        if ifault == 4:
            _warn(f"kmeans(): Quick-TRANSfer stage steps exceeded maximum (= {min(2 ** 31 - 1, 50 * a.shape[0])})")
        nc = np.bincount(cl, minlength=centers.shape[0])
    else:
        cl, c, wss, iterations, nc = (_lloyd if method == 2 else _macqueen)(a, centers, iter_max)
        ifault = 0
        if np.any(nc == 0):
            _warn("kmeans(): empty cluster: try a better set of initial centers")
    if iterations > iter_max:
        _warn(f"kmeans(): did not converge in {iter_max} iteration{'s' if iter_max != 1 else ''}")
        if method in (2, 3):
            ifault = 2
    return {"cluster": cl, "centers": c, "wss": wss, "iter": iterations, "size": nc, "ifault": ifault}


class KMeans(_Multivariate):
    """``kmeans``: a partition of the rows around ``k`` centres."""

    def __init__(self, data: Data, part: _Part, centers: Any, iter_max: int, nstart: int,
                 algorithm: str, rng: np.random.Generator):
        super().__init__("kmeans", data, part)
        a = part.x
        m = a.shape[0]
        method = {"Hartigan-Wong": 1, "Lloyd": 2, "Forgy": 2, "MacQueen": 3}[algorithm]
        unique = None
        if np.ndim(centers) == 0:
            k = int(centers)
            if k < 1:
                raise ValueError("kmeans(): number of cluster centres must lie between 1 and nrow(x)")
            start = None
            if nstart == 1:
                if k > m:
                    raise ValueError("kmeans(): more cluster centers than data points")
                start = a[rng.choice(m, k, replace=False)]
            if nstart >= 2 or _duplicated(start):
                unique = a[np.sort(np.unique(a, axis=0, return_index=True)[1])]
                if unique.shape[0] < k:
                    raise ValueError("kmeans(): more cluster centers than distinct data points.")
                start = unique[rng.choice(unique.shape[0], k, replace=False)]
        else:
            start = np.array(centers, dtype=float, ndmin=2)
            if _duplicated(start):
                raise ValueError("kmeans(): initial centers are not distinct")
            k = start.shape[0]
            if m < k:
                raise ValueError("kmeans(): more cluster centers than data points")
        if k == 1:
            method = 3
        if start.shape[1] != a.shape[1]:
            raise ValueError("kmeans(): must have same number of columns in 'x' and 'centers'")
        best = _kmeans_once(a, start, iter_max, method)
        if nstart >= 2 and unique is not None:
            for _ in range(1, nstart):
                start = unique[rng.choice(unique.shape[0], k, replace=False)]
                other = _kmeans_once(a, start, iter_max, method)
                if other["wss"].sum() < best["wss"].sum():
                    best = other
        self.cluster = best["cluster"] + 1
        self.centers, self.withinss = best["centers"], best["wss"]
        self.size, self.iter, self.ifault = best["size"], best["iter"], best["ifault"]
        self.tot_withinss = float(np.sum(self.withinss))
        self.totss = float(np.sum((a - a.mean(axis=0)) ** 2))
        self.betweenss = self.totss - self.tot_withinss
        self.algorithm = algorithm

    def _clusters(self, values) -> pl.Series:
        levels = [str(j) for j in range(1, self.centers.shape[0] + 1)]
        return pl.Series([str(v) for v in values], dtype=pl.Enum(levels))

    def tidy(self) -> pl.DataFrame:
        cols = {name: self.centers[:, j] for j, name in enumerate(self.names)}
        cols.update(size=self.size.astype(np.int64), withinss=self.withinss,
                    cluster=self._clusters(range(1, self.centers.shape[0] + 1)))
        return self._with_key(pl.DataFrame(cols))

    def glance(self) -> pl.DataFrame:
        return self._with_key(pl.DataFrame({"totss": [self.totss], "tot_withinss": [self.tot_withinss],
                                            "betweenss": [self.betweenss], "iter": [int(self.iter)]}))

    def augment(self, newdata: Any = None) -> pl.DataFrame:
        if newdata is not None:
            raise TypeError("augment(): kmeans() labels its own rows only")
        return pl.concat([self._part.rows, pl.DataFrame({".cluster": self._clusters(self.cluster)})],
                         how="horizontal")

    def __repr__(self) -> str:
        sizes = ", ".join(str(int(s)) for s in self.size)
        return (f"K-means clustering with {self.centers.shape[0]} clusters of sizes {sizes}{_key(self.key)}\n"
                f"{_table_text(self.tidy())}\n(between_SS / total_SS = {100 * self.betweenss / self.totss:.1f} %)")


def _duplicated(x: np.ndarray | None) -> bool:
    return x is not None and np.unique(x, axis=0).shape[0] < x.shape[0]


def _centers_for(centers: Any, part: _Part, groups: list[str]) -> Any:
    """Explicit centres for one group: a matrix, or a frame of centre rows
    (with the grouping columns, each group's own)."""
    if not is_frame(centers):
        return centers
    frame = Data.read(centers, None, "kmeans")._lf.collect()
    for g in groups:
        if g in frame.columns:
            frame = frame.filter(pl.col(g) == part.key[g])
    missing = [name for name in part.names if name not in frame.columns]
    if missing:
        raise ValueError(f"kmeans(): centers lack the columns {missing}")
    return frame.select(part.names).cast(pl.Float64).to_numpy()


@step
def kmeans(columns: Any = "~ .", centers: Any = 2, *, iter_max: int = 10, nstart: int = 1,
           algorithm: str = "Hartigan-Wong", seed: int | None = None, labels: Any = None) -> _ModelStep:
    """k-means clustering, as R's ``kmeans``: Hartigan-Wong (R's default),
    Lloyd (alias Forgy) or MacQueen, ported from R's code.

    ``centers`` is a number of clusters (random distinct rows start them;
    ``nstart`` tries several and keeps the smallest total within-cluster sum
    of squares) or the initial centres: a matrix, or a frame of centre
    rows, filtered by the grouping columns when it has them. With explicit
    centres the result is R's exactly; random starts use NumPy's generator
    (``seed=``), not R's, so only the objective is comparable with R.
    ``tidy`` gives the centres with ``size``, ``withinss`` and ``cluster``;
    ``glance`` ``totss``, ``tot_withinss``, ``betweenss`` and ``iter``;
    ``augment`` the rows with ``.cluster``. Rows with missing values are dropped.
    """
    algorithm = _match(algorithm, _KMEANS_ALGORITHMS, "algorithm", "kmeans")
    if isinstance(iter_max, bool) or not isinstance(iter_max, (int, np.integer)) or iter_max < 1:
        raise ValueError("kmeans(): 'iter.max' must be positive")
    if isinstance(nstart, bool) or not isinstance(nstart, (int, np.integer)) or nstart < 1:
        raise ValueError("kmeans(): nstart must be a positive integer")

    def _apply(data: Data) -> Any:
        rng = np.random.default_rng(seed)
        fits = [KMeans(data, part, _centers_for(centers, part, data._groups), int(iter_max), int(nstart),
                       algorithm, rng)
                for part in _read(data, columns, "kmeans", labels, kind="kmeans")]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, "kmeans")
