"""Penalized regression against R's glmnet, across the four data backends."""

import warnings

import numpy as np
import polars as pl
import pytest

from stats3 import coef, cv_glmnet, glance, glmnet, predict, tidy
from stats3.penalized import _interp, _r_format
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r


def mtcars():
    m = pl.read_csv(DATA/"mtcars.csv").drop("model")
    n = m.height
    return m.with_columns(w=(pl.int_range(1, n + 1) % 3 + 1).cast(pl.Float64),
                          off=pl.Series(np.linspace(-.2, .2, n)),
                          fold=pl.int_range(n) % 4 + 1)


def penguins():
    p = pl.read_csv(DATA/"penguins.csv", null_values="NA")
    return p.with_columns(fold=pl.int_range(p.height) % 5 + 1)


def lung():
    l = pl.read_csv(DATA/"lung_glmnet.csv")
    return l.with_columns(fold=pl.int_range(l.height) % 5 + 1)


CARS = "mpg ~ cyl + disp + hp + drat + wt + qsec + vs + am + gear + carb"
PEN = "species ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g"
LUNG = "Surv(time, status) ~ age + sex + ph_ecog + ph_karno + wt_loss"
AM = "am ~ wt + hp + disp + qsec + drat"

CASES = {
    "gaussian": (mtcars, CARS, {}, [1, .1]),
    "elastic": (mtcars, CARS, {"alpha": .5}, .3),
    "ridge": (mtcars, CARS, {"alpha": 0}, 2),
    "weighted": (mtcars, "mpg ~ wt + hp + disp + qsec + offset(off)", {"weights": "w"}, .2),
    "options": (mtcars, CARS, {"standardize": False, "intercept": False,
                               "penalty_factor": [0] + [1] * 9, "lower_limits": -1, "upper_limits": 2,
                               "nlambda": 30, "lambda_min_ratio": .001}, .5),
    "nonnegative": (mtcars, CARS, {"lower_limits": 0}, .5),
    "lambda": (mtcars, CARS, {"lambda_": [.01, 2, .5, .1]}, .3),
    "factors": (penguins, "body_mass_g ~ species + island + sex + bill_length_mm + flipper_length_mm", {}, 10),
    "binomial": (mtcars, AM, {"family": "binomial"}, .05),
    "binomial_offset": (mtcars, "vs ~ wt + hp + disp + qsec + offset(off)",
                        {"family": "binomial", "weights": "w", "alpha": .7}, .02),
    "poisson": (mtcars, "carb ~ cyl + disp + hp + drat + wt + qsec + offset(off)",
                {"family": "poisson", "weights": "w"}, .05),
    "multinomial": (penguins, PEN, {"family": "multinomial"}, .01),
    "cox": (lung, LUNG, {"family": "cox"}, .05),
    "efron": (lung, LUNG, {"family": "cox", "cox_ties": "efron", "alpha": .5}, .05),
    "default_gaussian": (mtcars, CARS, {}, .1),
    "default_binomial": (mtcars, AM, {"family": "binomial"}, .05),
}


def _fit(name, backend):
    data, formula, kwargs, s = CASES[name]
    frame = data()
    source, _ = as_input(frame, backend, [])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*fewer than 8")
        return glmnet(source, formula, **kwargs), frame, s


SEX = "sex ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g"
CV = {
    "mse": (mtcars, CARS, {"type_measure": "mse"}),
    "mae": (mtcars, CARS, {"type_measure": "mae", "alpha": .5}),
    "ungrouped": (mtcars, CARS, {"type_measure": "mse", "grouped": False}),
    "deviance": (mtcars, AM, {"family": "binomial", "type_measure": "deviance"}),
    "auc": (penguins, SEX, {"family": "binomial", "type_measure": "auc"}),
    "class": (penguins, SEX, {"family": "binomial", "type_measure": "class"}),
    "poisson": (mtcars, "carb ~ cyl + disp + hp + drat + wt + qsec + offset(off)",
                {"family": "poisson", "type_measure": "deviance", "weights": "w"}),
    "multinomial": (penguins, PEN, {"family": "multinomial", "type_measure": "deviance"}),
    "multinomial_class": (penguins, PEN, {"family": "multinomial", "type_measure": "class"}),
    "cox": (lung, LUNG, {"family": "cox", "type_measure": "deviance"}),
    "cindex": (lung, LUNG, {"family": "cox", "type_measure": "C"}),
}


def _cv(name, backend):
    data, formula, kwargs = CV[name]
    source, _ = as_input(data(), backend, [])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*fewer than 8")
        return cv_glmnet(source, formula, foldid="fold", **kwargs)


# glmnet stops coordinate descent at thresh; the oracle tightens it as far as
# each family's iterations converge, and stats3 solves to full precision, so
# the remaining difference is R's. Tolerances are relative to each column's
# largest value: what R's tightened Gaussian/Poisson paths reach (1e-8),
# binomial IRLS (1e-6), multinomial (2e-5) and Cox (5e-5) at thresh 1e-16 to
# 1e-20. The default_* cases keep R's default thresh=1e-7, whose coordinate
# descent stops with coefficients still moving in the third or fourth digit.
SCALE = {"binomial": 5e-6, "binomial_offset": 5e-6, "multinomial": 1e-4, "cox": 2e-4, "efron": 2e-4,
         "default_gaussian": 1e-2, "default_binomial": 1e-4}


def _loose(expected, tol):
    names = expected["columns"]
    names = [names] if isinstance(names, str) else names
    data = expected["data"]
    data = data if data and isinstance(data[0], list) else [data]
    out = {}
    for name, values in zip(names, data):
        numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numbers and name not in ("step", "nobs", "nzero", "lambda"):
            out[name] = (0.0, tol * max(1.0, max(abs(v) for v in numbers)) if name == "nulldev"
                         else tol * max(abs(v) for v in numbers) + 1e-12)
    return out


def _new(frame, fit):
    used = [c for c in frame.columns if c in fit.formula.text]
    return frame.drop_nulls(used).head(5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(CASES))
def test_paths_match_r(name, backend):
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    fit, frame, s = _fit(name, backend)
    new, _ = as_input(_new(frame, fit), backend, [])
    expected = _oracle("glmnet")[name]
    tables = {"tidy": lambda: tidy(fit), "tidy_zeros": lambda: tidy(fit, return_zeros=True),
              # npasses counts glmnet's own coordinate-descent passes; stats3's solver differs.
              "glance": lambda: as_polars(glance(fit)).select("nulldev", "nobs"),
              "coef": lambda: coef(fit, s=s), "link": lambda: predict(fit, new, s=s),
              "response": lambda: predict(fit, new, s=s, type="response"),
              "class": lambda: predict(fit, new, s=s, type="class"),
              "nonzero": lambda: predict(fit, s=s, type="nonzero")}
    tol = SCALE.get(name, 1e-7)
    for table, want in expected.items():
        if name.startswith("default") and table.startswith("tidy"):
            continue   # R's default-threshold path can stop a few lambdas later
        assert_matches_r(tables[table](), want, loose=_loose(want, tol))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(CV))
def test_cross_validation_matches_r(name, backend):
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    fit = _cv(name, backend)
    expected = _oracle("glmnet")["cv"][name]
    tol = 2e-4 if name in ("cox", "cindex") else 1e-4 if name.startswith("multinomial") else 1e-6
    for table, actual in (("tidy", tidy(fit)), ("glance", glance(fit)), ("coef", coef(fit))):
        assert_matches_r(actual, expected[table], loose=_loose(expected[table], tol))


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_fits_and_output_types(backend):
    source, by = as_input(mtcars(), backend, ["am"])
    fits = glmnet(source, "mpg ~ wt + hp + disp + qsec + drat", by=by)
    assert len(fits) == 2
    if R_COMMAND is not None:
        want = _oracle("glmnet")["grouped"]
        assert_matches_r(tidy(fits), want["tidy"], loose=_loose(want["tidy"], 1e-7))
        assert_matches_r(as_polars(glance(fits)).select("am", "nulldev", "nobs"), want["glance"])
    expected = {"polars": ("polars", "DataFrame"), "lazy": ("polars", "LazyFrame"),
                "pandas": ("pandas", "DataFrame"), "tidy3": ("tidy3", "TidyFrame")}[backend]
    source, _ = as_input(mtcars(), backend, [])
    fit = source >> glmnet(CARS)
    cv = cv_glmnet(source, CARS, foldid="fold")
    for output in [tidy(fit), glance(fit), coef(fit), predict(fit, source, s=.1), tidy(cv), glance(cv),
                   coef(cv), predict(cv, source), tidy(fits), coef(fits)]:
        assert (type(output).__module__.split(".")[0], type(output).__name__) == expected
    grouped_cv = cv_glmnet(*as_input(mtcars(), backend, ["am"])[:1], "mpg ~ wt + hp + disp + qsec",
                           foldid="fold", by=as_input(mtcars(), backend, ["am"])[1])
    assert as_polars(glance(grouped_cv)).height == 2


def test_matrix_interface_matches_formula():
    m = mtcars()
    x = m.select("cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb")
    by_formula = glmnet(m, CARS)
    by_matrix = glmnet(x=x, y=m["mpg"].to_numpy())
    np.testing.assert_array_equal(by_matrix.lambda_, by_formula.lambda_)
    np.testing.assert_allclose(by_matrix.path.beta, by_formula.path.beta, atol=1e-12)
    plain = glmnet(x=x.to_numpy(), y=m["mpg"].to_numpy())
    assert as_polars(coef(plain, s=.5))["term"].to_list()[:3] == ["(Intercept)", "V1", "V2"]
    np.testing.assert_allclose(predict(plain, x.to_numpy(), s=.5)["s=0.5"].to_numpy(),
                               predict(by_formula, m, s=.5)["s=0.5"].to_numpy(), atol=1e-10)
    labels = glmnet(x=x, y=np.where(m["am"].to_numpy() == 1, "manual", "auto"), family="binomial")
    assert set(predict(labels, x, s=.05, type="class")["s=0.05"]) <= {"auto", "manual"}
    cv = cv_glmnet(x=x, y=m["mpg"].to_numpy(), foldid=m["fold"].to_numpy())
    assert cv.lambda_min == cv_glmnet(m, CARS, foldid="fold").lambda_min
    random = cv_glmnet(x=x, y=m["mpg"].to_numpy(), nfolds=4, seed=1)
    assert sorted(set(random.foldid)) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="newoffset"):
        predict(glmnet(x=x, y=m["mpg"].to_numpy(), offset=m["off"].to_numpy()), x)


def test_interpolation_and_names():
    lam = np.array([1.0, .5, .25])
    left, right, frac = _interp(lam, [2, 1, .75, .25, .1])
    assert left.tolist() == [0, 0, 0, 2, 2] and right.tolist() == [0, 0, 1, 2, 2]
    np.testing.assert_allclose(frac, [1, 1, .5, 1, 1])
    assert _r_format([1, .1]) == ["1.0", "0.1"]
    assert _r_format([10, 1]) == ["10", " 1"]
    assert _r_format([1e-5]) == ["1e-05"]
    assert _r_format([0.05123456789]) == ["0.05123457"]


@pytest.mark.parametrize("kwargs, match", [
    ({"family": "gamma"}, "family"), ({"nlambda": 1}, "nlambda"), ({"lambda_min_ratio": 1}, "less than 1"),
    ({"lambda_": [-1, 1]}, "non-negative"), ({"lower_limits": 1}, "non-positive"),
    ({"upper_limits": -1}, "non-negative"), ({"penalty_factor": [1, 1]}, "penalty_factor"),
    ({"cox_ties": "exact"}, "cox_ties"), ({"na_action": "exclude"}, "exclude")])
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        glmnet(mtcars(), CARS, **kwargs)


def test_invalid_responses_and_warnings():
    m = mtcars()
    with pytest.raises(ValueError, match="2 or more columns"):
        glmnet(m, "mpg ~ wt")
    with pytest.raises(ValueError, match="multinomial"), pytest.warns(UserWarning, match="fewer than 8"):
        glmnet(m, "factor(cyl) ~ wt + hp", family="binomial")
    with pytest.raises(ValueError, match="family='cox'"):
        glmnet(lung(), LUNG)
    with pytest.raises(ValueError, match="numeric"):
        glmnet(penguins(), PEN)
    with pytest.raises(ValueError, match="same shape"):
        glmnet(penguins(), PEN + " + offset(year)", family="multinomial")
    with pytest.warns(UserWarning, match="fewer than 8"):
        glmnet(m, "factor(cyl) ~ wt + hp", family="multinomial")
    with pytest.raises(ValueError, match="nfolds"):
        cv_glmnet(m, CARS, foldid=np.resize([1, 2], 32))
    with pytest.warns(UserWarning, match="type_measure"):
        cv_glmnet(m, CARS, foldid="fold", type_measure="auc")
    with pytest.raises(ValueError, match="column name"):
        cv_glmnet(m, "mpg ~ wt + hp", foldid=np.resize([1, 2, 3], 32), by="am")
    fit = glmnet(m, CARS)
    with pytest.raises(ValueError, match="class"):
        predict(fit, m, type="class")
    assert "Df" in repr(fit) and "lambda_min" in repr(cv_glmnet(m, CARS, foldid="fold"))
