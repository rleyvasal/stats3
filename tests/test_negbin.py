"""Negative binomial fits against MASS, including shared model workflows."""

import subprocess
import warnings

import numpy as np
import polars as pl
import pytest

from stats3 import anova, augment, emmeans, glance, glm, glm_nb, negative_binomial, predict, tidy
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r


def counts():
    # Synthetic NB counts (seed 42), with zeros and nonconstant exposure.
    return pl.read_csv(DATA / "negbin.csv").with_row_index("i").with_columns(
        w0=pl.when(pl.col("i") % 10 == 0).then(0).otherwise(pl.col("w")))


FITS = {
    "basic": lambda d: glm_nb(d, "y ~ x + group"),
    "interaction": lambda d: glm_nb(d, "y ~ x * group"),
    "offset": lambda d: glm_nb(d, "y ~ x + group + offset(log(exposure))"),
    "offset_arg": lambda d: glm_nb(d, "y ~ x + group", offset="exposure"),
    "weighted": lambda d: glm_nb(d, "y ~ x + group", weights="w"),
    "zero_weights": lambda d: glm_nb(d, "y ~ x + group", weights="w0"),
    "initial": lambda d: glm_nb(d, "y ~ x + group", init_theta=1),
    "identity": lambda d: glm_nb(d, "y ~ group", link="identity"),
    "sqrt": lambda d: glm_nb(d, "y ~ group", link="sqrt"),
    "no_intercept": lambda d: glm_nb(d, "y ~ 0 + group + x"),
    "aliased": lambda d: glm_nb(d, "y ~ x + I(2 * x) + group"),
    "fixed": lambda d: glm(d, "y ~ x + group", negative_binomial(2)),
    "fixed_sqrt": lambda d: glm(d, "y ~ group", negative_binomial(2, "sqrt")),
}
TABLES = {"tidy": lambda f: tidy(f, conf_int=True), "wald": lambda f: tidy(f, conf_int=True, conf_method="wald"),
          "glance": glance, "augment": augment, "anova": anova}


@pytest.fixture(scope="module")
def r_negbin():
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    probe = subprocess.run([*R_COMMAND, "-e", "library(MASS); library(emmeans)"], capture_output=True)
    if probe.returncode:
        pytest.skip("Negative binomial oracle needs R's MASS and emmeans")
    return _oracle("negbin")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(FITS))
@pytest.mark.parametrize("table", sorted(TABLES))
def test_negbin_matches_r(name, table, backend, r_negbin):
    data, _ = as_input(counts(), backend, [])
    fit = FITS[name](data)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="tests made without re-estimating theta")
        result = TABLES[table](fit)
    assert type(result) is type(data)
    if table == "augment":
        result = as_polars(result).select(pl.col(r"^\..*$"))
    # A and C have exactly equal sample means: their identity/sqrt group
    # coefficient and statistic are zero up to QR rounding, about 1e-15.
    loose = {"estimate": (1e-9, 1e-12), "statistic": (1e-9, 1e-12)} if name in ("identity", "sqrt", "fixed_sqrt") else None
    assert_matches_r(result, r_negbin[f"{name}__{table}"], loose=loose)


@pytest.mark.parametrize("name", [n for n in FITS if not n.startswith("fixed")])
def test_theta_and_information_match_r(name, r_negbin):
    fit = FITS[name](counts())
    table = pl.DataFrame({"theta": [fit.theta], "se_theta": [fit.se_theta]})
    assert fit.theta_converged
    assert_matches_r(table, r_negbin[f"{name}__theta"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", ["link", "response"])
def test_predict_matches_r(backend, scale, r_negbin):
    data, _ = as_input(counts(), backend, [])
    new, _ = as_input(counts().head(5), backend, [])
    result = predict(FITS["basic"](data), new, type=scale, se_fit=True)
    assert type(result) is type(data)
    assert_matches_r(result, r_negbin[f"predict_{scale}"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_shared_workflows_match_r(backend, r_negbin):
    data, _ = as_input(counts(), backend, [])
    fit = FITS["basic"](data)
    for name, result in {
        "compare": anova(fit, glm_nb(data, "y ~ x")),
        "robust": tidy(fit, vcov="HC3", conf_int=True),
        "emmeans": tidy(emmeans(fit, "group", type="response")),
    }.items():
        assert type(result) is type(data)
        assert_matches_r(result, r_negbin[name])
    data, by = as_input(counts(), backend, ["batch"])
    result = tidy(glm_nb(data, "y ~ x + group", by=by), conf_int=True)
    assert type(result) is type(data)
    assert_matches_r(result, r_negbin["grouped"])


def test_missing_rows_and_pipe():
    data = counts().with_columns(pl.when(pl.col("i") == 3).then(None).otherwise(pl.col("x")).alias("x"))
    fit = data >> glm_nb("y ~ x + group", na_action="exclude")
    assert (fit >> tidy()).equals(tidy(fit))
    assert augment(fit).height == data.height
    assert predict(fit)["fit"].is_null().sum() == 1
    with pytest.raises(ValueError, match="missing values"):
        glm_nb(data, "y ~ x + group", na_action="fail")


@pytest.mark.parametrize("theta", [0, -1, np.nan, np.inf])
def test_invalid_theta(theta):
    with pytest.raises(ValueError, match="theta must be"):
        negative_binomial(theta)


def test_invalid_inputs_and_model_comparison():
    with pytest.raises(ValueError, match="link must be"):
        glm_nb(counts(), "y ~ x", link="probit")
    with pytest.raises(ValueError, match="maxit must be"):
        glm_nb(counts(), "y ~ x", maxit=1)
    with pytest.raises(ValueError, match="negative values"):
        glm_nb(counts().with_columns(y=-pl.col("y")), "y ~ x")
    with pytest.raises(TypeError, match="compare glm_nb fits"):
        anova(FITS["basic"](counts()), glm(counts(), "y ~ x", "poisson"))


def test_iteration_limit_is_reported():
    with pytest.warns(UserWarning) as notes:
        fit = glm_nb(counts(), "y ~ x + group", maxit=2)
    assert not fit.theta_converged
    assert any("alternation limit reached" in str(n.message) for n in notes)
