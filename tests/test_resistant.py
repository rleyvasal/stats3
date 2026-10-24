"""Robust and quantile regression against MASS/quantreg on every backend."""

import subprocess
import warnings

import numpy as np
import polars as pl
import pytest

from stats3 import anova, augment, emmeans, glance, lm, modelsummary, predict, rlm, rq, tidy, vcov
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r


def cars():
    return pl.read_csv(DATA / "mtcars.csv").with_row_index("i", offset=1).with_columns(
        w=1 + pl.col("i") % 3).with_columns(w0=pl.when(pl.col("i").is_in([2, 10, 20])).then(0).otherwise(pl.col("w")))


RLM = {
    "basic": lambda d: rlm(d, "mpg ~ wt + hp"),
    "factors": lambda d: rlm(d, "mpg ~ wt + factor(cyl)"),
    "bisquare": lambda d: rlm(d, "mpg ~ wt + hp", psi="bisquare"),
    "bisquare_custom": lambda d: rlm(d, "mpg ~ wt + hp", psi="bisquare", tuning=6),
    "hampel": lambda d: rlm(d, "mpg ~ wt + hp", psi="hampel"),
    "hampel_custom": lambda d: rlm(d, "mpg ~ wt + hp", psi="hampel", tuning=(1.5, 3, 6)),
    "tuning": lambda d: rlm(d, "mpg ~ wt + hp", tuning=2),
    "proposal": lambda d: rlm(d, "mpg ~ wt + hp", scale_est="Huber"),
    "proposal_k2": lambda d: rlm(d, "mpg ~ wt + hp", scale_est="Huber", k2=2),
    "weighted": lambda d: rlm(d, "mpg ~ wt + hp", weights="w"),
    "zero_weights": lambda d: rlm(d, "mpg ~ wt + hp", weights="w0"),
    "case": lambda d: rlm(d, "mpg ~ wt + hp", weights="w", wt_method="case"),
    "case_proposal": lambda d: rlm(d, "mpg ~ wt + hp", weights="w", wt_method="case", scale_est="proposal 2"),
    "offset": lambda d: rlm(d, "mpg ~ wt + offset(hp)"),
    "no_intercept": lambda d: rlm(d, "mpg ~ 0 + wt + hp"),
    "xtwx": lambda d: rlm(d, "mpg ~ wt + hp", cov_method="XtWX"),
    "case_xtwx": lambda d: rlm(d, "mpg ~ wt + hp", weights="w", wt_method="case", cov_method="XtWX"),
}
RQ = {
    "median": lambda d: rq(d, "mpg ~ wt + hp"),
    "lower": lambda d: rq(d, "mpg ~ wt + hp", tau=.25),
    "upper": lambda d: rq(d, "mpg ~ wt + hp", tau=.75),
    "factors": lambda d: rq(d, "mpg ~ wt + factor(cyl)"),
    "weighted": lambda d: rq(d, "mpg ~ wt + hp", weights="w"),
    "zero_weights": lambda d: rq(d, "mpg ~ wt + hp", weights="w0"),
    "no_intercept": lambda d: rq(d, "mpg ~ 0 + wt + hp"),
    "iid": lambda d: rq(d, "mpg ~ wt + hp", se="iid"),
    "weighted_iid": lambda d: rq(d, "mpg ~ wt + hp", se="iid", weights="w"),
}
TABLES = {"tidy": lambda f: tidy(f, conf_int=True), "vcov": vcov, "augment": augment, "glance": glance,
          "predict": lambda f: predict(f, cars().head(5), interval="confidence", se_fit=True)}


@pytest.fixture(scope="module", params=["rlm", "rq"])
def reference(request):
    kind = request.param
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    packages = "library(MASS); library(emmeans)" if kind == "rlm" else "library(quantreg); library(emmeans)"
    probe = subprocess.run([*R_COMMAND, "-e", packages], capture_output=True)
    if probe.returncode:
        pytest.skip(f"R packages for {kind} oracle are unavailable")
    return kind, _oracle(kind)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("table", sorted(TABLES))
def test_tables_match_r(reference, backend, table):
    kind, expected = reference
    data, _ = as_input(cars(), backend, [])
    for name, make in (RLM if kind == "rlm" else RQ).items():
        fit = make(data)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"rq\(\): .*non-positive density")
            if table == "predict":
                new, _ = as_input(cars().head(5), backend, [])
                result = predict(fit, new, interval="confidence", se_fit=True)
            else:
                result = TABLES[table](fit)
        assert type(result) is type(data)
        if table == "augment":
            result = as_polars(result).select(".fitted", ".resid")
        # LP residuals at zero are rounded differently by the two solvers.
        loose = {".resid": (1e-9, 1e-10)} if kind == "rq" else None
        assert_matches_r(result, expected[f"{name}__{table}"], loose=loose)


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_and_marginal_means_match_r(reference, backend):
    kind, expected = reference
    make = rlm if kind == "rlm" else rq
    data, by = as_input(cars(), backend, ["am"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = tidy(make(data, "mpg ~ wt + hp", by=by), conf_int=True)
    assert type(result) is type(data)
    assert_matches_r(result, expected["grouped"])
    data, _ = as_input(cars(), backend, [])
    result = tidy(emmeans(make(data, "mpg ~ wt + factor(cyl)"), "cyl"))
    assert_matches_r(result, expected["emmeans"])


def test_robust_weights_match_r(reference):
    kind, expected = reference
    if kind != "rlm":
        return
    for name, make in RLM.items():
        fit = make(cars())
        result = pl.DataFrame({"weight": fit.robust_weights, "wresid": fit.wresid})
        # Weighted QR leaves machine-scale residuals on zero-weight rows.
        assert_matches_r(result, expected[f"{name}__weights"], loose={"wresid": (1e-9, 1e-12)})


def test_quantile_comparison_matches_r(reference):
    kind, expected = reference
    if kind != "rq":
        return
    result = anova(rq(cars(), "mpg ~ wt"), rq(cars(), "mpg ~ wt + hp"))
    assert_matches_r(result, expected["compare"])


@pytest.mark.parametrize("make", [rlm, rq])
def test_missing_rows_pipe_and_summary(make):
    data = cars().with_columns(pl.when(pl.col("i") == 3).then(None).otherwise(pl.col("wt")).alias("wt"))
    fit = data >> make("mpg ~ wt + hp", na_action="exclude")
    assert (fit >> tidy()).equals(tidy(fit))
    assert augment(fit).height == data.height
    assert predict(fit)["fit"].is_null().sum() == 1
    assert modelsummary({"OLS": lm(data, "mpg ~ wt + hp"), "Fit": fit}, stars=True).height > 0
    with pytest.raises(ValueError, match="missing values"):
        make(data, "mpg ~ wt + hp", na_action="fail")


def test_robust_fit_downweights_a_vertical_outlier():
    x = np.arange(21, dtype=float)
    y = 3 + 2*x + np.sin(x)
    y[10] += 100
    data = pl.DataFrame({"x": x, "y": y})
    fit = rlm(data, "y ~ x")
    assert fit.robust_weights[10] < .1
    assert abs(fit.coef[0] - 3) < abs(lm(data, "y ~ x").coef[0] - 3)
    assert tidy(fit)["p_value"].null_count() == 2


@pytest.mark.parametrize(("tau", "expected"), [(.25, 2.), (.5, 4.), (.75, 9.)])
def test_intercept_only_quantiles(tau, expected):
    fit = rq({"y": [1., 2., 4., 9., 15.]}, "y ~ 1", tau=tau)
    assert fit.coef[0] == pytest.approx(expected)


def test_weighted_median_changes_the_optimum():
    data = {"y": [1., 2., 4., 9., 15.], "w": [1., 1., 1., 10., 1.]}
    assert rq(data, "y ~ 1").objective == pytest.approx(10.5)
    weighted = rq(data, "y ~ 1", weights="w")
    assert weighted.coef[0] == pytest.approx(9.)
    assert weighted.objective == pytest.approx(13.)


def test_robust_zero_scale_and_iteration_limit():
    fit = rlm({"x": [0., 1., 2., 3., 4.], "y": [0.] * 5}, "y ~ x")
    assert fit.converged
    assert predict(fit)["fit"].to_list() == [0.] * 5
    assert tidy(fit)["std_error"].null_count() == 2
    with pytest.warns(UserWarning, match="failed to converge"):
        limited = rlm(cars(), "mpg ~ wt + hp", maxit=1, acc=1e-12)
    assert not limited.converged


@pytest.mark.parametrize("make", [rlm, rq])
def test_singular_and_unsupported_inference(make):
    with pytest.raises(ValueError, match="singular"):
        make(cars(), "mpg ~ wt + I(2 * wt)")
    fit = make(cars(), "mpg ~ wt")
    with pytest.raises(ValueError, match="own covariance"):
        tidy(fit, vcov="HC3")
    with pytest.raises(TypeError, match="single-model"):
        anova(fit)
