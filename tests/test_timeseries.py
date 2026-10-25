"""Time series (acf, pacf, ccf, Box.test, arima) against R, on the four data backends."""

import warnings

import numpy as np
import polars as pl
import pytest

from stats3 import acf, arima, augment, box_test, ccf, glance, modelsummary, pacf, predict, tidy, vcov
from stats3.timeseries import _arima_like, _complex_gradient, _make_arima, _partrans, _invpartrans
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r


def series() -> pl.DataFrame:
    return pl.read_csv(DATA / "timeseries.csv", schema_overrides={"time": pl.Float64})


def one(name: str, **na) -> pl.DataFrame:
    """One dataset, with missing values at the given (1-based) positions."""
    frame = series().filter(pl.col("dataset") == name)
    for column, at in na.items():
        frame = frame.with_columns(pl.when(pl.col("t").is_in(at)).then(None).otherwise(pl.col("value")).alias(column))
    return frame


def pair(half: str = "a") -> pl.DataFrame:
    us = one("USAccDeaths")["value"].to_list()
    ap = one("AirPassengers")["value"].to_list()
    return pl.DataFrame({"g": half, "t": range(1, 73), "x": us, "y": ap[:72] if half == "a" else ap[72:]})


def r(*path):
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    out = _oracle("timeseries")
    for key in path:
        out = out[key]
    return out


LK_NA = [10, 40, 41]

ACF_CASES = {
    "lh_cor": (lambda: one("lh"), lambda: acf("value"), []),
    "lh_cov": (lambda: one("lh"), lambda: acf("value", lag_max=5, type="covariance"), []),
    "lh_partial": (lambda: one("lh"), lambda: pacf("value"), []),
    "lh_partial_type": (lambda: one("lh"), lambda: acf("value", lag_max=8, type="partial"), []),
    "lk_ma": (lambda: one("LakeHuron"), lambda: acf("value", ci_type="ma"), []),
    "us_nodemean": (lambda: one("USAccDeaths"), lambda: acf("value", lag_max=30, demean=False), []),
    "lk_na_pass": (lambda: one("LakeHuron", x=LK_NA), lambda: acf("x", na_action="pass"), []),
    "lk_na_pass_partial": (lambda: one("LakeHuron", x=LK_NA), lambda: pacf("x", na_action="pass"), []),
    "lk_na_omit": (lambda: one("LakeHuron", x=[1, 2, 98]), lambda: acf("x", na_action="omit"), []),
    "lk_contiguous": (lambda: one("LakeHuron", x=[10, 50]), lambda: acf("x", na_action="contiguous"), []),
    # Rows shuffled: order_by= puts each series back in time order.
    "grouped": (lambda: series().sample(fraction=1, shuffle=True, seed=3), lambda: acf("value", order_by="t"),
                ["dataset"]),
    "grouped_partial": (lambda: series(), lambda: pacf("value", lag_max=6), ["dataset"]),
    "ccf": (lambda: pair(), lambda: ccf("x", "y"), []),
    "ccf_cov": (lambda: pair(), lambda: ccf("x", "y", lag_max=6, type="covariance"), []),
    "ccf_grouped": (lambda: pl.concat([pair("a"), pair("b")]), lambda: ccf("x", "y", lag_max=4), ["g"]),
}

BOX_CASES = {
    "lh_bp": (lambda: one("lh"), lambda: box_test("value"), []),
    "lk_lb": (lambda: one("LakeHuron"), lambda: box_test("value", lag=10, type="Ljung-Box", fitdf=2), []),
    "lk_na": (lambda: one("LakeHuron", x=LK_NA), lambda: box_test("x", lag=5, type="Ljung-Box"), []),
    "grouped": (lambda: series(), lambda: box_test("value", lag=6, type="Ljung-Box"), ["dataset"]),
}


def _run(spec, backend):
    data, make, by = spec
    frame, by_arg = as_input(data(), backend, by)
    step = make()
    step.by = by_arg
    return frame >> step


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(ACF_CASES))
def test_correlations_match_r(name, backend):
    got = _run(ACF_CASES[name], backend)
    expected = {"polars": "DataFrame", "lazy": "LazyFrame", "pandas": "DataFrame", "tidy3": "TidyFrame"}
    assert type(got).__name__ == expected[backend]
    # stats3 removes the correctly rounded mean; R's colMeans sums in double
    # (long double is double on arm64), off by ~2e-14 on LakeHuron's level
    # near 580, which shifts the correlations by up to ~1e-14 absolute.
    assert_matches_r(got, r("acf", name), rel=1e-12, loose={"acf": (1e-12, 5e-14)})


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(BOX_CASES))
def test_box_test_matches_r(name, backend):
    assert_matches_r(_run(BOX_CASES[name], backend), r("box", name), rel=1e-12)


def lake() -> pl.DataFrame:
    return one("LakeHuron").with_columns(yr=pl.col("time") - 1920)


FITS = {
    "lh100": (lambda: one("lh"), dict(order=(1, 0, 0)), 3),
    "lh300": (lambda: one("lh"), dict(order=(3, 0, 0)), 12),
    "lh101_ml": (lambda: one("lh"), dict(order=(1, 0, 1), method="ML"), 3),
    "lh300_css": (lambda: one("lh"), dict(order=(3, 0, 0), method="CSS"), 3),
    "lh_fixed": (lambda: one("lh"), dict(order=(3, 0, 0), fixed=[None, 0, None, None], transform_pars=False), 3),
    "lh_notrans": (lambda: one("lh"), dict(order=(1, 0, 1), method="ML", transform_pars=False), 3),
    "lh_nomean": (lambda: one("lh"), dict(order=(1, 0, 0), include_mean=False), 3),
    "lh_ncond": (lambda: one("lh"), dict(order=(1, 0, 0), method="CSS", n_cond=3), 3),
    "lk_xreg": (lake, dict(order=(2, 0, 0), xreg="yr"), 5),
    "lk_101": (lambda: one("LakeHuron"), dict(order=(1, 0, 1)), 3),
    "lk_na": (lambda: one("LakeHuron", value=LK_NA), dict(order=(1, 0, 1)), 3),
    "lk_110": (lambda: one("LakeHuron"), dict(order=(1, 1, 0)), 3),
    "us": (lambda: one("USAccDeaths"), dict(order=(0, 1, 1), seasonal=(0, 1, 1), period=12), 12),
    "us_sar": (lambda: one("USAccDeaths"), dict(order=(1, 1, 0), seasonal=(1, 1, 0), period=12), 12),
    "ap": (lambda: one("AirPassengers").with_columns(pl.col("value").log()),
           dict(order=(0, 1, 1), seasonal=(0, 1, 1), period=12), 12),
}

# R's optim minimises with central-difference gradients (step 1e-3 on its
# parameter scale) and stops at reltol 1e-8, so its estimates sit off the
# exact optimum by up to ~2.5e-4 relative (lh, LakeHuron; ~1e-5 absolute).
# stats3 finds the optimum to ~1e-10 (its log likelihood is never below
# R's: see test_arima_likelihood_at_least_r). Everything that moves with the
# estimates to first order (standard errors, covariances, sigma2, forecasts,
# residuals) carries R's error; the log likelihood, flat at the optimum, only
# to second order (1.5e-9).
ESTIMATES = {c: (5e-4, 1e-4) for c in ("estimate", "std_error", "statistic", "conf_low", "conf_high")}
FIT_STATISTICS = {"sigma2": 2e-6, "logLik": 5e-9, "AIC": 5e-9, "AICc": 5e-9, "BIC": 5e-9}
FORECASTS = {"fit": 5e-5, "se_fit": 1e-4, ".resid": (2e-3, 1e-4)}


@pytest.fixture(scope="module", params=BACKENDS)
def fits(request):
    backend = request.param
    out = {}
    for name, (data, kwargs, _) in FITS.items():
        frame, _ = as_input(data(), backend, [])
        out[name] = arima(frame, "value", **kwargs)
    return backend, out


@pytest.mark.parametrize("table", ["tidy", "glance", "vcov", "predict", "resid"])
def test_arima_matches_r(fits, table):
    backend, models = fits
    for name, fit in models.items():
        n_ahead = FITS[name][2]
        new = None
        if name == "lk_xreg":
            new, _ = as_input(pl.DataFrame({"yr": [53.0, 54, 55, 56, 57]}), backend, [])
        actual = {"tidy": lambda: tidy(fit, conf_int=True), "glance": lambda: glance(fit),
                  "vcov": lambda: vcov(fit), "predict": lambda: predict(fit, new, n_ahead=n_ahead),
                  "resid": lambda: as_polars(augment(fit)).select(".resid")}[table]()
        loose = {**ESTIMATES, **FIT_STATISTICS, **FORECASTS}
        if table == "vcov":
            loose.update({n: (2e-3, 2e-6) for n in fit.names})
        assert_matches_r(actual, r("arima", name, table), loose=loose)


def test_arima_likelihood_at_least_r(fits):
    _, models = fits
    for name, fit in models.items():
        want = r("arima", name, "glance")["data"][1][0]
        assert fit.loglik >= want - 1e-12 * abs(want), name


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_arima(backend):
    frame, by = as_input(series().sample(fraction=1, shuffle=True, seed=5), backend, ["dataset"])
    models = arima(frame, "value", (0, 1, 1), order_by="t", by=by)
    assert len(models) == 4
    loose = {**ESTIMATES, **FIT_STATISTICS, **FORECASTS}
    assert_matches_r(tidy(models, conf_int=True), r("arima", "grouped", "tidy"), loose=loose)
    assert_matches_r(glance(models), r("arima", "grouped", "glance"), loose=loose)
    assert_matches_r(predict(models, n_ahead=3), r("arima", "grouped", "predict"), loose=loose)
    expected = {"polars": ("polars", "DataFrame"), "lazy": ("polars", "LazyFrame"),
                "pandas": ("pandas", "DataFrame"), "tidy3": ("tidy3", "TidyFrame")}[backend]
    for output in [tidy(models), glance(models), predict(models), augment(models), box_test(models, lag=4)]:
        assert (type(output).__module__.split(".")[0], type(output).__name__) == expected
    assert as_polars(modelsummary(models)).height > 0


def test_box_test_on_residuals():
    fit = arima(one("lh"), "value", (3, 0, 0))
    assert_matches_r(box_test(fit, lag=8, type="Ljung-Box", fitdf=3), r("box", "residuals"), rel=1e-4)


def test_likelihood_and_transformations():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=4)
    np.testing.assert_allclose(_invpartrans(_partrans(raw)), raw, rtol=1e-12)
    y = one("lh")["value"].to_numpy() - 2.4

    def f(par):
        mod = _make_arima(par[:2], par[2:], np.array([]), 1e6)
        ssq, sumlog, nu, _ = _arima_like(y, mod)
        return 0.5 * (np.log(ssq / nu) + sumlog / nu)

    par = np.array([.5, -.2, .3])
    h = 1e-6
    numeric = [(f(par + h * e) - f(par - h * e)).real / (2 * h) for e in np.eye(3)]
    np.testing.assert_allclose(_complex_gradient(f, par), numeric, rtol=1e-6)


def test_print_and_errors():
    fit = arima(one("lh"), "value", (1, 0, 0))
    assert "sigma^2 estimated as" in repr(fit)
    assert list(fit.coefficients) == ["ar1", "intercept"]
    with pytest.raises(ValueError, match="period"):
        arima(one("lh"), "value", (1, 0, 0), seasonal=(1, 0, 0))
    with pytest.raises(ValueError, match="order"):
        arima(one("lh"), "value", (1, 0))
    with pytest.raises(ValueError, match="method"):
        arima(one("lh"), "value", method="REML")
    with pytest.raises(ValueError, match="wrong length"):
        arima(one("lh"), "value", (1, 0, 0), fixed=[0.5])
    with pytest.raises(ValueError, match="vcov"):
        tidy(fit, vcov="HC3")
    with pytest.raises(TypeError, match="ANOVA"):
        from stats3 import anova
        anova(fit)
    with pytest.raises(ValueError, match="xreg"):
        predict(arima(lake(), "value", (1, 0, 0), xreg="yr"), n_ahead=2)
    with pytest.warns(UserWarning, match="AR parameters were fixed"):
        arima(one("lh"), "value", (2, 0, 0), fixed=[0.5, None, None])
    css = arima(one("lh"), "value", (1, 0, 0), method="CSS")
    assert glance(css)["AIC"].to_list() == [None]
    assert "part log likelihood" in repr(css)


def test_series_errors():
    missing = one("LakeHuron", x=LK_NA)
    with pytest.raises(ValueError, match="missing values in object"):
        acf(missing, "x")
    with pytest.raises(ValueError, match="internal NAs"):
        acf(missing, "x", na_action="omit")
    with pytest.raises(ValueError, match="na_action"):
        acf(missing, "x", na_action="drop")
    with pytest.raises(ValueError, match="type"):
        acf(one("lh"), "value", type="spectrum")
    with pytest.raises(ValueError, match="ci_type"):
        acf(one("lh"), "value", type="covariance", ci_type="ma")
    with pytest.raises(ValueError, match="lag"):
        box_test(one("lh"), "value", lag=0)
    with pytest.raises(ValueError, match="type"):
        box_test(one("lh"), "value", type="Ljung")
    with pytest.raises(ValueError, match="order_by"):
        acf(one("lh").with_columns(t=pl.lit(None)), "value", order_by="t")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert one("lh") >> acf("value") is not None
    first = as_polars(pacf(one("LakeHuron", x=[33, 66]), "x", na_action="contiguous", lag_max=3))
    alone = as_polars(pacf(one("LakeHuron").head(32), "value", lag_max=3))
    assert first.equals(alone)
