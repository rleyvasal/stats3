"""loess, lowess, smooth.spline and ksmooth against base R on every backend."""

import warnings

import numpy as np
import polars as pl
import pytest

from stats3 import anova, augment, glance, ksmooth, loess, lm, lowess, predict, smooth_spline, tidy, vcov
from oracle import BACKENDS, DATA, _oracle, as_input, as_polars, assert_matches_r


def cars():
    return pl.read_csv(DATA / "cars.csv").with_row_index("i", offset=1).with_columns(
        w=1 + pl.col("i") % 3).drop("i")


def mtcars():
    return pl.read_csv(DATA / "mtcars.csv")


def penguins():
    return pl.read_csv(DATA / "penguins.csv", null_values="NA")


DATASETS = {"cars": cars, "mtcars": mtcars, "penguins": penguins}
NEW = {"cars": pl.DataFrame({"speed": [3., 4., 10.5, 25., 26.]}),
       "mtcars": pl.DataFrame({"wt": [1.5, 2.62, 3.44, 5.4, 6.], "hp": [60., 110., 175., 230., 100.],
                               "disp": [100., 160., 360., 400., 300.]}),
       "penguins": pl.DataFrame({"flipper_length_mm": [171., 190.5, 210., 231., 240.],
                                 "bill_length_mm": [40., 39.1, 45., 50., 59.6]})}

LOESS = {
    "cars_default": lambda d: loess(d, "dist ~ speed"),
    "cars_degree1": lambda d: loess(d, "dist ~ speed", span=.5, degree=1),
    "cars_degree0": lambda d: loess(d, "dist ~ speed", span=.6, degree=0),
    "cars_direct": lambda d: loess(d, "dist ~ speed", surface="direct"),
    "cars_symmetric": lambda d: loess(d, "dist ~ speed", family="symmetric"),
    "cars_exact": lambda d: loess(d, "dist ~ speed", statistics="exact"),
    "cars_trace_approx": lambda d: loess(d, "dist ~ speed", trace_hat="approximate"),
    "cars_direct_symmetric": lambda d: loess(d, "dist ~ speed", family="symmetric", surface="direct"),
    "cars_weighted": lambda d: loess(d, "dist ~ speed", weights="w"),
    "cars_cell": lambda d: loess(d, "dist ~ speed", span=.9, cell=.1, iterations=2, family="symmetric"),
    "cars_enp": lambda d: loess(d, "dist ~ speed", enp_target=6),
    "mtcars_2d": lambda d: loess(d, "mpg ~ wt + hp"),
    "mtcars_2d_symmetric": lambda d: loess(d, "mpg ~ wt + hp", family="symmetric"),
    "mtcars_2d_raw": lambda d: loess(d, "mpg ~ wt * hp", normalize=False, span=.9),
    "mtcars_2d_direct": lambda d: loess(d, "mpg ~ wt + hp", surface="direct"),
    "mtcars_2d_exact": lambda d: loess(d, "mpg ~ wt + hp", degree=1, statistics="exact"),
    "mtcars_3d": lambda d: loess(d, "mpg ~ wt + hp + disp", span=1),
    "penguins_1d": lambda d: loess(d, "body_mass_g ~ flipper_length_mm"),
    "penguins_2d": lambda d: loess(d, "body_mass_g ~ flipper_length_mm + bill_length_mm", span=.3),
}

SPLINES = {
    "cars_gcv": lambda d: smooth_spline(d, "speed", "dist"),
    "cars_cv": lambda d: smooth_spline(d, "speed", "dist", cv=True, spar=.6),
    "cars_df": lambda d: smooth_spline(d, "speed", "dist", df=5),
    "cars_spar": lambda d: smooth_spline(d, "speed", "dist", spar=.5),
    "cars_lambda": lambda d: smooth_spline(d, "speed", "dist", lambda_=1e-3),
    "cars_none": lambda d: smooth_spline(d, "speed", "dist", spar=.7, cv=None),
    "cars_weighted": lambda d: smooth_spline(d, "speed", "dist", w="w"),
    "cars_all_knots": lambda d: smooth_spline(d, "speed", "dist", all_knots=True, df=8),
    "cars_nknots": lambda d: smooth_spline(d, "speed", "dist", nknots=10, spar=.3),
    "cars_penalty": lambda d: smooth_spline(d, "speed", "dist", penalty=1.4, df_offset=1),
    "mtcars_gcv": lambda d: smooth_spline(d, "wt", "mpg"),
    "mtcars_cv": lambda d: smooth_spline(d, "hp", "mpg", cv=True),
    "penguins_gcv": lambda d: smooth_spline(d, "flipper_length_mm", "body_mass_g"),
    "penguins_df": lambda d: smooth_spline(d, "bill_length_mm", "body_mass_g", df=6),
}
AT = {"cars": [2, 4, 10.5, 17, 25, 28], "mtcars": [1, 1.513, 2.5, 3.44, 5.424, 6],
      "penguins": [150, 172, 190.5, 205, 231, 240], "penguins_df": [30, 32.1, 45, 59.6, 62],
      "mtcars_cv": [40, 52, 150, 335, 360]}
# Where R searches for spar, Brent's parabolic steps amplify last-bit
# differences in the criterion (R's compiled code fuses multiply-adds), so
# the chosen spar agrees to about 1e-9 relative; R locates it only to 1e-4.
SEARCHED = {"cars_gcv", "cars_weighted", "cars_penalty", "mtcars_gcv", "mtcars_cv", "penguins_gcv"}


@pytest.fixture(scope="module")
def expected():
    return _oracle("smoothing")


def _quiet(make, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return make(data)


@pytest.mark.parametrize("backend", BACKENDS)
def test_loess_matches_r(expected, backend):
    for name, make in LOESS.items():
        source = name.split("_")[0]
        data, _ = as_input(DATASETS[source](), backend, [])
        fit = _quiet(make, data)
        result = glance(fit)
        assert type(result) is type(data)
        assert_matches_r(result, expected[f"{name}__glance"])
        assert_matches_r(as_polars(augment(fit)).select(".fitted", ".resid"), expected[f"{name}__augment"])
        assert_matches_r(pl.DataFrame({"weight": fit.robust_weights}), expected[f"{name}__robust"])
        new, _ = as_input(NEW[source], backend, [])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = predict(fit, new, se=True)
            se = as_polars(augment(fit, se_fit=True)).select(pl.col(".se_fit").alias("se_fit"))
        assert type(result) is type(new)
        assert_matches_r(result, expected[f"{name}__predict"])
        assert_matches_r(se, expected[f"{name}__se"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_loess_anova_and_groups_match_r(expected, backend):
    data, _ = as_input(cars(), backend, [])
    result = anova(loess(data, "dist ~ speed", span=.4), loess(data, "dist ~ speed"))
    assert type(result) is type(data)
    assert_matches_r(result, expected["anova"])
    data, by = as_input(penguins(), backend, ["species"])
    result = glance(loess(data, "body_mass_g ~ flipper_length_mm", by=by))
    assert type(result) is type(data)
    assert_matches_r(result, expected["loess_grouped"])


LOWESS = {
    "lowess_default": ("cars", lambda d, by: lowess(d, "speed", "dist", by=by)),
    "lowess_tight": ("cars", lambda d, by: lowess(d, "speed", "dist", f=.3, iter=0, by=by)),
    "lowess_delta": ("cars", lambda d, by: lowess(d, "speed", "dist", f=.5, delta=0, by=by)),
    "lowess_mtcars": ("mtcars", lambda d, by: lowess(d, "wt", "mpg", f=.4, iter=5, delta=.3, by=by)),
    "lowess_penguins": ("penguins", lambda d, by: lowess(d, "flipper_length_mm", "body_mass_g", by=by)),
    "ksmooth_box": ("cars", lambda d, by: ksmooth(d, "speed", "dist", by=by)),
    "ksmooth_normal": ("cars", lambda d, by: ksmooth(d, "speed", "dist", kernel="normal", bandwidth=2, by=by)),
    "ksmooth_points": ("cars", lambda d, by: ksmooth(d, "speed", "dist", bandwidth=3, x_points=[30, 2, 10.5, 15, 1], by=by)),
    "ksmooth_range": ("mtcars", lambda d, by: ksmooth(d, "wt", "mpg", kernel="normal", bandwidth=.5, range_x=(1, 6),
                                                      n_points=40, by=by)),
}


@pytest.mark.parametrize("backend", BACKENDS)
def test_lowess_and_ksmooth_match_r(expected, backend):
    for name, (source, make) in LOWESS.items():
        data, by = as_input(DATASETS[source](), backend, [])
        result = make(data, by)
        assert type(result) is type(data)
        assert_matches_r(result, expected[name])
    data, by = as_input(penguins(), backend, ["species"])
    assert_matches_r(lowess(data, "flipper_length_mm", "body_mass_g", f=.5, by=by), expected["lowess_grouped"])
    result = ksmooth(data, "flipper_length_mm", "body_mass_g", kernel="normal", bandwidth=5, n_points=20, by=by)
    assert type(result) is type(data)
    assert_matches_r(result, expected["ksmooth_grouped"])


def _scaled(want, rel):
    """Tolerances relative to each column's scale, for values near zero."""
    return {c: (rel, rel * max(abs(v) for v in d if v is not None)) for c, d in zip(want["columns"], want["data"])}


def _check_spline(fit, name, source, expected, backend, rel, deriv_rel):
    for got, key in ((as_polars(augment(fit)).select(".fitted", ".resid"), "augment"), (predict(fit), "fit")):
        want = expected[f"ss_{name}__{key}"]
        assert_matches_r(got, want, loose=_scaled(want, rel), rel=rel)
    at = AT.get(name, AT[source])
    for deriv in range(4):
        want = expected[f"ss_{name}__deriv{deriv}"]
        # A natural spline's f'' vanishes at the boundary knots, leaving only
        # cancellation there: compare relative to the derivative's scale.
        scale = max(abs(v) for v in want["data"][1])
        assert_matches_r(predict(fit, at, deriv=deriv), want, loose={"y": (deriv_rel, deriv_rel * scale)})
    frame, _ = as_input(pl.DataFrame({fit.formula.variables()[0].strip("`"): [float(v) for v in at]}), backend, [])
    result = predict(fit, frame, deriv=1)
    assert type(result) is type(frame)
    assert_matches_r(result, expected[f"ss_{name}__deriv1"], rel=deriv_rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_smooth_spline_matches_r(expected, backend):
    for name, make in SPLINES.items():
        source = name.split("_")[0]
        data, _ = as_input(DATASETS[source](), backend, [])
        fit = _quiet(make, data)
        want = expected[f"ss_{name}__glance"]
        result = glance(fit)
        assert type(result) is type(data)
        if name not in SEARCHED:
            assert_matches_r(result, want)
            _check_spline(fit, name, source, expected, backend, 1e-9, 1e-9)
            continue
        # Searched spar: within 1e-8 (lambda, 16^(6 spar), within 1e-7), and
        # derivatives, which amplify the spar difference, within 1e-6 ...
        assert_matches_r(result, want, rel=1e-8, loose={"lambda": 1e-7})
        _check_spline(fit, name, source, expected, backend, 1e-8, 1e-6)
        # ... while the fit at R's final lambda (the last one it evaluated)
        # matches R's to 1e-9.
        lam = want["data"][want["columns"].index("lambda")][0]
        x, y = fit.formula.variables()[0].strip("`"), fit.formula.response.strip("`")
        opts = {"cv": fit.cv, "penalty": 1.4, "df_offset": 1} if name == "cars_penalty" else {"cv": fit.cv}
        exact = _quiet(lambda d: smooth_spline(d, x, y, lambda_=lam, w="w" if name == "cars_weighted" else None,
                                              **opts), data)
        assert exact.df == pytest.approx(want["data"][0][0], rel=1e-9)
        assert exact.pen_crit == pytest.approx(want["data"][want["columns"].index("pen_crit")][0], rel=1e-9)
        _check_spline(exact, name, source, expected, backend, 1e-9, 1e-9)
    data, by = as_input(penguins(), backend, ["species"])
    result = glance(smooth_spline(data, "flipper_length_mm", "body_mass_g", df=4, by=by))
    assert type(result) is type(data)
    assert_matches_r(result, expected["spline_grouped"])


def test_pipes_missing_rows_and_errors():
    data = cars().with_columns(pl.when(pl.col("speed") == 7).then(None).otherwise(pl.col("dist")).alias("dist"))
    fit = data >> loess("dist ~ speed", na_action="exclude")
    assert augment(fit).height == data.height
    assert predict(fit)["fit"].null_count() == 2
    assert (data >> lowess("speed", "dist")).height == data.height - 2
    assert (data >> smooth_spline("speed", "dist")).n == data.height - 2
    assert "Equivalent Number of Parameters" in repr(fit)
    assert "GCV" in repr(smooth_spline(cars(), "speed", "dist"))
    with pytest.raises(TypeError, match="no coefficients"):
        tidy(fit)
    with pytest.raises(TypeError, match="no coefficients"):
        vcov(fit)
    with pytest.raises(TypeError, match="anova"):
        anova(fit)
    with pytest.raises(TypeError, match="loess fits with loess"):
        anova(loess(cars(), "dist ~ speed"), lm(cars(), "dist ~ speed"))
    with pytest.raises(ValueError, match="numeric"):
        loess(penguins(), "body_mass_g ~ species")
    with pytest.raises(ValueError, match="degree"):
        loess(cars(), "dist ~ speed", degree=3)
    with pytest.raises(ValueError, match="span is too small"):
        loess(cars(), "dist ~ speed", span=.001)
    with pytest.raises(ValueError, match="both"):
        smooth_spline(cars(), "speed", "dist", spar=.5, lambda_=1)
    with pytest.raises(ValueError, match="four unique"):
        smooth_spline({"x": [1., 2., 3., 1.], "y": [1., 2., 3., 4.]}, "x", "y")
    with pytest.raises(ValueError, match="kernel"):
        ksmooth(cars(), "speed", "dist", kernel="tri")
    with pytest.raises(ValueError, match="'f'"):
        lowess(cars(), "speed", "dist", f=0)
    with pytest.raises(ValueError, match="deriv"):
        predict(smooth_spline(cars(), "speed", "dist"), [1.], deriv=-1)


def test_smoothers_reproduce_simple_shapes():
    x = np.linspace(0, 1, 40)
    data = pl.DataFrame({"x": x, "y": 2 + 3 * x})
    # Local linear and quadratic fits reproduce a straight line.
    assert loess(data, "y ~ x", degree=1).fitted == pytest.approx(2 + 3 * x)
    assert lowess(data, "x", "y")["y"].to_numpy() == pytest.approx(2 + 3 * x)
    spline = smooth_spline(data, "x", "y", spar=1.2)
    assert predict(spline, [.5], deriv=1)["y"][0] == pytest.approx(3)
    assert predict(spline, [.5], deriv=2)["y"][0] == pytest.approx(0, abs=1e-6)
    assert ksmooth(data, "x", "y", x_points=[5.])["y"].to_list() == [None]
