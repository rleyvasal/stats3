"""Nonlinear least squares against R's nls, across the four data backends."""

import warnings

import numpy as np
import polars as pl
import pytest

from stats3 import anova, augment, glance, lm, nls, predict, tidy, vcov
from stats3.nonlinear import _Expression, _derivative, _evaluate
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r


def puromycin():
    return pl.read_csv(DATA / "Puromycin.csv").with_row_index("i").with_columns(
        treated=(pl.col("state") == "treated").cast(pl.Float64),
        w=pl.when(pl.col("i") == 4).then(0.0).otherwise(pl.Series([1., 2., .5, 1.5] * 6)[:23])).drop("i")


def treated():
    return puromycin().filter(pl.col("state") == "treated")


def dnase1():
    return pl.read_csv(DATA / "DNase.csv").filter(pl.col("Run") == 1)


def chicks():
    return pl.read_csv(DATA / "ChickWeight.csv")


TIGHT = {"tol": 1e-7}
MICMEN = "rate ~ Vm * conc / (K + conc)"
# name: (data, model, keyword arguments)
CASES = {
    "micmen": (treated, MICMEN, {"start": {"Vm": 200, "K": .1}, "control": TIGHT}),
    "rise": (dnase1, "density ~ a * (1 - exp(-b * conc))", {"start": {"a": 2, "b": .3}, "control": TIGHT}),
    "functions": (dnase1, "density ~ a * log1p(conc^p) / (1 + sqrt(conc))",
                  {"start": {"a": 1, "p": 1}, "control": TIGHT}),
    "weighted": (puromycin, MICMEN, {"start": {"Vm": 200, "K": .1}, "weights": "w", "control": TIGHT}),
    "exclude": (lambda: treated().with_row_index("i").with_columns(
                    rate=pl.when(pl.col("i") == 2).then(None).otherwise(pl.col("rate"))).drop("i"),
                MICMEN, {"start": {"Vm": 200, "K": .1}, "na_action": "exclude", "control": TIGHT}),
    "plinear": (treated, "rate ~ conc / (K + conc)", {"start": {"K": .1}, "algorithm": "plinear", "control": TIGHT}),
    "plinear_cbind": (dnase1, "density ~ cbind(1, 1 - exp(-exp(lrc) * conc))",
                      {"start": {"lrc": -1}, "algorithm": "plinear", "control": TIGHT}),
    "port": (treated, MICMEN, {"start": {"Vm": 200, "K": .1}, "algorithm": "port", "control": TIGHT}),
    "port_bounded": (treated, MICMEN, {"start": {"Vm": 200, "K": .1}, "algorithm": "port",
                                       "lower": [0, .07], "upper": [250, 1], "control": TIGHT}),
    "logis": (dnase1, "density ~ SSlogis(log(conc), Asym, xmid, scal)", {"control": TIGHT}),
    "asymp": (dnase1, "density ~ SSasymp(conc, Asym, R0, lrc)", {"control": TIGHT}),
    "micmen_ss": (treated, "rate ~ SSmicmen(conc, Vm, K)", {"control": TIGHT}),
    "gompertz": (dnase1, "density ~ SSgompertz(log(conc), Asym, b2, b3)", {"control": TIGHT}),
    "fpl": (lambda: chicks().filter(pl.col("Chick") == 1), "weight ~ SSfpl(Time, A, B, xmid, scal)", {"control": TIGHT}),
    "weibull": (lambda: chicks().filter((pl.col("Chick") == 6) & (pl.col("Time") > 0)),
                "weight ~ SSweibull(Time, Asym, Drop, lrc, pwr)", {"control": TIGHT}),
    "logis_default": (dnase1, "density ~ SSlogis(log(conc), Asym, xmid, scal)", {}),
    "micmen_default": (treated, MICMEN, {"start": {"Vm": 200, "K": .1}}),
}
NEW = {"micmen": treated, "exclude": treated, "weighted": puromycin}


@pytest.fixture(scope="module", params=BACKENDS)
def fits(request):
    backend = request.param
    return backend, {name: nls(as_input(data(), backend, [])[0], model, **kwargs)
                     for name, (data, model, kwargs) in CASES.items()}


def _loose(name, table, fit):
    names = ("estimate", "std_error", "statistic", "sigma", "logLik", "AIC", "BIC", "deviance",
             ".fitted", "fit", *fit.parameter_names)
    loose, rel = {}, None
    if name in ("logis", "asymp", "micmen_ss", "gompertz", "fpl", "weibull", "port", "port_bounded"):
        # Both stop within tol = 1e-7 of the optimum but from different points:
        # selfStart initial values come from R's forward-difference plinear fits,
        # and PORT takes its own path, so the estimates agree to a few 1e-7.
        rel = 5e-7
    if name.endswith("_default"):
        # At R's default tol = 1e-5 the fit stops short of the optimum; with
        # selfStart, from initial values that differ by ~1e-8 (see above).
        rel = 1e-6
        loose["fin_tol"] = 1e-2
    if rel is not None:
        loose.update({c: rel for c in names}, p_value=10 * rel)
        # Residuals are differences of fitted values: compare on their scale.
        loose[".resid"] = (rel, rel * float(np.max(np.abs(fit.fitted))))
    # confint.nls interpolates a spline through about ten profile points;
    # stats3 solves for the end points (R's intervals approach them as
    # profile()'s delta.t shrinks), so they agree to R's interpolation error.
    loose.update(conf_low=6e-4, conf_high=6e-4)
    # Tight fits stop below tol = 1e-7 by rounding-level amounts.
    loose["fin_tol"] = (loose.get("fin_tol", 1.0), 1e-7)
    return loose


@pytest.mark.parametrize("table", ["tidy", "vcov", "glance", "augment", "predict"])
def test_tables_match_r(fits, table):
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    backend, models = fits
    for name, fit in models.items():
        expected = _oracle("nls")[name][table]
        new = as_input(NEW.get(name, CASES[name][0])().head(5), backend, [])[0]
        actual = {"tidy": lambda: tidy(fit, conf_int="conf_low" in expected["columns"]),
                  "vcov": lambda: vcov(fit), "glance": lambda: glance(fit),
                  "augment": lambda: as_polars(augment(fit)).select(".fitted", ".resid"),
                  "predict": lambda: predict(fit, new)}[table]()
        try:
            assert_matches_r(actual, expected, loose=_loose(name, table, fit))
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from error


@pytest.mark.parametrize("backend", BACKENDS)
def test_groups_comparisons_and_output_types(backend):
    source, by = as_input(puromycin(), backend, ["state"])
    grouped = nls(source, "rate ~ SSmicmen(conc, Vm, K)", by=by, control=TIGHT)
    assert len(grouped) == 2
    if R_COMMAND is not None:
        # As the selfStart cases in _loose.
        loose = {c: 5e-7 for c in ("estimate", "std_error", "statistic", "sigma", "logLik", "AIC", "BIC", "deviance")}
        loose.update(p_value=5e-6, conf_low=6e-4, conf_high=6e-4, fin_tol=(1.0, 1e-7))
        for table, fn in [("tidy", lambda f: tidy(f, conf_int=True)), ("glance", glance)]:
            assert_matches_r(fn(grouped), _oracle("nls")["grouped"][table], loose=loose)
    source, _ = as_input(puromycin(), backend, [])
    common = nls(source, "rate ~ SSmicmen(conc, Vm, K)", control=TIGHT)
    shifted = source >> nls("rate ~ (Vm + dV * treated) * conc / (K + conc)",
                            start={"Vm": 160, "dV": 50, "K": .05}, control=TIGHT)
    both = nls(source, "rate ~ (Vm + dV * treated) * conc / (K + dK * treated + conc)",
               start={"Vm": 160, "dV": 50, "K": .05, "dK": .01}, control=TIGHT)
    if R_COMMAND is not None:
        # The shared selfStart fit agrees to about 1e-8 (see _loose).
        loose = {c: 5e-7 for c in ("rss", "sumsq", "statistic")}
        loose["p_value"] = 5e-6
        assert_matches_r(anova(common, shifted, both), _oracle("nls")["compare"], loose=loose)
        assert_matches_r(anova(both, common), _oracle("nls")["compare_down"], loose=loose)
    expected = {"polars": ("polars", "DataFrame"), "lazy": ("polars", "LazyFrame"),
                "pandas": ("pandas", "DataFrame"), "tidy3": ("tidy3", "TidyFrame")}[backend]
    for output in [tidy(both), glance(both), predict(both), vcov(both), augment(both), anova(common, both)]:
        assert (type(output).__module__.split(".")[0], type(output).__name__) == expected
    assert as_polars(both >> tidy()).height == 4


def test_r_error_messages():
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    messages = dict(zip(*_oracle("nls")["errors"]["data"]))
    calls = {
        "maxiter": lambda: nls(treated(), MICMEN, start={"Vm": 200, "K": .1}, control={"maxiter": 2}),
        "min_factor": lambda: nls(dnase1(), "density ~ a * (1 - exp(-b * conc))", start={"a": 2, "b": .3},
                                  control={"min_factor": .9, "tol": 1e-12}),
        "singular": lambda: nls(treated(), "rate ~ (a + b) * conc / (K + conc)", start={"a": 100, "b": 100, "K": .1}),
    }
    for case, call in calls.items():
        with pytest.raises(ValueError) as error:
            call()
        assert str(error.value) == f"nls(): {messages[case]}"


def test_warn_only_returns_the_unconverged_fit():
    with pytest.warns(UserWarning, match="number of iterations exceeded maximum of 2"):
        fit = nls(treated(), MICMEN, start={"Vm": 200, "K": .1}, control={"maxiter": 2, "warn_only": True})
    assert not fit.converged and glance(fit)["is_conv"].item() is False
    with pytest.raises(ValueError, match="initial par violates constraints"):
        nls(treated(), MICMEN, start={"Vm": 200, "K": .1}, algorithm="port", lower={"K": .2})


@pytest.mark.parametrize("model", ["a * exp(-b * x) + c / x", "a ^ (b * x)", "sin(a * x) * cos(b) - tan(a / x)",
                                   "log(a * x, b) + log2(a) * log10(x) + sqrt(abs(a - x))",
                                   "pnorm(a * x) + dnorm(b - x) + plogis(a) * dlogis(b * x)",
                                   "gamma(a) + lgamma(b * x) + digamma(a * x) + trigamma(b) + psigamma(a, 2)",
                                   "asin(a / 4) + acos(b / 4) + atan(a * x) + sinh(b) * cosh(a) / tanh(x)",
                                   "expm1(a * x) + log1p(b * x) - -a^-b^2 + factorial(a) + lfactorial(b)",
                                   "sinpi(a * x) + cospi(b) + tanpi(a / 7)"])
def test_symbolic_gradient_matches_central_differences(model):
    expr = _Expression(f"y ~ {model}", "nls")
    env = {"x": np.array([.4, 1.1, 2.2]), "a": 1.3, "b": .7, "c": 2.0, "pi": np.pi}
    for name in ("a", "b"):
        exact = _evaluate(_derivative(expr.columns[0], name), env, {})
        h = 1e-6
        up, down = dict(env, **{name: env[name] + h}), dict(env, **{name: env[name] - h})
        numeric = (_evaluate(expr.columns[0], up, {}) - _evaluate(expr.columns[0], down, {})) / (2 * h)
        np.testing.assert_allclose(np.broadcast_to(exact, (3,)), numeric, rtol=1e-6, atol=1e-8)


def test_parser_follows_r_precedence():
    env = {"a": 2.0, "b": 3.0}
    for text, value in [("-a^2", -4), ("a^-1*3", 1.5), ("a^b^2", 2 ** 9), ("a**2", 4), ("-a*b", -6),
                        ("2e-1 * `a`", .4), ("a - b - 1", -2), ("a / b / 2", 1 / 3), ("pi", np.pi)]:
        expr = _Expression(f"~ {text}", "nls")
        assert _evaluate(expr.columns[0], dict(env, pi=np.pi), {}) == pytest.approx(value)
    for bad in ["y ~ a * ", "y ~ foo(a)", "y ~ (a", "y ~ a $ b", "a + b"]:
        with pytest.raises(ValueError):
            _Expression(bad, "nls")


def test_missing_rows_starting_values_and_arguments():
    source = treated().with_row_index("i").with_columns(
        rate=pl.when(pl.col("i") == 2).then(None).otherwise(pl.col("rate"))).drop("i")
    fit = nls(source, MICMEN, start={"Vm": 200, "K": .1}, na_action="exclude")
    assert augment(fit)[".fitted"].null_count() == 1 and predict(fit)["fit"].null_count() == 1
    assert nls(source, MICMEN, start={"Vm": 200, "K": .1}).n == 11
    with pytest.raises(ValueError, match="missing"):
        nls(source, MICMEN, start={"Vm": 200, "K": .1}, na_action="fail")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nls(treated(), MICMEN, control={"warn_only": True})
    assert "Initializing 'Vm', 'K' to '1.'" in str(caught[0].message)
    with pytest.raises(ValueError, match="parameters without starting value in 'data': K"):
        nls(treated(), MICMEN, start={"Vm": 200})
    with pytest.warns(UserWarning, match="bounds ignored"):
        nls(treated(), MICMEN, start={"Vm": 200, "K": .1}, lower=0)
    with pytest.raises(ValueError, match="plinear"):
        nls(treated(), "rate ~ cbind(1, conc)", start={"K": 1})
    with pytest.raises(ValueError, match="algorithm"):
        nls(treated(), MICMEN, start={"Vm": 200, "K": .1}, algorithm="lm")
    with pytest.raises(ValueError, match="unknown control"):
        nls(treated(), MICMEN, start={"Vm": 200, "K": .1}, control={"maxit": 3})
    with pytest.raises(ValueError, match="too few distinct input values"):
        nls(treated().head(4), "rate ~ SSlogis(conc, Asym, xmid, scal)")
    with pytest.raises(ValueError, match="not numeric"):
        nls(puromycin(), "rate ~ Vm * state", start={"Vm": 1})
    fit = nls(treated(), MICMEN, start={"Vm": 200, "K": .1})
    with pytest.raises(TypeError, match="sequences of nls"):
        anova(fit)
    with pytest.raises(TypeError, match="nls fits"):
        anova(fit, lm(treated(), "rate ~ conc"))
    with pytest.raises(ValueError, match="vcov"):
        tidy(fit, vcov="HC3")
    assert "Residual standard error" in repr(fit)
    np.testing.assert_allclose(predict(fit, pl.DataFrame({"conc": [.5]}))["fit"].to_numpy(),
                               fit.coefficients["Vm"] * .5 / (fit.coefficients["K"] + .5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert list(nls(treated(), "rate ~ SSmicmen(input = conc, K = K, Vm = Vm)").coefficients) == ["Vm", "K"]
