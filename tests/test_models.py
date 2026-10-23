"""lm and glm against R: coefficients, fit statistics, influence, anova
tables, model comparisons and predictions, on every kind of input."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from stats3 import (
    Gamma,
    anova,
    augment,
    binomial,
    gaussian,
    glance,
    glm,
    inverse_gaussian,
    lm,
    poisson,
    predict,
    quasipoisson,
    tidy,
)
from oracle import BACKENDS, DATA, _oracle, as_input, as_polars, assert_matches_r, penguins


def mtcars() -> pl.DataFrame:
    return pl.read_csv(DATA / "mtcars.csv")


P, M = penguins, mtcars

FITS = {
    "lm_simple": (P, lambda: lm("body_mass_g ~ flipper_length_mm")),
    "lm_factor": (P, lambda: lm("body_mass_g ~ flipper_length_mm + species")),
    "lm_interaction": (P, lambda: lm("body_mass_g ~ flipper_length_mm * species")),
    "lm_two_factors": (P, lambda: lm("body_mass_g ~ species * sex")),
    "lm_no_intercept": (P, lambda: lm("body_mass_g ~ 0 + species")),
    "lm_margins": (P, lambda: lm("body_mass_g ~ species + species:sex")),
    "lm_transform": (P, lambda: lm("log(body_mass_g) ~ log(flipper_length_mm) + I(bill_length_mm^2)")),
    "lm_weights": (M, lambda: lm("mpg ~ wt + hp", weights="cyl")),
    "lm_factor_numeric": (M, lambda: lm("mpg ~ wt + factor(cyl)")),
    "lm_aliased": (M, lambda: lm("mpg ~ wt + I(2 * wt) + hp")),
    "lm_dot": (lambda: M().select("mpg", "wt", "hp", "qsec"), lambda: lm("mpg ~ .")),
    "glm_binomial": (P, lambda: glm("factor(sex) ~ body_mass_g + species", binomial())),
    "glm_binomial_01": (M, lambda: glm("am ~ wt + hp", binomial())),
    "glm_probit": (M, lambda: glm("am ~ wt", binomial(link="probit"))),
    "glm_cloglog": (M, lambda: glm("am ~ wt", binomial(link="cloglog"))),
    "glm_cbind": (M, lambda: glm("cbind(carb, gear) ~ wt + am", "binomial")),
    "glm_poisson": (M, lambda: glm("carb ~ hp + factor(cyl)", poisson())),
    "glm_poisson_offset": (M, lambda: glm("carb ~ hp + offset(log(gear))", poisson())),
    "glm_quasipoisson": (M, lambda: glm("carb ~ hp", quasipoisson())),
    "glm_gamma_log": (M, lambda: glm("mpg ~ wt + hp", Gamma(link="log"))),
    "glm_gamma_inverse": (M, lambda: glm("mpg ~ wt", Gamma())),
    "glm_inverse_gaussian": (M, lambda: glm("mpg ~ wt", inverse_gaussian())),
    "glm_gaussian_log": (M, lambda: glm("mpg ~ wt", gaussian(link="log"))),
}

TABLES = {
    "tidy": lambda fit: tidy(fit, conf_int=True),
    "glance": glance,
    "augment": lambda fit: augment(fit),
    "anova": anova,
}

# Values near zero are compared absolutely: models of 0/1 data that separate
# well have tiny influence values, and a residual of a row the model fits
# almost exactly (body mass near 4000 g, residual 0.0067) is the difference
# of two nearly equal numbers, so R (from Q) and stats3 (y - fitted) agree to
# about 1e-14 of the response's size, not relatively.
LOOSE = {".cooksd": (1e-9, 1e-12), ".hat": (1e-9, 1e-12), ".resid": (1e-9, 1e-8), ".std_resid": (1e-9, 1e-11)}


def _fit(name: str, backend: str, by: list[str] | None = None):
    data, make = FITS[name] if name in FITS else (None, None)
    frame, by_arg = as_input(data(), backend, by or [])
    step = make()
    step.by = by_arg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return frame >> step


def _expected_type(result, backend):
    expected = {"polars": "DataFrame", "lazy": "LazyFrame", "pandas": "DataFrame", "tidy3": "TidyFrame"}
    assert type(result).__name__ == expected[backend]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("table", sorted(TABLES))
@pytest.mark.parametrize("name", sorted(FITS))
def test_model_tables_match_r(name, table, backend):
    fit = _fit(name, backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TABLES[table](fit)
    _expected_type(result, backend)
    if table == "augment":
        result = as_polars(result).select(pl.col(r"^\..*$"))
    assert_matches_r(result, _oracle("models")[f"{name}__{table}"], loose=LOOSE)


GROUPED = {
    "lm_by_species__tidy": (P, lambda: lm("body_mass_g ~ flipper_length_mm"), lambda f: tidy(f, conf_int=True)),
    "lm_by_species__glance": (P, lambda: lm("body_mass_g ~ flipper_length_mm"), glance),
    "glm_by_species__tidy": (P, lambda: glm("factor(sex) ~ body_mass_g", binomial()), lambda f: tidy(f, conf_int=True)),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(GROUPED))
def test_grouped_models_match_r(name, backend):
    data, make, table = GROUPED[name]
    frame, by_arg = as_input(data(), backend, ["species"])
    step = make()
    step.by = by_arg
    models = frame >> step
    assert len(models) == 3
    result = table(models)
    _expected_type(result, backend)
    assert_matches_r(result, _oracle("models")[name])


def test_wald_intervals_match_confint_default():
    fit = glm(P(), "factor(sex) ~ body_mass_g + species", binomial())
    assert_matches_r(tidy(fit, conf_int=True, conf_method="wald"), _oracle("models")["glm_binomial__wald"])


def test_model_comparisons_match_r():
    m = M()
    lm_fits = [lm(m, f) for f in ("mpg ~ wt", "mpg ~ wt + hp", "mpg ~ wt + hp + qsec")]
    assert_matches_r(anova(*lm_fits), _oracle("models")["compare_lm"])
    glm_fits = [glm(m, f, binomial()) for f in ("am ~ wt", "am ~ wt + hp")]
    assert_matches_r(anova(*glm_fits), _oracle("models")["compare_glm"])
    gamma = [glm(m, f, Gamma(link="log")) for f in ("mpg ~ wt", "mpg ~ wt + hp")]
    assert_matches_r(anova(*gamma), _oracle("models")["compare_gamma"])


def test_predictions_match_r():
    fit = lm(P(), "body_mass_g ~ flipper_length_mm + species")
    new = pl.DataFrame({"flipper_length_mm": [190.0, 210.0], "species": ["Adelie", "Gentoo"]})
    assert_matches_r(predict(fit, new, interval="confidence"), _oracle("models")["predict_confidence"])
    assert_matches_r(predict(fit, new, interval="prediction", level=0.9), _oracle("models")["predict_prediction"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logistic = glm(M(), "am ~ wt + hp", binomial())
    cars = pl.DataFrame({"wt": [2.5, 3.5], "hp": [100.0, 200.0]})
    assert_matches_r(predict(logistic, cars, type="response", se_fit=True), _oracle("models")["predict_glm"])


# behaviour --------------------------------------------------------------------------


def test_pipe_and_data_first_forms_agree():
    a = tidy(P() >> lm("body_mass_g ~ flipper_length_mm"))
    b = tidy(lm(P(), "body_mass_g ~ flipper_length_mm"))
    c = lm(P(), "body_mass_g ~ flipper_length_mm") >> tidy()
    assert a.equals(b) and a.equals(c)


def test_printed_summary_reads_like_r():
    text = repr(lm(P(), "body_mass_g ~ flipper_length_mm + species"))
    assert text.startswith("lm(body_mass_g ~ flipper_length_mm + species)")
    assert "speciesGentoo" in text and "R² 0.7826" in text and "F(3, 338)" in text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = repr(glm(M(), "am ~ wt", binomial()))
    assert "binomial(link = 'logit')" in text and "AIC" in text


def test_aliased_coefficients_are_reported():
    fit = lm(M(), "mpg ~ wt + I(2 * wt) + hp")
    row = tidy(fit).filter(pl.col("term") == "I(2 * wt)").row(0, named=True)
    assert row["estimate"] is None and row["std_error"] is None
    assert "not estimated (aliased): I(2 * wt)" in repr(fit)


def test_enum_levels_set_the_reference_level():
    frame = P().with_columns(pl.col("species").cast(pl.Enum(["Gentoo", "Adelie", "Chinstrap"])))
    terms = tidy(lm(frame, "body_mass_g ~ species"))["term"].to_list()
    assert terms == ["(Intercept)", "speciesAdelie", "speciesChinstrap"]


def test_text_and_logical_responses_are_factors_for_binomial():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        as_text = tidy(glm(P(), "sex ~ body_mass_g", binomial()))
        as_factor = tidy(glm(P(), "factor(sex) ~ body_mass_g", binomial()))
        as_bool = tidy(glm(P().with_columns(male=pl.col("sex") == "male"), "male ~ body_mass_g", binomial()))
    assert as_text.equals(as_factor)
    np.testing.assert_allclose(as_text["estimate"], as_bool["estimate"])


def test_augment_with_newdata_adds_fitted_values():
    fit = lm(P(), "body_mass_g ~ flipper_length_mm")
    new = pl.DataFrame({"flipper_length_mm": [190.0, 210.0]})
    out = augment(fit, newdata=new, interval="confidence")
    assert out.columns == ["flipper_length_mm", ".fitted", ".lower", ".upper"]


def test_models_give_their_table_to_any_tidy():
    fit = lm(P(), "body_mass_g ~ flipper_length_mm + species")
    assert fit.__tidy__(conf_int=True).equals(tidy(fit, conf_int=True))
    models = lm(P(), "body_mass_g ~ flipper_length_mm", by="species")
    assert models.__tidy__().equals(tidy(models))


def test_tidy3_tidy_works_on_models_when_imported_last():
    tidy3 = pytest.importorskip("tidy3")
    fit = lm(P(), "body_mass_g ~ flipper_length_mm + species")
    try:
        result = tidy3.tidy(fit, conf_int=True)
    except TypeError:
        pytest.skip("this tidy3 has no __tidy__ hook yet")
    assert result.equals(tidy(fit, conf_int=True))
    assert (fit >> tidy3.tidy()).equals(tidy(fit))


def test_tidy_hands_data_to_tidy3():
    tidy3 = pytest.importorskip("tidy3")
    assert type(tidy({"x": [1, 2]})).__name__ == "TidyFrame"
    assert tidy3 is not None


def test_exponentiate_gives_odds_ratios():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = glm(M(), "am ~ wt", binomial())
    plain = tidy(fit)["estimate"].to_numpy()
    np.testing.assert_allclose(tidy(fit, exponentiate=True)["estimate"].to_numpy(), np.exp(plain))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: lm(P(), "body_mass_g ~ nothing"), "does not exist"),
        (lambda: lm(P(), "species ~ body_mass_g"), "must be numeric"),
        (lambda: lm(P(), "body_mass_g"), "formula must look like"),
        (lambda: glm(M(), "mpg ~ wt", Gamma(link="probit")), "not available"),
        (lambda: glm(M(), "mpg ~ wt", "student"), "family must be"),
        (lambda: glm(M(), "carb - 5 ~ wt", poisson()), "negative values not allowed"),
        (lambda: lm(M(), "mpg ~ poly(wt, 2)"), "cannot read"),
        (lambda: predict(lm(P(), "body_mass_g ~ species"), pl.DataFrame({"species": ["Emperor"]})), "new levels"),
        (lambda: lm(P(), "body_mass_g ~ species", by="species"), "grouping column"),
    ],
)
def test_errors(call, message):
    with pytest.raises((ValueError, KeyError, TypeError), match=message):
        call()


SUM = {"species": "sum", "sex": "sum"}
TYPES = {
    "type2_lm": lambda: anova(lm(P(), "body_mass_g ~ species * sex"), type=2),
    "type3_lm_sum": lambda: anova(lm(P(), "body_mass_g ~ species * sex", contrasts=SUM), type=3),
    "tidy_lm_sum": lambda: tidy(lm(P(), "body_mass_g ~ species * sex", contrasts=SUM), conf_int=True),
    "type2_mixed": lambda: anova(lm(P(), "body_mass_g ~ flipper_length_mm * species + sex"), type="II"),
    "type3_mixed": lambda: anova(lm(P(), "body_mass_g ~ flipper_length_mm * species + sex"), type="III"),
    "type2_glm": lambda: anova(glm(P(), "factor(sex) ~ body_mass_g * species", binomial()), type=2),
    "type3_glm_sum": lambda: anova(glm(P(), "factor(sex) ~ body_mass_g * species", binomial(),
                                       contrasts={"species": "sum"}), type=3),
    "type2_gamma": lambda: anova(glm(M(), "mpg ~ wt * factor(am) + hp", Gamma(link="log")), type=2),
}


@pytest.mark.parametrize("name", sorted(TYPES))
def test_type_2_and_3_tables_match_r(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TYPES[name]()
    assert_matches_r(result, _oracle("models")[name])


def test_type_2_and_3_need_one_model():
    m = M()
    with pytest.raises(ValueError, match="for one model"):
        anova(lm(m, "mpg ~ wt"), lm(m, "mpg ~ wt + hp"), type=2)
    with pytest.raises(ValueError, match="type must be"):
        anova(lm(m, "mpg ~ wt"), type=4)
    with pytest.raises(ValueError, match="contrasts must be"):
        lm(m, "mpg ~ factor(cyl)", contrasts="helmert")
