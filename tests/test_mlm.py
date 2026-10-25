"""Multivariate linear models against R: mlm, summary.manova, anova.mlm, mauchly.test."""

import numpy as np
import polars as pl
import pytest

from stats3 import anova, lm, manova, mauchly_test, predict, tidy, vcov
from oracle import BACKENDS, DATA, _oracle, as_input, assert_matches_r, penguins


def iris():
    return pl.read_csv(DATA / "iris.csv").rename(lambda c: c.lower().replace(".", "_"))


def weighted_penguins():
    return penguins().with_row_index("i").with_columns(w=((pl.col("i") + 1) % 5) + 1)


FITS = {
    "iris": (iris, "cbind(sepal_length, sepal_width, petal_length, petal_width) ~ species"),
    "numeric": (iris, "cbind(petal_length, petal_width) ~ sepal_length + sepal_width"),
    "penguins": (penguins, "cbind(bill_length_mm, bill_depth_mm, flipper_length_mm) ~ species + sex + body_mass_g"),
    "interaction": (penguins, "cbind(bill_length_mm, bill_depth_mm) ~ species * sex"),
    "expressions": (penguins, "cbind(log(bill_length_mm), bill_depth_mm / 10) ~ island + year"),
    "no_intercept": (penguins, "cbind(bill_length_mm, bill_depth_mm, flipper_length_mm) ~ 0 + species"),
    "aliased": (penguins, "cbind(bill_length_mm, bill_depth_mm) ~ body_mass_g + I(2 * body_mass_g) + species"),
}
TESTS = {"pillai": "Pillai", "wilks": "Wilks", "hl": "Hotelling-Lawley", "roy": "Roy"}
IDATA = pl.DataFrame({"part": ["sepal", "sepal", "petal", "petal"], "dim": ["length", "width", "length", "width"]})
DESIGNS = {
    "identity": {},
    "within": {"X": "~ 1"},
    "part": {"X": "~ part", "idata": IDATA},
    "dim": {"M": "~ part * dim", "X": "~ dim", "idata": IDATA},
    "interaction": {"M": "~ part * dim", "X": "~ part + dim", "idata": IDATA},
    "one_row": {"M": "~ part + dim", "X": "~ part", "idata": IDATA},
    "matrix": {"M": np.column_stack([np.ones(4), [1, 1, -1, -1], [1, -1, 1, -1]]), "X": np.ones((4, 1))},
}
YEAR = "cbind(bill_length_mm, bill_depth_mm, flipper_length_mm) ~ species + sex"


@pytest.fixture(scope="module")
def r_mlm():
    return _oracle("mlm")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(FITS))
def test_fits_match_r(name, backend, r_mlm):
    load, formula = FITS[name]
    data, _ = as_input(load(), backend, [])
    fit = lm(data, formula)
    result = tidy(fit, conf_int=True)
    assert type(result) is type(data)
    assert_matches_r(result, r_mlm[f"{name}__tidy"])
    for column, test in TESTS.items():
        assert_matches_r(manova(data, formula, test=test), r_mlm[f"{name}__manova_{column}"])
        assert_matches_r(anova(fit, test=test), r_mlm[f"{name}__anova_{column}"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_runs_match_r(backend, r_mlm):
    data, by = as_input(penguins(), backend, ["year"])
    assert_matches_r(manova(data, YEAR, by=by), r_mlm["by_year_manova"])
    fit = lm(data, YEAR, by=by)
    assert_matches_r(tidy(fit, conf_int=True), r_mlm["by_year_tidy"])
    assert_matches_r(anova(fit, test="Wilks"), r_mlm["by_year_anova"])
    assert_matches_r(mauchly_test(fit, X="~ 1"), r_mlm["by_year_mauchly"])
    assert_matches_r(anova(fit, test="Spherical", X="~ 1"), r_mlm["by_year_spherical"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_within_designs_match_r(name, backend, r_mlm):
    data, _ = as_input(iris(), backend, [])
    fit = lm(data, FITS["iris"][1])
    args = DESIGNS[name]
    assert_matches_r(mauchly_test(fit, **args), r_mlm[f"mauchly__{name}"])
    assert_matches_r(anova(fit, test="Spherical", **args), r_mlm[f"spherical__{name}"])
    if f"transformed__{name}" in r_mlm:
        assert_matches_r(anova(fit, **args), r_mlm[f"transformed__{name}"])


def test_one_contrast_multivariate_equals_univariate():
    # R's anova.mlm fails for a one-row contrast; with one row the
    # multivariate F is the univariate F.
    fit = lm(iris(), FITS["iris"][1])
    multi = anova(fit, **DESIGNS["one_row"])["statistic"].to_list()
    uni = anova(fit, test="Spherical", **DESIGNS["one_row"])["statistic"].to_list()
    assert multi[:-1] == pytest.approx(uni[:-1], rel=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_comparisons_predict_and_vcov_match_r(backend, r_mlm):
    data, _ = as_input(iris(), backend, [])
    big = lm(data, FITS["iris"][1])
    small = lm(data, "cbind(sepal_length, sepal_width, petal_length, petal_width) ~ 1")
    for column, test in TESTS.items():
        assert_matches_r(anova(small, big, test=test), r_mlm[f"compare_{column}"])
    assert_matches_r(anova(big, small), r_mlm["compare_reversed"])
    assert_matches_r(anova(small, big, test="Spherical", X="~ 1"), r_mlm["compare_spherical"])
    p, _ = as_input(penguins(), backend, [])
    new, _ = as_input(penguins()[[0, 1, 2, 4, 5]], backend, [])
    assert_matches_r(predict(lm(p, FITS["penguins"][1]), new), r_mlm["predict"])
    assert_matches_r(vcov(lm(p, FITS["aliased"][1])), r_mlm["vcov"])


def test_weighted_manova_matches_r(r_mlm):
    result = manova(weighted_penguins(), FITS["penguins"][1], weights="w")
    assert_matches_r(result, r_mlm["manova_weighted"])
    assert_matches_r(manova(iris(), FITS["iris"][1], test="Wilks", intercept=True), r_mlm["manova_intercept"])


def test_errors_and_verbs():
    fit = lm(iris(), FITS["iris"][1])
    assert (fit >> anova(test="Spherical", X="~ 1")).equals(anova(fit, test="Spherical", X="~ 1"))
    assert (fit >> mauchly_test(X="~ 1")).equals(mauchly_test(fit, X="~ 1"))
    with pytest.raises(ValueError, match="need multiple responses"):
        manova(iris(), "sepal_length ~ species")
    with pytest.raises(ValueError, match="weights are not supported"):
        anova(lm(weighted_penguins(), FITS["penguins"][1], weights="w"))
    with pytest.raises(ValueError, match="X does not define a subspace of M"):
        mauchly_test(fit, M="~ part", X="~ dim", idata=IDATA)
    with pytest.raises(ValueError, match="test must be one of"):
        anova(fit, test="Bartlett")
    with pytest.raises(TypeError, match="for multivariate lm fits"):
        anova(lm(iris(), "sepal_length ~ species"), X="~ 1")
    with pytest.raises(TypeError, match="glance"):
        from stats3 import glance
        glance(fit)
