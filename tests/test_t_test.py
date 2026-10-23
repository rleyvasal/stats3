"""t_test(): R's t.test is the reference; scipy is a second one that needs no R."""

from __future__ import annotations

import math

import polars as pl
import pytest
import scipy.stats as scipy_stats

from stats3 import t_test
from oracle import (
    BACKENDS,
    _oracle,
    assert_matches_r,
    assert_same_frame,
    case,
    penguins,
    run_case,
)

P = penguins


def _welch():
    return t_test("body_mass_g ~ sex")


def _student():
    return t_test("body_mass_g ~ sex", var_equal=True)


def _paired():
    return t_test("bill_length_mm", "bill_depth_mm", paired=True)


def _one():
    return t_test("flipper_length_mm", mu=200)


CASES = {
    "welch": case(P, _welch),
    "student": case(P, _student),
    "paired": case(P, _paired),
    "one_sample": case(P, _one),
    "one_sample_formula": case(P, lambda: t_test("flipper_length_mm ~ 1", mu=200)),
    "two_columns": case(P, lambda: t_test("bill_length_mm", "bill_depth_mm")),
    "welch_mu_less": case(P, lambda: t_test("body_mass_g ~ sex", mu=-500, alternative="less", conf_level=0.9)),
    "student_greater": case(P, lambda: t_test("body_mass_g ~ sex", var_equal=True, alternative="greater", conf_level=0.99)),
    "paired_less": case(P, lambda: t_test("bill_length_mm", "bill_depth_mm", paired=True, mu=27, alternative="less")),
    "one_sample_greater": case(P, lambda: t_test("flipper_length_mm", mu=200, alternative="greater")),
    "welch_by_species": case(P, _welch, "species"),
    "student_by_species": case(P, _student, "species"),
    "paired_by_species": case(P, _paired, "species"),
    "one_sample_by_species": case(P, _one, "species"),
    "welch_by_species_island": case(P, _welch, "species", "island"),
    "student_by_island_less": case(P, lambda: t_test("body_mass_g ~ sex", var_equal=True, alternative="less"), "island"),
    "welch_by_sex_species_levels": case(
        lambda: P().filter(pl.col("species") != "Chinstrap"),
        lambda: t_test("flipper_length_mm ~ species"),
        "sex",
    ),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_t_test_matches_r(name, backend):
    assert_matches_r(run_case(CASES[name], backend), _oracle("t_test")[name])


def test_one_sided_intervals_are_infinite_on_the_open_side():
    less = t_test(P(), "body_mass_g ~ sex", alternative="less")
    greater = t_test(P(), "body_mass_g ~ sex", alternative="greater")
    assert less["conf_low"][0] == -math.inf and math.isfinite(less["conf_high"][0])
    assert greater["conf_high"][0] == math.inf and math.isfinite(greater["conf_low"][0])


# scipy reference -----------------------------------------------------------


def _values(name):
    return P()[name].drop_nulls().cast(pl.Float64).to_numpy()


def _assert_scipy(row, ref, df):
    ci = ref.confidence_interval(0.95)
    assert row["statistic"] == pytest.approx(ref.statistic, rel=1e-10)
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)
    assert row["df"] == pytest.approx(df, rel=1e-10)
    assert row["conf_low"] == pytest.approx(ci.low, rel=1e-10)
    assert row["conf_high"] == pytest.approx(ci.high, rel=1e-10)


@pytest.mark.parametrize("var_equal", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_two_sample_matches_scipy(var_equal, alternative):
    # Grouping rows need both columns present, as R's formula method does.
    frame = P().drop_nulls(["body_mass_g", "sex"])
    female = frame.filter(pl.col("sex") == "female")["body_mass_g"].cast(pl.Float64).to_numpy()
    male = frame.filter(pl.col("sex") == "male")["body_mass_g"].cast(pl.Float64).to_numpy()
    ref = scipy_stats.ttest_ind(female, male, equal_var=var_equal, alternative=alternative)
    row = t_test(
        P(), "body_mass_g ~ sex", var_equal=var_equal, alternative=alternative.replace("-", "_")
    ).row(0, named=True)
    _assert_scipy(row, ref, ref.df)
    assert row["estimate"] == pytest.approx(female.mean() - male.mean(), rel=1e-12)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_one_sample_matches_scipy(alternative):
    x = _values("flipper_length_mm")
    ref = scipy_stats.ttest_1samp(x, 200, alternative=alternative)
    row = t_test(P(), "flipper_length_mm", mu=200, alternative=alternative.replace("-", "_")).row(0, named=True)
    _assert_scipy(row, ref, len(x) - 1)


def test_paired_matches_scipy():
    frame = P().drop_nulls(["bill_length_mm", "bill_depth_mm"])
    x, y = frame["bill_length_mm"].to_numpy(), frame["bill_depth_mm"].to_numpy()
    ref = scipy_stats.ttest_rel(x, y)
    row = t_test(P(), "bill_length_mm", "bill_depth_mm", paired=True).row(0, named=True)
    _assert_scipy(row, ref, len(x) - 1)


def test_grouped_rows_equal_tests_on_each_subset():
    grouped = t_test(P(), "body_mass_g ~ sex", by="species")
    for species in grouped["species"]:
        alone = t_test(P().filter(pl.col("species") == species), "body_mass_g ~ sex")
        assert_same_frame(grouped.filter(pl.col("species") == species).drop("species"), alone)


# behaviour -----------------------------------------------------------------


def test_groups_come_out_in_sorted_order():
    keys = t_test(P(), "body_mass_g ~ sex", by=["island", "species"]).select("island", "species").rows()
    assert keys == sorted(keys)


def test_enum_levels_choose_the_first_group():
    frame = P().with_columns(pl.col("sex").cast(pl.Enum(["male", "female"])))
    row = t_test(frame, "body_mass_g ~ sex").row(0, named=True)
    assert row["estimate1"] > row["estimate2"]
    assert row["estimate"] == pytest.approx(683.4117965367964)


def test_column_forms_agree():
    by_name = t_test(P(), "flipper_length_mm", mu=200)
    by_expr = t_test(P(), pl.col("flipper_length_mm"), mu=200)
    by_formula = t_test(P(), "flipper_length_mm ~ 1", mu=200)
    assert_same_frame(by_name, by_expr)
    assert_same_frame(by_name, by_formula)


def test_pipe_function_and_data_first_forms_agree():
    piped = P() >> t_test("body_mass_g ~ sex", by="species")
    called = t_test("body_mass_g ~ sex", by="species")(P())
    data_first = t_test(P(), "body_mass_g ~ sex", by="species")
    # Polars may sum a group's floats in a different order from run to run.
    assert_same_frame(piped, called)
    assert_same_frame(piped, data_first)


def test_nan_and_null_are_both_missing():
    row = t_test({"x": [1.0, 2.0, 4.0, float("nan"), None]}, "x").row(0, named=True)
    ref = scipy_stats.ttest_1samp([1.0, 2.0, 4.0], 0)
    assert row["df"] == 2
    assert row["statistic"] == pytest.approx(ref.statistic)


def test_rows_missing_the_group_are_dropped():
    data = {"y": [1.0, 2.0, 3.0, 5.0, 8.0, 100.0], "g": ["a", "a", "b", "b", "b", None]}
    row = t_test(data, "y ~ g").row(0, named=True)
    ref = scipy_stats.ttest_ind([1.0, 2.0], [3.0, 5.0, 8.0], equal_var=False)
    assert row["statistic"] == pytest.approx(ref.statistic)


@pytest.mark.parametrize(
    ("frame", "call", "message"),
    [
        ({"y": [1.0, 2.0, 3.0], "g": ["a"] * 3}, lambda: t_test("y ~ g"), "exactly 2 levels"),
        ({"y": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]}, lambda: t_test("y ~ g"), "exactly 2 levels"),
        ({"y": [1.0, 2.0, 3.0], "g": ["a", "a", "b"]}, lambda: t_test("y ~ g"), "not enough 'y'"),
        ({"x": [1.0]}, lambda: t_test("x"), "not enough 'x'"),
        ({"x": [3.0, 3.0, 3.0]}, lambda: t_test("x"), "essentially constant"),
        ({"x": [1.0, 2.0]}, lambda: t_test("x ~ y"), "does not exist"),
        ({"x": [1.0, 2.0]}, lambda: t_test("x ~"), "formula must look like"),
    ],
)
def test_errors_follow_r(frame, call, message):
    with pytest.raises((ValueError, KeyError), match=message):
        pl.DataFrame(frame) >> call()


def test_student_test_needs_three_observations_not_two_per_group():
    row = t_test({"y": [1.0, 2.0, 4.0], "g": ["a", "a", "b"]}, "y ~ g", var_equal=True).row(0, named=True)
    ref = scipy_stats.ttest_ind([1.0, 2.0], [4.0], equal_var=True)
    assert row["statistic"] == pytest.approx(ref.statistic)


def test_error_names_the_group():
    data = {"k": ["p", "p", "q", "q"], "y": [1.0, 2.0, 3.0, 5.0], "g": ["a", "b", "a", "a"]}
    with pytest.raises(ValueError, match="k = 'q'"):
        t_test(data, "y ~ g", by="k")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alternative": "both"}, "alternative must be"),
        ({"conf_level": 1.0}, "conf_level must be"),
        ({"mu": float("nan")}, "mu must be"),
        ({"paired": True}, "paired=True needs two columns"),
    ],
)
def test_argument_checks(kwargs, message):
    with pytest.raises(ValueError, match=message):
        t_test("x", **kwargs)


def test_formula_and_paired_or_y_are_refused():
    with pytest.raises(ValueError, match="not a formula"):
        t_test("y ~ g", paired=True)
    with pytest.raises(ValueError, match="not both"):
        t_test("y ~ g", "x")
