"""t_test(): R's t.test is the reference; scipy is a second one that needs no R."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from tidy3 import col, fct_relevel, filter, group_by, mutate, tidy
from stats3 import t_test
from oracle import PENGUINS, _oracle, assert_matches_r, assert_same_frame

scipy_stats = pytest.importorskip("scipy.stats")

def _penguins():
    return tidy(pl.read_csv(PENGUINS, null_values="NA"))


def _welch(p):
    return p >> t_test("body_mass_g ~ sex")


def _student(p):
    return p >> t_test("body_mass_g ~ sex", var_equal=True)


def _paired(p):
    return p >> t_test("bill_length_mm", "bill_depth_mm", paired=True)


def _one(p):
    return p >> t_test("flipper_length_mm", mu=200)


def _grouped(test, *groups):
    return lambda p: test(p >> group_by(*groups))


CASES = {
    "welch": _welch,
    "student": _student,
    "paired": _paired,
    "one_sample": _one,
    "one_sample_formula": lambda p: p >> t_test("flipper_length_mm ~ 1", mu=200),
    "two_columns": lambda p: p >> t_test("bill_length_mm", "bill_depth_mm"),
    "welch_mu_less": lambda p: p >> t_test(
        "body_mass_g ~ sex", mu=-500, alternative="less", conf_level=0.9
    ),
    "student_greater": lambda p: p >> t_test(
        "body_mass_g ~ sex", var_equal=True, alternative="greater", conf_level=0.99
    ),
    "paired_less": lambda p: p >> t_test(
        "bill_length_mm", "bill_depth_mm", paired=True, mu=27, alternative="less"
    ),
    "one_sample_greater": lambda p: p >> t_test(
        "flipper_length_mm", mu=200, alternative="greater"
    ),
    "welch_by_species": _grouped(_welch, "species"),
    "student_by_species": _grouped(_student, "species"),
    "paired_by_species": _grouped(_paired, "species"),
    "one_sample_by_species": _grouped(_one, "species"),
    "welch_by_species_island": _grouped(_welch, "species", "island"),
    "student_by_island_less": lambda p: p >> group_by("island") >> t_test(
        "body_mass_g ~ sex", var_equal=True, alternative="less"
    ),
    "welch_by_sex_species_levels": lambda p: p
    >> filter(col("species") != "Chinstrap")
    >> group_by("sex")
    >> t_test("flipper_length_mm ~ species"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_t_test_matches_r(case):
    assert_matches_r(CASES[case](_penguins()), _oracle("t_test")[case])


def test_one_sided_intervals_are_infinite_on_the_open_side():
    p = _penguins()
    less = (p >> t_test("body_mass_g ~ sex", alternative="less")).collect()
    greater = (p >> t_test("body_mass_g ~ sex", alternative="greater")).collect()
    assert less["conf_low"][0] == -math.inf and math.isfinite(less["conf_high"][0])
    assert greater["conf_high"][0] == math.inf and math.isfinite(greater["conf_low"][0])


# scipy reference -----------------------------------------------------------


def _values(name, where=None):
    frame = pl.read_csv(PENGUINS, null_values="NA")
    if where is not None:
        frame = frame.filter(where)
    return frame[name].drop_nulls().cast(pl.Float64).to_numpy()


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
    frame = pl.read_csv(PENGUINS, null_values="NA").drop_nulls(["body_mass_g", "sex"])
    female = frame.filter(pl.col("sex") == "female")["body_mass_g"].cast(pl.Float64).to_numpy()
    male = frame.filter(pl.col("sex") == "male")["body_mass_g"].cast(pl.Float64).to_numpy()
    ref = scipy_stats.ttest_ind(female, male, equal_var=var_equal, alternative=alternative)
    row = (_penguins() >> t_test(
        "body_mass_g ~ sex", var_equal=var_equal, alternative=alternative.replace("-", "_")
    )).collect().row(0, named=True)
    _assert_scipy(row, ref, ref.df)
    assert row["estimate"] == pytest.approx(female.mean() - male.mean(), rel=1e-12)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_one_sample_matches_scipy(alternative):
    x = _values("flipper_length_mm")
    ref = scipy_stats.ttest_1samp(x, 200, alternative=alternative)
    row = (_penguins() >> t_test(
        "flipper_length_mm", mu=200, alternative=alternative.replace("-", "_")
    )).collect().row(0, named=True)
    _assert_scipy(row, ref, len(x) - 1)


def test_paired_matches_scipy():
    frame = pl.read_csv(PENGUINS, null_values="NA").drop_nulls(["bill_length_mm", "bill_depth_mm"])
    x, y = frame["bill_length_mm"].to_numpy(), frame["bill_depth_mm"].to_numpy()
    ref = scipy_stats.ttest_rel(x, y)
    row = (_penguins() >> t_test("bill_length_mm", "bill_depth_mm", paired=True)).collect().row(0, named=True)
    _assert_scipy(row, ref, len(x) - 1)


def test_grouped_rows_equal_tests_on_each_subset():
    grouped = (_penguins() >> group_by("species") >> t_test("body_mass_g ~ sex")).collect()
    for row in grouped.iter_rows(named=True):
        alone = (
            _penguins() >> filter(col("species") == row["species"]) >> t_test("body_mass_g ~ sex")
        ).collect()
        part = grouped.filter(pl.col("species") == row["species"]).drop("species")
        assert_same_frame(part, alone)


# behaviour -----------------------------------------------------------------


def test_result_is_ungrouped_with_groups_in_dplyr_order():
    result = _penguins() >> group_by("island", "species") >> t_test("body_mass_g ~ sex")
    assert result._groups is None
    keys = result.collect().select("island", "species").rows()
    assert keys == sorted(keys)


def test_factor_levels_choose_the_first_group():
    p = _penguins() >> mutate(sex=fct_relevel(col("sex"), "male"))
    row = (p >> t_test("body_mass_g ~ sex")).collect().row(0, named=True)
    assert row["estimate1"] > row["estimate2"]
    assert row["estimate"] == pytest.approx(683.4117965367964)


def test_column_forms_agree():
    p = _penguins()
    by_name = (p >> t_test("flipper_length_mm", mu=200)).collect()
    by_col = (p >> t_test(col("flipper_length_mm"), mu=200)).collect()
    by_formula = (p >> t_test("flipper_length_mm ~ 1", mu=200)).collect()
    assert by_name.equals(by_col) and by_name.equals(by_formula)


def test_called_as_a_function_and_on_lazy_input():
    lazy = tidy(pl.scan_csv(PENGUINS, null_values="NA"))
    assert_same_frame(t_test("body_mass_g ~ sex")(lazy).collect(), _welch(_penguins()).collect())


def test_nan_and_null_are_both_missing():
    frame = tidy({"x": [1.0, 2.0, 4.0, float("nan"), None]})
    row = (frame >> t_test("x")).collect().row(0, named=True)
    ref = scipy_stats.ttest_1samp([1.0, 2.0, 4.0], 0)
    assert row["df"] == 2
    assert row["statistic"] == pytest.approx(ref.statistic)


def test_rows_missing_the_group_are_dropped():
    frame = tidy({"y": [1.0, 2.0, 3.0, 5.0, 8.0, 100.0], "g": ["a", "a", "b", "b", "b", None]})
    row = (frame >> t_test("y ~ g")).collect().row(0, named=True)
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
        tidy(frame) >> call()


def test_student_test_needs_three_observations_not_two_per_group():
    frame = tidy({"y": [1.0, 2.0, 4.0], "g": ["a", "a", "b"]})
    row = (frame >> t_test("y ~ g", var_equal=True)).collect().row(0, named=True)
    ref = scipy_stats.ttest_ind([1.0, 2.0], [4.0], equal_var=True)
    assert row["statistic"] == pytest.approx(ref.statistic)


def test_error_names_the_group():
    frame = tidy({"k": ["p", "p", "q", "q"], "y": [1.0, 2.0, 3.0, 5.0], "g": ["a", "b", "a", "a"]})
    with pytest.raises(ValueError, match="k = 'q'"):
        frame >> group_by("k") >> t_test("y ~ g")


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
