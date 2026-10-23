"""What each kind of input brings: pandas categories, Polars enums, tidy3
groups, factor levels and expressions; and the output frame types."""

from __future__ import annotations

import polars as pl
import pytest

from stats3 import binom_test, p_adjust, power_t_test, t_test
from oracle import assert_same_frame, as_polars, penguins

P = penguins


# pandas -------------------------------------------------------------------------


def test_pandas_categories_order_the_levels_and_survive_as_group_columns():
    pd = pytest.importorskip("pandas")
    frame = P().to_pandas()
    frame["sex"] = pd.Categorical(frame["sex"], categories=["male", "female"])
    frame["species"] = pd.Categorical(frame["species"], categories=["Gentoo", "Chinstrap", "Adelie"])
    result = t_test(frame, "body_mass_g ~ sex", by="species")
    assert isinstance(result, pd.DataFrame)
    assert list(result["species"]) == ["Gentoo", "Chinstrap", "Adelie"]
    assert list(result["species"].cat.categories) == ["Gentoo", "Chinstrap", "Adelie"]
    assert (result["estimate"] > 0).all()  # male minus female


def test_pandas_results_match_polars():
    pytest.importorskip("pandas")
    from_pandas = t_test(P().to_pandas(), "body_mass_g ~ sex", by="species")
    assert_same_frame(as_polars(from_pandas), t_test(P(), "body_mass_g ~ sex", by="species"))


# Polars -----------------------------------------------------------------------------


def test_lazy_input_gives_lazy_output():
    result = P().lazy() >> t_test("body_mass_g ~ sex")
    assert isinstance(result, pl.LazyFrame)


def test_as_converts_the_output():
    pd = pytest.importorskip("pandas")
    assert isinstance(t_test(P(), "body_mass_g ~ sex", as_="pandas"), pd.DataFrame)
    assert isinstance(t_test(P().to_pandas(), "body_mass_g ~ sex", as_="polars"), pl.DataFrame)
    with pytest.raises(ValueError, match="as_ must be one of"):
        t_test("body_mass_g ~ sex", as_="arrow")


def test_dict_of_columns_is_read_as_polars():
    result = t_test({"x": [1.0, 2.0, 4.0]}, "x")
    assert isinstance(result, pl.DataFrame)


def test_other_inputs_are_refused():
    with pytest.raises(TypeError, match="expected a Polars or pandas DataFrame"):
        [1, 2, 3] >> t_test("x")


def test_by_accepts_polars_column_expressions():
    assert_same_frame(
        t_test(P(), "body_mass_g ~ sex", by=pl.col("species")),
        t_test(P(), "body_mass_g ~ sex", by="species"),
    )


# tidy3 -------------------------------------------------------------------------------


@pytest.fixture
def tidy3():
    return pytest.importorskip("tidy3")


def test_tidy3_groups_and_factor_levels_are_read(tidy3):
    frame = tidy3.tidy(P()) >> tidy3.mutate(sex=tidy3.fct_relevel(tidy3.col("sex"), "male"))
    result = frame >> tidy3.group_by("species") >> t_test("body_mass_g ~ sex")
    assert type(result).__name__ == "TidyFrame"
    assert (result.collect()["estimate"] > 0).all()  # male first, from fct_relevel


def test_tidy3_output_keeps_group_factor_levels(tidy3):
    frame = tidy3.tidy(P()) >> tidy3.mutate(
        species=tidy3.fct_relevel(tidy3.col("species"), "Gentoo", "Chinstrap", "Adelie")
    )
    result = frame >> tidy3.group_by("species") >> t_test("body_mass_g ~ sex")
    assert result.collect()["species"].to_list() == ["Gentoo", "Chinstrap", "Adelie"]
    arranged = result >> tidy3.arrange("species")
    assert arranged.collect()["species"].to_list() == ["Gentoo", "Chinstrap", "Adelie"]


def test_tidy3_expressions_are_compiled(tidy3):
    frame = tidy3.tidy(P())
    with_tidy3 = frame >> binom_test(tidy3.col("sex") == "male")
    with_polars = frame >> binom_test(pl.col("sex") == "male")
    assert_same_frame(with_tidy3.collect(), with_polars.collect())
    by_col = frame >> t_test(tidy3.col("flipper_length_mm"), mu=200)
    assert_same_frame(by_col.collect(), (frame >> t_test("flipper_length_mm", mu=200)).collect())


def test_tidy3_mutate_adjusts_p_values_per_group(tidy3):
    tests = tidy3.tidy(P()) >> tidy3.group_by("species", "island") >> t_test("body_mass_g ~ sex")
    via_tidy3 = (tests >> tidy3.group_by("species") >> tidy3.mutate(adj=p_adjust("p_value", "BH"))).collect()
    via_polars = tests.collect().with_columns(adj=p_adjust("p_value", "BH").over("species"))
    assert via_tidy3["adj"].to_list() == pytest.approx(via_polars["adj"].to_list())


def test_tidy3_lazy_frames_work(tidy3):
    from oracle import PENGUINS

    lazy = tidy3.tidy(pl.scan_csv(PENGUINS, null_values="NA"))
    assert_same_frame(t_test("body_mass_g ~ sex")(lazy).collect(), t_test(P(), "body_mass_g ~ sex"))


def test_by_is_refused_on_a_grouped_tidy3_frame(tidy3):
    grouped = tidy3.tidy(P()) >> tidy3.group_by("species")
    with pytest.raises(ValueError, match="by= cannot be used on a grouped frame"):
        grouped >> t_test("body_mass_g ~ sex", by="island")


def test_power_as_tidy3(tidy3):
    assert type(power_t_test(n=20, delta=1, as_="tidy3")).__name__ == "TidyFrame"
