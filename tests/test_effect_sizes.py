"""effect_size= on t_test, wilcox_test, kruskal_test, chisq_test and aov,
against R's effectsize package (the measure ``effectsize()`` gives for each
R test, with its default interval)."""

from __future__ import annotations

import math
import warnings

import polars as pl
import pytest

from stats3 import aov, chisq_test, kruskal_test, t_test, wilcox_test
from stats3.effects import EFFECT_COLUMNS
from oracle import (
    BACKENDS, _oracle, as_polars, assert_matches_r, assert_same_frame, case, penguins, ranks, run_case,
)


def P() -> pl.DataFrame:
    return penguins().with_columns(depth_25=pl.col("bill_depth_mm") - 25)


def S() -> pl.DataFrame:
    return ranks().with_columns(c_floor=pl.col("c").floor())


CASES = {
    "t_welch": case(P, lambda: t_test("body_mass_g ~ sex", effect_size=True)),
    "t_student_less": case(P, lambda: t_test("body_mass_g ~ sex", var_equal=True, alternative="less",
                                             conf_level=0.9, effect_size="cohens_d")),
    "t_g_by_species": case(P, lambda: t_test("body_mass_g ~ sex", effect_size="hedges_g"), "species"),
    "t_g_mu": case(P, lambda: t_test("body_mass_g ~ sex", mu=-500, alternative="greater", effect_size="hedges_g")),
    "t_one": case(P, lambda: t_test("flipper_length_mm", mu=200, alternative="greater", effect_size=True)),
    "t_paired": case(P, lambda: t_test("bill_length_mm", "bill_depth_mm", paired=True, effect_size=True)),
    "t_columns": case(S, lambda: t_test("a", "b", effect_size=True)),
    "w_sex": case(P, lambda: wilcox_test("body_mass_g ~ sex", effect_size=True)),
    "w_mu_greater": case(P, lambda: wilcox_test("body_mass_g ~ sex", mu=-500, alternative="greater",
                                                effect_size=True)),
    "w_one": case(P, lambda: wilcox_test("bill_length_mm", mu=44, effect_size=True)),
    "w_one_less": case(S, lambda: wilcox_test("c", alternative="less", effect_size="rank_biserial")),
    "w_paired": case(P, lambda: wilcox_test("bill_length_mm", "depth_25", paired=True, effect_size=True)),
    "w_paired_zeros": case(S, lambda: wilcox_test("c", "c_floor", paired=True, effect_size=True)),
    "w_by_species": case(P, lambda: wilcox_test("body_mass_g ~ sex", effect_size=True), "species"),
    "w_columns": case(S, lambda: wilcox_test("a", "b", effect_size=True)),
    "k_species": case(P, lambda: kruskal_test("body_mass_g ~ species", effect_size=True)),
    "k_by_sex": case(P, lambda: kruskal_test("flipper_length_mm ~ island", effect_size=True), "sex"),
    "c_species_island": case(P, lambda: chisq_test("species", "island", effect_size=True)),
    "c_sex_island": case(P, lambda: chisq_test("sex", "island", effect_size=True)),
    "c_2x2": case(P, lambda: chisq_test("sex", "dream", effect_size="cramers_v")),
    "c_by_year": case(P, lambda: chisq_test("species", "island", effect_size=True), "year"),
    "c_gof": case(P, lambda: chisq_test("species", effect_size=True)),
    "c_gof_p": case(P, lambda: chisq_test("species", p=[0.5, 0.2, 0.3], effect_size="fei")),
    "a_species": case(P, lambda: aov("body_mass_g ~ species", effect_size=True)),
    "a_two_way": case(P, lambda: aov("body_mass_g ~ species * sex", effect_size=True)),
    "a_covariate": case(P, lambda: aov("body_mass_g ~ flipper_length_mm + species", effect_size=True)),
    "a_by_sex": case(P, lambda: aov("flipper_length_mm ~ island", effect_size=True), "sex"),
}

# effectsize finds the noncentral t, chi-squared and F bounds with optim
# (abstol 1e-9 on the quantiles), good to about 1e-7; stats3 solves them
# exactly, as test_tier2 checks against uniroot. Rank-biserial and rank
# epsilon squared are closed forms, compared at full precision.
_OPTIM = {"effect_low": 1e-6, "effect_high": 1e-6}
LOOSE = {name: _OPTIM for name in CASES if name[0] in "tca"}
# R's pt is approximate for ncp above 37.62 (?pt): the paired d interval
# sits at t = 79.5, as test_tier2's d_paired.
LOOSE["t_paired"] = {"effect_low": 1e-3, "effect_high": 1e-3}


def _effects(result, groups: list[str]) -> pl.DataFrame:
    frame = as_polars(result)
    if "term" in frame.columns:
        frame = frame.filter(pl.col("term") != "Residuals")
        groups = [*groups, "term"]
    return frame.select(*groups, *EFFECT_COLUMNS)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_effectsize(name, backend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_case(CASES[name], backend)
    assert_matches_r(_effects(result, CASES[name][2]), _oracle("effect_sizes")[name], loose=LOOSE.get(name))


# Without R ------------------------------------------------------------------------


def test_off_by_default_leaves_columns_unchanged():
    data = penguins()
    for make in (lambda **k: t_test("body_mass_g ~ sex", **k), lambda **k: wilcox_test("body_mass_g ~ sex", **k),
                 lambda **k: kruskal_test("body_mass_g ~ species", **k),
                 lambda **k: chisq_test("species", "island", **k), lambda **k: aov("body_mass_g ~ species", **k)):
        plain = data >> make()
        assert_same_frame(data >> make(effect_size=False), plain)
        with_effect = data >> make(effect_size=True)
        assert not set(EFFECT_COLUMNS) & set(plain.columns)
        assert [c for c in with_effect.columns if c not in EFFECT_COLUMNS] == plain.columns
        if "method" in plain.columns:  # the effect columns come just before method
            at = with_effect.columns.index("effect")
            assert with_effect.columns[at + 4] == "method"


def test_test_columns_are_unchanged_by_effect_size():
    data = penguins()
    plain = data >> t_test("body_mass_g ~ sex", var_equal=True)
    effect = data >> t_test("body_mass_g ~ sex", var_equal=True, effect_size=True)
    assert_same_frame(effect.select(plain.columns), plain)


def test_effect_columns_are_the_standalone_verbs():
    from stats3 import cohens_d, eta_squared, hedges_g

    data = penguins()
    d = data >> t_test("body_mass_g ~ sex", var_equal=True, effect_size=True)
    ref = data >> cohens_d("body_mass_g ~ sex")
    assert d["effect_size"].to_list() == pytest.approx(ref["estimate"].to_list(), rel=1e-14)
    g = data >> t_test("body_mass_g ~ sex", effect_size="hedges_g")
    ref = data >> hedges_g("body_mass_g ~ sex", pooled_sd=False)
    assert g["effect_low"].to_list() == pytest.approx(ref["conf_low"].to_list(), rel=1e-12)
    e = data >> aov("body_mass_g ~ species", effect_size=True)
    ref = data >> eta_squared("body_mass_g ~ species")
    assert e["effect_low"][0] == pytest.approx(ref["conf_low"][0], rel=1e-12)


def test_rank_biserial_is_signed_by_the_first_level():
    frame = pl.DataFrame({"y": [1.0, 2, 3, 4, 5, 6], "g": ["a", "a", "a", "b", "b", "b"]})
    out = frame >> wilcox_test("y ~ g", effect_size=True)
    assert out["effect_size"][0] == -1.0
    assert out["effect_low"][0] == -1.0 and out["effect_high"][0] == -1.0


def test_bad_effect_size_names_the_choices():
    with pytest.raises(ValueError, match="'cohens_d', 'hedges_g'"):
        t_test("y ~ g", effect_size="rank_biserial")
    with pytest.raises(ValueError, match="'cramers_v'"):
        chisq_test("a", "b", effect_size="fei")
    with pytest.raises(ValueError, match="'fei'"):
        chisq_test("a", effect_size="cramers_v")
    with pytest.raises(ValueError, match="effect_size must be"):
        aov("y ~ g", effect_size="omega_squared")


def test_kruskal_interval_is_missing():
    out = penguins() >> kruskal_test("body_mass_g ~ species", effect_size=True)
    assert out["effect_low"].is_null().all() and out["effect_high"].is_null().all()
    assert math.isfinite(out["effect_size"][0])
