"""Base-R tests against R: oneway_test, fligner_test, friedman_test,
mcnemar_test, mantelhaen_test, poisson_test, pairwise_t_test,
pairwise_wilcox_test, pairwise_prop_test and power_anova_test."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.stats as scipy_stats

from stats3 import (
    fligner_test,
    friedman_test,
    mantelhaen_test,
    mcnemar_test,
    oneway_test,
    pairwise_prop_test,
    pairwise_t_test,
    pairwise_wilcox_test,
    poisson_test,
    power_anova_test,
)
from oracle import BACKENDS, _oracle, assert_matches_r, case, penguins, ranks, run_case

DATA = Path(__file__).parent / "data"


def mtcars() -> pl.DataFrame:
    return pl.read_csv(DATA / "mtcars.csv")


def heavy() -> pl.DataFrame:
    return penguins().with_columns(heavy=pl.col("body_mass_g") > 4000)


def long() -> pl.DataFrame:
    """ranks' a, b and e stacked: block ``id``, treatment ``k``, value ``v``."""
    r = ranks()
    return pl.concat([
        r.select("id", k=pl.lit(k), v=pl.col(k), g="g") for k in ("a", "b", "e")
    ])


def cars() -> pl.DataFrame:
    """mtcars' cyl, gear and carb stacked per car: integer values, many ties."""
    m = mtcars()
    return pl.concat([
        m.select(id="model", k=pl.lit(k), v=pl.col(k).cast(pl.Float64)) for k in ("cyl", "gear", "carb")
    ])


def counts() -> pl.DataFrame:
    """Carburettors (events) and cars (time) per gear and transmission."""
    return (
        mtcars().group_by("gear", "am").agg(events=pl.col("carb").sum(), time=pl.len())
        .sort("gear", "am")
    )


def gear4() -> pl.DataFrame:
    return counts().filter(pl.col("gear") == 4)


def single(x: float, t: float):
    return lambda: pl.DataFrame({"x": [x], "t": [t]})


def props() -> pl.DataFrame:
    return (
        penguins().drop_nulls("sex").group_by("year", "island")
        .agg(male=(pl.col("sex") == "male").sum(), n=pl.len()).sort("year", "island")
    )


def props_year(year: int):
    return lambda: props().filter(pl.col("year") == year)


P, S, M, H, L = penguins, ranks, mtcars, heavy, long

CASES = {
    "o_welch": case(P, lambda: oneway_test("body_mass_g ~ species")),
    "o_equal": case(P, lambda: oneway_test("body_mass_g ~ species", var_equal=True)),
    "o_by_year": case(P, lambda: oneway_test("flipper_length_mm ~ island"), "year"),
    "fl_species": case(P, lambda: fligner_test("body_mass_g ~ species")),
    "fl_by_sex": case(P, lambda: fligner_test("flipper_length_mm ~ island"), "sex"),
    "fr_ranks": case(L, lambda: friedman_test("v ~ k | id")),
    "fr_ties": case(cars, lambda: friedman_test("v ~ k | id")),
    "fr_by_g": case(L, lambda: friedman_test("v ~ k | id"), "g"),
    "mc_cars": case(M, lambda: mcnemar_test("vs", "am")),
    "mc_three": case(M, lambda: mcnemar_test("cyl", "gear")),
    "mc_logical": case(P, lambda: mcnemar_test("dream", "first_year")),
    "mc_no_correct": case(P, lambda: mcnemar_test("dream", "first_year", correct=False)),
    "mc_by_species": case(H, lambda: mcnemar_test("heavy", "first_year"), "species"),
    "mh_species": case(H, lambda: mantelhaen_test("sex", "heavy", "species")),
    "mh_less": case(H, lambda: mantelhaen_test(
        "sex", "heavy", "species", alternative="less", correct=False, conf_level=0.9)),
    "mh_greater": case(P, lambda: mantelhaen_test("sex", "dream", "year", alternative="greater")),
    "mh_exact": case(H, lambda: mantelhaen_test("sex", "heavy", "species", exact=True)),
    "mh_exact_less": case(P, lambda: mantelhaen_test("sex", "dream", "year", exact=True, alternative="less")),
    "mh_general": case(P, lambda: mantelhaen_test("island", "sex", "year")),
    "mh_by_species": case(H, lambda: mantelhaen_test("sex", "heavy", "year"), "species"),
    "po_one": case(single(137, 24.19893), lambda: poisson_test("x", "t")),
    "po_one_r": case(single(137, 0), lambda: poisson_test("x", 24.19893, r=5, alternative="greater")),
    "po_small": case(single(2, 1), lambda: poisson_test("x", r=6)),
    "po_zero": case(single(0, 2), lambda: poisson_test("x", "t", alternative="less", conf_level=0.9)),
    "po_two": case(gear4, lambda: poisson_test("events", "time")),
    "po_two_less": case(gear4, lambda: poisson_test(
        "events", "time", r=1.5, alternative="less", conf_level=0.9)),
    "po_by_gear": case(counts, lambda: poisson_test("events", "time"), "gear"),
    "pt_pooled": case(P, lambda: pairwise_t_test("body_mass_g ~ species")),
    "pt_welch": case(P, lambda: pairwise_t_test(
        "body_mass_g ~ species", pool_sd=False, p_adjust_method="BH")),
    "pt_less": case(P, lambda: pairwise_t_test(
        "flipper_length_mm ~ island", alternative="less", p_adjust_method="none")),
    "pt_paired": case(L, lambda: pairwise_t_test("v ~ k", paired=True, p_adjust_method="bonferroni")),
    "pt_by_year": case(P, lambda: pairwise_t_test("body_mass_g ~ species"), "year"),
    "pt_numeric": case(M, lambda: pairwise_t_test("mpg ~ cyl", p_adjust_method="hommel")),
    "pw_species": case(P, lambda: pairwise_wilcox_test("flipper_length_mm ~ species")),
    "pw_exact": case(S, lambda: pairwise_wilcox_test("a ~ h", p_adjust_method="BH")),
    "pw_paired": case(L, lambda: pairwise_wilcox_test("v ~ k", paired=True)),
    "pw_by_year": case(P, lambda: pairwise_wilcox_test("body_mass_g ~ species"), "year"),
    "pp_islands": case(props_year(2008), lambda: pairwise_prop_test("male", "n", labels="island")),
    "pp_numbered": case(props_year(2009), lambda: pairwise_prop_test(
        "male", "n", p_adjust_method="BY", correct=False)),
    "pp_by_year": case(props, lambda: pairwise_prop_test("male", "n", labels="island"), "year"),
}

# Calculators: no data, so one run each.
POWER = {
    "pa_power": lambda: power_anova_test(groups=4, n=5, between_var=1, within_var=3),
    "pa_n": lambda: power_anova_test(groups=4, between_var=1, within_var=3, power=0.8),
    "pa_groups": lambda: power_anova_test(n=10, between_var=1, within_var=3, power=0.8),
    "pa_within": lambda: power_anova_test(groups=3, n=10, between_var=1, power=0.9),
    "pa_between": lambda: power_anova_test(groups=3, n=10, within_var=2, power=0.9),
    "pa_sig": lambda: power_anova_test(groups=3, n=10, between_var=1, within_var=2, power=0.9, sig_level=None),
}

# Where R's own numerics are approximate, compare to R's accuracy (stats3
# computes these to full precision):
# - uniroot at tol = .Machine$double.eps^0.25: power_anova_test's solved
#   value, and the exact Mantel-Haenszel estimate and interval (found on
#   (0, 1) for the odds ratio or its inverse);
# - noncentral F: R's pf with ncp is good to about 1e-9 relative;
# - R 4.6's exact Wilcoxon p-values with ties lose about 1e-14 absolute to
#   cancellation (some come out negative); stats3's stay positive.
# The exact Mantel-Haenszel odds ratio and its bounds above 1 are found as
# 1 / uniroot(..., c(eps, 1)), so R is accurate to about 1.2e-4 on 1 / value
# (2% on an upper bound near 3,400): RECIPROCAL compares those on that scale.
# stats3's estimate solves R's estimating equation exactly in R itself.
RECIPROCAL = {"mh_exact": ("estimate", "conf_low", "conf_high"),
              "mh_exact_less": ("estimate", "conf_low", "conf_high")}
_UNIROOT = (0.0, 1.3e-4)
LOOSE = {
    **{name: dict.fromkeys(columns, _UNIROOT) for name, columns in RECIPROCAL.items()},
    "pw_by_year": {"p_value": (1e-9, 2e-14)},
    "pa_n": {"n": 2e-4},
    "pa_groups": {"groups": 2e-4},
    "pa_within": {"within_var": 2e-4},
    "pa_between": {"between_var": 2e-4},
    "pa_sig": {"sig_level": 2e-4},
    "pa_power": {"power": 1e-8},
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_r(name, backend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_case(CASES[name], backend)
    expected = _oracle("stats_tier3")[name]
    if name in RECIPROCAL:
        result, expected = _reciprocal(result, expected, RECIPROCAL[name])
    assert_matches_r(result, expected, loose=LOOSE.get(name))


def _reciprocal(result, expected, columns):
    """The result and R's values with ``columns`` as 1 / value. R writes an
    infinite bound as null, so null becomes 0 and 0 becomes infinite (which
    the comparison reads as null, as for R)."""
    from oracle import as_polars

    frame = as_polars(result).with_columns(
        (1 / pl.col(c)).fill_nan(None).alias(c) for c in columns
    )
    data = [
        [0.0 if v is None else None if v == 0 else 1 / v for v in values] if name in columns else values
        for name, values in zip(expected["columns"], expected["data"])
    ]
    return frame, {**expected, "data": data}


@pytest.mark.parametrize("name", sorted(POWER))
def test_power_matches_r(name):
    assert_matches_r(POWER[name](), _oracle("stats_tier3")[name], loose=LOOSE.get(name))


# SciPy references --------------------------------------------------------------


def test_fligner_matches_scipy():
    p = penguins().drop_nulls("body_mass_g")
    samples = [p.filter(pl.col("species") == s)["body_mass_g"].to_numpy() for s in ("Adelie", "Chinstrap", "Gentoo")]
    ref = scipy_stats.fligner(*samples)
    got = fligner_test(p, "body_mass_g ~ species")
    assert got["statistic"][0] == pytest.approx(ref.statistic, rel=1e-12)
    assert got["p_value"][0] == pytest.approx(ref.pvalue, rel=1e-12)


def test_friedman_matches_scipy():
    r = ranks()
    ref = scipy_stats.friedmanchisquare(r["a"], r["b"], r["e"])
    got = friedman_test(long(), "v ~ k | id")
    assert got["statistic"][0] == pytest.approx(ref.statistic, rel=1e-12)
    assert got["p_value"][0] == pytest.approx(ref.pvalue, rel=1e-12)


def test_welch_anova_two_levels_is_welch_t():
    p = penguins()
    t = scipy_stats.ttest_ind(
        *(p.filter(pl.col("sex") == s)["body_mass_g"].drop_nulls().to_numpy() for s in ("female", "male")),
        equal_var=False,
    )
    got = oneway_test(p, "body_mass_g ~ sex")
    assert got["statistic"][0] == pytest.approx(t.statistic**2, rel=1e-12)
    assert got["p_value"][0] == pytest.approx(t.pvalue, rel=1e-10)


def test_poisson_one_sample_matches_scipy():
    got = poisson_test(pl.DataFrame({"x": [3]}), "x", 1, r=2, alternative="greater")
    assert got["p_value"][0] == pytest.approx(scipy_stats.poisson.sf(2, 2), rel=1e-13)


def test_mcnemar_matches_its_definition():
    m = mtcars()
    table = pl.DataFrame(m).group_by("vs", "am").len()
    cell = {(r["vs"], r["am"]): r["len"] for r in table.iter_rows(named=True)}
    b, c = cell.get((0, 1), 0), cell.get((1, 0), 0)
    got = mcnemar_test(m, "vs", "am")
    assert got["statistic"][0] == pytest.approx((abs(b - c) - 1) ** 2 / (b + c), rel=1e-14)


# shape and errors ----------------------------------------------------------------


def test_pairwise_keeps_the_level_type():
    out = pairwise_t_test(mtcars(), "mpg ~ cyl")
    assert out.schema["group1"] == pl.Int64
    assert out.select("group1", "group2").rows() == [(6, 4), (8, 4), (8, 6)]


def test_pairwise_enum_levels_order_pairs():
    p = penguins().with_columns(pl.col("species").cast(pl.Enum(["Gentoo", "Chinstrap", "Adelie"])))
    out = pairwise_t_test(p, "body_mass_g ~ species")
    assert out.select("group1", "group2").rows() == [
        ("Chinstrap", "Gentoo"), ("Adelie", "Gentoo"), ("Adelie", "Chinstrap"),
    ]


def test_mantelhaen_without_two_by_two_has_test_columns_only():
    out = mantelhaen_test(penguins(), "island", "sex", "year")
    assert out.columns == ["statistic", "df", "p_value", "method"]


def test_poisson_groups_fill_the_parameter_that_applies():
    out = poisson_test(counts(), "events", "time", by="gear")
    assert out["time_base"].null_count() == 1
    assert out["expected_count1"].null_count() == 2


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: friedman_test(long().head(89), "v ~ k | id"), "unreplicated complete block"),
        (lambda: friedman_test(long(), "v ~ k"), "y ~ groups | blocks"),
        (lambda: mcnemar_test(mtcars(), "vs", "cyl"), "same number of levels"),
        (lambda: mantelhaen_test(heavy(), "sex", "heavy", "body_mass_g"), "each stratum must be > 1"),
        (lambda: poisson_test(pl.DataFrame({"x": [1.5]}), "x"), "integer"),
        (lambda: poisson_test(pl.DataFrame({"x": [1, 2, 3]}), "x"), "k > 2"),
        (lambda: poisson_test(pl.DataFrame({"x": [1]}), "x", r=-1), "positive number"),
        (lambda: pairwise_t_test(penguins(), "body_mass_g ~ species", paired=True, pool_sd=True), "pooling"),
        (lambda: pairwise_t_test(penguins(), "body_mass_g ~ species", p_adjust_method="tukey"), "p_adjust_method"),
        (lambda: pairwise_t_test(penguins(), "body_mass_g ~ species", paired=True), "same number of values"),
        (lambda: pairwise_prop_test(pl.DataFrame({"x": [1], "n": [3]}), "x", "n"), "too few groups"),
        (lambda: oneway_test(penguins().filter(pl.col("sex").is_null()), "body_mass_g ~ island"), "not enough"),
        (lambda: power_anova_test(groups=3, n=10), "exactly one"),
        (lambda: power_anova_test(groups=1, n=10, between_var=1, within_var=1), "at least 2"),
    ],
)
def test_errors(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_power_anova_output_type():
    pd = pytest.importorskip("pandas")
    assert isinstance(power_anova_test(groups=3, n=10, between_var=1, within_var=2, as_="pandas"), pd.DataFrame)
    np.testing.assert_allclose(
        power_anova_test(groups=4, n=5, between_var=1, within_var=3)["power"][0], 0.3535594, rtol=1e-6
    )
