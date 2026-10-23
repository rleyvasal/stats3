"""Tier 2 against R: shapiro_test, bartlett_test, levene_test, var_test,
ks_test, p_adjust, cohens_d, hedges_g, eta_squared, power_t_test and
power_prop_test."""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest
import scipy.stats as scipy_stats

from stats3 import (
    bartlett_test,
    cohens_d,
    eta_squared,
    hedges_g,
    ks_test,
    levene_test,
    p_adjust,
    power_prop_test,
    power_t_test,
    shapiro_test,
    t_test,
    var_test,
)
from stats3.effects import adjust_p
from oracle import (
    BACKENDS,
    _oracle,
    assert_matches_r,
    case,
    penguins,
    ranks,
    run_case,
)

P, S = penguins, ranks
PVALUES = [0.01, 0.04, None, 0.03, 0.2, 0.0005, 0.04, 0.6, 0.012]
METHODS = ("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none")

CASES = {
    "s_mass": case(P, lambda: shapiro_test("body_mass_g")),
    "s_by_species": case(P, lambda: shapiro_test("flipper_length_mm"), "species"),
    "s_small": case(S, lambda: shapiro_test("a")),
    "b_species": case(P, lambda: bartlett_test("body_mass_g ~ species")),
    "b_by_year": case(P, lambda: bartlett_test("flipper_length_mm ~ species"), "year"),
    "l_species": case(P, lambda: levene_test("body_mass_g ~ species")),
    "l_mean": case(P, lambda: levene_test("body_mass_g ~ species", center="mean")),
    "l_by_sex": case(P, lambda: levene_test("flipper_length_mm ~ island"), "sex"),
    "v_sex": case(P, lambda: var_test("body_mass_g ~ sex")),
    "v_columns_less": case(S, lambda: var_test("a", "b", ratio=2, alternative="less", conf_level=0.9)),
    "v_greater": case(S, lambda: var_test("a", "e", alternative="greater")),
    "v_by_species": case(P, lambda: var_test("body_mass_g ~ sex"), "species"),
    "k1_pnorm": case(P, lambda: ks_test("flipper_length_mm", "pnorm", mean=200, sd=14)),
    "k1_exact": case(S, lambda: ks_test("a", "pnorm")),
    "k1_greater": case(S, lambda: ks_test("a", "pnorm", mean=0.4, alternative="greater")),
    "k1_less": case(S, lambda: ks_test("a", "pnorm", mean=0.4, alternative="less")),
    "k1_punif": case(S, lambda: ks_test("a", "punif", min=-3, max=3)),
    "k1_asymp": case(S, lambda: ks_test("a", "pnorm", exact=False)),
    "k1_asymp_less": case(S, lambda: ks_test("a", "pnorm", exact=False, alternative="less")),
    "k2_sex": case(P, lambda: ks_test("body_mass_g ~ sex")),
    "k2_exact": case(S, lambda: ks_test("a", "b")),
    "k2_ties": case(S, lambda: ks_test("c", "b")),
    "k2_greater": case(S, lambda: ks_test("a", "e", alternative="greater")),
    "k2_less_ties": case(S, lambda: ks_test("c", "e", alternative="less")),
    "k2_by_species": case(P, lambda: ks_test("flipper_length_mm ~ sex"), "species"),
    "d_pooled": case(P, lambda: cohens_d("body_mass_g ~ sex")),
    "d_unpooled": case(P, lambda: cohens_d("body_mass_g ~ sex", pooled_sd=False)),
    "g_pooled": case(P, lambda: hedges_g("body_mass_g ~ sex")),
    "d_one": case(P, lambda: cohens_d("flipper_length_mm", mu=200)),
    "d_paired": case(P, lambda: cohens_d("bill_length_mm", "bill_depth_mm", paired=True)),
    "d_by_species": case(P, lambda: cohens_d("body_mass_g ~ sex"), "species"),
    "e_species": case(P, lambda: eta_squared("body_mass_g ~ species")),
    "e_by_sex": case(P, lambda: eta_squared("flipper_length_mm ~ island"), "sex"),
    "e_small": case(S, lambda: eta_squared("a ~ h")),
}

# Calculators: no data, so one run each.
POWER = {
    "pw_t_power": lambda: power_t_test(n=20, delta=1),
    "pw_t_n": lambda: power_t_test(power=0.9, delta=1),
    "pw_t_delta": lambda: power_t_test(n=30, power=0.8, type="paired", alternative="one.sided"),
    "pw_t_sd": lambda: power_t_test(n=25, delta=2, sd=None, power=0.85, type="one.sample"),
    "pw_t_sig": lambda: power_t_test(n=20, delta=1, sig_level=None, power=0.9),
    "pw_t_strict": lambda: power_t_test(n=10, delta=0.5, strict=True),
    "pw_p_power": lambda: power_prop_test(n=50, p1=0.5, p2=0.75),
    "pw_p_n": lambda: power_prop_test(p1=0.5, p2=0.75, power=0.9),
    "pw_p_p2": lambda: power_prop_test(n=50, p1=0.5, power=0.9),
    "pw_p_strict": lambda: power_prop_test(n=30, p1=0.4, p2=0.6, strict=True),
}

# Where R's own numerics are approximate, compare to R's accuracy (stats3
# computes these to full precision):
# - power: uniroot at tol = .Machine$double.eps^0.25, about 1.2e-4 on the
#   solved value;
# - asymptotic two-sided Kolmogorov: R's limit series keeps only its first
#   term below 1 (k_max = 3 for tol = 1e-6), an error near 2e-5 here;
# - noncentral t with ncp above 37.62 (documented in ?pt): at t = 79.5 R's
#   paired d interval is off by 7e-4; a 40-digit mpmath integration agrees
#   with scipy's bound (probability 0.975, against R's 0.97596);
# - noncentral F: R's pf with ncp is good to about 1e-9 relative.
_SOLVED = {"pw_t_n": "n", "pw_t_delta": "delta", "pw_t_sd": "sd", "pw_t_sig": "sig_level",
           "pw_p_n": "n", "pw_p_p2": "p2"}
_ETA = {"conf_low": 1e-8}
LOOSE = {
    **{name: {column: (0.0, 2e-4)} for name, column in _SOLVED.items()},
    "k1_asymp": {"p_value": (0.0, 5e-5)},
    "k2_sex": {"p_value": (0.0, 5e-5)},
    "d_paired": {"conf_low": 1e-3, "conf_high": 1e-3},
    "e_species": _ETA, "e_by_sex": _ETA, "e_small": _ETA,
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_r(name, backend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_case(CASES[name], backend)
    assert_matches_r(result, _oracle("stats_tier2")[name], loose=LOOSE.get(name))


@pytest.mark.parametrize("name", sorted(POWER))
def test_power_matches_r(name):
    assert_matches_r(POWER[name](), _oracle("stats_tier2")[name], loose=LOOSE.get(name))


ADJUST = {f"pa_{m}": (m, None) for m in METHODS} | {"pa_hommel_n": ("hommel", 12), "pa_BH_n": ("BH", 12)}


@pytest.mark.parametrize("name", sorted(ADJUST))
def test_p_adjust_matches_r(name):
    method, n = ADJUST[name]
    frame = pl.DataFrame({"p": PVALUES}).select(adj=p_adjust("p", method, n=n))
    assert_matches_r(frame, _oracle("stats_tier2")[name])


# p_adjust on values and columns -------------------------------------------------


def test_p_adjust_returns_the_type_it_is_given():
    want = adjust_p(np.array([np.nan if v is None else v for v in PVALUES]), "BH")
    as_array = p_adjust(np.array(PVALUES, dtype=float), "BH")
    assert isinstance(as_array, np.ndarray)
    np.testing.assert_allclose(as_array, want, equal_nan=True)
    np.testing.assert_allclose(p_adjust(PVALUES, "BH"), want, equal_nan=True)
    series = p_adjust(pl.Series("p", PVALUES), "BH")
    assert isinstance(series, pl.Series) and series.null_count() == 1
    assert isinstance(p_adjust("p", "BH"), pl.Expr)


def test_p_adjust_keeps_a_pandas_series_index():
    pd = pytest.importorskip("pandas")
    want = adjust_p(np.array([np.nan if v is None else v for v in PVALUES]), "BH")
    indexed = pd.Series(PVALUES, index=list("abcdefghi"), name="p", dtype=float)
    out = p_adjust(indexed, "BH")
    assert isinstance(out, pd.Series) and list(out.index) == list("abcdefghi")
    np.testing.assert_allclose(out.to_numpy(), want, equal_nan=True)


def test_p_adjust_adjusts_each_group_on_its_own():
    tests = t_test(P(), "body_mass_g ~ sex", by=["species", "island"])
    grouped = tests.with_columns(adj=p_adjust("p_value", "BH").over("species"))
    for species in ("Adelie", "Chinstrap", "Gentoo"):
        part = grouped.filter(pl.col("species") == species)
        assert part["adj"].to_list() == pytest.approx(
            list(adjust_p(part["p_value"].to_numpy(), "BH"))
        )


def test_power_round_trips():
    n = power_t_test(power=0.9, delta=1)["n"][0]
    assert power_t_test(n=n, delta=1)["power"][0] == pytest.approx(0.9, abs=1e-10)
    n = power_prop_test(p1=0.5, p2=0.75, power=0.9)["n"][0]
    assert power_prop_test(n=n, p1=0.5, p2=0.75)["power"][0] == pytest.approx(0.9, abs=1e-10)


def test_power_as_pandas():
    pytest.importorskip("pandas")
    frame = power_t_test(n=20, delta=1, as_="pandas")
    assert type(frame).__module__.startswith("pandas")
    assert frame["power"].iloc[0] == pytest.approx(0.8689528, rel=1e-6)


# scipy reference (runs without R) ---------------------------------------------


def _by_species(name):
    frame = P().drop_nulls(name)
    return [frame.filter(pl.col("species") == s)[name].cast(pl.Float64).to_numpy()
            for s in ("Adelie", "Chinstrap", "Gentoo")]


def test_checks_match_scipy():
    row = shapiro_test(P(), "body_mass_g").row(0, named=True)
    ref = scipy_stats.shapiro(P()["body_mass_g"].drop_nulls().cast(pl.Float64).to_numpy())
    assert (row["statistic"], row["p_value"]) == pytest.approx((ref.statistic, ref.pvalue), rel=1e-10)
    row = bartlett_test(P(), "body_mass_g ~ species").row(0, named=True)
    ref = scipy_stats.bartlett(*_by_species("body_mass_g"))
    assert (row["statistic"], row["p_value"]) == pytest.approx((ref.statistic, ref.pvalue), rel=1e-10)
    row = levene_test(P(), "body_mass_g ~ species").row(0, named=True)
    ref = scipy_stats.levene(*_by_species("body_mass_g"), center="median")
    assert (row["statistic"], row["p_value"]) == pytest.approx((ref.statistic, ref.pvalue), rel=1e-10)


def test_ks_matches_scipy_without_ties():
    a, b = S()["a"].to_numpy(), S()["b"].to_numpy()
    row = ks_test(S(), "a", "pnorm").row(0, named=True)
    ref = scipy_stats.kstest(a, "norm", method="exact")
    assert (row["statistic"], row["p_value"]) == pytest.approx((ref.statistic, ref.pvalue), rel=1e-10)
    row = ks_test(S(), "a", "b").row(0, named=True)
    ref = scipy_stats.ks_2samp(a, b, method="exact")
    assert (row["statistic"], row["p_value"]) == pytest.approx((ref.statistic, ref.pvalue), rel=1e-9)


# warnings and errors follow R ---------------------------------------------------


def test_ties_warn_as_r_does():
    with pytest.warns(UserWarning, match="ties should not be present"):
        ks_test(P(), "flipper_length_mm", "pnorm", mean=200, sd=14)
    with pytest.warns(UserWarning, match="approximate in the presence of ties"):
        ks_test(P(), "body_mass_g ~ sex")


@pytest.mark.parametrize(
    ("frame", "call", "message"),
    [
        ({"x": [1.0, 2.0]}, lambda: shapiro_test("x"), "between 3 and 5000"),
        ({"x": [2.0, 2.0, 2.0]}, lambda: shapiro_test("x"), "identical"),
        ({"y": [1.0, 2.0, 3.0], "g": ["a", "a", "b"]}, lambda: bartlett_test("y ~ g"), "at least 2 observations"),
        ({"y": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]}, lambda: var_test("y ~ g"), "exactly 2 levels"),
        ({"x": [0.1, 0.2]}, lambda: ks_test("x", "pbeta"), "one of"),
        ({"x": [0.1, 0.2]}, lambda: ks_test("x", "pnorm", rate=2), "takes"),
        ({"x": [0.1, 0.2]}, lambda: ks_test("x", "pgamma"), "needs shape"),
    ],
)
def test_errors_follow_r(frame, call, message):
    with pytest.raises(ValueError, match=message):
        pl.DataFrame(frame) >> call()


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: p_adjust("p", "sidak"), "method must be"),
        (lambda: levene_test("y ~ g", center="trimmed"), "center must be"),
        (lambda: var_test("x", "y", ratio=0), "ratio must be"),
        (lambda: power_t_test(n=10, delta=1, power=0.8), "exactly one"),
        (lambda: power_t_test(delta=1, power=0.8, type="three"), "type must be"),
        (lambda: cohens_d("y ~ g", paired=True), "pass a formula"),
    ],
)
def test_argument_checks(make, message):
    with pytest.raises(ValueError, match=message):
        make()
