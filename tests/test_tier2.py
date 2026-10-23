"""Tier 2 against R: shapiro_test, bartlett_test, levene_test, var_test,
ks_test, p_adjust, cohens_d, hedges_g, eta_squared, power_t_test and
power_prop_test."""

from __future__ import annotations

import math
import warnings

import polars as pl
import pytest

from tidy3 import (
    col,
    filter,
    group_by,
    mutate,
    select,
    tidy,
)
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
from oracle import PENGUINS as PENGUINS_CSV
from oracle import RANKS as RANKS_CSV
from oracle import _oracle, assert_matches_r, penguins, ranks

scipy_stats = pytest.importorskip("scipy.stats")

PVALUES = [0.01, 0.04, None, 0.03, 0.2, 0.0005, 0.04, 0.6, 0.012]


def _adjust(method, n=None):
    return tidy({"p": PVALUES}) >> mutate(adj=p_adjust("p", method, n=n)) >> select("adj")


def _p():
    return penguins()


def _s():
    return ranks()


CASES = {
    "s_mass": lambda: _p() >> shapiro_test("body_mass_g"),
    "s_by_species": lambda: _p() >> group_by("species") >> shapiro_test("flipper_length_mm"),
    "s_small": lambda: _s() >> shapiro_test("a"),
    "b_species": lambda: _p() >> bartlett_test("body_mass_g ~ species"),
    "b_by_year": lambda: _p() >> group_by("year") >> bartlett_test("flipper_length_mm ~ species"),
    "l_species": lambda: _p() >> levene_test("body_mass_g ~ species"),
    "l_mean": lambda: _p() >> levene_test("body_mass_g ~ species", center="mean"),
    "l_by_sex": lambda: _p() >> group_by("sex") >> levene_test("flipper_length_mm ~ island"),
    "v_sex": lambda: _p() >> var_test("body_mass_g ~ sex"),
    "v_columns_less": lambda: _s() >> var_test("a", "b", ratio=2, alternative="less", conf_level=0.9),
    "v_greater": lambda: _s() >> var_test("a", "e", alternative="greater"),
    "v_by_species": lambda: _p() >> group_by("species") >> var_test("body_mass_g ~ sex"),
    "k1_pnorm": lambda: _p() >> ks_test("flipper_length_mm", "pnorm", mean=200, sd=14),
    "k1_exact": lambda: _s() >> ks_test("a", "pnorm"),
    "k1_greater": lambda: _s() >> ks_test("a", "pnorm", mean=0.4, alternative="greater"),
    "k1_less": lambda: _s() >> ks_test("a", "pnorm", mean=0.4, alternative="less"),
    "k1_punif": lambda: _s() >> ks_test("a", "punif", min=-3, max=3),
    "k1_asymp": lambda: _s() >> ks_test("a", "pnorm", exact=False),
    "k1_asymp_less": lambda: _s() >> ks_test("a", "pnorm", exact=False, alternative="less"),
    "k2_sex": lambda: _p() >> ks_test("body_mass_g ~ sex"),
    "k2_exact": lambda: _s() >> ks_test("a", "b"),
    "k2_ties": lambda: _s() >> ks_test("c", "b"),
    "k2_greater": lambda: _s() >> ks_test("a", "e", alternative="greater"),
    "k2_less_ties": lambda: _s() >> ks_test("c", "e", alternative="less"),
    "k2_by_species": lambda: _p() >> group_by("species") >> ks_test("flipper_length_mm ~ sex"),
    **{f"pa_{m}": (lambda m=m: _adjust(m)) for m in
       ("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none")},
    "pa_hommel_n": lambda: _adjust("hommel", n=12),
    "pa_BH_n": lambda: _adjust("BH", n=12),
    "d_pooled": lambda: _p() >> cohens_d("body_mass_g ~ sex"),
    "d_unpooled": lambda: _p() >> cohens_d("body_mass_g ~ sex", pooled_sd=False),
    "g_pooled": lambda: _p() >> hedges_g("body_mass_g ~ sex"),
    "d_one": lambda: _p() >> cohens_d("flipper_length_mm", mu=200),
    "d_paired": lambda: _p() >> cohens_d("bill_length_mm", "bill_depth_mm", paired=True),
    "d_by_species": lambda: _p() >> group_by("species") >> cohens_d("body_mass_g ~ sex"),
    "e_species": lambda: _p() >> eta_squared("body_mass_g ~ species"),
    "e_by_sex": lambda: _p() >> group_by("sex") >> eta_squared("flipper_length_mm ~ island"),
    "e_small": lambda: _s() >> eta_squared("a ~ h"),
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

# Where R's own numerics are approximate, compare to R's accuracy (tidy3
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
    **{case: {column: (0.0, 2e-4)} for case, column in _SOLVED.items()},
    "k1_asymp": {"p_value": (0.0, 5e-5)},
    "k2_sex": {"p_value": (0.0, 5e-5)},
    "d_paired": {"conf_low": 1e-3, "conf_high": 1e-3},
    "e_species": _ETA, "e_by_sex": _ETA, "e_small": _ETA,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_r(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CASES[case]()
    assert_matches_r(result, _oracle("stats_tier2")[case], loose=LOOSE.get(case))


# behaviour -------------------------------------------------------------------


def test_p_adjust_adjusts_each_group_on_its_own():
    tests = _p() >> group_by("species", "island") >> t_test("body_mass_g ~ sex")
    grouped = (tests >> group_by("species") >> mutate(adj=p_adjust("p_value", "BH"))).collect()
    for species in ("Adelie", "Chinstrap", "Gentoo"):
        alone = (tests >> filter(col("species") == species) >> mutate(adj=p_adjust("p_value", "BH"))).collect()
        assert grouped.filter(pl.col("species") == species)["adj"].to_list() == alone["adj"].to_list()


def test_p_adjust_methods_match_on_numpy_and_in_mutate():
    from stats3.effects import adjust_p
    import numpy as np

    values = np.array([np.nan if v is None else v for v in PVALUES])
    for method in ("holm", "hochberg", "hommel", "bonferroni", "BH", "BY"):
        got = _adjust(method).collect()["adj"].to_list()
        want = [None if math.isnan(v) else v for v in adjust_p(values, method)]
        assert got == want, method


def test_power_round_trips():
    n = power_t_test(power=0.9, delta=1).collect()["n"][0]
    assert power_t_test(n=n, delta=1).collect()["power"][0] == pytest.approx(0.9, abs=1e-10)
    n = power_prop_test(p1=0.5, p2=0.75, power=0.9).collect()["n"][0]
    assert power_prop_test(n=n, p1=0.5, p2=0.75).collect()["power"][0] == pytest.approx(0.9, abs=1e-10)


# scipy reference (runs without R) ---------------------------------------------


def _column(name, where=None):
    frame = pl.read_csv(PENGUINS_CSV, null_values="NA")
    if where is not None:
        frame = frame.filter(where)
    return frame[name].drop_nulls().cast(pl.Float64).to_numpy()


def _by_species(name):
    return [_column(name, pl.col("species") == s) for s in ("Adelie", "Chinstrap", "Gentoo")]


def test_checks_match_scipy():
    row = (_p() >> shapiro_test("body_mass_g")).collect().row(0, named=True)
    ref = scipy_stats.shapiro(_column("body_mass_g"))
    assert (row["statistic"], row["p_value"]) == pytest.approx((ref.statistic, ref.pvalue), rel=1e-10)
    row = (_p() >> bartlett_test("body_mass_g ~ species")).collect().row(0, named=True)
    ref = scipy_stats.bartlett(*_by_species("body_mass_g"))
    assert (row["statistic"], row["p_value"]) == pytest.approx((ref.statistic, ref.pvalue), rel=1e-10)
    row = (_p() >> levene_test("body_mass_g ~ species")).collect().row(0, named=True)
    ref = scipy_stats.levene(*_by_species("body_mass_g"), center="median")
    assert (row["statistic"], row["p_value"]) == pytest.approx((ref.statistic, ref.pvalue), rel=1e-10)


def test_ks_matches_scipy_without_ties():
    s = pl.read_csv(RANKS_CSV)
    a, b = s["a"].to_numpy(), s["b"].to_numpy()
    row = (_s() >> ks_test("a", "pnorm")).collect().row(0, named=True)
    ref = scipy_stats.kstest(a, "norm", method="exact")
    assert (row["statistic"], row["p_value"]) == pytest.approx((ref.statistic, ref.pvalue), rel=1e-10)
    row = (_s() >> ks_test("a", "b")).collect().row(0, named=True)
    ref = scipy_stats.ks_2samp(a, b, method="exact")
    assert (row["statistic"], row["p_value"]) == pytest.approx((ref.statistic, ref.pvalue), rel=1e-9)


# warnings and errors follow R ---------------------------------------------------


def test_ties_warn_as_r_does():
    with pytest.warns(UserWarning, match="ties should not be present"):
        _p() >> ks_test("flipper_length_mm", "pnorm", mean=200, sd=14)
    with pytest.warns(UserWarning, match="approximate in the presence of ties"):
        _p() >> ks_test("body_mass_g ~ sex")


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
        tidy(frame) >> call()


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
