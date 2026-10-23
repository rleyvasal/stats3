"""Tier 1 tests against R: wilcox_test, kruskal_test, cor_test, aov,
tukey_hsd, chisq_test, fisher_test, prop_test and binom_test."""

from __future__ import annotations

import warnings

import polars as pl
import pytest
import scipy.stats as scipy_stats

from stats3 import (
    aov,
    binom_test,
    chisq_test,
    cor_test,
    fisher_test,
    kruskal_test,
    prop_test,
    tukey_hsd,
    wilcox_test,
)
from oracle import (
    BACKENDS,
    _oracle,
    assert_matches_r,
    assert_same_frame,
    case,
    penguins,
    ranks,
    run_case,
)

P, S = penguins, ranks


def _first(k):
    return lambda: S().filter(pl.col("id") <= k)


def _props(frame):
    return (
        frame.drop_nulls("sex").group_by("island")
        .agg(x=(pl.col("sex") == "male").sum(), n=pl.len()).sort("island")
    )


def _all_islands():
    return _props(P())


def _two_islands():
    return _props(P().filter(pl.col("island") != "Biscoe"))


def _island(name):
    return lambda: _props(P()).filter(pl.col("island") == name)


def _counts(x, n):
    return lambda: pl.DataFrame({"x": [x], "n": [n]})


MALE = pl.col("sex") == "male"

CASES = {
    "w_two": case(P, lambda: wilcox_test("body_mass_g ~ sex")),
    "w_two_no_correct": case(P, lambda: wilcox_test("body_mass_g ~ sex", correct=False)),
    "w_two_exact": case(S, lambda: wilcox_test("a ~ g")),
    "w_two_exact_less_mu": case(S, lambda: wilcox_test("a ~ g", mu=0.3, alternative="less")),
    "w_two_edgeworth": case(S, lambda: wilcox_test("a ~ g", exact=False, correct=2)),
    "w_two_exact_ties_by": case(P, lambda: wilcox_test("body_mass_g ~ sex"), "species", "island"),
    "w_columns": case(P, lambda: wilcox_test("bill_length_mm", "bill_depth_mm")),
    "w_one": case(P, lambda: wilcox_test("flipper_length_mm", mu=200)),
    "w_one_exact": case(S, lambda: wilcox_test("a", mu=0.1)),
    "w_one_exact_zeros": case(S, lambda: wilcox_test("c", alternative="greater")),
    "w_one_zeros_asymp": case(S, lambda: wilcox_test("c", exact=False)),
    "w_one_edgeworth": case(S, lambda: wilcox_test("a", exact=False, correct=3)),
    "w_paired": case(P, lambda: wilcox_test("bill_length_mm", "bill_depth_mm", paired=True)),
    "w_paired_exact": case(S, lambda: wilcox_test("e", "a", paired=True, alternative="greater")),
    "w_by_species": case(P, lambda: wilcox_test("body_mass_g ~ sex"), "species"),
    "k_species": case(P, lambda: kruskal_test("body_mass_g ~ species")),
    "k_by_sex": case(P, lambda: kruskal_test("flipper_length_mm ~ island"), "sex"),
    "k_small": case(S, lambda: kruskal_test("a ~ h")),
    "c_pearson": case(P, lambda: cor_test("bill_length_mm", "bill_depth_mm")),
    "c_pearson_by": case(P, lambda: cor_test("bill_length_mm", "bill_depth_mm"), "species"),
    "c_pearson_less": case(P, lambda: cor_test("bill_length_mm", "bill_depth_mm", alternative="less", conf_level=0.9)),
    "c_pearson_four": case(_first(4), lambda: cor_test("a", "b", alternative="greater")),
    "c_pearson_three": case(_first(3), lambda: cor_test("a", "b")),
    "c_spearman": case(P, lambda: cor_test("bill_length_mm", "bill_depth_mm", method="spearman")),
    "c_spearman_by_h": case(S, lambda: cor_test("a", "b", method="spearman"), "h"),
    "c_spearman_all": case(S, lambda: cor_test("a", "b", method="spearman")),
    "c_spearman_greater_by_h": case(S, lambda: cor_test("a", "e", method="spearman", alternative="greater"), "h"),
    "c_spearman_less": case(S, lambda: cor_test("a", "b", method="spearman", alternative="less")),
    "c_kendall": case(P, lambda: cor_test("bill_length_mm", "bill_depth_mm", method="kendall")),
    "c_kendall_continuity": case(P, lambda: cor_test("bill_length_mm", "bill_depth_mm", method="kendall", continuity=True)),
    "c_kendall_exact": case(S, lambda: cor_test("a", "b", method="kendall")),
    "c_kendall_by_h": case(S, lambda: cor_test("a", "e", method="kendall", alternative="less"), "h"),
    "a_species": case(P, lambda: aov("body_mass_g ~ species")),
    "a_by_sex": case(P, lambda: aov("flipper_length_mm ~ island"), "sex"),
    "a_small": case(S, lambda: aov("a ~ h")),
    "t_species": case(P, lambda: tukey_hsd("body_mass_g ~ species")),
    "t_by_sex": case(P, lambda: tukey_hsd("flipper_length_mm ~ island"), "sex"),
    "t_small_99": case(S, lambda: tukey_hsd("a ~ h", conf_level=0.99)),
    "x_species_island": case(P, lambda: chisq_test("species", "island")),
    "x_yates": case(P, lambda: chisq_test("sex", "dream")),
    "x_no_yates": case(P, lambda: chisq_test("sex", "dream", correct=False)),
    "x_gof": case(P, lambda: chisq_test("species")),
    "x_gof_p": case(P, lambda: chisq_test("species", p=[0.5, 0.2, 0.3])),
    "x_gof_rescale": case(P, lambda: chisq_test("species", p={"Adelie": 5, "Chinstrap": 2, "Gentoo": 3}, rescale_p=True)),
    "x_by_year": case(P, lambda: chisq_test("species", "sex"), "year"),
    "f_sex_dream": case(P, lambda: fisher_test("sex", "dream")),
    "f_by_species": case(P, lambda: fisher_test("sex", "first_year"), "species"),
    "f_less": case(P, lambda: fisher_test("sex", "dream", alternative="less", conf_level=0.9)),
    "f_greater": case(P, lambda: fisher_test("sex", "first_year", alternative="greater")),
    "p_three": case(_all_islands, lambda: prop_test("x", "n")),
    "p_three_given": case(_all_islands, lambda: prop_test("x", "n", p=[0.5, 0.4, 0.6])),
    "p_two": case(_two_islands, lambda: prop_test("x", "n")),
    "p_two_less": case(_two_islands, lambda: prop_test("x", "n", alternative="less", conf_level=0.9)),
    "p_two_no_correct": case(_two_islands, lambda: prop_test("x", "n", correct=False)),
    "p_one": case(_island("Biscoe"), lambda: prop_test("x", "n", p=0.4)),
    "p_one_greater": case(_island("Dream"), lambda: prop_test("x", "n", alternative="greater")),
    "b_male": case(P, lambda: binom_test(MALE)),
    "b_counts": case(_counts(7, 20), lambda: binom_test("x", "n", p=0.4, alternative="greater")),
    "b_zero": case(_counts(0, 12), lambda: binom_test("x", "n", p=0.1)),
    "b_less": case(_counts(9, 30), lambda: binom_test("x", "n", alternative="less", conf_level=0.9)),
    "b_by_species": case(P, lambda: binom_test(MALE, p=0.45), "species"),
}

# Where R's own numerics are approximate, stats3 (through scipy) matches it to
# R's accuracy: qtukey's iteration puts Tukey intervals within about 1e-7;
# ptukey is off by up to 3e-5 (relative) at 6 residual df, where scipy agrees
# with a 30-digit mpmath integration to 12 digits, and loses relative
# accuracy below about 1e-9; fisher.test finds the conditional odds ratio and
# its interval with uniroot (measured: 2e-6 and 3e-5).
LOOSE = {
    "t_": {"conf_low": 1e-6, "conf_high": 1e-6, "adj_p_value": (5e-5, 1e-9)},
    "f_": {"estimate": 1e-5, "conf_low": 1e-4, "conf_high": 1e-4},
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_r(name, backend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_case(CASES[name], backend)
    assert_matches_r(result, _oracle("stats_tier1")[name], loose=LOOSE.get(name[:2]))


# grouped results are the per-subset results ---------------------------------

GROUPED = {
    "wilcox_test": lambda: wilcox_test("body_mass_g ~ sex"),
    "kruskal_test": lambda: kruskal_test("body_mass_g ~ island"),
    "cor_test": lambda: cor_test("bill_length_mm", "flipper_length_mm", method="kendall"),
    "aov": lambda: aov("body_mass_g ~ island"),
    "chisq_test": lambda: chisq_test("sex", "island"),
    "binom_test": lambda: binom_test(MALE),
}


@pytest.mark.parametrize("name", sorted(GROUPED))
def test_grouped_rows_equal_tests_on_each_subset(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        step = GROUPED[name]()
        step.by = "year"
        grouped = P() >> step
        for year in (2007, 2008, 2009):
            alone = P().filter(pl.col("year") == year) >> GROUPED[name]()
            assert_same_frame(grouped.filter(pl.col("year") == year).drop("year"), alone)


def test_results_are_in_group_order():
    keys = wilcox_test(P(), "body_mass_g ~ sex", by=["species", "island"]).select("species", "island").rows()
    assert keys == sorted(keys)


# scipy reference (runs without R) ---------------------------------------------


def _row(result):
    return result.row(0, named=True)


def _by_species(name):
    frame = P().drop_nulls(name)
    return [frame.filter(pl.col("species") == s)[name].to_numpy() for s in ("Adelie", "Chinstrap", "Gentoo")]


def test_rank_sum_matches_scipy():
    d = P().drop_nulls(["body_mass_g", "sex"])
    f = d.filter(pl.col("sex") == "female")["body_mass_g"].to_numpy()
    m = d.filter(pl.col("sex") == "male")["body_mass_g"].to_numpy()
    ref = scipy_stats.mannwhitneyu(f, m, method="asymptotic", use_continuity=True)
    row = _row(wilcox_test(P(), "body_mass_g ~ sex"))
    assert row["statistic"] == ref.statistic
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)


def test_signed_rank_exact_matches_scipy():
    ref = scipy_stats.wilcoxon(S()["a"].to_numpy() - 0.1, method="exact")
    row = _row(wilcox_test(S(), "a", mu=0.1))
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-12)


def test_kruskal_matches_scipy():
    ref = scipy_stats.kruskal(*_by_species("body_mass_g"))
    row = _row(kruskal_test(P(), "body_mass_g ~ species"))
    assert row["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_correlations_match_scipy(method):
    d = P().drop_nulls(["bill_length_mm", "bill_depth_mm"])
    x, y = d["bill_length_mm"].to_numpy(), d["bill_depth_mm"].to_numpy()
    ref = {"pearson": scipy_stats.pearsonr, "spearman": scipy_stats.spearmanr,
           "kendall": scipy_stats.kendalltau}[method](x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = _row(cor_test(P(), "bill_length_mm", "bill_depth_mm", method=method))
    assert row["estimate"] == pytest.approx(ref.statistic, rel=1e-12)
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)


def test_aov_and_tukey_match_scipy():
    samples = _by_species("body_mass_g")
    ref = scipy_stats.f_oneway(*samples)
    assert _row(aov(P(), "body_mass_g ~ species"))["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    tukey = scipy_stats.tukey_hsd(*samples)
    ci = tukey.confidence_interval()
    got = tukey_hsd(P(), "body_mass_g ~ species")
    for (a, b), row in zip([(1, 0), (2, 0), (2, 1)], got.iter_rows(named=True)):
        assert row["estimate"] == pytest.approx(tukey.statistic[a, b], rel=1e-12)
        assert row["adj_p_value"] == pytest.approx(tukey.pvalue[a, b], rel=1e-6, abs=1e-12)
        assert row["conf_low"] == pytest.approx(ci.low[a, b], rel=1e-6)


def test_chisq_and_binom_match_scipy():
    d = P()
    table = (d.group_by("species", "island").len()
             .pivot("island", index="species", values="len").fill_null(0).sort("species"))
    ref = scipy_stats.chi2_contingency(table.drop("species").to_numpy())
    assert _row(chisq_test(d, "species", "island"))["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    males = d["sex"].drop_nulls()
    ref = scipy_stats.binomtest(int((males == "male").sum()), len(males))
    row = _row(binom_test(d, MALE))
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-12)
    ci = ref.proportion_ci()
    assert (row["conf_low"], row["conf_high"]) == pytest.approx((ci.low, ci.high), rel=1e-10)


# warnings and errors follow R ---------------------------------------------------


def test_small_expected_counts_warn_as_r_does():
    with pytest.warns(UserWarning, match="Chi-squared approximation may be incorrect"):
        chisq_test(S(), "g", "h")


def test_exact_rank_correlation_with_ties_warns_as_r_does():
    with pytest.warns(UserWarning, match="cannot compute exact p-value with ties") as caught:
        cor_test(P(), "bill_length_mm", "bill_depth_mm", method="spearman")
    # The warning names the caller's line, not a line inside stats3.
    assert caught[0].filename == __file__


@pytest.mark.parametrize(
    ("frame", "call", "message"),
    [
        ({"y": [1.0, 2.0], "g": ["a", "a"]}, lambda: wilcox_test("y ~ g"), "exactly 2 levels"),
        ({"y": [1.0, 2.0], "g": ["a", "a"]}, lambda: kruskal_test("y ~ g"), "same group"),
        ({"y": [1.0, 2.0], "g": ["a", "b"]}, lambda: aov("y ~ g + h"), "one grouping column"),
        ({"x": [1.0, 2.0], "y": [3.0, 1.0]}, lambda: cor_test("x", "y"), "not enough finite"),
        ({"x": ["a", "a"], "y": ["b", "c"]}, lambda: chisq_test("x", "y"), "at least 2 levels"),
        ({"x": ["a", "b"]}, lambda: chisq_test("x", p=[0.2, 0.2]), "sum to 1"),
        ({"x": ["a", "b"]}, lambda: chisq_test("x", p={"a": 0.5}), "no probability"),
        ({"x": ["a", "b", "c", "a"], "y": ["u", "v", "u", "v"]}, lambda: fisher_test("x", "y"), "only 2 x 2"),
        ({"x": [5], "n": [3]}, lambda: prop_test("x", "n"), "not be greater"),
        ({"x": [1, 2], "n": [3, 4]}, lambda: binom_test("x", "n"), "one row per group"),
    ],
)
def test_errors_follow_r(frame, call, message):
    with pytest.raises(ValueError, match=message):
        pl.DataFrame(frame) >> call()


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: wilcox_test("y ~ g", paired=True), "not a formula"),
        (lambda: wilcox_test("y", correct=5), "correct must be"),
        (lambda: cor_test("x", "y", method="tau"), "method must be"),
        (lambda: tukey_hsd("y ~ g", conf_level=2), "conf_level"),
        (lambda: binom_test("x", p=1.5), "between 0 and 1"),
        (lambda: chisq_test("x", "y", p=[0.5, 0.5]), "p is for one column"),
    ],
)
def test_argument_checks(make, message):
    with pytest.raises(ValueError, match=message):
        make()
