"""Tier 1 tests against R: wilcox_test, kruskal_test, cor_test, aov,
tukey_hsd, chisq_test, fisher_test, prop_test and binom_test."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from tidy3 import (
    col,
    drop_na,
    filter,
    group_by,
    n,
    summarise,
    sum as tidy_sum,
    tidy,
)
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
from oracle import PENGUINS as PENGUINS_CSV
from oracle import RANKS as RANKS_CSV
from oracle import _oracle, assert_matches_r, assert_same_frame, penguins, penguins_flags, ranks

scipy_stats = pytest.importorskip("scipy.stats")


def _p():
    return penguins_flags()


def _s():
    return ranks()


def _first(k):
    return _s() >> filter(col("id") <= k)


def _props(frame):
    return (
        frame >> drop_na("sex") >> group_by("island")
        >> summarise(x=tidy_sum(col("sex") == "male"), n=n())
    )


def _two():
    return _props(_p() >> filter(col("island") != "Biscoe"))


def _one(index):
    return _props(_p()) >> filter(col("island") == ["Biscoe", "Dream", "Torgersen"][index])


CASES = {
    "w_two": lambda: _p() >> wilcox_test("body_mass_g ~ sex"),
    "w_two_no_correct": lambda: _p() >> wilcox_test("body_mass_g ~ sex", correct=False),
    "w_two_exact": lambda: _s() >> wilcox_test("a ~ g"),
    "w_two_exact_less_mu": lambda: _s() >> wilcox_test("a ~ g", mu=0.3, alternative="less"),
    "w_two_edgeworth": lambda: _s() >> wilcox_test("a ~ g", exact=False, correct=2),
    "w_two_exact_ties_by": lambda: _p() >> group_by("species", "island") >> wilcox_test("body_mass_g ~ sex"),
    "w_columns": lambda: _p() >> wilcox_test("bill_length_mm", "bill_depth_mm"),
    "w_one": lambda: _p() >> wilcox_test("flipper_length_mm", mu=200),
    "w_one_exact": lambda: _s() >> wilcox_test("a", mu=0.1),
    "w_one_exact_zeros": lambda: _s() >> wilcox_test("c", alternative="greater"),
    "w_one_zeros_asymp": lambda: _s() >> wilcox_test("c", exact=False),
    "w_one_edgeworth": lambda: _s() >> wilcox_test("a", exact=False, correct=3),
    "w_paired": lambda: _p() >> wilcox_test("bill_length_mm", "bill_depth_mm", paired=True),
    "w_paired_exact": lambda: _s() >> wilcox_test("e", "a", paired=True, alternative="greater"),
    "w_by_species": lambda: _p() >> group_by("species") >> wilcox_test("body_mass_g ~ sex"),
    "k_species": lambda: _p() >> kruskal_test("body_mass_g ~ species"),
    "k_by_sex": lambda: _p() >> group_by("sex") >> kruskal_test("flipper_length_mm ~ island"),
    "k_small": lambda: _s() >> kruskal_test("a ~ h"),
    "c_pearson": lambda: _p() >> cor_test("bill_length_mm", "bill_depth_mm"),
    "c_pearson_by": lambda: _p() >> group_by("species") >> cor_test("bill_length_mm", "bill_depth_mm"),
    "c_pearson_less": lambda: _p() >> cor_test("bill_length_mm", "bill_depth_mm", alternative="less", conf_level=0.9),
    "c_pearson_four": lambda: _first(4) >> cor_test("a", "b", alternative="greater"),
    "c_pearson_three": lambda: _first(3) >> cor_test("a", "b"),
    "c_spearman": lambda: _p() >> cor_test("bill_length_mm", "bill_depth_mm", method="spearman"),
    "c_spearman_by_h": lambda: _s() >> group_by("h") >> cor_test("a", "b", method="spearman"),
    "c_spearman_all": lambda: _s() >> cor_test("a", "b", method="spearman"),
    "c_spearman_greater_by_h": lambda: _s() >> group_by("h") >> cor_test("a", "e", method="spearman", alternative="greater"),
    "c_spearman_less": lambda: _s() >> cor_test("a", "b", method="spearman", alternative="less"),
    "c_kendall": lambda: _p() >> cor_test("bill_length_mm", "bill_depth_mm", method="kendall"),
    "c_kendall_continuity": lambda: _p() >> cor_test("bill_length_mm", "bill_depth_mm", method="kendall", continuity=True),
    "c_kendall_exact": lambda: _s() >> cor_test("a", "b", method="kendall"),
    "c_kendall_by_h": lambda: _s() >> group_by("h") >> cor_test("a", "e", method="kendall", alternative="less"),
    "a_species": lambda: _p() >> aov("body_mass_g ~ species"),
    "a_by_sex": lambda: _p() >> group_by("sex") >> aov("flipper_length_mm ~ island"),
    "a_small": lambda: _s() >> aov("a ~ h"),
    "t_species": lambda: _p() >> tukey_hsd("body_mass_g ~ species"),
    "t_by_sex": lambda: _p() >> group_by("sex") >> tukey_hsd("flipper_length_mm ~ island"),
    "t_small_99": lambda: _s() >> tukey_hsd("a ~ h", conf_level=0.99),
    "x_species_island": lambda: _p() >> chisq_test("species", "island"),
    "x_yates": lambda: _p() >> chisq_test("sex", "dream"),
    "x_no_yates": lambda: _p() >> chisq_test("sex", "dream", correct=False),
    "x_gof": lambda: _p() >> chisq_test("species"),
    "x_gof_p": lambda: _p() >> chisq_test("species", p=[0.5, 0.2, 0.3]),
    "x_gof_rescale": lambda: _p() >> chisq_test("species", p={"Adelie": 5, "Chinstrap": 2, "Gentoo": 3}, rescale_p=True),
    "x_by_year": lambda: _p() >> group_by("year") >> chisq_test("species", "sex"),
    "f_sex_dream": lambda: _p() >> fisher_test("sex", "dream"),
    "f_by_species": lambda: _p() >> group_by("species") >> fisher_test("sex", "first_year"),
    "f_less": lambda: _p() >> fisher_test("sex", "dream", alternative="less", conf_level=0.9),
    "f_greater": lambda: _p() >> fisher_test("sex", "first_year", alternative="greater"),
    "p_three": lambda: _props(_p()) >> prop_test("x", "n"),
    "p_three_given": lambda: _props(_p()) >> prop_test("x", "n", p=[0.5, 0.4, 0.6]),
    "p_two": lambda: _two() >> prop_test("x", "n"),
    "p_two_less": lambda: _two() >> prop_test("x", "n", alternative="less", conf_level=0.9),
    "p_two_no_correct": lambda: _two() >> prop_test("x", "n", correct=False),
    "p_one": lambda: _one(0) >> prop_test("x", "n", p=0.4),
    "p_one_greater": lambda: _one(1) >> prop_test("x", "n", alternative="greater"),
    "b_male": lambda: _p() >> binom_test(col("sex") == "male"),
    "b_counts": lambda: tidy({"x": [7], "n": [20]}) >> binom_test("x", "n", p=0.4, alternative="greater"),
    "b_zero": lambda: tidy({"x": [0], "n": [12]}) >> binom_test("x", "n", p=0.1),
    "b_less": lambda: tidy({"x": [9], "n": [30]}) >> binom_test("x", "n", alternative="less", conf_level=0.9),
    "b_by_species": lambda: _p() >> group_by("species") >> binom_test(col("sex") == "male", p=0.45),
}

# Where R's own numerics are approximate, tidy3 (through scipy) matches it to
# R's accuracy: qtukey's iteration puts Tukey intervals within about 1e-7;
# ptukey is off by up to 3e-5 (relative) at 6 residual df, where scipy agrees
# with a 30-digit mpmath integration to 12 digits, and loses relative
# accuracy below about 1e-9; fisher.test finds the conditional odds ratio and
# its interval with uniroot (measured: 2e-6 and 3e-5).
LOOSE = {
    "t_": {"conf_low": 1e-6, "conf_high": 1e-6, "adj_p_value": (5e-5, 1e-9)},
    "f_": {"estimate": 1e-5, "conf_low": 1e-4, "conf_high": 1e-4},
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_r(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CASES[case]()
    assert_matches_r(result, _oracle("stats_tier1")[case], loose=LOOSE.get(case[:2]))


# grouped results are the per-subset results ---------------------------------

GROUPED = {
    "wilcox_test": lambda: wilcox_test("body_mass_g ~ sex"),
    "kruskal_test": lambda: kruskal_test("body_mass_g ~ island"),
    "cor_test": lambda: cor_test("bill_length_mm", "flipper_length_mm", method="kendall"),
    "aov": lambda: aov("body_mass_g ~ island"),
    "chisq_test": lambda: chisq_test("sex", "island"),
    "binom_test": lambda: binom_test(col("sex") == "male"),
}


@pytest.mark.parametrize("name", sorted(GROUPED))
def test_grouped_rows_equal_tests_on_each_subset(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = (penguins() >> group_by("year") >> GROUPED[name]()).collect()
        for year in (2007, 2008, 2009):
            alone = (penguins() >> filter(col("year") == year) >> GROUPED[name]()).collect()
            part = grouped.filter(pl.col("year") == year).drop("year")
            assert_same_frame(part, alone)


def test_results_are_ungrouped_and_in_group_order():
    result = penguins() >> group_by("species", "island") >> wilcox_test("body_mass_g ~ sex")
    assert result._groups is None
    keys = result.collect().select("species", "island").rows()
    assert keys == sorted(keys)


# scipy reference (runs without R) ---------------------------------------------


def _frame():
    return pl.read_csv(PENGUINS_CSV, null_values="NA")



def _row(result):
    return result.collect().row(0, named=True)


def test_rank_sum_matches_scipy():
    d = _frame().drop_nulls(["body_mass_g", "sex"])
    f = d.filter(pl.col("sex") == "female")["body_mass_g"].to_numpy()
    m = d.filter(pl.col("sex") == "male")["body_mass_g"].to_numpy()
    ref = scipy_stats.mannwhitneyu(f, m, method="asymptotic", use_continuity=True)
    row = _row(penguins() >> wilcox_test("body_mass_g ~ sex"))
    assert row["statistic"] == ref.statistic
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)


def test_signed_rank_exact_matches_scipy():
    s = pl.read_csv(RANKS_CSV)
    ref = scipy_stats.wilcoxon(s["a"].to_numpy() - 0.1, method="exact")
    row = _row(ranks() >> wilcox_test("a", mu=0.1))
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-12)



def test_kruskal_matches_scipy():
    d = _frame().drop_nulls(["body_mass_g"])
    ref = scipy_stats.kruskal(*[d.filter(pl.col("species") == s)["body_mass_g"].to_numpy()
                                for s in ("Adelie", "Chinstrap", "Gentoo")])
    row = _row(penguins() >> kruskal_test("body_mass_g ~ species"))
    assert row["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_correlations_match_scipy(method):
    d = _frame().drop_nulls(["bill_length_mm", "bill_depth_mm"])
    x, y = d["bill_length_mm"].to_numpy(), d["bill_depth_mm"].to_numpy()
    ref = {"pearson": scipy_stats.pearsonr, "spearman": scipy_stats.spearmanr,
           "kendall": scipy_stats.kendalltau}[method](x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = _row(penguins() >> cor_test("bill_length_mm", "bill_depth_mm", method=method))
    assert row["estimate"] == pytest.approx(ref.statistic, rel=1e-12)
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)


def test_aov_and_tukey_match_scipy():
    d = _frame().drop_nulls(["body_mass_g"])
    samples = [d.filter(pl.col("species") == s)["body_mass_g"].to_numpy()
               for s in ("Adelie", "Chinstrap", "Gentoo")]
    ref = scipy_stats.f_oneway(*samples)
    row = _row(penguins() >> aov("body_mass_g ~ species"))
    assert row["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    tukey = scipy_stats.tukey_hsd(*samples)
    got = (penguins() >> tukey_hsd("body_mass_g ~ species")).collect()
    ci = tukey.confidence_interval()
    for (a, b), row in zip([(1, 0), (2, 0), (2, 1)], got.iter_rows(named=True)):
        assert row["estimate"] == pytest.approx(tukey.statistic[a, b], rel=1e-12)
        assert row["adj_p_value"] == pytest.approx(tukey.pvalue[a, b], rel=1e-6, abs=1e-12)
        assert row["conf_low"] == pytest.approx(ci.low[a, b], rel=1e-6)


def test_chisq_and_binom_match_scipy():
    d = _frame()
    table = d.group_by("species", "island").len().pivot("island", index="species", values="len").fill_null(0).sort("species")
    ref = scipy_stats.chi2_contingency(table.drop("species").to_numpy())
    row = _row(penguins() >> chisq_test("species", "island"))
    assert row["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    males = d["sex"].drop_nulls()
    ref = scipy_stats.binomtest(int((males == "male").sum()), len(males))
    row = _row(penguins() >> binom_test(col("sex") == "male"))
    assert row["p_value"] == pytest.approx(ref.pvalue, rel=1e-12)
    ci = ref.proportion_ci()
    assert (row["conf_low"], row["conf_high"]) == pytest.approx((ci.low, ci.high), rel=1e-10)


# warnings and errors follow R ---------------------------------------------------


def test_small_expected_counts_warn_as_r_does():
    with pytest.warns(UserWarning, match="Chi-squared approximation may be incorrect"):
        (ranks() >> chisq_test("g", "h")).collect()


def test_exact_rank_correlation_with_ties_warns_as_r_does():
    with pytest.warns(UserWarning, match="cannot compute exact p-value with ties"):
        penguins() >> cor_test("bill_length_mm", "bill_depth_mm", method="spearman")


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
        tidy(frame) >> call()


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
