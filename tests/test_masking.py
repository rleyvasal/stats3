"""Notebook shorthand: unquoted formulas and bare column names in stats3 calls."""

from __future__ import annotations

import polars as pl
import pytest

from stats3.masking import mask_source, quote_formulas
from oracle import PENGUINS, assert_same_frame, penguins

NS = {"penguins", "df", "mask", "group_by", "mutate", "stats3", "log", "x", "pl"}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("penguins >> t_test(body_mass_g ~ sex)", "penguins >> t_test('body_mass_g ~ sex')"),
        ("t_test(penguins, body_mass_g ~ sex)", "t_test(penguins, 'body_mass_g ~ sex')"),
        ("wilcox_test(log(y)   ~ g, mu=0.3)", "wilcox_test('log(y) ~ g', mu=0.3)"),
        ("stats3.t_test(\n    body_mass_g ~\n      sex,\n)", "stats3.t_test('body_mass_g ~ sex')"),
        ('t_test("y ~ g")', "t_test('y ~ g')"),
        ("friedman_test(v ~ k | id)", "friedman_test('v ~ k | id')"),
        ("pairwise_t_test(y ~ g, pool_sd=False)", "pairwise_t_test('y ~ g', pool_sd=False)"),
        ("glm_nb(y ~ x + g)", "glm_nb('y ~ x + g')"),
        ("df.filter(~mask)", "df.filter(~mask)"),
        ("x = ~x", "x = ~x"),
    ],
)
def test_formulas_are_quoted_only_in_stats3_calls(source, expected):
    assert mask_source(source, NS) == expected


def test_formulas_elsewhere_are_left_alone():
    # Not a stats3 function: left for Python to report.
    assert quote_formulas("other(y ~ g)") == "other(y ~ g)"
    assert quote_formulas("text = 'a ~ b'") == "text = 'a ~ b'"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('binom_test(sex == "male")', "binom_test(__stats3_col__('sex') == 'male')"),
        ("tidy(fit, vcov=vcov_cl(island))", "tidy(fit, vcov=vcov_cl(__stats3_col__('island')))"),
        ("cor_test(a, b, by=species)",
         "cor_test(__stats3_col__('a'), __stats3_col__('b'), by=__stats3_col__('species'))"),
        ("x = 3\nbinom_test(sex == 'male', p=x)",
         "x = 3\nbinom_test(__stats3_col__('sex') == 'male', p=x)"),
        ("t_test(penguins, flipper_length_mm, mu=200)",
         "t_test(penguins, __stats3_col__('flipper_length_mm'), mu=200)"),
        ("mutate(adj=p_adjust(p_value, 'BH'))", "mutate(adj=p_adjust(__stats3_col__('p_value'), 'BH'))"),
        ("t_test(pl.col('x'))", "t_test(pl.col('x'))"),
        ("t_test(len)", "t_test(len)"),
        ("group_by(species)", "group_by(species)"),
    ],
)
def test_undefined_names_become_columns_inside_stats3_calls(source, expected):
    assert mask_source(source, NS) == expected


def _shell():
    pytest.importorskip("IPython")
    from IPython.testing.globalipapp import get_ipython

    return get_ipython()


@pytest.mark.filterwarnings("ignore:cannot compute exact p-value with ties")
def test_shorthand_runs_in_ipython():
    # Importing stats3 in a notebook turns the shorthand on; here stats3 was
    # imported before the shell existed, so load it as an extension.
    ip = _shell()
    cells = [
        f"%load_ext stats3.jupyter\nimport polars as pl\nfrom stats3 import *\npenguins = pl.read_csv({str(PENGUINS)!r}, null_values='NA')",
        "r1 = penguins >> t_test(body_mass_g ~ sex, by=species)",
        "r2 = penguins >> binom_test(sex == 'male')",
        "r3 = cor_test(penguins, bill_length_mm, bill_depth_mm, method='spearman')",
    ]
    for cell in cells:
        result = ip.run_cell(cell)
        assert result.success, (cell, result.error_before_exec or result.error_in_exec)
    from stats3 import binom_test, cor_test, t_test

    assert_same_frame(ip.user_ns["r1"], t_test(penguins(), "body_mass_g ~ sex", by="species"))
    assert_same_frame(ip.user_ns["r2"], binom_test(penguins(), pl.col("sex") == "male"))
    with pytest.warns(UserWarning):
        expected = cor_test(penguins(), "bill_length_mm", "bill_depth_mm", method="spearman")
    assert_same_frame(ip.user_ns["r3"], expected)


def test_shorthand_runs_alongside_tidy3_in_ipython():
    pytest.importorskip("tidy3")
    ip = _shell()
    cells = [
        f"%load_ext stats3.jupyter\nimport polars as pl\nfrom tidy3 import *\nfrom stats3 import *\n"
        f"pt = tidy(pl.read_csv({str(PENGUINS)!r}, null_values='NA'))",
        "r4 = (pt\n>> group_by(species)\n>> t_test(body_mass_g ~ sex)\n)",
        "r5 = (pt\n>> group_by(species, island)\n>> wilcox_test(body_mass_g ~ sex)\n"
        ">> group_by(species)\n>> mutate(adj = p_adjust(p_value, 'BH'))\n)",
    ]
    for cell in cells:
        result = ip.run_cell(cell)
        assert result.success, (cell, result.error_before_exec or result.error_in_exec)
    assert type(ip.user_ns["r4"]).__name__ == "TidyFrame"
    assert "adj" in ip.user_ns["r5"].collect().columns


def test_model_formulas_run_unquoted_in_ipython():
    ip = _shell()
    cells = [
        f"%load_ext stats3.jupyter\nimport polars as pl\nfrom stats3 import *\n"
        f"penguins = pl.read_csv({str(PENGUINS)!r}, null_values='NA')",
        "fit = penguins >> lm(body_mass_g ~ flipper_length_mm + species)",
        "logit = glm(penguins, sex ~ body_mass_g, family=binomial)",
        "coefs = fit >> tidy()",
    ]
    for cell in cells:
        result = ip.run_cell(cell)
        assert result.success, (cell, result.error_before_exec or result.error_in_exec)
    assert ip.user_ns["coefs"]["term"].to_list()[-1] == "speciesGentoo"
    assert ip.user_ns["logit"].family.family == "binomial"
