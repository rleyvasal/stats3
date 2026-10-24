"""Robust standard errors against R's sandwich and lmtest, row alignment
against R's na.exclude, and modelsummary tables."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from stats3 import (
    Gamma,
    augment,
    binomial,
    glance,
    glm,
    lm,
    modelsummary,
    poisson,
    predict,
    tidy,
    vcov,
    vcov_cl,
    vcov_hc,
    vcov_nw,
)
from oracle import BACKENDS, _oracle, as_input, as_polars, assert_matches_r, penguins

DATA = Path(__file__).parent / "data"


def mtcars() -> pl.DataFrame:
    return pl.read_csv(DATA / "mtcars.csv")


BASE = "body_mass_g ~ flipper_length_mm + species"
P, M = penguins, mtcars


def base(data):
    return lm(data, BASE)


def weighted(data):
    return lm(data, BASE, weights="bill_depth_mm")


def logit(data):
    return glm(data, "am ~ wt + hp", family=binomial())


def gamma(data):
    return glm(data, "mpg ~ wt + hp", family=Gamma(link="log"))


def pois(data):
    return glm(data, "carb ~ hp + offset(log(gear))", family=poisson())


# (data, fit, vcov) per R case; each runs on every backend.
CASES = {
    **{f"hc_{t}": (P, base, t) for t in ("const", "HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5")},
    "hc_weighted": (P, weighted, "HC3"),
    "hc_weighted_hc1": (P, weighted, vcov_hc("HC1")),
    "cl_island": (P, base, vcov_cl("island")),
    "cl_two_way": (P, base, vcov_cl(["island", "year"])),
    "cl_year_hc2": (P, base, vcov_cl("year", "HC2")),
    "cl_year_hc3": (P, base, vcov_cl("year", "HC3")),
    "cl_weighted_hc2": (P, weighted, vcov_cl("year", "HC2")),
    "cl_no_adjust": (P, base, vcov_cl("island", "HC0", cadjust=False)),
    "nw_default": (P, base, vcov_nw()),
    "nw_lag": (P, base, vcov_nw(4, prewhite=False, adjust=True)),
    "nw_order": (P, base, vcov_nw(order_by="flipper_length_mm")),
    "glm_hc3": (M, logit, "HC3"),
    "glm_hc0": (M, logit, "HC0"),
    "glm_cl": (M, logit, vcov_cl("cyl")),
    "gamma_hc1": (M, gamma, "HC1"),
    "gamma_nw": (M, gamma, "NW"),
    "pois_cl": (M, pois, vcov_cl(["cyl", "am"])),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_robust_tidy_matches_sandwich(name, backend):
    data, make, v = CASES[name]
    frame, _ = as_input(data(), backend, [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tidy(make(frame), conf_int=True, vcov=v)
    # sandwich and lmtest are exact; stats3's QR and R's agree to about 1e-12.
    assert_matches_r(result, _oracle("robust")[name])


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_clustered(backend):
    frame, by = as_input(penguins(), backend, ["species"])
    fits = lm(frame, "body_mass_g ~ flipper_length_mm + sex", by=by)
    assert_matches_r(tidy(fits, conf_int=True, vcov=vcov_cl("year")), _oracle("robust")["by_species"])


def test_robust_wald_f_matches_waldtest():
    got = glance(base(penguins()), vcov="HC1").select("statistic", "p_value")
    assert_matches_r(got, _oracle("robust")["wald_hc1"])


def test_vcov_matrices():
    fit = base(penguins())
    assert_matches_r(vcov(fit, "HC3"), _oracle("robust")["vcov_hc3"])
    assert_matches_r(vcov(fit), _oracle("robust")["vcov_classical"])


def test_classical_vcov_matches_tidy():
    fit = base(penguins())
    v = vcov(fit, "const").drop("term").to_numpy()
    assert np.sqrt(np.diag(v)) == pytest.approx(tidy(fit)["std_error"].to_numpy(), rel=1e-12)


def test_glm_with_vcov_uses_wald_intervals():
    fit = logit(mtcars())
    with pytest.raises(ValueError, match="profile intervals"):
        fit.tidy(conf_int=True, conf_method="profile", vcov="HC0")
    wald = tidy(fit, conf_int=True, vcov="HC0")
    assert wald["conf_low"][1] < wald["estimate"][1] < wald["conf_high"][1]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: vcov_hc("HC9"), "type must be one of"),
        (lambda: vcov_cl("island", "HC4"), "type must be one of"),
        (lambda: vcov_cl([]), "at least one"),
        (lambda: vcov_nw(lag=-1), "lag must be"),
        (lambda: tidy(base(penguins()), vcov=3), "vcov must be"),
        (lambda: tidy(lm(penguins(), "body_mass_g ~ flipper_length_mm"), vcov=vcov_cl("sex")), "missing values"),
        (lambda: glance(logit(mtcars()), vcov="HC1"), "vcov is for lm"),
    ],
)
def test_errors(call, message):
    with pytest.raises((ValueError, TypeError), match=message):
        call()


# row alignment -------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_exclude_keeps_every_row(backend):
    frame, _ = as_input(penguins(), backend, [])
    fit = lm(frame, "body_mass_g ~ flipper_length_mm + sex", na_action="exclude")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_matches_r(predict(fit, interval="confidence"), _oracle("robust")["exclude_lm"])
    got = as_polars(augment(fit))
    assert got.height == 344
    assert_matches_r(got.select(".fitted", ".resid"), _oracle("robust")["exclude_resid"])


def test_exclude_glm_predict():
    fit = glm(penguins(), "sex ~ body_mass_g + species", family=binomial(), na_action="exclude")
    assert_matches_r(predict(fit, type="response", se_fit=True), _oracle("robust")["exclude_glm"])


def test_exclude_by_group_lines_up_with_each_group():
    p = penguins()
    fits = lm(p, "body_mass_g ~ flipper_length_mm + sex", by="species", na_action="exclude")
    out = augment(fits)
    assert out.height == p.height
    assert out.group_by("species").len().sort("species")["len"].to_list() == [152, 68, 124]
    missing = out.filter(pl.col("sex").is_null() | pl.col("body_mass_g").is_null())
    assert missing[".fitted"].null_count() == missing.height


def test_omit_is_the_default_and_fail_raises():
    p = penguins()
    assert augment(lm(p, "body_mass_g ~ sex")).height == 333
    with pytest.raises(ValueError, match="na_action='fail'"):
        lm(p, "body_mass_g ~ sex", na_action="fail")
    assert augment(lm(mtcars(), "mpg ~ wt", na_action="fail")).height == 32
    with pytest.raises(ValueError, match="na_action must be"):
        lm(p, "body_mass_g ~ sex", na_action="pass")


# modelsummary --------------------------------------------------------------------


def test_modelsummary_frame_matches_tidy_and_glance():
    p = penguins()
    a, b = lm(p, "body_mass_g ~ flipper_length_mm"), lm(p, BASE)
    table = modelsummary({"Base": a, "Species": b})
    assert table.columns == ["part", "term", "statistic", "Base", "Species"]
    estimates = table.filter(pl.col("statistic") == "estimate")
    assert estimates["term"].to_list() == tidy(b)["term"].to_list()
    assert estimates.filter(pl.col("term") == "speciesGentoo")["Base"][0] == ""
    assert estimates.filter(pl.col("term") == "speciesGentoo")["Species"][0] == f"{tidy(b)['estimate'][3]:.3f}"
    se = table.filter((pl.col("term") == "flipper_length_mm") & (pl.col("statistic") == "std_error"))
    assert se["Species"][0] == f"({tidy(b)['std_error'][1]:.3f})"
    gof = dict(table.filter(pl.col("part") == "gof").select("term", "Base").iter_rows())
    assert gof["Num.Obs."] == "342"
    assert gof["R2"] == f"{glance(a)['r_squared'][0]:.3f}"


def test_modelsummary_stars_robust_and_intervals():
    p = penguins()
    fit = lm(p, BASE)
    table = modelsummary(fit, stars=True, vcov="HC1", statistic="conf_int", digits=2)
    robust = tidy(fit, conf_int=True, vcov="HC1")
    row = table.filter((pl.col("term") == "speciesChinstrap") & (pl.col("statistic") == "conf_int"))
    assert row["(1)"][0] == f"[{robust['conf_low'][2]:.2f}, {robust['conf_high'][2]:.2f}]"
    assert table.filter(pl.col("statistic") == "estimate")["(1)"][2].endswith("***")
    assert table.filter(pl.col("term") == "Std.Errors")["(1)"][0] == "HC1"


def test_modelsummary_text_outputs():
    p = penguins()
    fits = lm(p, "body_mass_g ~ flipper_length_mm", by="species")
    text = modelsummary(fits, output="markdown")
    assert [c.strip() for c in text.splitlines()[0].split("|")[2:5]] == ["Adelie", "Chinstrap", "Gentoo"]
    assert "Num.Obs." in text and "(5.076)" in text
    page = modelsummary(fits, output="html", stars=True)
    assert page.startswith("<table>") and "<th>Gentoo</th>" in page and "p &lt; 0.05" in page


def test_modelsummary_glm_and_frame_type():
    pd = pytest.importorskip("pandas")
    fit = glm(penguins(), "sex ~ body_mass_g", family=binomial())
    table = modelsummary(fit, exponentiate=True, as_="pandas")
    assert isinstance(table, pd.DataFrame)
    assert "R2" not in set(table["term"])
    assert table.loc[table["statistic"] == "estimate", "(1)"].iloc[1] == f"{np.exp(tidy(fit)['estimate'][1]):.3f}"


def test_modelsummary_errors():
    fit = lm(penguins(), BASE)
    with pytest.raises(ValueError, match="statistic must be"):
        modelsummary(fit, statistic="se")
    with pytest.raises(ValueError, match="unknown gof"):
        modelsummary(fit, gof=["r2"])
    with pytest.raises(TypeError, match="expected models"):
        modelsummary(penguins())
    with pytest.raises(ValueError, match="2 vcov for 1 models"):
        modelsummary(fit, vcov=["HC1", "HC3"])
