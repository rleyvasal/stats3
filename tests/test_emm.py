"""Marginal means and contrasts against R's emmeans on every backend."""

import subprocess

import numpy as np
import polars as pl
import pytest

from stats3 import Gamma, contrast, emmeans, gaussian, glm, lm, pairs, tidy, vcov_cl
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r, penguins


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("comparison", [None, pairs, lambda e: contrast(e, "eff")])
def test_public_tidy_and_pipe(backend, grouped, comparison):
    data, by = as_input(penguins(), backend, ["species"] if grouped else [])
    result = emmeans(lm(data, "body_mass_g ~ sex", by=by), "sex")
    if comparison is not None:
        result = comparison(result)
    table = tidy(result)
    assert type(table) is type(data)
    assert as_polars(table).equals(result.tidy())
    assert as_polars(result >> tidy()).equals(as_polars(table))


def test_public_means_and_pairwise_difference():
    data = {"y": [1., 2., 3., 5., 6., 7.], "g": ["a"] * 3 + ["b"] * 3}
    means = emmeans(lm(data, "y ~ g"), "g")
    np.testing.assert_allclose(tidy(means)["estimate"], [2., 6.])
    np.testing.assert_allclose(tidy(pairs(means))["estimate"], [-4.])


@pytest.mark.parametrize("options", [{"conf_int": True}, {"conf_level": 0.9}, {"exponentiate": True}])
def test_tidy_does_not_silently_ignore_model_options(options):
    means = emmeans(lm(penguins(), "body_mass_g ~ species"), "species")
    with pytest.raises(TypeError, match="set intervals and scale"):
        tidy(means, **options)


BASE = "body_mass_g ~ flipper_length_mm + species"
TWO = "body_mass_g ~ species * sex"
LOGIT = "am ~ wt + factor(cyl)"
POISSON = "carb ~ hp + factor(cyl) + offset(log(gear))"


def basic(data):
    return emmeans(lm(data, BASE), "species")


@pytest.fixture(scope="module")
def r_emmeans():
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    probe = subprocess.run([*R_COMMAND, "-e", "library(emmeans)"], capture_output=True)
    if probe.returncode:
        pytest.skip("R's emmeans package is unavailable")
    return _oracle("emmeans")


R_CASES = {
    "lm_means": basic,
    "lm_pairs": lambda d: pairs(basic(d)),
    "reverse": lambda d: pairs(basic(d), reverse=True),
    "control": lambda d: contrast(basic(d), "trt.vs.ctrl", ref=2),
    "effects": lambda d: contrast(basic(d), "eff"),
    "custom": lambda d: contrast(basic(d), {"first_vs_rest": [1, -.5, -.5]}),
    "level90": lambda d: emmeans(lm(d, BASE), "species", level=.9),
    "pairs_level90": lambda d: pairs(basic(d), level=.9),
    "at": lambda d: emmeans(lm(d, "body_mass_g ~ flipper_length_mm * species"), "species",
                            at={"flipper_length_mm": [190, 210]}),
    "at_by": lambda d: emmeans(lm(d, BASE), "species", by="flipper_length_mm",
                               at={"flipper_length_mm": [190, 210]}),
    "subset": lambda d: emmeans(lm(d, BASE), "species", at={"species": ["Gentoo", "Adelie"]}),
    "two_factors": lambda d: emmeans(lm(d, TWO), ["species", "sex"]),
    "by": lambda d: emmeans(lm(d, TWO), "species", by="sex"),
    "by_pairs": lambda d: pairs(emmeans(lm(d, TWO), "species", by="sex")),
    "weighted": lambda d: emmeans(lm(d, "body_mass_g ~ species + flipper_length_mm", weights="bill_depth_mm"), "species"),
    "robust": lambda d: emmeans(lm(d, BASE), "species", vcov="HC3"),
    "aliased": lambda d: emmeans(lm(d, "body_mass_g ~ species * island"), ["species", "island"]),
    "aliased_pairs": lambda d: pairs(emmeans(lm(d, "body_mass_g ~ species * island"), "species", by="island")),
    "binary": lambda d: emmeans(lm(d, "mpg ~ wt * am"), "am"),
    "binary_average": lambda d: emmeans(lm(d, "mpg ~ wt * am"), "wt"),
    "no_intercept": lambda d: emmeans(lm(d, "body_mass_g ~ 0 + species"), "species"),
    "robust_pairs": lambda d: pairs(emmeans(lm(d, BASE), "species", vcov="HC3")),
    "custom_scheffe": lambda d: contrast(basic(d), {"a": [1, -1, 0], "b": [2, -2, 0]}, adjust="scheffe"),
    "offset_arg": lambda d: pairs(emmeans(glm(d, "carb ~ hp + factor(cyl) + gear", "poisson", offset="gear"), "gear", at={"gear": [3, 5]})),
    "logit_average": lambda d: contrast(emmeans(glm(d, LOGIT, "binomial"), "cyl", type="response"), {"average": [.5, .5, 0]}),
    "offset_pairs": lambda d: pairs(emmeans(glm(d, POISSON, "poisson"), "gear", at={"gear": [3, 5]})),
    "gamma": lambda d: emmeans(glm(d, "mpg ~ wt + factor(cyl)", Gamma(link="log")), "cyl", type="response"),
    "gaussian": lambda d: emmeans(glm(d, "mpg ~ wt + factor(cyl)", gaussian(link="log")), "cyl", type="response"),
}
for adjustment in ("none", "tukey", "sidak", "scheffe", "bonferroni", "holm", "hochberg", "hommel", "BH", "BY", "fdr"):
    R_CASES[f"pairs_{adjustment}"] = lambda d, a=adjustment: pairs(basic(d), adjust=a)
for weighting in ("equal", "proportional", "cells"):
    R_CASES[f"weights_{weighting}"] = lambda d, w=weighting: emmeans(lm(d, TWO), "species", weights=w)
    R_CASES[f"offset_weights_{weighting}"] = lambda d, w=weighting: emmeans(
        glm(d, "carb ~ hp + factor(cyl) + offset(am)", "poisson"), "cyl", weights=w)
    R_CASES[f"offset_weights_pairs_{weighting}"] = lambda d, w=weighting: pairs(emmeans(
        glm(d, "carb ~ hp + factor(cyl) + offset(am)", "poisson"), "cyl", weights=w))
for family, formula in (("logit", LOGIT), ("poisson", POISSON)):
    for scale in ("link", "response"):
        R_CASES[f"{family}_{scale}"] = lambda d, f=formula, t=scale, fam=family: emmeans(
            glm(d, f, "binomial" if fam == "logit" else "poisson"), "cyl", type=t)
        R_CASES[f"{family}_pairs_{scale}"] = lambda d, f=formula, t=scale, fam=family: pairs(emmeans(
            glm(d, f, "binomial" if fam == "logit" else "poisson"), "cyl", type=t))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(R_CASES))
def test_emmeans_matches_r(name, backend, r_emmeans):
    import polars as pl

    cars = name.startswith(("logit", "poisson", "binary", "offset")) or name in ("gamma", "gaussian")
    data = pl.read_csv(DATA / "mtcars.csv") if cars else penguins()
    frame, _ = as_input(data, backend, [])
    result = tidy(R_CASES[name](frame))
    assert type(result) is type(frame)
    # R's ptukey loses about 1e-12 in the tail; qtukey's root iteration
    # limits interval accuracy. Dunnett's blend uses uniroot with tol=5e-4.
    # Keep coefficients and standard errors at the usual 1e-9.
    tukey = name in ("lm_pairs", "reverse", "by_pairs", "pairs_level90", "pairs_tukey", "aliased_pairs", "robust_pairs") or "_pairs_" in name
    loose = {}
    if tukey or name == "control":
        loose = {"p_value": (1e-6, 1e-12), "conf_low": (1e-7, 1e-6), "conf_high": (1e-7, 1e-6)}
    if name == "control":
        loose.update(conf_low=(1e-6, 1e-5), conf_high=(1e-6, 1e-5))
    assert_matches_r(result, r_emmeans[name], loose=loose)


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_emmeans_matches_r(backend, r_emmeans):
    frame, by = as_input(penguins(), backend, ["species"])
    result = tidy(emmeans(lm(frame, "body_mass_g ~ flipper_length_mm + sex", by=by), "sex"))
    assert type(result) is type(frame)
    assert_matches_r(result, r_emmeans["grouped"])


def _housing():
    import warnings

    from test_ordinal import housing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return housing().with_columns(Rev=pl.col("Sat"))


def _polr(data, formula="Sat ~ Infl + Type + Cont", **kwargs):
    from stats3 import polr

    return polr(data, formula, weights="Freq", **kwargs)


POLR_CASES = {
    "polr_latent": lambda d: emmeans(_polr(d), "Infl"),
    "polr_latent_pairs": lambda d: pairs(emmeans(_polr(d), "Infl")),
    "polr_latent_effects": lambda d: contrast(emmeans(_polr(d), "Infl"), "eff"),
    "polr_prob": lambda d: emmeans(_polr(d), "Infl", by="Sat", mode="prob"),
    "polr_prob_pairs": lambda d: pairs(emmeans(_polr(d), "Infl", by="Sat", mode="prob")),
    "polr_prob_two": lambda d: emmeans(_polr(d), ["Infl", "Sat"], mode="prob"),
    "polr_mean_class": lambda d: emmeans(_polr(d), "Infl", mode="mean.class"),
    "polr_mean_class_pairs": lambda d: pairs(emmeans(_polr(d), "Infl", mode="mean.class")),
    "polr_by": lambda d: emmeans(_polr(d), "Infl", by="Cont"),
    "polr_proportional": lambda d: emmeans(_polr(d), "Infl", weights="proportional"),
    "polr_cells": lambda d: emmeans(_polr(d), "Infl", weights="cells"),
    "polr_prob_cells": lambda d: emmeans(_polr(d), "Infl", by="Sat", mode="prob", weights="cells"),
    "polr_offset": lambda d: emmeans(_polr(d, "Sat ~ Infl + Type + offset(off)"), "Infl"),
    "polr_offset_prob": lambda d: emmeans(_polr(d, "Sat ~ Infl + Type + offset(off)"), "Infl", by="Sat", mode="prob"),
    "polr_probit_prob": lambda d: emmeans(_polr(d, method="probit"), "Infl", by="Sat", mode="prob"),
    "polr_cloglog_mean_class": lambda d: emmeans(_polr(d, method="cloglog"), "Infl", mode="mean.class"),
    "polr_cauchit_prob": lambda d: emmeans(_polr(d, method="cauchit"), "Infl", by="Sat", mode="prob"),
    "polr_loglog_prob": lambda d: emmeans(_polr(d, method="loglog"), "Infl", by="Sat", mode="prob"),
    "polr_loglog_mean_class": lambda d: emmeans(_polr(d, method="loglog"), "Infl", mode="mean.class"),
    "polr_robust": lambda d: emmeans(_polr(d), "Infl", vcov=vcov_cl("Type")),
    "polr_robust_prob": lambda d: emmeans(_polr(d), "Infl", by="Sat", mode="prob", vcov=vcov_cl("Type", type="HC1")),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(POLR_CASES))
def test_polr_emmeans_matches_r(name, backend, r_emmeans):
    frame, _ = as_input(_housing(), backend, [])
    result = tidy(POLR_CASES[name](frame))
    assert type(result) is type(frame)
    # The polr estimates match MASS at optimizer accuracy (see test_ordinal),
    # so the means and their standard errors do too. Cauchit is looser: MASS's
    # optimum is set by its stopping rule (see test_ordinal). Cluster score
    # sums nearly cancel, so robust standard errors amplify that error.
    rel, absolute = (2e-6, 1e-7)
    if "cauchit" in name:
        rel, absolute = (3e-5, 3e-6)
    if "robust" in name:
        rel, absolute = (3e-4, 1e-4)
    loose = {c: (rel, absolute) for c in ("estimate", "std_error", "statistic", "conf_low", "conf_high")}
    # Tukey p-values and intervals use R's ptukey/qtukey accuracy, as above.
    loose["p_value"] = (1e-6, 1e-9)
    if "pairs" in name:
        loose.update(conf_low=(max(rel, 1e-6), max(absolute, 1e-6)), conf_high=(max(rel, 1e-6), max(absolute, 1e-6)))
    assert_matches_r(result, r_emmeans[name], loose=loose)


def _multinom_data():
    return (penguins().drop_nulls(["species", "body_mass_g", "sex"]).with_row_index("i", offset=1)
            .with_columns(mass=pl.col("body_mass_g") / 1000, yf=pl.col("year").cast(pl.String),
                          w=1 + pl.col("i") % 3))


def _multinom(data, **kwargs):
    from stats3 import multinom

    return multinom(data, "species ~ mass + sex + yf", **kwargs)


MULTINOM_CASES = {
    "multinom_prob": lambda d: emmeans(_multinom(d), "species", by="sex"),
    "multinom_prob_pairs": lambda d: pairs(emmeans(_multinom(d), "species", by="sex")),
    "multinom_latent": lambda d: emmeans(_multinom(d), "species", by="sex", mode="latent"),
    "multinom_species": lambda d: emmeans(_multinom(d), "species"),
    "multinom_proportional": lambda d: emmeans(_multinom(d), "species", by="sex", weights="proportional"),
    "multinom_weighted": lambda d: emmeans(_multinom(d, weights="w"), "species", by="sex"),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(MULTINOM_CASES))
def test_multinom_emmeans_matches_r(name, backend, r_emmeans):
    frame, _ = as_input(_multinom_data(), backend, [])
    result = tidy(MULTINOM_CASES[name](frame))
    assert type(result) is type(frame)
    # nnet reaches the MLE to about 1e-5 (see test_multinom), so the means,
    # which are smooth functions of the coefficients, agree to that accuracy.
    loose = {c: (5e-5, 1e-6) for c in ("estimate", "std_error", "statistic", "conf_low", "conf_high")}
    loose["p_value"] = (1e-4, 1e-9)
    assert_matches_r(result, r_emmeans[name], loose=loose)


def test_multinom_emmeans_modes():
    fit = _multinom(_multinom_data())
    with pytest.raises(ValueError, match=r"mode must be one of \['prob', 'latent'\]"):
        emmeans(fit, "species", mode="mean.class")
    with pytest.raises(ValueError, match="multinom fits choose their scale with mode="):
        emmeans(fit, "species", type="response")
    # Each sex's probabilities sum to one.
    probs = tidy(emmeans(fit, "species", by="sex"))
    assert probs.group_by("sex").agg(pl.col("estimate").sum())["estimate"].to_list() == pytest.approx([1, 1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_polr_emmeans_matches_r(backend, r_emmeans):
    frame, by = as_input(_housing(), backend, ["Cont"])
    result = tidy(emmeans(_polr(frame, "Sat ~ Infl + Type", by=by), "Infl"))
    assert type(result) is type(frame)
    loose = {c: (2e-6, 1e-7) for c in ("estimate", "std_error", "conf_low", "conf_high")}
    assert_matches_r(result, r_emmeans["polr_grouped"], loose=loose)
