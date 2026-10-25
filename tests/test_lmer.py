"""Linear mixed models against lme4 on every backend."""

import subprocess
import warnings

import numpy as np
import polars as pl
import pytest

import stats3.mixed as mixed
from stats3 import anova, augment, glance, lm, lmer, modelsummary, predict, ranef, tidy, vcov
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r, assert_same_frame


def sleep():
    # lme4's sleepstudy, with a weights column of 1, 2, 3.
    return pl.read_csv(DATA / "sleepstudy.csv").with_row_index("i", offset=1).with_columns(w=1 + pl.col("i") % 3)


def penguins():
    return pl.read_csv(DATA / "penguins.csv", null_values="NA").with_row_index("i", offset=1).with_columns(
        w=1 + pl.col("i") % 3)


# name: (data, formula, arguments), as in tests/r_lmer_cases.R
FITS = {
    "intercept": (sleep, "Reaction ~ Days + (1 | Subject)", {}),
    "slopes": (sleep, "Reaction ~ Days + (Days || Subject)", {}),
    "slope_only": (sleep, "Reaction ~ Days + (0 + Days | Subject)", {}),
    "ml": (sleep, "Reaction ~ Days + (Days || Subject)", {"reml": False}),
    "weighted": (sleep, "Reaction ~ Days + (Days || Subject)", {"weights": "w"}),
    "weighted_ml": (sleep, "Reaction ~ Days + (Days || Subject)", {"weights": "w", "reml": False}),
    "offset": (sleep, "Reaction ~ Days + offset(0.5 * Days) + (1 | Subject)", {}),
    "offset_arg": (sleep, "Reaction ~ Days + (1 | Subject)", {"offset": 0.5 * pl.col("Days")}),
    "no_intercept": (sleep, "Reaction ~ 0 + Days + (1 | Subject)", {}),
    "transformed": (sleep, "log(Reaction) ~ I(Days^2) + (1 | Subject)", {}),
    "factors": (penguins, "body_mass_g ~ flipper_length_mm + sex + (1 | species)", {}),
    "interaction": (penguins, "body_mass_g ~ flipper_length_mm * sex + (1 | species)", {}),
    "weighted_factors": (penguins, "body_mass_g ~ flipper_length_mm + sex + (1 | species)", {"weights": "w"}),
    "crossed": (penguins, "body_mass_g ~ flipper_length_mm + (1 | species) + (1 | island)", {}),
    "interaction_group": (penguins, "body_mass_g ~ flipper_length_mm + sex + (1 | species:island)", {}),
    "nested": (penguins, "body_mass_g ~ flipper_length_mm + sex + (1 | species/sex)", {}),
    "island_slopes": (penguins, "body_mass_g ~ sex + (bill_depth_mm || island)", {}),
    "species_slope": (penguins, "bill_length_mm ~ sex + (0 + bill_depth_mm | species)", {}),
}
TABLES = {
    "tidy": lambda f: tidy(f, conf_int=True), "glance": glance,
    "augment": lambda f: augment(f), "ranef": ranef, "vcov": vcov, "anova": anova,
    "theta": lambda f: pl.DataFrame({"theta": f.theta, "sigma": f.sigma}),
}
# Residuals and Cook's distances pass through zero, so their relative error is
# larger than that of the fit itself; everything else agrees to 1e-7.
LOOSE_TIGHT = {".resid": 1e-6, ".cooksd": 1e-6}


def fit(name, data=None):
    make, formula, kwargs = FITS[name]
    return lmer(make() if data is None else data, formula, **kwargs)


def table(name, fit_):
    result = TABLES[name](fit_)
    if name == "augment":
        return as_polars(result).select(pl.col(r"^\..*$"))
    return result


def scaled(expected, rel):
    """Tolerances of ``rel`` relative to each column's largest value: for
    values near zero a relative tolerance means nothing."""
    names = expected["columns"]
    out = {}
    for name, values in zip([names] if isinstance(names, str) else names, expected["data"]):
        numbers = [abs(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numbers:
            out[name] = (rel, rel * max(numbers))
    return out


@pytest.fixture(scope="module")
def r_lmer():
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    probe = subprocess.run([*R_COMMAND, "-e", "library(lme4)"], capture_output=True)
    if probe.returncode:
        pytest.skip("Mixed-model oracle needs R's lme4")
    return _oracle("lmer")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(FITS))
@pytest.mark.parametrize("which", sorted(TABLES))
def test_tables_match_lme4(name, which, backend, r_lmer):
    make, _, _ = FITS[name]
    data, _ = as_input(make(), backend, [])
    result = TABLES[which](fit(name, data))
    if which != "theta":
        assert type(result) is type(data)
    if which == "augment":
        result = as_polars(result).select(pl.col(r"^\..*$"))
    # The reference is lme4's own deviance function minimized to 1e-9 in theta
    # (its modular interface with Newton steps; see tests/r_lmer_cases.R).
    assert_matches_r(result, r_lmer[f"{name}__{which}__tight"], rel=1e-7, loose=LOOSE_TIGHT)


@pytest.mark.parametrize("name", sorted(FITS))
@pytest.mark.parametrize("which", sorted(TABLES))
def test_tables_match_default_lmer(name, which, r_lmer):
    # lme4's default optimizer (nloptwrap's BOBYQA) stops on the criterion's
    # value, which is flat near the optimum: its theta carries up to 3e-4
    # relative error on weakly identified components (an island variance from
    # three levels), and fitted values, conditional modes and Cook's
    # distances inherit that. The minimized criterion itself agrees to 1e-11
    # (the ML deviance at a REML fit is not minimized, and moves with theta).
    expected = r_lmer[f"{name}__{which}"]
    loose = scaled(expected, 5e-4)
    for exact in ("logLik", "AIC", "BIC", "REMLcrit"):
        loose[exact] = 1e-10
    loose["deviance"] = 1e-7
    assert_matches_r(table(which, fit(name)), expected, rel=5e-4, loose=loose)


@pytest.mark.parametrize("name", ["intercept", "slopes", "weighted", "crossed", "nested"])
@pytest.mark.parametrize("which", sorted(TABLES))
def test_sparse_factorization_matches_dense(name, which, monkeypatch):
    dense = table(which, fit(name))
    monkeypatch.setattr(mixed, "SPARSE_FROM", 0)
    sparse = table(which, fit(name))
    through_zero = [c for c in (".resid", ".cooksd") if c in dense.columns]
    assert_same_frame(sparse.drop(through_zero), dense.drop(through_zero), rel=1e-8)
    for column in through_zero:  # relative to the column's scale, as these pass through zero
        scale = dense[column].abs().max()
        assert sparse[column].to_numpy() == pytest.approx(dense[column].to_numpy(), abs=1e-8 * scale)


@pytest.mark.parametrize("backend", BACKENDS)
def test_predict_matches_lme4(backend, r_lmer):
    data, _ = as_input(sleep(), backend, [])
    slopes = fit("slopes", data)
    known, _ = as_input(pl.DataFrame({"Days": [0, 5, 9], "Subject": [308, 372, 334]}), backend, [])
    new, _ = as_input(pl.DataFrame({"Days": [0, 5, 9], "Subject": [308, 372, 999]}), backend, [])
    for result, key in [(predict(slopes, new, re_form="population"), "predict_population"),
                        (predict(slopes, known), "predict_random"),
                        (predict(slopes, new, allow_new_levels=True), "predict_new_levels")]:
        assert type(result) is type(data)
        assert_matches_r(result, r_lmer[key], rel=1e-7)
    with pytest.raises(ValueError, match="new levels detected in newdata for 'Subject': 999"):
        predict(slopes, new)
    rows, _ = as_input(penguins()[[0, 1, 2, 152, 276]], backend, [])
    data, _ = as_input(penguins(), backend, [])
    for name in ("factors", "crossed", "nested"):
        assert_matches_r(predict(fit(name, data), rows), r_lmer[f"predict_{name}"], rel=1e-7)
    augmented = as_polars(augment(fit("factors", data), rows))
    assert augmented.columns[-1] == ".fitted" and augmented.height == 5


def test_model_comparisons_match_lme4(r_lmer):
    intercept, slopes, slope_only = fit("intercept"), fit("slopes"), fit("slope_only")
    with pytest.warns(UserWarning, match="refitting model"):
        assert_matches_r(anova(slopes, intercept), r_lmer["compare"], rel=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert_matches_r(anova(intercept, slopes, slope_only), r_lmer["compare_three"], rel=1e-6)
    ml1 = lmer(sleep(), "Reaction ~ Days + (1 | Subject)", reml=False)
    assert_matches_r(anova(ml1, fit("ml")), r_lmer["compare_ml"], rel=1e-7)
    assert fit("ml").refit_ml() is fit("ml") or not fit("ml").reml


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_fits_match_lme4(backend, r_lmer):
    data, by = as_input(penguins(), backend, ["sex"])
    with pytest.warns(UserWarning, match="singular"):
        result = tidy(lmer(data, "body_mass_g ~ flipper_length_mm + (1 | species)", by=by), conf_int=True)
    assert type(result) is type(data)
    # The rows without a sex form a group whose species variance is zero: lme4's
    # bounded optimizer stops exactly at zero, the unbounded one here at 1e-10.
    assert_matches_r(result, r_lmer["grouped"], rel=1e-7, loose={"estimate": (1e-7, 1e-6)})


def test_missing_rows_are_aligned(r_lmer):
    data = penguins()
    with_na = lmer(data, "body_mass_g ~ flipper_length_mm + sex + (1 | species)", na_action="exclude")
    result = augment(with_na)
    assert result.height == data.height and result[".fitted"].null_count() == 11
    assert_matches_r(result.select(".fitted", ".resid"), r_lmer["exclude"], rel=1e-7, loose=LOOSE_TIGHT)
    assert predict(with_na)["fit"].null_count() == 11
    assert predict(with_na, re_form="population").height == data.height
    assert glance(with_na)["nobs"][0] == 333
    assert augment(fit("factors")).height == 333
    with pytest.raises(ValueError, match="missing values"):
        lmer(data, "body_mass_g ~ flipper_length_mm + sex + (1 | species)", na_action="fail")


def test_tidy_effects_pipes_and_summary():
    slopes = fit("slopes")
    full = tidy(slopes)
    assert full.columns == ["effect", "group", "term", "estimate", "std_error", "statistic", "p_value"]
    assert full["effect"].to_list() == ["fixed", "fixed", "ran_pars", "ran_pars", "ran_pars"]
    assert full["group"].to_list() == [None, None, "Subject", "Subject.1", "Residual"]
    assert full["term"].to_list() == ["(Intercept)", "Days", "sd__(Intercept)", "sd__Days", "sd__Observation"]
    assert full["p_value"].null_count() == 5
    assert tidy(slopes, effects="fixed").height == 2
    assert tidy(slopes, effects="ran_pars")["std_error"].null_count() == 3
    modes = tidy(slopes, effects="ran_vals")
    assert modes.columns == ["effect", "group", "term", "level", "estimate", "std_error"]
    assert modes.height == 36 and modes["level"][0] == "308"
    assert slopes.coefficients.keys() == {"(Intercept)", "Days"}
    assert slopes.var_corr()["group"].to_list() == ["Subject", "Subject.1", "Residual"]
    assert "REML criterion" in repr(slopes) and "random effects" in repr(slopes)
    assert "deviance" in repr(fit("ml"))
    assert (sleep() >> lmer("Reaction ~ Days + (Days || Subject)") >> tidy()).equals(full)
    assert (slopes >> glance()).equals(glance(slopes))
    assert (slopes >> anova()).equals(anova(slopes))
    assert glance(slopes).columns == ["nobs", "sigma", "logLik", "AIC", "BIC", "REMLcrit", "deviance", "df_residual"]
    assert glance(fit("ml"))["REMLcrit"][0] is None
    summary = modelsummary({"Intercepts": fit("intercept"), "Slopes": slopes}, stars=True)
    terms = summary.filter(pl.col("statistic") == "estimate")["term"].to_list()
    assert terms == ["(Intercept)", "Days", "SD ((Intercept) Subject)", "SD (Observations)", "SD (Days Subject.1)"]
    assert "AIC" in summary["term"].to_list()


def test_grouping_levels_follow_factor_order():
    order = ["Gentoo", "Chinstrap", "Adelie"]
    data = penguins().with_columns(pl.col("species").cast(pl.Enum(order)))
    assert ranef(lmer(data, "body_mass_g ~ flipper_length_mm + (1 | species)"))["level"].to_list() == order
    pandas = pytest.importorskip("pandas")
    frame = penguins().to_pandas()
    frame["species"] = frame["species"].astype("category").cat.set_categories(order)
    result = ranef(lmer(frame, "body_mass_g ~ flipper_length_mm + (1 | species)"))
    assert isinstance(result, pandas.DataFrame) and result["level"].tolist() == order
    nested = fit("nested")
    assert nested.groups_unique == ["sex:species", "species"]
    assert ranef(nested)["level"][0] == "female:Adelie"


def test_thousands_of_levels_use_the_sparse_path():
    rng = np.random.default_rng(1)
    n, q = 30000, 2000
    group = rng.integers(0, q, n)
    x = rng.normal(size=n)
    y = 1 + 2 * x + 2 * rng.normal(size=q)[group] + 0.5 * rng.normal(size=q)[group] * x + rng.normal(size=n)
    model = lmer(pl.DataFrame({"y": y, "x": x, "g": group}), "y ~ x + (x || g)")
    assert hasattr(model._solution.factor, "_lu") and model._solution.factor._dense is None
    assert model.theta == pytest.approx([2, 0.5], rel=0.1)
    assert glance(model)["sigma"][0] == pytest.approx(1, rel=0.05)
    assert ranef(model).height == 2 * q
    assert predict(model, pl.DataFrame({"x": [0.0], "g": [group[0]]})).height == 1


def test_invalid_models_and_arguments():
    data = sleep()
    with pytest.raises(ValueError, match=r"correlated random effects \(1 \+ Days \| Subject\).*use \(1 \+ Days \|\| Subject\)"):
        lmer(data, "Reaction ~ Days + (1 + Days | Subject)")
    with pytest.raises(ValueError, match="correlated random effects"):
        lmer(penguins(), "body_mass_g ~ 1 + (0 + sex | species)")
    with pytest.raises(ValueError, match="no random effects"):
        lmer(data, "Reaction ~ Days")
    with pytest.raises(ValueError, match="cannot be subtracted"):
        lmer(data, "Reaction ~ Days - (1 | Subject)")
    with pytest.raises(ValueError, match="grouping column names"):
        lmer(data, "Reaction ~ Days + (1 | log(Subject))")
    with pytest.raises(ValueError, match="weights must be positive"):
        lmer(data, "Reaction ~ Days + (1 | Subject)", weights=pl.col("Days"))
    with pytest.raises(ValueError, match="> 1 sampled level"):
        lmer(penguins().filter(pl.col("species") == "Gentoo"), "body_mass_g ~ flipper_length_mm + (1 | species)")
    with pytest.raises(ValueError, match="must be numeric|one numeric"):
        lmer(penguins(), "sex ~ flipper_length_mm + (1 | species)")
    with pytest.raises(ValueError, match="reml must be"):
        lmer(data, "Reaction ~ Days + (1 | Subject)", reml="yes")
    with pytest.raises(ValueError, match="na_action"):
        lmer(data, "Reaction ~ Days + (1 | Subject)", na_action="drop")
    slopes = fit("slopes")
    with pytest.raises(TypeError, match="compare lmer fits with lmer fits"):
        anova(slopes, lm(data, "Reaction ~ Days"))
    with pytest.raises(TypeError, match="type II/III"):
        anova(slopes, type=2)
    with pytest.raises(ValueError, match="test= is not used"):
        anova(slopes, test="F")
    with pytest.raises(ValueError, match="same response, rows and weights"):
        anova(slopes, fit("weighted"))
    with pytest.raises(ValueError, match="own covariance"):
        tidy(slopes, vcov="HC3")
    with pytest.raises(TypeError, match="effects= is for lmer"):
        tidy(lm(data, "Reaction ~ Days"), effects="fixed")
    with pytest.raises(ValueError, match="effects must be chosen"):
        tidy(slopes, effects="random")
    with pytest.raises(ValueError, match="ran_vals.*alone"):
        tidy(slopes, effects=("fixed", "ran_vals"))
    with pytest.raises(ValueError, match="re_form must be"):
        predict(slopes, re_form="~(1 | Subject)")
    with pytest.raises(TypeError, match="expected a model from lmer"):
        ranef(lm(data, "Reaction ~ Days"))
    with pytest.warns(UserWarning, match="rank deficient so dropping 1 column"):
        aliased = lmer(data, "Reaction ~ Days + I(2 * Days) + (1 | Subject)")
    assert aliased.names == ["(Intercept)", "Days"]
    assert tidy(aliased, effects="fixed")["estimate"].to_list() == pytest.approx(tidy(fit("intercept"), effects="fixed")["estimate"].to_list())
