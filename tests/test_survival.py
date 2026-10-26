"""Survival analysis against R's survival package (and broom), on the four backends."""

import numpy as np
import polars as pl
import pytest

from stats3 import anova, augment, cox_zph, coxph, glance, predict, survdiff, survfit, survreg, tidy, vcov
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r


def lung():
    frame = pl.read_csv(DATA / "lung.csv", null_values="NA").rename({"ph.ecog": "ecog"})
    i = pl.int_range(1, pl.len() + 1)
    frame = frame.with_columns(entry=pl.min_horizontal(pl.col("time") - 1, (i % 5) * 20.0),
                               w=(i % 3) + 1.0, wf=.5 + (i % 4) / 4, off=((i % 7) - 3) / 10)
    # Interval-censored deaths, as tests/r_survival_cases.R makes them.
    dead = pl.col("status") == 2
    frame = frame.with_columns(lo=pl.when(dead).then((pl.col("time") // 30) * 30.0).otherwise(pl.col("time").cast(pl.Float64)))
    frame = frame.with_columns(hi=pl.when(dead).then(pl.col("lo") + 30))
    frame = frame.with_columns(lo=pl.when(dead & ((pl.col("time") < 60) | (pl.col("lo") == 0))).then(None).otherwise(pl.col("lo")))
    return frame.with_columns(
        ev=pl.when(pl.col("hi").is_null()).then(0).when(pl.col("lo").is_null()).then(2).otherwise(3),
        t1=pl.coalesce("lo", "hi"), t2=pl.coalesce("hi", "lo"))


def veteran():
    levels = ["squamous", "smallcell", "adeno", "large"]
    return pl.read_csv(DATA / "veteran.csv").with_columns(pl.col("celltype").cast(pl.Enum(levels)))


def complete(frame):
    return frame.drop_nulls(["age", "sex", "ecog"])


S = "Surv(time, status)"


def km(*args, **kwargs):
    def make(data):
        fit = survfit(data, *args, **kwargs)
        return {"tidy": tidy(fit), "glance": glance(fit)}
    return make


def sdiff(*args, **kwargs):
    def make(data):
        fit = survdiff(data, *args, **kwargs)
        return {"tidy": tidy(fit), "glance": glance(fit)}
    return make


def cox(formula, new=None, zph=(), **kwargs):
    def make(data):
        fit = coxph(data, formula, **kwargs)
        out = {"tidy": tidy(fit, conf_int=True), "glance": glance(fit), "vcov": vcov(fit),
               "lp": predict(fit, se_fit=True), "risk": predict(fit, type="risk", se_fit=True),
               "expected": predict(fit, type="expected")}
        if new is not None:
            rows = as_input(new(), BACKEND[0], [])[0]
            out["new_lp"] = predict(fit, rows, se_fit=True)
            out["new_risk"] = predict(fit, rows, type="risk")
            out["new_expected"] = predict(fit, rows, type="expected")
        for name, opts in zph:
            out[name] = cox_zph(fit, **opts)
        return out
    return make


def curves(formula, newdata=None, **kwargs):
    def make(data):
        fit = coxph(data, formula, **{k: v for k, v in kwargs.items() if k in ("ties", "weights")})
        rows = None if newdata is None else as_input(newdata(), BACKEND[0], [])[0]
        curve = survfit(fit, newdata=rows, **{k: v for k, v in kwargs.items() if k == "conf_type"})
        return {"tidy": tidy(curve), "glance": glance(curve)}
    return make


def reg(formula, new, **kwargs):
    def make(data):
        fit = survreg(data, formula, **kwargs)
        rows = as_input(new(), BACKEND[0], [])[0]
        return {"tidy": tidy(fit, conf_int=True), "glance": glance(fit), "response": predict(fit),
                "new_response": predict(fit, rows, se_fit=True),
                "new_lp": predict(fit, rows, type="lp", se_fit=True),
                "new_quantile": predict(fit, rows, type="quantile", p=[.25, .5], se_fit=True),
                "new_uquantile": predict(fit, rows, type="uquantile", p=.5)}
    return make


BACKEND = ["polars"]
NEW_LUNG = lambda: complete(lung()).head(5)  # noqa: E731
NEW_VET = lambda: veteran().head(5)  # noqa: E731
ZPH = [("zph", {})]

CASES = {
    "km_sex": (lung, km(f"{S} ~ sex")),
    "km_one": (veteran, km(f"{S} ~ 1")),
    "km_two": (lung, km(f"{S} ~ sex + ecog")),
    "km_weights": (lung, km(f"{S} ~ sex", weights="w")),
    "km_counting": (lung, km("Surv(entry, time, status) ~ sex")),
    "km_factor": (veteran, km(f"{S} ~ celltype")),
    **{f"km_{t}": (lung, km(f"{S} ~ sex", conf_type=t, conf_int=.9)) for t in ("log-log", "plain", "logit", "arcsin")},
    "km_none": (lung, lambda d: {"tidy": tidy(survfit(d, f"{S} ~ sex", conf_type="none"))}),
    "km_grouped": (veteran, km(f"{S} ~ celltype", by="trt")),
    "diff_sex": (lung, sdiff(f"{S} ~ sex")),
    "diff_rho": (lung, sdiff(f"{S} ~ sex", rho=1)),
    "diff_strata": (lung, sdiff(f"{S} ~ sex + strata(ecog)")),
    "diff_cell": (veteran, sdiff(f"{S} ~ celltype")),
    "diff_two": (lung, sdiff(f"{S} ~ sex + ecog", rho=.5)),
    "diff_grouped": (veteran, sdiff(f"{S} ~ celltype", by="trt")),
    "cox_efron": (lung, cox(f"{S} ~ age + sex + ecog", NEW_LUNG, ZPH + [
        ("zph_rank", {"transform": "rank"}), ("zph_identity", {"transform": "identity", "global_": False})])),
    "cox_breslow": (lung, cox(f"{S} ~ age + sex", NEW_LUNG, ZPH, ties="breslow")),
    "cox_exact": (veteran, cox(f"{S} ~ karno + trt", ties="exact")),
    "cox_strata": (lung, cox(f"{S} ~ age + ecog + strata(sex)", NEW_LUNG, ZPH)),
    "cox_cluster": (lung, cox(f"{S} ~ age + sex + cluster(inst)")),
    "cox_robust": (lung, cox(f"{S} ~ age + sex", robust=True)),
    "cox_weights": (lung, cox(f"{S} ~ age + sex", NEW_LUNG, weights="w")),
    "cox_fweights": (lung, cox(f"{S} ~ age + sex", weights="wf")),
    "cox_offset": (lung, cox(f"{S} ~ age + sex + offset(off)")),
    "cox_counting": (lung, cox("Surv(entry, time, status) ~ age + sex", zph=ZPH)),
    "cox_factor": (veteran, cox(f"{S} ~ trt + celltype + karno", NEW_VET, ZPH + [("zph_terms", {"terms": False})])),
    "cox_curve": (lung, curves(f"{S} ~ age + sex + ecog")),
    "cox_curve_strata": (lung, curves(f"{S} ~ age + ecog + strata(sex)")),
    "cox_curve_new": (lung, curves(f"{S} ~ age + sex + ecog", lambda: complete(lung()).head(2))),
    "cox_curve_breslow": (lung, curves(f"{S} ~ age + sex", lambda: complete(lung()).head(1), ties="breslow",
                                       conf_type="log-log")),
    "cox_curve_counting": (lung, curves("Surv(entry, time, status) ~ age + sex")),
    "cox_curve_weights": (lung, curves(f"{S} ~ age + sex", weights="w")),
    **{f"reg_{d}": (lung, reg(f"{S} ~ age + sex", NEW_LUNG, dist=d))
       for d in ("weibull", "exponential", "lognormal", "loglogistic", "gaussian")},
    "reg_factor": (veteran, reg(f"{S} ~ trt + celltype + karno", NEW_VET)),
    "reg_weights": (lung, reg(f"{S} ~ age + sex", NEW_LUNG, weights="w")),
    "reg_offset": (lung, reg(f"{S} ~ age + sex + offset(off)", NEW_LUNG)),
    "reg_scale": (lung, reg(f"{S} ~ age + sex", NEW_LUNG, scale=1.5)),
    "reg_intercept": (lung, reg(f"{S} ~ 1", NEW_LUNG, dist="lognormal")),
    **{f"reg_interval_{d}": (lung, reg('Surv(lo, hi, type="interval2") ~ age + sex', NEW_LUNG, dist=d))
       for d in ("weibull", "lognormal", "loglogistic")},
    "reg_interval_scale": (lung, reg('Surv(lo, hi, type="interval2") ~ age', NEW_LUNG, dist="gaussian", scale=200)),
    "reg_interval_codes": (lung, reg('Surv(t1, t2, ev, type="interval") ~ age + sex', NEW_LUNG)),
    "reg_left": (lung, reg('Surv(time, status == 2, type="left") ~ age + sex', NEW_LUNG)),
    "km_interval": (lung, km('Surv(lo, hi, type="interval2") ~ sex')),
    "km_left": (lung, km('Surv(time, status == 2, type="left") ~ 1')),
    "km_fweights": (lung, km(f"{S} ~ sex", weights="wf")),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(CASES))
def test_matches_r(name, backend):
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    data, make = CASES[name]
    source, _ = as_input(data(), backend, [])
    BACKEND[0] = backend
    expected = _oracle("survival")[name]
    tables = make(source)
    assert set(expected) == set(tables)
    module = {"lazy": "polars"}.get(backend, backend)
    loose = {}
    if name in ("km_interval", "km_left"):
        # Turnbull's counts are sums of fractional weights: once the
        # remaining weight is rounding residue (1e-15), only its absolute size
        # means anything; compare on the scale of the number of records.
        records = complete(lung()).height if name == "km_interval" else lung().height
        loose = {c: (1e-9, 1e-9 * records) for c in ("n_risk", "n_event", "n_censor")}
        # and survival at the tail is that residue's product (3e-17 in both).
        loose.update({c: (1e-9, 1e-12) for c in ("estimate", "std_error", "conf_low", "conf_high")})
    for table, result in tables.items():
        assert type(result).__module__.split(".")[0] == module, (table, type(result))
        assert_matches_r(result, expected[table], loose=loose)


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_cox_and_survreg(backend):
    source, by = as_input(veteran(), backend, ["trt"])
    fits = coxph(source, f"{S} ~ celltype + karno", by=by)
    assert len(fits) == 2
    expected = _oracle("survival")["cox_grouped"]
    assert_matches_r(tidy(fits), expected["tidy"])
    assert_matches_r(glance(fits), expected["glance"])
    assert as_polars(cox_zph(fits)).height == 6
    source, by = as_input(lung(), backend, ["sex"])
    fits = survreg(source, f"{S} ~ age", by=by)
    expected = _oracle("survival")["reg_grouped"]
    assert_matches_r(tidy(fits), expected["tidy"])
    assert_matches_r(glance(fits), expected["glance"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_anova_and_missing_rows(backend):
    source, _ = as_input(lung(), backend, [])
    fit = coxph(source, f"{S} ~ age + sex + ecog")
    assert_matches_r(anova(fit), _oracle("survival")["cox_anova"])
    rows, _ = as_input(complete(lung()), backend, [])
    small, big = coxph(rows, f"{S} ~ age"), rows >> coxph(f"{S} ~ age + sex + ecog")
    assert_matches_r(anova(small, big), _oracle("survival")["cox_compare"])
    assert "p_value" not in as_polars(anova(small, big, test="none")).columns
    excluded = coxph(source, f"{S} ~ age + ecog", na_action="exclude")
    lp, expected = as_polars(predict(excluded)), as_polars(predict(excluded, type="expected"))
    assert lp.height == 228 and lp["fit"].null_count() == 1
    assert_matches_r(pl.DataFrame({"fit": lp["fit"], "expected": expected["fit"]}),
                     _oracle("survival")["cox_exclude"])
    assert as_polars(augment(excluded)).height == 228


def test_surv_forms_and_errors():
    data = lung()
    coded = data.with_columns(dead=pl.col("status") == 2, event=pl.col("status") - 1)
    base = coxph(data, f"{S} ~ age").coef
    for formula in ("Surv(time, dead) ~ age", "Surv(time, event) ~ age", "Surv(time, status == 2) ~ age"):
        np.testing.assert_allclose(coxph(coded, formula).coef, base, rtol=1e-12)
    assert coxph(data, f"{S} ~ age").nevent == 165
    assert tidy(survfit(data, "Surv(time) ~ 1"))["n_censor"].sum() == 0
    with pytest.raises(ValueError, match="Surv"):
        coxph(data, "time ~ age")
    with pytest.raises(ValueError, match="status"):
        coxph(data.with_columns(status=pl.col("status") + 5), f"{S} ~ age")
    with pytest.raises(ValueError, match="greater than start"):
        coxph(data, "Surv(time, entry, status) ~ age")
    with pytest.raises(ValueError, match="ties"):
        coxph(data, f"{S} ~ age", ties="average")
    with pytest.raises(ValueError, match="dist"):
        survreg(data, f"{S} ~ age", dist="t")
    with pytest.raises(ValueError, match="counting"):
        survdiff(data, "Surv(entry, time, status) ~ sex")
    with pytest.raises(ValueError, match="conf_type"):
        survfit(data, f"{S} ~ sex", conf_type="wide")
    with pytest.raises(ValueError, match="interaction"):
        coxph(data, f"{S} ~ age:strata(sex)")
    interval = 'Surv(lo, hi, type="interval2")'
    with pytest.raises(ValueError, match='Cox model doesn\'t support "interval" survival data'):
        coxph(data, f"{interval} ~ age")
    with pytest.raises(ValueError, match='Cox model doesn\'t support "left" survival data'):
        coxph(data, 'Surv(time, status == 2, type="left") ~ age')
    with pytest.raises(ValueError, match="Right censored data only"):
        survdiff(data, f"{interval} ~ sex")
    with pytest.raises(ValueError, match="type must be one of"):
        survreg(data, 'Surv(time, status, type="sideways") ~ age')
    with pytest.raises(ValueError, match="takes 3 arguments"):
        survreg(data, 'Surv(time, status, type="interval") ~ age')
    with pytest.raises(ValueError, match="start > stop"):
        survreg(data.with_columns(hi=pl.col("lo") - 1), f"{interval} ~ age")
    with pytest.raises(ValueError, match="needs both times"):
        survreg(data.with_columns(t2=pl.lit(None, pl.Float64)), 'Surv(t1, t2, ev, type="interval") ~ age')
    # Rows missing both bounds are missing rows, as in R.
    both = data.with_columns(lo=pl.when(pl.col("hi").is_null()).then(None).otherwise(pl.col("lo")))
    assert survreg(both, f"{interval} ~ age").n == both.filter(~(pl.col("lo").is_null() & pl.col("hi").is_null())).height
    with pytest.raises(ValueError, match="robust"):
        anova(coxph(data, f"{S} ~ age + cluster(inst)"))
    with pytest.raises(TypeError):
        cox_zph(survreg(data, f"{S} ~ age"))
    piped = data >> survfit(f"{S} ~ sex")
    assert as_polars(tidy(piped)).equals(as_polars(tidy(survfit(data, f"{S} ~ sex"))))
    assert as_polars(augment(survreg(data, f"{S} ~ age"))).height == 228
    assert "coxph(" in repr(coxph(data, f"{S} ~ age"))
    assert "survreg(" in repr(survreg(data, f"{S} ~ age"))
    assert "survfit(" in repr(survfit(data, f"{S} ~ sex"))
    assert "survdiff(" in repr(survdiff(data, f"{S} ~ sex"))
