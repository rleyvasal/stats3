"""Generalized additive models against mgcv, on every backend.

The oracle fits run mgcv with tightened stopping rules, so its smoothing
parameters are the optima of its criteria to rounding error; stats3 finds
the same optima, and results agree to 1e-7 or better unless a case says
otherwise. Thin plate bases come from an eigen-decomposition whose vectors
LAPACK may return with either sign, so tp coefficients and covariances are
only compared up to sign (here: not at all); fitted values, effective
degrees of freedom, smoothing parameters and tests do not depend on them.
"""

import subprocess
import warnings

import numpy as np
import polars as pl
import pytest

from stats3 import Gamma, anova, augment, binomial, gam, glance, inverse_gaussian, poisson, predict, quasipoisson, tidy, vcov
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r


def data():
    return pl.read_csv(DATA / "gam.csv")


def large():
    return pl.read_csv(DATA / "gam_large.csv")


FITS = {
    "gaussian": lambda d: gam(d, "y ~ s(x0) + s(x1) + s(x2) + f"),
    "gaussian_reml": lambda d: gam(d, "y ~ s(x0) + s(x3) + f", method="REML"),
    "cr": lambda d: gam(d, "y ~ s(x0, bs='cr') + s(x1, bs='cr', k=15) + f", method="REML"),
    "cs": lambda d: gam(d, "y ~ s(x0, bs='cs', k=8) + s(x2, bs='cs', k=10)", method="REML"),
    "ps": lambda d: gam(d, "y ~ s(x0, bs='ps') + s(x2, bs='ps', m=c(2, 1), k=20)"),
    "re": lambda d: gam(d, "y ~ s(x2) + s(subject, bs='re')", method="REML"),
    "re_slope": lambda d: gam(d, "y ~ s(x2) + s(subject, x0, bs='re') + s(f, bs='re')", method="REML"),
    "by_factor": lambda d: gam(d, "y ~ f + s(x2, by=f)", method="REML"),
    "by_numeric": lambda d: gam(d, "y ~ s(x2, by=x0) + s(x1)"),
    "te": lambda d: gam(d, "y ~ te(x0, x1) + s(x2)", method="REML"),
    "te_mixed": lambda d: gam(d, "y ~ te(x0, x2, bs=c('tp', 'ps'), k=c(4, 6))"),
    "te_re": lambda d: gam(d, "y ~ te(x0, subject, bs=c('cr', 're'))", method="REML"),
    "tp2": lambda d: gam(d, "y ~ s(x0, x1) + s(x2, k=20)", method="REML"),
    "tp3": lambda d: gam(d, "y ~ s(x0, x1, x2, k=40)", method="REML"),
    "fixed": lambda d: gam(d, "y ~ s(x0, fx=TRUE, k=5) + s(x2)", method="REML"),
    "m1_cr_by": lambda d: gam(d, "y ~ s(x0, m=1) + s(x2, bs='cr', by=f) + f", method="REML"),
    "weighted": lambda d: gam(d, "y ~ s(x0) + s(x2)", weights="w"),
    "poisson": lambda d: gam(d, "count ~ s(x0) + s(x2) + offset(log(exposure))", poisson()),
    "poisson_offset_arg": lambda d: gam(d, "count ~ s(x0) + s(x2)", poisson(), method="REML",
                                        offset=pl.col("exposure").log()),
    "binomial": lambda d: gam(d, "binary ~ s(x0) + s(x3)", binomial(), method="REML"),
    "probit": lambda d: gam(d, "cbind(succ, fail) ~ s(x0) + s(x2)", binomial("probit"), method="REML"),
    "gamma": lambda d: gam(d, "pos ~ s(x0) + s(x1) + f", Gamma("log")),
    "gamma_reml": lambda d: gam(d, "pos ~ s(x0) + s(x1)", Gamma("log"), method="REML"),
    "quasipoisson": lambda d: gam(d, "count ~ s(x0) + s(x2)", quasipoisson(), method="REML"),
    "inverse_gaussian": lambda d: gam(d, "pos ~ s(x0) + s(x2)", inverse_gaussian(), method="REML"),
}
DEFAULTS = {"default_gaussian": FITS["gaussian"], "default_reml": FITS["gamma_reml"]}
TABLES = {
    "tidy": lambda f: tidy(f, conf_int=True),
    "smooth": lambda f: tidy(f, parametric=False),
    "glance": glance,
    "augment": augment,
    "anova": anova,
}

# Per-case tolerances, with the reason R's numbers are less exact than 1e-7.
LOOSE = {
    # s(x3) is penalized to a straight line: its smoothing parameter is
    # effectively infinite, the REML score is flat in it and mgcv stops
    # where its gradient falls below tolerance.
    "binomial": 1e-6,
    # The 60-coefficient te(x0, subject) has no centring constraint and a
    # nearly singular Hessian; mgcv's rank handling moves results by ~1e-8.
    "te_re": 1e-6,
    # s(f, bs="re") has three levels: REML is flat to 1e-13 over a 1e-7
    # relative range of the neighbouring s(subject, x0) variance.
    "re_slope": 1e-6,
    # Davies' algorithm (psum.chisq) is accurate to 2e-5 in each p-value,
    # but deterministic: p-values match R's to ~1e-9 unless noted.
}
# mgcv's own stopping rules: magic stops at 1e-7 relative in the GCV score,
# Newton at 1e-6 relative gradient; results agree to what those reach.
DEFAULT_TOL = 1e-4


@pytest.fixture(scope="module")
def r_gam():
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    probe = subprocess.run([*R_COMMAND, "-e", "library(mgcv)"], capture_output=True)
    if probe.returncode:
        pytest.skip("gam oracle needs R's mgcv")
    return _oracle("gam")


def _floors(expected, rel):
    """Per-row values (fits, residuals) near zero agree to rel times the
    column's scale, not to rel of themselves."""
    out = {}
    for name, values in zip(expected["columns"], expected["data"]):
        numbers = [abs(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numbers:
            out[name] = (rel, rel * max(numbers))
    return out


def _check(result, expected, rel, table, p_rel=None):
    loose = {}
    if table == "augment":
        result = as_polars(result).select(pl.col(r"^\..*$"))
        loose = _floors(expected, rel)
    # p-values below 1e-12 come from Davies' integration or pt/pnorm tails,
    # which agree in absolute terms.
    loose["p_value"] = (p_rel or rel, 1e-12)
    assert_matches_r(result, expected, rel=rel, loose=loose)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(FITS))
@pytest.mark.parametrize("table", sorted(TABLES))
def test_gam_matches_r(name, table, backend, r_gam, cs_same_basis):
    _skip_unless_same_cs_basis(name, cs_same_basis)
    frame, _ = as_input(data(), backend, [])
    fit = FITS[name](frame)
    result = TABLES[table](fit)
    assert type(result) is type(frame)
    _check(result, r_gam[f"{name}__{table}"], LOOSE.get(name, 1e-7), table)


@pytest.mark.parametrize("name", sorted(DEFAULTS))
@pytest.mark.parametrize("table", sorted(TABLES))
def test_mgcv_default_controls(name, table, r_gam):
    # far-tail p-values (1e-34) move 100 times more than their statistics
    _check(TABLES[table](DEFAULTS[name](data())), r_gam[f"{name}__{table}"], DEFAULT_TOL, table, p_rel=1e-2)


@pytest.mark.parametrize("name", sorted(FITS))
def test_smoothing_parameters(name, r_gam, cs_same_basis):
    _skip_unless_same_cs_basis(name, cs_same_basis)
    fit = FITS[name](data())
    table = pl.DataFrame({"term": list(fit.smoothing_parameters), "sp": list(fit.smoothing_parameters.values())})
    want = r_gam[f"{name}__sp"]
    # an infinite smoothing parameter is only located to where the score
    # stops changing (mgcv: past 1e10; stats3 stops at e^25): compare the
    # finite ones
    keep = [i for i, v in enumerate(want["data"][1]) if v < 1e8]
    if len(keep) < len(want["data"][1]):
        assert all(table["sp"][i] > 1e8 for i in range(table.height) if i not in keep)
        want = {"columns": want["columns"], "data": [[c[i] for i in keep] for c in want["data"]]}
        table = table[keep]
    # Large smoothing parameters sit where the criterion is flat; mgcv's
    # Newton method locates them to about 1e-6 even with tight controls,
    # while the fits they give agree to 1e-7.
    assert_matches_r(table, want, rel=1e-5)


@pytest.mark.parametrize("name", ["cr", "ps", "re", "by_factor"])
def test_coefficients_and_covariances(name, r_gam):
    """cr and ps bases are mgcv's exactly; the re and by_factor fits include
    tp smooths, whose basis columns (and so coefficients and covariances)
    match mgcv's up to sign."""
    signed = name in ("cr", "ps")

    def strip(frame):
        frame = as_polars(frame)
        return frame if signed else frame.with_columns(pl.col(pl.Float64).abs())

    def strip_r(want):
        if signed:
            return want
        data = [col if i == 0 else [abs(v) for v in col] for i, col in enumerate(want["data"])]
        return {**want, "data": data}

    fit = FITS[name](data())
    coef = pl.DataFrame({"term": list(fit.coefficients), "estimate": list(fit.coefficients.values())})
    want = r_gam[f"{name}__coef"]
    assert_matches_r(strip(coef), strip_r(want), rel=1e-7, loose=_floors(want, 1e-7))
    for kind in ("Vp", "Ve", "Vc"):
        want = r_gam[f"{name}__{kind}"]
        # Vc adds the smoothing parameter uncertainty from the REML Hessian,
        # analytic in mgcv and differenced from analytic gradients here.
        rel = 1e-6 if kind == "Vc" else 1e-7
        assert_matches_r(strip(vcov(fit, None if kind == "Vp" else kind)), strip_r(want), rel=rel,
                         loose=_floors(want, rel))


CS_PENALTIES = (("x0", 8), ("x2", 10))


def _cs_penalties(r_gam):
    """stats3's and R's cs penalty matrices, for each cs smooth in FITS."""
    from stats3.gam_smooth import cr_basis

    return [(cr_basis(data()[name].to_numpy(), k, 0.1).S[0],
             np.array(r_gam[f"cs_penalty_{name}"]["data"], dtype=float).T) for name, k in CS_PENALTIES]


@pytest.fixture(scope="module")
def cs_same_basis(r_gam):
    """Whether R's LAPACK returned the same null-space eigenvectors as
    stats3's here. mgcv's cs shrinks the cr penalty's 2-d null space with 0.1
    and 0.01 times its smallest positive eigenvalue, on whichever null
    eigenvectors LAPACK returns first. Any orthonormal pair is valid and the
    choice depends on the LAPACK build (stats3 and R both call dsyevr, but
    NumPy's and R's libraries can differ), so mgcv's own cs fits differ
    across platforms; they are comparable with R only where both agree."""
    return all(np.allclose(ours, want, rtol=1e-9, atol=1e-9 * np.abs(want).max()) for ours, want in _cs_penalties(r_gam))


def _skip_unless_same_cs_basis(name, cs_same_basis):
    if name == "cs" and not cs_same_basis:
        pytest.skip("R's LAPACK chose a different cs null-space basis on this platform; "
                    "test_cs_penalties_match_r checks the basis-independent parts")


def test_cs_penalties_match_r(r_gam, cs_same_basis):
    """The cs penalty's eigenvalues, its shrunk null space and its action on
    the complement match R on every platform. Where R's LAPACK chose the same
    null-space eigenvectors, the matrices match entry by entry too."""
    for ours, want in _cs_penalties(r_gam):
        scale = np.abs(want).max()
        values, vectors = np.linalg.eigh((ours + ours.T) / 2)
        want_values, want_vectors = np.linalg.eigh((want + want.T) / 2)
        assert values == pytest.approx(want_values, rel=1e-9, abs=1e-9 * scale)
        # the two shrunk directions are the two smallest eigenvalues
        null, want_null = vectors[:, :2] @ vectors[:, :2].T, want_vectors[:, :2] @ want_vectors[:, :2].T
        assert np.abs(null - want_null).max() < 1e-8
        rest = np.eye(len(ours)) - null
        assert np.abs(rest @ ours @ rest - rest @ want @ rest).max() < 1e-9 * scale
        if cs_same_basis:
            assert np.abs(ours - want).max() < 1e-9 * scale


def _new_rows():
    rows = data()[[0, 6, 49, 119, 199]]
    return rows.with_columns(x0=pl.Series([rows["x0"][0], 1.05, *rows["x0"][2:]]))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["gaussian", "cr", "ps", "by_factor", "te", "gamma", "poisson_offset_arg"])
@pytest.mark.parametrize("scale", ["link", "response", "terms"])
def test_predict_matches_r(name, scale, backend, r_gam):
    frame, _ = as_input(data(), backend, [])
    new, _ = as_input(_new_rows(), backend, [])
    result = predict(FITS[name](frame), new, type=scale, se_fit=True)
    assert type(result) is type(frame)
    want = r_gam[f"{name}__predict_{scale}"]
    assert_matches_r(result, want, rel=1e-7, loose=_floors(want, 1e-7))


@pytest.mark.parametrize("backend", BACKENDS)
def test_comparisons_and_groups(backend, r_gam):
    frame, _ = as_input(data(), backend, [])
    result = anova(gam(frame, "y ~ s(x0) + f"), gam(frame, "y ~ s(x0) + s(x1) + f"), test="F")
    assert type(result) is type(frame)
    assert_matches_r(result, r_gam["compare_f"], rel=1e-7)
    result = anova(gam(frame, "count ~ s(x0)", poisson(), method="REML"),
                   gam(frame, "count ~ s(x0) + s(x2)", poisson(), method="REML"))
    assert_matches_r(result, r_gam["compare_chisq"], rel=1e-7)
    frame, by = as_input(data(), backend, ["g"])
    fits = gam(frame, "y ~ s(x0) + s(x2)", method="REML", by=by)
    for table, key in ((lambda f: tidy(f, parametric=False), "grouped"), (glance, "grouped_glance")):
        result = table(fits)
        assert type(result) is type(frame)
        assert_matches_r(result, r_gam[key], rel=1e-7, loose={"p_value": (1e-7, 1e-12)})


@pytest.mark.parametrize("formula,key", [("y ~ s(x)", "large__ysx"), ("y ~ s(x, z)", "large__ysxz")])
def test_knot_subsampling(formula, key, r_gam):
    """More than 2000 unique covariate values: mgcv's tp basis uses 2000
    knots sampled with R's RNG at seed 1, which stats3 reproduces."""
    assert_matches_r(glance(gam(large(), formula, method="REML")), r_gam[key], rel=1e-7)


def test_basis_dimensions_and_names():
    fit = FITS["by_factor"](data())
    assert list(fit.smoothing_parameters) == ["s(x2):fa", "s(x2):fb", "s(x2):fc"]
    assert len(fit.coefficients) == 3 + 3 * 9
    assert list(fit.coefficients)[3] == "s(x2):fa.1"
    te = FITS["te"](data())
    assert list(te.smoothing_parameters) == ["te(x0,x1)1", "te(x0,x1)2", "s(x2)"]
    assert te.edf.size == 1 + 24 + 9


def test_missing_rows_pipe_and_errors():
    d = data().with_row_index("i").with_columns(
        x0=pl.when(pl.col("i") == 3).then(None).otherwise(pl.col("x0")))
    fit = d >> gam("y ~ s(x0) + f", na_action="exclude")
    assert (fit >> tidy()).equals(tidy(fit))
    assert augment(fit).height == d.height
    assert predict(fit)["fit"].is_null().sum() == 1
    with pytest.raises(ValueError, match="missing values"):
        gam(d, "y ~ s(x0)", na_action="fail")
    with pytest.raises(ValueError, match="method must be"):
        gam(data(), "y ~ s(x0)", method="ML")
    with pytest.raises(ValueError, match="bs must be"):
        gam(data(), "y ~ s(x0, bs='ad')")
    with pytest.raises(ValueError, match="not supported"):
        gam(data(), "y ~ s(x0, id=1)")
    with pytest.raises(ValueError, match="fewer unique"):
        gam(data().head(8), "y ~ s(x0)")
    with pytest.raises(ValueError, match="vcov must be"):
        vcov(FITS["gaussian"](data()), "HC3")
    with pytest.raises(ValueError, match="new levels"):
        predict(FITS["by_factor"](data()), data().head(2).with_columns(f=pl.lit("z")))
    with pytest.raises(TypeError, match="gam fits with gam fits"):
        from stats3 import glm

        anova(FITS["gaussian"](data()), glm(data(), "y ~ x0"))


def test_repr_and_warnings():
    fit = FITS["gaussian_reml"](data())
    text = repr(fit)
    assert "R-sq.(adj)" in text and "-REML" in text
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FITS["gamma"](data())


def test_nested_smooths_are_refused():
    with pytest.raises(ValueError, match="nested"):
        gam(data(), "y ~ s(x0) + te(x0, x1)")
    with pytest.raises(ValueError, match="nested"):
        gam(data(), "y ~ s(x2) + s(x2, by=f)")
