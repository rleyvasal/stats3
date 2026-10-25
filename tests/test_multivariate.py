"""Exploratory multivariate methods against R's stats, across the four backends."""

import warnings

import numpy as np
import polars as pl
import pytest

from stats3 import (
    augment, cmdscale, cutree, dist, factanal, glance, hclust, kmeans, prcomp, predict, princomp,
    promax, tidy, varimax,
)
from oracle import BACKENDS, DATA, R_COMMAND, _oracle, as_input, as_polars, assert_matches_r, penguins


def usarrests():
    return pl.read_csv(DATA / "usarrests.csv")


def iris():
    return pl.read_csv(DATA / "iris.csv").rename(lambda c: c.replace(".", "_")).with_columns(
        pl.col("Species").cast(pl.Enum(["setosa", "versicolor", "virginica"])))


def mtcars():
    return pl.read_csv(DATA / "mtcars.csv")


US = ["Murder", "Assault", "UrbanPop", "Rape"]
IRIS = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
PENGUINS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
MTCARS = ["mpg", "disp", "hp", "drat", "wt", "qsec"]
R_IRIS = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]


def oracle(*path):
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable")
    out = _oracle("multivariate")
    for name in path:
        out = out[name]
    return out


def renamed(expected, mapping):
    """R's column names for iris use dots; stats3's frame uses underscores."""
    expected = dict(expected)
    expected["columns"] = [mapping.get(c, c) for c in expected["columns"]]
    return expected


def values(expected, column):
    return expected["data"][expected["columns"].index(column)]


IRIS_NAMES = dict(zip(R_IRIS, IRIS))


def iris_expected(expected):
    """Variable labels inside R's iris tables, renamed to the frame's."""
    expected = renamed(expected, IRIS_NAMES)
    expected["data"] = [[IRIS_NAMES.get(v, v) if isinstance(v, str) else v for v in col]
                        for col in expected["data"]]
    return expected


def signs(fit_tables, expected, pcs):
    """Component signs relative to R's: +1 or -1 per component, from the loadings."""
    have = as_polars(fit_tables).sort(["PC", "column"])
    want = pl.DataFrame(dict(zip(expected["columns"], expected["data"]))).sort(["PC", "column"])
    out = {}
    for pc in range(1, pcs + 1):
        a = have.filter(pl.col("PC") == pc)["value"].to_numpy()
        b = want.filter(pl.col("PC") == pc)["value"].to_numpy()
        out[pc] = float(np.sign(a @ b))
    return out


def flip(frame, sign, value="value", by="PC"):
    frame = as_polars(frame)
    return frame.with_columns(pl.col(value) * pl.col(by).replace_strict(sign, default=1.0))


PRCOMP = {
    "usarrests": (usarrests, US, {"scale_": True, "labels": "state"}),
    "iris": (iris, IRIS, {}),
    "penguins": (penguins, PENGUINS, {"scale_": True, "rank_": 2}),
    "tol": (iris, IRIS, {"tol": 0.2}),
    "uncentred": (usarrests, US, {"center": False, "scale_": True, "labels": "state"}),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(PRCOMP))
def test_prcomp_matches_r(backend, name):
    make, columns, kwargs = PRCOMP[name]
    source, _ = as_input(make(), backend, [])
    fit = prcomp(source, columns if name != "usarrests" else "~ " + " + ".join(columns), **kwargs)
    want = {t: iris_expected(oracle("prcomp", name, t)) for t in ("x", "rotation", "pcs", "augment")}
    pcs = len(set(values(want["rotation"], "PC")))
    sign = signs(tidy(fit, matrix="rotation"), want["rotation"], pcs)
    # LAPACK's dgesdd fixes the signs; R and NumPy share it here, so no component flips.
    assert set(sign.values()) == {1.0}
    assert_matches_r(tidy(fit, matrix="rotation"), want["rotation"], rel=1e-9)
    assert_matches_r(tidy(fit), want["x"], loose={"value": (1e-9, 1e-12)})
    assert_matches_r(tidy(fit, matrix="pcs"), want["pcs"])
    fitted = as_polars(augment(fit)).select(pl.selectors.starts_with(".fittedPC"))
    assert_matches_r(fitted, want["augment"], loose={c: (1e-9, 1e-12) for c in want["augment"]["columns"]})


def test_prcomp_predict_and_grouped():
    fit = prcomp(usarrests(), US, scale_=True)
    assert_matches_r(predict(fit, usarrests().head(5)), oracle("prcomp", "predict"))
    assert_matches_r(fit >> tidy(matrix="pcs"), oracle("prcomp", "usarrests", "pcs"))
    grouped = prcomp(iris(), IRIS, scale_=True, by="Species")
    assert len(grouped) == 3
    assert_matches_r(tidy(grouped, matrix="pcs"), oracle("grouped", "prcomp"))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,make,columns,kwargs", [
    ("usarrests", usarrests, US, {"cor": True, "labels": "state"}),
    ("iris", iris, IRIS, {}),
    ("penguins", penguins, PENGUINS, {"cor": True}),
])
def test_princomp_matches_r(backend, name, make, columns, kwargs):
    source, _ = as_input(make(), backend, [])
    fit = princomp(source, columns, **kwargs)
    loose = {"value": (1e-9, 1e-12)}
    for table in ("x", "rotation", "pcs"):
        assert_matches_r(tidy(fit, matrix=table), iris_expected(oracle("princomp", name, table)), loose=loose)
    want = oracle("princomp", name, "augment")
    fitted = as_polars(augment(fit)).select(pl.selectors.starts_with(".fittedComp"))
    assert_matches_r(fitted, want, loose={c: (1e-9, 1e-12) for c in want["columns"]})


# R stops L-BFGS-B at factr * epsilon relative change (the oracle tightens it
# to 1e3); stats3 runs to full precision. Uniquenesses and loadings agree to
# the optimizer's accuracy, the objective-based statistic more closely.
FA_LOOSE = {c: (1e-5, 1e-6) for c in ("uniqueness", "fl1", "fl2", "fl3", "total_variance", ".fs1", ".fs2")}
FA_LOOSE.update(statistic=(1e-6, 0), p_value=(1e-5, 0))

FACTANAL = {
    "mtcars2": (mtcars, MTCARS, 2, {}),
    "mtcars2_none": (mtcars, MTCARS, 2, {"rotation": "none", "scores": "regression"}),
    "mtcars2_promax": (mtcars, MTCARS, 2, {"rotation": "promax", "scores": "Bartlett"}),
    "mtcars3": (mtcars, MTCARS, 3, {}),
    "usarrests": (usarrests, US, 1, {}),
    "iris": (iris, IRIS, 1, {"scores": "regression"}),
    "penguins": (penguins, PENGUINS, 1, {}),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(FACTANAL))
def test_factanal_matches_r(backend, name):
    make, columns, factors, kwargs = FACTANAL[name]
    source, _ = as_input(make(), backend, [])
    fit = factanal(source, columns, factors, **kwargs)
    assert_matches_r(tidy(fit), iris_expected(oracle("factanal", name, "tidy")), loose=FA_LOOSE)
    assert_matches_r(glance(fit), oracle("factanal", name, "glance"), loose=FA_LOOSE)
    if "scores" in kwargs:
        want = oracle("factanal", name, "scores")
        assert_matches_r(as_polars(augment(fit)).select(pl.selectors.starts_with(".fs")), want, loose=FA_LOOSE)


def test_factanal_grouped_and_rotations():
    grouped = factanal(iris(), IRIS, 1, by="Species")
    assert_matches_r(tidy(grouped), iris_expected(oracle("grouped", "factanal")), loose=FA_LOOSE)
    fit = factanal(mtcars(), MTCARS, 2, rotation="none")
    rotated = varimax(fit.loadings)
    np.testing.assert_allclose(rotated.loadings, fit.loadings @ rotated.rotmat)
    oblique = promax(fit.loadings)
    np.testing.assert_allclose(oblique.loadings, fit.loadings @ oblique.rotmat, atol=1e-12)
    custom = factanal(mtcars(), MTCARS, 2, rotation=varimax)
    np.testing.assert_allclose(custom.loadings, factanal(mtcars(), MTCARS, 2).loadings)
    with pytest.raises(ValueError, match="too many"):
        factanal(usarrests(), US, 3)
    with pytest.raises(ValueError, match="at least three"):
        factanal(usarrests(), US[:2], 1)
    with pytest.raises(ValueError, match="scores"):
        augment(fit)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski"])
def test_dist_matches_r(backend, method):
    source, _ = as_input(usarrests(), backend, [])
    assert_matches_r(dist(source, US, method=method, p=3, labels="state"), oracle("dist", method))


def test_dist_missing_values():
    small = iris().head(6).with_row_index().with_columns(
        pl.when(pl.col("index") == 1).then(None).otherwise(pl.col("Petal_Length")).alias("Petal_Length"),
        pl.when(pl.col("index") == 3).then(None).otherwise(pl.col("Sepal_Length")).alias("Sepal_Length"),
        pl.when(pl.col("index") == 4).then(0.0).otherwise(pl.col("Sepal_Width")).alias("Sepal_Width"))
    assert_matches_r(dist(small, IRIS), oracle("dist", "missing"))
    assert_matches_r(dist(small, IRIS, method="canberra"), oracle("dist", "missing_canberra"))


HCLUST = ["complete", "single", "average", "mcquitty", "ward.D", "ward.D2", "centroid", "median"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", HCLUST)
def test_hclust_matches_r(backend, method):
    source, _ = as_input(usarrests(), backend, [])
    d = dist(source, US, labels="state")
    if method in ("centroid", "median"):
        d = as_input(as_polars(d).with_columns(pl.col("distance") ** 2), backend, [])[0]
    fit = d >> hclust(method)
    want = oracle("hclust", method)
    assert_matches_r(tidy(fit), want["merge"])
    assert_matches_r(tidy(fit, matrix="order"), want["order"])
    assert_matches_r(cutree(fit, k=4), want["cutree"])


@pytest.mark.parametrize("method", ["complete", "single", "average", "ward.D2"])
def test_hclust_ties_match_r(method):
    fit = hclust(dist(iris(), IRIS), method)
    want = oracle("hclust", f"iris_{method}")
    assert_matches_r(tidy(fit), want["merge"])
    assert_matches_r(tidy(fit, matrix="order"), want["order"])
    assert_matches_r(cutree(fit, k=3), want["cutree"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_hclust_grouped_and_cut_height(backend):
    fit = hclust(dist(usarrests(), US, labels="state"), "average")
    assert_matches_r(cutree(fit, h=50), oracle("hclust", "cut_h"))
    source, by = as_input(iris(), backend, ["Species"])
    grouped = dist(source, IRIS, by=by) >> hclust("average")
    assert len(grouped) == 3
    assert_matches_r(tidy(grouped), oracle("grouped", "hclust"))
    assert_matches_r(cutree(grouped, k=3), oracle("grouped", "cutree"))
    with pytest.raises(ValueError, match="either"):
        cutree(fit)
    with pytest.raises(ValueError, match="not sorted"):
        cutree(hclust(dist(usarrests(), US), "centroid"), h=10)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,make,columns,k,kwargs", [
    ("usarrests", usarrests, US, 2, {"labels": "state"}),
    ("iris", iris, IRIS, 3, {}),
    ("manhattan", usarrests, US, 3, {"labels": "state", "method": "manhattan"}),
])
def test_cmdscale_matches_r(backend, name, make, columns, k, kwargs):
    source, _ = as_input(make(), backend, [])
    fit = dist(source, columns, **kwargs) >> cmdscale(k)
    want = oracle("cmdscale", name)
    # dsyevr fixes the signs; R and SciPy share it here, so no axis flips.
    loose = {f"dim{j}": (1e-9, 1e-10) for j in range(1, k + 1)}
    assert_matches_r(tidy(fit), want["points"], loose=loose)
    # Eigenvalues past the k-th are zero up to rounding of the largest.
    assert_matches_r(tidy(fit, matrix="eig"), want["eig"], loose={"eigenvalue": (1e-9, 1e-9)})
    assert_matches_r(glance(fit), want["glance"])


KMEANS = {"iris": (iris, IRIS, [0, 50, 100], {}), "usarrests": (usarrests, US, [0, 1, 2, 3], {}),
          "penguins": (penguins, PENGUINS, [0, 49, 199], {"iter_max": 30})}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("algorithm", ["Hartigan-Wong", "Lloyd", "MacQueen"])
@pytest.mark.parametrize("name", list(KMEANS))
def test_kmeans_matches_r(backend, algorithm, name):
    make, columns, rows, kwargs = KMEANS[name]
    frame = make()
    complete = frame.drop_nulls(columns)
    centers = complete.select(columns).to_numpy()[rows]
    source, _ = as_input(frame, backend, [])
    fit = kmeans(source, columns, centers, algorithm=algorithm, **kwargs)
    want = renamed(oracle("kmeans", f"{name}_{algorithm}")["tidy"], IRIS_NAMES)
    assert_matches_r(tidy(fit), want)
    assert_matches_r(glance(fit), oracle("kmeans", f"{name}_{algorithm}", "glance"))
    assert_matches_r(as_polars(augment(fit)).select(".cluster"), oracle("kmeans", f"{name}_{algorithm}", "augment"))


def test_kmeans_two_clusters_random_starts_and_groups():
    centers = usarrests().select(US).to_numpy()[[1, 8]]
    assert_matches_r(tidy(kmeans(usarrests(), US, centers)), oracle("kmeans", "k2", "tidy"))
    # Random starts draw from NumPy, not R's generator: only the objective compares.
    fit = kmeans(iris(), IRIS, 3, nstart=25, seed=1)
    assert fit.tot_withinss == pytest.approx(values(oracle("kmeans", "random"), "tot_withinss")[0], rel=1e-9)
    starts = iris().group_by("Species", maintain_order=True).head(2)
    grouped = kmeans(iris(), IRIS, starts, by="Species")
    assert_matches_r(tidy(grouped), renamed(oracle("grouped", "kmeans"), IRIS_NAMES))
    with pytest.raises(ValueError, match="not distinct"):
        kmeans(iris(), IRIS, iris().select(IRIS).to_numpy()[[0, 0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kmeans(penguins(), PENGUINS, penguins().drop_nulls(PENGUINS).select(PENGUINS).to_numpy()[[0, 49, 199]],
               iter_max=1)
    assert any("did not converge" in str(w.message) for w in caught)


@pytest.mark.parametrize("backend", BACKENDS)
def test_output_types(backend):
    source, by = as_input(iris(), backend, ["Species"])
    expected = {"polars": ("polars", "DataFrame"), "lazy": ("polars", "LazyFrame"),
                "pandas": ("pandas", "DataFrame"), "tidy3": ("tidy3", "TidyFrame")}[backend]
    fits = [prcomp(source, IRIS, by=by), princomp(source, IRIS, by=by), factanal(source, IRIS, 1, by=by),
            kmeans(source, IRIS, 2, seed=3, by=by)]
    outputs = [tidy(f) for f in fits] + [glance(fits[3]), augment(fits[3]), dist(source, IRIS, by=by)]
    for output in outputs:
        assert (type(output).__module__.split(".")[0], type(output).__name__) == expected
    assert as_polars(augment(fits[3])).height == 150


def test_invalid_arguments():
    with pytest.raises(ValueError, match="response not allowed"):
        prcomp(iris(), "Sepal_Length ~ Petal_Length")
    with pytest.raises(ValueError, match="numerical"):
        prcomp(iris(), "~ .")
    with pytest.raises(ValueError, match="constant"):
        prcomp(iris().with_columns(c=pl.lit(1.0)), [*IRIS, "c"], scale_=True)
    with pytest.raises(ValueError, match="matrix"):
        tidy(prcomp(iris(), IRIS), matrix="nope")
    with pytest.raises(ValueError, match="invalid distance"):
        dist(iris(), IRIS, method="nope")
    with pytest.raises(ValueError, match="invalid clustering"):
        hclust(method="nope")
    with pytest.raises(ValueError, match="more units"):
        princomp(iris().head(3), IRIS)
    with pytest.raises(TypeError, match="glance"):
        glance(prcomp(iris(), IRIS))
    with pytest.raises(ValueError, match="distance table"):
        hclust(iris())
