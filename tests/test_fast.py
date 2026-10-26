"""The optional compiled path (``stats3[fast]``) and the NumPy path give the
same results; the vectorized fused multiply-add is exact; the studentized
range ports give R's values."""

from fractions import Fraction

import numpy as np
import polars as pl
import pytest

import stats3._fast as fast
from stats3 import coxph, dist, glance, hclust, kmeans, tidy
from stats3._tukey import ptukey, qtukey
from stats3.multivariate import R_FUSES_MULTIPLY_ADD, _fma, _fma_array
from oracle import DATA

IRIS = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]


def iris():
    return pl.read_csv(DATA / "iris.csv").rename(lambda c: c.replace(".", "_"))


def lung():
    return pl.read_csv(DATA / "lung.csv", null_values="NA")


@pytest.fixture
def both_paths():
    """Runs a function with the compiled kernels (when Numba is installed)
    and without; returns both results."""
    pytest.importorskip("numba")

    def run(make):
        fast._KERNELS.clear()
        fast.COMPILED = True
        try:
            with_kernels = make()
            fast.COMPILED = False
            without = make()
        finally:
            fast.COMPILED = True
        return with_kernels, without

    return run


def _same(a: pl.DataFrame, b: pl.DataFrame) -> None:
    assert a.columns == b.columns
    for name in a.columns:
        x, y = a[name], b[name]
        if x.dtype.is_numeric():
            assert np.array_equal(x.to_numpy(), y.to_numpy(), equal_nan=True), name
        else:
            assert x.to_list() == y.to_list(), name


@pytest.mark.parametrize("algorithm", ["Hartigan-Wong", "Lloyd", "MacQueen"])
def test_kmeans_paths_agree(both_paths, algorithm):
    data = iris()
    centers = data.select(IRIS).to_numpy()[[0, 50, 100]]
    with_kernels, without = both_paths(lambda: kmeans(data, IRIS, centers, algorithm=algorithm))
    assert np.array_equal(with_kernels.cluster, without.cluster)
    assert np.array_equal(with_kernels.centers, without.centers)
    assert with_kernels.iter == without.iter
    # kmns.f adds the within sums of squares column by column; the NumPy path row by row.
    np.testing.assert_allclose(with_kernels.withinss, without.withinss, rtol=1e-12)


def test_kmeans_paths_agree_on_random_data(both_paths):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3000, 8))
    x[:, 0] = np.round(x[:, 0], 1)  # ties in one column
    frame = pl.DataFrame({f"x{j}": x[:, j] for j in range(8)})
    columns = list(frame.columns)
    for algorithm in ("Hartigan-Wong", "Lloyd", "MacQueen"):
        with_kernels, without = both_paths(lambda: kmeans(frame, columns, x[:6], algorithm=algorithm, iter_max=20))
        assert np.array_equal(with_kernels.cluster, without.cluster), algorithm
        assert np.array_equal(with_kernels.centers, without.centers), algorithm


def test_dist_paths_agree(both_paths):
    data = iris().with_columns(pl.when(pl.int_range(pl.len()) % 17 == 0).then(None).otherwise(pl.col("Sepal_Width")).alias("Sepal_Width"))
    with_kernels, without = both_paths(lambda: dist(data, IRIS))
    _same(with_kernels, without)


@pytest.mark.parametrize("method", ["ward.D", "single", "complete", "average", "mcquitty", "median", "centroid", "ward.D2"])
def test_hclust_paths_agree(both_paths, method):
    d = dist(iris(), IRIS)
    with_kernels, without = both_paths(lambda: hclust(d, method))
    _same(tidy(with_kernels), tidy(without))
    _same(tidy(with_kernels, matrix="order"), tidy(without, matrix="order"))


def test_concordance_paths_agree(both_paths):
    data = lung().rename({"ph.ecog": "ecog"})
    with_kernels, without = both_paths(
        lambda: glance(coxph(data, "Surv(time, status) ~ age + sex + strata(ecog)")))
    np.testing.assert_allclose(with_kernels["concordance"].to_numpy(), without["concordance"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(with_kernels["std_error_concordance"].to_numpy(),
                               without["std_error_concordance"].to_numpy(), rtol=1e-12)


def test_vectorized_fma_is_exact():
    rng = np.random.default_rng(3)
    n = 20000
    x = np.r_[rng.normal(size=n), np.round(rng.normal(size=n), 1), rng.integers(-50, 50, n) / 10, [1e-160, 0.0, 3.0]]
    z = np.r_[np.abs(rng.normal(size=n)) * 3, np.cumsum(np.round(rng.normal(size=n), 1) ** 2)[::-1] / 7,
              rng.integers(0, 1000, n) / 100, [1.0, 1.0, 1e-300]]
    got = _fma_array(x, x, z)
    if R_FUSES_MULTIPLY_ADD:
        want = np.array([float(Fraction(a) * Fraction(a) + Fraction(c)) for a, c in zip(x.tolist(), z.tolist())])
    else:
        want = x * x + z
    assert np.array_equal(got, want)
    assert np.array_equal(np.array([_fma(a, a, c) for a, c in zip(x.tolist(), z.tolist())]), want)


# R 4.6.1's values
R_PTUKEY = [
    ((3.1, 2, 98), 0.96925494382038502),
    ((2.5, 4, 6), 0.62923111789959052),
    ((4.5, 10, 500), 0.95057350734981294),
    ((0.7, 3, 30000), 0.12628920315009459),
    ((6, 5, 2), 0.84899738064969898),
    ((3.3, 3, np.inf), 0.94868909358101361),
]
R_QTUKEY = [
    ((0.95, 2, 98), 2.8064607908484471),
    ((0.99, 4, 6), 7.0332629951006789),
    ((0.9, 10, 500), 4.1441973953885514),
    ((0.5, 3, 30000), 1.5877877614514313),
    ((0.95, 3, np.inf), 3.3144931575530041),
]


def test_ptukey_matches_r():
    q, k, df = (np.array([case[i] for case, _ in R_PTUKEY]) for i in range(3))
    want = np.array([v for _, v in R_PTUKEY])
    np.testing.assert_allclose(ptukey(q, k, df), want, rtol=1e-12)
    np.testing.assert_allclose(ptukey(q, k, df, lower_tail=False), 1 - want, rtol=1e-9)
    assert ptukey(0.0, 3, 10) == 0 and ptukey(np.inf, 3, 10) == 1 and np.isnan(ptukey(2.0, 1, 10))


def test_qtukey_matches_r():
    p, k, df = (np.array([case[i] for case, _ in R_QTUKEY]) for i in range(3))
    want = np.array([v for _, v in R_QTUKEY])
    np.testing.assert_allclose(qtukey(p, k, df), want, rtol=1e-11)
    assert qtukey(0.0, 3, 10) == 0 and qtukey(1.0, 3, 10) == np.inf and np.isnan(qtukey(0.5, 3, 1))
