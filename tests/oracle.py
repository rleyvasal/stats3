"""The R oracle: runs tests/r_oracle_cases.R and compares stats3 with R.

The cases need R with dplyr, jsonlite, sandwich and lmtest; without them the comparisons skip.
Marginal means comparisons additionally need emmeans.
Negative binomial comparisons need MASS and emmeans.
Set STATS3_R_ORACLE_MANIFEST to a pixi manifest to run R through pixi.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import polars as pl
import pytest


ROOT = Path(__file__).resolve().parents[1]
R_CASES = ROOT / "tests" / "r_oracle_cases.R"
DATA = ROOT / "tests" / "data"
PENGUINS = DATA / "penguins.csv"
RANKS = DATA / "ranks.csv"


def penguins() -> pl.DataFrame:
    """R's penguins, with two logical columns for the 2 x 2 tables."""
    return pl.read_csv(PENGUINS, null_values="NA").with_columns(
        dream=pl.col("island") == "Dream", first_year=pl.col("year") == 2007
    )


def ranks() -> pl.DataFrame:
    """Small tie-free data for the exact rank tests."""
    return pl.read_csv(RANKS)


# Every R case runs on each kind of input stats3 reads.
BACKENDS = ("polars", "lazy", "pandas", "tidy3")


def case(data, make, *by):
    """An R oracle case: Polars data, a function making the stats3 step,
    and the grouping columns."""
    return (data, make, list(by))


def as_input(frame: pl.DataFrame, backend: str, by: list[str]):
    """``frame`` as ``backend``'s frame, and the ``by=`` to pass with it."""
    if backend == "polars":
        return frame, by or None
    if backend == "lazy":
        return frame.lazy(), by or None
    if backend == "pandas":
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        return frame.to_pandas(), by or None
    if backend == "tidy3":
        tidy3 = pytest.importorskip("tidy3")
        data = tidy3.tidy(frame)
        return (data >> tidy3.group_by(*by) if by else data), None
    raise ValueError(backend)


def run_case(spec, backend: str):
    """Run one case on one backend; check the result is that backend's type."""
    data, make, by = spec
    frame, by_arg = as_input(data(), backend, by)
    step = make()
    step.by = by_arg
    result = frame >> step
    expected = {"polars": "DataFrame", "lazy": "LazyFrame", "pandas": "DataFrame", "tidy3": "TidyFrame"}
    assert type(result).__name__ == expected[backend], type(result)
    assert type(result).__module__.split(".")[0] == {"lazy": "polars"}.get(backend, backend)
    return result


def _r_command() -> list[str] | None:
    manifest = os.environ.get("STATS3_R_ORACLE_MANIFEST")
    pixi = shutil.which("pixi")
    if manifest and pixi:
        return [pixi, "run", "--manifest-path", manifest, "Rscript"]
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None
    probe = subprocess.run(
        [rscript, "-e", "library(dplyr); library(jsonlite); library(sandwich); library(lmtest)"], capture_output=True
    )
    return [rscript] if probe.returncode == 0 else None


R_COMMAND = _r_command()


@lru_cache(maxsize=None)
def _oracle(case: str) -> dict:
    """R's results for one oracle case, as parsed JSON."""
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable; install R with dplyr, jsonlite, sandwich and lmtest")
    completed = subprocess.run(
        [*R_COMMAND, str(R_CASES), case],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def as_polars(result) -> pl.DataFrame:
    """Any result (Polars, Polars lazy, pandas, tidy3) as a Polars DataFrame."""
    if isinstance(result, pl.DataFrame):
        return result
    if isinstance(result, pl.LazyFrame):
        return result.collect()
    if type(result).__module__.startswith("pandas"):
        return pl.from_pandas(result)
    return result.collect()


def _columns(result) -> dict[str, list]:
    frame = as_polars(result)
    # jsonlite writes infinite interval bounds as null.
    return {
        name: [None if isinstance(v, float) and math.isinf(v) else v for v in frame[name]]
        for name in frame.columns
    }


def _numeric(values: list) -> bool:
    present = [v for v in values if v is not None]
    return bool(present) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in present
    )


def assert_matches_r(result, expected: dict, *, rel: float = 1e-9, loose: dict | None = None) -> None:
    """Columns and values equal R's; numbers to ``rel`` (``loose`` per column).

    p-values also pass within 1e-15 absolute: R 4.6 computes some two-sided
    normal p-values as 1 - p, which rounds p-values below 1e-16 to 0.
    """
    loose = loose or {}
    names = expected["columns"]
    names = [names] if isinstance(names, str) else names
    data = expected["data"]
    got = _columns(result)
    assert list(got) == names
    for name, want in zip(names, data):
        have = got[name]
        if _numeric(want) or _numeric(have):
            assert [v is None for v in have] == [v is None for v in want], name
            pairs = [(h, w) for h, w in zip(have, want) if w is not None]
            tol = loose.get(name, rel)
            absolute = 1e-15 if "p_value" in name else 0.0
            if isinstance(tol, tuple):
                tol, absolute = tol
            assert [h for h, _ in pairs] == pytest.approx(
                [w for _, w in pairs], rel=tol, abs=absolute
            ), name
        else:
            assert have == want, name


def assert_same_frame(have: pl.DataFrame, want: pl.DataFrame, rel: float = 1e-12) -> None:
    """Equal frames up to floating-point rounding: Polars may sum a group
    and the same rows on their own in a different order."""
    assert have.columns == want.columns
    for name in have.columns:
        a, b = have[name].to_list(), want[name].to_list()
        if have[name].dtype.is_float():
            assert [v is None for v in a] == [v is None for v in b], name
            assert [v for v in a if v is not None] == pytest.approx(
                [v for v in b if v is not None], rel=rel, abs=0
            ), name
        else:
            assert a == b, name
