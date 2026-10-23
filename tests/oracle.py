"""The R oracle: runs tests/r_oracle_cases.R and compares stats3 with R.

The cases need R with dplyr and jsonlite; without them the comparisons skip.
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

from tidy3 import col, mutate, tidy

ROOT = Path(__file__).resolve().parents[1]
R_CASES = ROOT / "tests" / "r_oracle_cases.R"
DATA = ROOT / "tests" / "data"
PENGUINS = DATA / "penguins.csv"
RANKS = DATA / "ranks.csv"


def penguins():
    return tidy(pl.read_csv(PENGUINS, null_values="NA"))


def penguins_flags():
    return penguins() >> mutate(
        dream=col("island") == "Dream", first_year=col("year") == 2007
    )


def ranks():
    return tidy(pl.read_csv(RANKS))


def _r_command() -> list[str] | None:
    manifest = os.environ.get("STATS3_R_ORACLE_MANIFEST")
    pixi = shutil.which("pixi")
    if manifest and pixi:
        return [pixi, "run", "--manifest-path", manifest, "Rscript"]
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None
    probe = subprocess.run(
        [rscript, "-e", "library(dplyr); library(jsonlite)"], capture_output=True
    )
    return [rscript] if probe.returncode == 0 else None


R_COMMAND = _r_command()


@lru_cache(maxsize=None)
def _oracle(case: str) -> dict:
    """R's results for one oracle case, as parsed JSON."""
    if R_COMMAND is None:
        pytest.skip("R oracle unavailable; install R with dplyr and jsonlite")
    completed = subprocess.run(
        [*R_COMMAND, str(R_CASES), case],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def _columns(result) -> dict[str, list]:
    frame = result.collect()
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
