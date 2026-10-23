"""Notebook shorthand: unquoted formulas and bare column names in stats3 calls.

Importing stats3 in IPython or Jupyter turns it on; ``%load_ext
stats3.jupyter`` does too, and ``disable()`` turns it off. From the next cell
on::

    penguins >> t_test(body_mass_g ~ sex, by=species)
    penguins >> binom_test(sex == "male")

(IPython reads a whole cell before running any of it, so the cell that
imports stats3 cannot use the shorthand yet.) Plain ``.py`` scripts are not
rewritten: there, write ``"body_mass_g ~ sex"`` and ``pl.col("sex")``.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from stats3.masking import (
    COL_NAME,
    Stats3MaskTransformer,
    is_stats3_transform,
    stats3_formula_transform,
)


def _shell(ipython: Any | None) -> Any | None:
    if ipython is not None:
        return ipython
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def enable(ipython: Any | None = None) -> bool:
    """Install the formula and bare-name rewrites; True when in IPython."""
    shell = _shell(ipython)
    if shell is None:
        return False
    namespace = getattr(shell, "user_ns", None)
    if isinstance(namespace, dict):
        namespace[COL_NAME] = pl.col
    # Source pass: before parsing, and before tidy3's pipe rewriting when
    # tidy3 was loaded first. Quoting is idempotent, so both lists are safe.
    for attr in ("input_transformers_cleanup", "input_transformers_post"):
        transformers = getattr(shell, attr, None)
        if isinstance(transformers, list):
            transformers[:] = [t for t in transformers if not is_stats3_transform(t)]
            transformers.insert(0, stats3_formula_transform)
    ast_transformers = getattr(shell, "ast_transformers", None)
    if isinstance(ast_transformers, list):
        ast_transformers[:] = [t for t in ast_transformers if not is_stats3_transform(t)]
        ast_transformers.append(Stats3MaskTransformer(namespace if isinstance(namespace, dict) else {}))
    return True


def disable(ipython: Any | None = None) -> None:
    """Remove the rewrites."""
    shell = _shell(ipython)
    if shell is None:
        return
    for attr in ("input_transformers_cleanup", "input_transformers_post", "ast_transformers"):
        transformers = getattr(shell, attr, None)
        if isinstance(transformers, list):
            transformers[:] = [t for t in transformers if not is_stats3_transform(t)]


def load_ipython_extension(ipython: Any) -> None:
    enable(ipython)


def unload_ipython_extension(ipython: Any) -> None:
    disable(ipython)
