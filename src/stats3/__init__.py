"""R's statistical tests on Polars, as pipe steps for tidy3.

    from tidy3 import *
    from stats3 import *

    penguins >> group_by(species) >> t_test("body_mass_g ~ sex")

Each test returns one row per group, with R's defaults and R's results.
"""

from stats3.__version__ import __version__
from stats3.checks import bartlett_test, ks_test, levene_test, shapiro_test, var_test
from stats3.core import aov, t_test, tukey_hsd
from stats3.counts import binom_test, chisq_test, fisher_test, prop_test
from stats3.effects import (
    cohens_d,
    eta_squared,
    hedges_g,
    p_adjust,
    power_prop_test,
    power_t_test,
)
from stats3.ranks import cor_test, kruskal_test, wilcox_test

__all__ = [
    "aov",
    "bartlett_test",
    "binom_test",
    "chisq_test",
    "cohens_d",
    "cor_test",
    "eta_squared",
    "fisher_test",
    "hedges_g",
    "kruskal_test",
    "ks_test",
    "levene_test",
    "p_adjust",
    "power_prop_test",
    "power_t_test",
    "prop_test",
    "shapiro_test",
    "t_test",
    "tukey_hsd",
    "var_test",
    "wilcox_test",
]
