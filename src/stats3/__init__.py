"""R's statistical tests on Polars, for Polars, pandas and tidy3 data.

    from stats3 import *

    penguins >> t_test("body_mass_g ~ sex", by="species")    # Polars or pandas
    t_test(penguins, "body_mass_g ~ sex", by="species")      # data first
    penguins >> group_by(species) >> t_test(body_mass_g ~ sex)  # tidy3, notebook

Each test returns one row per group, with R's defaults and R's results, in
the kind of frame it was given.
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
from stats3.models import (
    Gamma,
    anova,
    augment,
    binomial,
    gaussian,
    glance,
    glm,
    inverse_gaussian,
    lm,
    poisson,
    predict,
    quasibinomial,
    quasipoisson,
    tidy,
)
from stats3.ranks import cor_test, kruskal_test, wilcox_test

# In IPython / Jupyter, turn on unquoted formulas and bare column names in
# stats3 calls (from the next cell on), as tidy3 does for its verbs.
try:
    from stats3.jupyter import enable as _enable_notebook

    _enable_notebook()
except Exception:  # pragma: no cover - never let the shell block an import
    pass

__all__ = [
    "anova",
    "aov",
    "augment",
    "bartlett_test",
    "binom_test",
    "binomial",
    "chisq_test",
    "cohens_d",
    "cor_test",
    "eta_squared",
    "fisher_test",
    "Gamma",
    "gaussian",
    "glance",
    "glm",
    "hedges_g",
    "inverse_gaussian",
    "kruskal_test",
    "ks_test",
    "levene_test",
    "lm",
    "p_adjust",
    "poisson",
    "power_prop_test",
    "power_t_test",
    "predict",
    "prop_test",
    "quasibinomial",
    "quasipoisson",
    "shapiro_test",
    "t_test",
    "tidy",
    "tukey_hsd",
    "var_test",
    "wilcox_test",
]
