"""R's statistical tests on Polars, for Polars, pandas and tidy3 data.

    from stats3 import *

    penguins >> t_test("body_mass_g ~ sex", by="species")    # Polars or pandas
    t_test(penguins, "body_mass_g ~ sex", by="species")      # data first
    penguins >> group_by(species) >> t_test(body_mass_g ~ sex)  # tidy3, notebook

Each test returns one row per group, with R's defaults and R's results, in
the kind of frame it was given.
"""

from stats3.__version__ import __version__
from stats3.bayes import (
    bayes_binom_test, bayes_compare, bayes_diagnostics, bayes_lm, bayes_poisson_test,
    hypothesis, posterior_draws, posterior_epred, posterior_predict, posterior_summary,
    pp_check, prior, prior_predict,
)
from stats3.bayes_sampling import bayes_glm
from stats3.bayes_marginal import bayes_emmeans
from stats3.checks import (
    bartlett_test,
    fligner_test,
    ks_test,
    levene_test,
    shapiro_test,
    var_test,
)
from stats3.core import aov, oneway_test, t_test, tukey_hsd
from stats3.counts import (
    binom_test,
    chisq_test,
    fisher_test,
    mantelhaen_test,
    mcnemar_test,
    poisson_test,
    prop_test,
)
from stats3.effects import (
    cohens_d,
    eta_squared,
    hedges_g,
    p_adjust,
    power_anova_test,
    power_prop_test,
    power_t_test,
)
from stats3.emm import contrast, emmeans, pairs
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
    vcov,
)
from stats3.pairwise import pairwise_prop_test, pairwise_t_test, pairwise_wilcox_test
from stats3.negbin import glm_nb, negative_binomial
from stats3.ordinal import polr
from stats3.penalized import coef, cv_glmnet, glmnet
from stats3.ranks import cor_test, friedman_test, kruskal_test, wilcox_test
from stats3.report import modelsummary
from stats3.resistant import rlm, rq
from stats3.robust import vcov_cl, vcov_hc, vcov_nw

# In IPython / Jupyter, turn on unquoted formulas and bare column names in
# stats3 calls (from the next cell on), as tidy3 does for its verbs.
try:
    from stats3.jupyter import enable as _enable_notebook

    _enable_notebook()
except Exception:  # pragma: no cover - never let the shell block an import
    pass

__all__ = [
    "bayes_compare",
    "bayes_binom_test",
    "bayes_poisson_test",
    "bayes_diagnostics",
    "bayes_glm",
    "bayes_emmeans",
    "bayes_lm",
    "hypothesis",
    "posterior_draws",
    "posterior_epred",
    "posterior_predict",
    "posterior_summary",
    "prior",
    "prior_predict",
    "pp_check",
    "anova",
    "aov",
    "augment",
    "bartlett_test",
    "binom_test",
    "binomial",
    "chisq_test",
    "coef",
    "cohens_d",
    "contrast",
    "cor_test",
    "cv_glmnet",
    "emmeans",
    "eta_squared",
    "fisher_test",
    "fligner_test",
    "friedman_test",
    "Gamma",
    "gaussian",
    "glance",
    "glm",
    "glm_nb",
    "glmnet",
    "hedges_g",
    "inverse_gaussian",
    "kruskal_test",
    "ks_test",
    "levene_test",
    "lm",
    "mantelhaen_test",
    "mcnemar_test",
    "modelsummary",
    "negative_binomial",
    "oneway_test",
    "p_adjust",
    "pairwise_prop_test",
    "pairwise_t_test",
    "pairwise_wilcox_test",
    "pairs",
    "poisson",
    "polr",
    "poisson_test",
    "power_anova_test",
    "power_prop_test",
    "power_t_test",
    "predict",
    "prop_test",
    "quasibinomial",
    "quasipoisson",
    "rlm",
    "rq",
    "shapiro_test",
    "t_test",
    "tidy",
    "tukey_hsd",
    "var_test",
    "vcov",
    "vcov_cl",
    "vcov_hc",
    "vcov_nw",
    "wilcox_test",
]
