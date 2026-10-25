# Loose distributional reference: brms posteriors with the priors stats3 is given.
# Writes posterior means and SDs, by brms's parameter names, to
# tests/data/brms_groups.json, which tests/test_bayes_groups.py reads; brms
# compiles and samples for about two minutes, so the output is kept in the repo.
# Regenerate with: Rscript tests/r_brms_cases.R  (brms 2.23, rstan backend)
suppressPackageStartupMessages({library(brms); library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
if (!length(args)) args <- "tests/data/brms_groups.json"
d <- read.csv("tests/data/bayes_groups.csv")
d$choice <- factor(d$choice, levels=c("a", "b", "c"))
summaries <- function(fit, pattern) {
  draws <- as_draws_df(fit)
  draws <- as.data.frame(draws)[, grepl(pattern, names(draws)), drop=FALSE]
  list(term=names(draws), mean=unname(colMeans(draws)), sd=unname(apply(draws, 2, sd)))
}
options(mc.cores=4)
gaussian_fit <- brm(y ~ x + (1 + x | g), data=d, family=gaussian(),
  prior=c(prior(normal(0, 2.5), class=b), prior(normal(0, 5), class=Intercept),
          prior(exponential(1), class=sigma), prior(normal(0, 1), class=sd),
          prior(lkj(2), class=cor)),
  chains=4, iter=3000, warmup=1000, seed=11, refresh=0, silent=2, backend="rstan")
categorical_fit <- brm(choice ~ x + (1 | g), data=d, family=categorical(),
  prior=c(prior(normal(0, 2.5), class=b, dpar=mub), prior(normal(0, 2.5), class=b, dpar=muc),
          prior(normal(0, 5), class=Intercept, dpar=mub), prior(normal(0, 5), class=Intercept, dpar=muc),
          prior(normal(0, 1), class=sd, dpar=mub), prior(normal(0, 1), class=sd, dpar=muc)),
  chains=4, iter=3000, warmup=1000, seed=12, refresh=0, silent=2, backend="rstan")
out <- list(gaussian=summaries(gaussian_fit, "^(b_|sd_|cor_|sigma$|r_g\\[)"),
            categorical=summaries(categorical_fit, "^(b_|sd_|r_g__)"))
writeLines(toJSON(out, digits=NA), args[[1]])
