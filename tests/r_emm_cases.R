# emmeans itself supplies every number; only table column names are adapted.
emm_cases <- function() {
  suppressPackageStartupMessages(library(emmeans))
  p <- read.csv("tests/data/penguins.csv", na.strings="NA")
  m <- read.csv("tests/data/mtcars.csv")
  row <- function(emm, keys, comparisons=FALSE) {
    s <- as.data.frame(summary(emm, infer=c(TRUE, comparisons)))
    estimate <- intersect(c("emmean", "estimate", "prob", "rate", "response", "ratio", "odds.ratio"), names(s))[1]
    lower <- intersect(c("lower.CL", "asymp.LCL"), names(s))[1]
    upper <- intersect(c("upper.CL", "asymp.UCL"), names(s))[1]
    out <- s[keys]
    out$estimate <- s[[estimate]]
    out$std_error <- s$SE
    out$df <- s$df
    if (comparisons) {
      statistic <- intersect(c("t.ratio", "z.ratio"), names(s))[1]
      out$statistic <- s[[statistic]]
      out$p_value <- s$p.value
    }
    out$conf_low <- s[[lower]]
    out$conf_high <- s[[upper]]
    out
  }
  base <- lm(body_mass_g ~ flipper_length_mm + species, p)
  e <- emmeans(base, "species")
  two <- lm(body_mass_g ~ species * sex, p)
  logit <- glm(am ~ wt + factor(cyl), binomial, m)
  pois <- glm(carb ~ hp + factor(cyl) + offset(log(gear)), poisson, m)
  out <- list(lm_means=row(e, "species"), lm_pairs=row(pairs(e), "contrast", TRUE))
  for (adjust in c("none", "tukey", "sidak", "scheffe", "bonferroni", "holm", "hochberg", "hommel", "BH", "BY", "fdr")) {
    out[[paste0("pairs_", adjust)]] <- row(pairs(e, adjust=adjust), "contrast", TRUE)
  }
  out$reverse <- row(pairs(e, reverse=TRUE), "contrast", TRUE)
  out$control <- row(contrast(e, "trt.vs.ctrl", ref=2), "contrast", TRUE)
  out$effects <- row(contrast(e, "eff"), "contrast", TRUE)
  out$custom <- row(contrast(e, list(first_vs_rest=c(1, -.5, -.5))), "contrast", TRUE)
  out$level90 <- row(emmeans(base, "species", level=.9), "species")
  out$pairs_level90 <- row(pairs(e, level=.9), "contrast", TRUE)
  out$at <- row(emmeans(lm(body_mass_g ~ flipper_length_mm * species, p), "species",
                        at=list(flipper_length_mm=c(190, 210))), "species")
  out$at_by <- row(emmeans(base, "species", by="flipper_length_mm",
                           at=list(flipper_length_mm=c(190, 210))), c("flipper_length_mm", "species"))
  out$subset <- row(emmeans(base, "species", at=list(species=c("Gentoo", "Adelie"))), "species")
  for (weights in c("equal", "proportional", "cells")) {
    out[[paste0("weights_", weights)]] <- row(emmeans(two, "species", weights=weights), "species")
    offset_fit <- glm(carb ~ hp + factor(cyl) + offset(am), poisson, m)
    offset_means <- emmeans(offset_fit, "cyl", weights=weights)
    # emmeans 2.0.4 uses equal offset weights with weights="cells".
    # stats3 deliberately uses cell weights for both the design and offset;
    # supply those offsets explicitly to R to validate the planned behavior.
    if (weights == "cells") {
      offset_means <- emmeans(offset_fit, "cyl", weights=weights,
                              offset=as.numeric(tapply(m$am, m$cyl, mean)))
    }
    out[[paste0("offset_weights_", weights)]] <- row(offset_means, "cyl")
    out[[paste0("offset_weights_pairs_", weights)]] <- row(pairs(offset_means), "contrast", TRUE)
  }
  out$two_factors <- row(emmeans(two, c("species", "sex")), c("species", "sex"))
  out$by <- row(emmeans(two, "species", by="sex"), c("sex", "species"))
  out$by_pairs <- row(pairs(emmeans(two, "species", by="sex")), c("sex", "contrast"), TRUE)
  out$grouped <- p |> group_by(species) |>
    group_modify(function(d, k) row(emmeans(lm(body_mass_g ~ flipper_length_mm + sex, d), "sex"), "sex")) |>
    ungroup() |> as.data.frame()
  out$weighted <- row(emmeans(lm(body_mass_g ~ species + flipper_length_mm, p, weights=bill_depth_mm), "species"), "species")
  out$robust <- row(emmeans(base, "species", vcov.=sandwich::vcovHC(base, type="HC3")), "species")
  out$aliased <- row(emmeans(lm(body_mass_g ~ species * island, p), c("species", "island")), c("species", "island"))
  out$aliased_pairs <- row(pairs(emmeans(lm(body_mass_g ~ species * island, p), "species", by="island")), c("island", "contrast"), TRUE)
  out$binary <- row(emmeans(lm(mpg ~ wt * am, m), "am"), "am")
  out$binary_average <- row(emmeans(lm(mpg ~ wt * am, m), "wt"), "wt")
  out$no_intercept <- row(emmeans(lm(body_mass_g ~ 0 + species, p), "species"), "species")
  out$robust_pairs <- row(pairs(emmeans(base, "species", vcov.=sandwich::vcovHC(base, type="HC3"))), "contrast", TRUE)
  out$custom_scheffe <- row(contrast(e, list(a=c(1, -1, 0), b=c(2, -2, 0)), adjust="scheffe"), "contrast", TRUE)
  out$offset_arg <- row(pairs(emmeans(glm(carb ~ hp + factor(cyl) + gear, poisson, m, offset=gear), "gear", at=list(gear=c(3, 5)))), "contrast", TRUE)
  out$logit_average <- row(contrast(emmeans(logit, "cyl", type="response"), list(average=c(.5, .5, 0))), "contrast", TRUE)
  for (type in c("link", "response")) {
    out[[paste0("logit_", type)]] <- row(emmeans(logit, "cyl", type=type), "cyl")
    out[[paste0("logit_pairs_", type)]] <- row(pairs(emmeans(logit, "cyl", type=type)), "contrast", TRUE)
    out[[paste0("poisson_", type)]] <- row(emmeans(pois, "cyl", type=type), "cyl")
    out[[paste0("poisson_pairs_", type)]] <- row(pairs(emmeans(pois, "cyl", type=type)), "contrast", TRUE)
  }
  out$offset_pairs <- row(pairs(emmeans(pois, "gear", at=list(gear=c(3, 5)))), "contrast", TRUE)
  out$gamma <- row(emmeans(glm(mpg ~ wt + factor(cyl), Gamma(link="log"), m), "cyl", type="response"), "cyl")
  out$gaussian <- row(emmeans(glm(mpg ~ wt + factor(cyl), gaussian(link="log"), m), "cyl", type="response"), "cyl")
  out
}
