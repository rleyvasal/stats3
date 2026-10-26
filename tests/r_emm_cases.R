# emmeans itself supplies every number; only table column names are adapted.
emm_cases <- function() {
  suppressPackageStartupMessages(library(emmeans))
  p <- read.csv("tests/data/penguins.csv", na.strings="NA")
  m <- read.csv("tests/data/mtcars.csv")
  row <- function(emm, keys, comparisons=FALSE) {
    s <- as.data.frame(summary(emm, infer=c(TRUE, comparisons)))
    estimate <- intersect(c("emmean", "estimate", "prob", "mean.class", "rate", "response", "ratio", "odds.ratio"), names(s))[1]
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
  c(out, polr_emm_cases(row), multinom_emm_cases(row))
}

multinom_emm_cases <- function(row) {
  suppressPackageStartupMessages(library(nnet))
  d <- read.csv("tests/data/penguins.csv", na.strings="NA")
  d <- d[complete.cases(d[, c("species", "body_mass_g", "sex")]), ]
  d$species <- factor(d$species); d$sex <- factor(d$sex); d$yf <- factor(d$year)
  d$mass <- d$body_mass_g / 1000
  d$w <- 1 + seq_len(nrow(d)) %% 3
  fit <- function(weights=NULL) {
    # do.call embeds the data in the call, so emmeans can recover it.
    do.call(multinom, list(species ~ mass + sex + yf, data=d, weights=weights, trace=FALSE,
                           maxit=10000, reltol=1e-15))
  }
  f <- fit()
  out <- list()
  prob <- emmeans(f, ~ species | sex)
  out$multinom_prob <- row(prob, c("sex", "species"))
  out$multinom_prob_pairs <- row(pairs(prob), c("sex", "contrast"), TRUE)
  out$multinom_latent <- row(emmeans(f, ~ species | sex, mode="latent"), c("sex", "species"))
  out$multinom_species <- row(emmeans(f, "species"), "species")
  out$multinom_proportional <- row(emmeans(f, ~ species | sex, weights="proportional"), c("sex", "species"))
  weighted <- fit(d$w)
  out$multinom_weighted <- row(emmeans(weighted, ~ species | sex), c("sex", "species"))
  # Where nnet stopped (its deviance), to judge the accuracy the means can have.
  out$multinom_deviance <- data.frame(plain=f$deviance, weighted=weighted$deviance)
  out
}

polr_emm_cases <- function(row) {
  suppressPackageStartupMessages(library(MASS))
  h <- housing
  h$off <- seq(-.3, .3, length.out=nrow(h))
  h$Rev <- factor(h$Sat, levels=rev(levels(h$Sat)))
  fit <- function(formula, method="logistic", data=h, ...) {
    # do.call embeds the data in the call, so emmeans can recover it.
    do.call(polr, list(formula, data=data, weights=data$Freq, Hess=TRUE, method=method,
                       control=list(reltol=1e-13, maxit=1000), ...))
  }
  f <- fit(Sat ~ Infl + Type + Cont)
  out <- list()
  latent <- emmeans(f, "Infl")
  out$polr_latent <- row(latent, "Infl")
  out$polr_latent_pairs <- row(pairs(latent), "contrast", TRUE)
  out$polr_latent_effects <- row(contrast(latent, "eff"), "contrast", TRUE)
  prob <- emmeans(f, ~ Infl | Sat, mode="prob")
  out$polr_prob <- row(prob, c("Sat", "Infl"))
  out$polr_prob_pairs <- row(pairs(prob), c("Sat", "contrast"), TRUE)
  out$polr_prob_two <- row(emmeans(f, c("Infl", "Sat"), mode="prob"), c("Infl", "Sat"))
  mean_class <- emmeans(f, "Infl", mode="mean.class")
  out$polr_mean_class <- row(mean_class, "Infl")
  out$polr_mean_class_pairs <- row(pairs(mean_class), "contrast", TRUE)
  out$polr_by <- row(emmeans(f, "Infl", by="Cont"), c("Cont", "Infl"))
  out$polr_proportional <- row(emmeans(f, "Infl", weights="proportional"), "Infl")
  out$polr_cells <- row(emmeans(f, "Infl", weights="cells"), "Infl")
  out$polr_prob_cells <- row(emmeans(f, ~ Infl | Sat, mode="prob", weights="cells"), c("Sat", "Infl"))
  offset <- fit(Sat ~ Infl + Type + offset(off))
  out$polr_offset <- row(emmeans(offset, "Infl"), "Infl")
  out$polr_offset_prob <- row(emmeans(offset, ~ Infl | Sat, mode="prob"), c("Sat", "Infl"))
  out$polr_probit_prob <- row(emmeans(fit(Sat ~ Infl + Type + Cont, "probit"), ~ Infl | Sat, mode="prob"), c("Sat", "Infl"))
  out$polr_cloglog_mean_class <- row(emmeans(fit(Sat ~ Infl + Type + Cont, "cloglog"), "Infl", mode="mean.class"), "Infl")
  # MASS's default start fails for cauchit here; start from the logistic fit.
  cauchit <- fit(Sat ~ Infl + Type + Cont, "cauchit", start=c(coef(f), f$zeta))
  out$polr_cauchit_prob <- row(emmeans(cauchit, ~ Infl | Sat, mode="prob"), c("Sat", "Infl"))
  # emmeans 2.0.4 back-transforms MASS's loglog with the wrong tail (it gives
  # negative probabilities). A loglog fit is a cloglog fit of the reversed
  # response, so validate loglog through that fit, mapped back.
  reversed <- fit(Rev ~ Infl + Type + Cont, "cloglog")
  loglog_prob <- row(emmeans(reversed, ~ Infl | Rev, mode="prob"), c("Rev", "Infl"))
  names(loglog_prob)[1] <- "Sat"
  loglog_prob$Sat <- factor(as.character(loglog_prob$Sat), levels=levels(h$Sat))
  out$polr_loglog_prob <- loglog_prob[order(loglog_prob$Sat, loglog_prob$Infl), ]
  rownames(out$polr_loglog_prob) <- NULL
  loglog_mean <- row(emmeans(reversed, "Infl", mode="mean.class"), "Infl")
  classes <- nlevels(h$Sat) + 1
  loglog_mean[c("estimate", "conf_low", "conf_high")] <- classes - loglog_mean[c("estimate", "conf_high", "conf_low")]
  out$polr_loglog_mean_class <- loglog_mean
  out$polr_robust <- row(emmeans(f, "Infl", vcov.=sandwich::vcovCL(f, cluster=~Type)), "Infl")
  out$polr_robust_prob <- row(emmeans(f, ~ Infl | Sat, mode="prob", vcov.=sandwich::vcovCL(f, cluster=~Type, type="HC1")), c("Sat", "Infl"))
  out$polr_grouped <- h |> group_by(Cont) |>
    group_modify(function(d, k) row(emmeans(fit(Sat ~ Infl + Type, data=d), "Infl"), "Infl")) |>
    ungroup() |> as.data.frame()
  out
}
