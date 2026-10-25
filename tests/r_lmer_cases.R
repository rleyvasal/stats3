# lme4 supplies the mixed-model fits; these functions only name and align tables.
#
# Each model is fitted twice: as lmer() fits it, and "tight". lme4's default
# optimizer stops on the criterion's value, which is flat near the optimum, so
# its theta carries up to about 1e-5 relative error; the tight fit runs BOBYQA
# to a 1e-12 trust region and then Newton steps on central differences of
# lme4's own deviance function (its modular interface: lFormula, mkLmerDevfun,
# mkMerMod), which locates the optimum to about 1e-9.
lmer_cases <- function() {
  suppressPackageStartupMessages(library(lme4))
  s <- read.csv("tests/data/sleepstudy.csv")
  s$w <- 1 + seq_len(nrow(s)) %% 3
  p <- read.csv("tests/data/penguins.csv", na.strings="NA")
  p$w <- 1 + seq_len(nrow(p)) %% 3
  polish <- function(devfun, theta) {
    k <- length(theta)
    value <- devfun(theta)
    for (iteration in 1:20) {
      h <- 1e-4 * pmax(1, abs(theta))
      gradient <- numeric(k)
      hessian <- matrix(0, k, k)
      up <- down <- numeric(k)
      for (i in 1:k) {
        e <- replace(numeric(k), i, h[i])
        up[i] <- devfun(theta + e)
        down[i] <- devfun(theta - e)
        gradient[i] <- (up[i] - down[i]) / (2 * h[i])
        hessian[i, i] <- (up[i] - 2 * value + down[i]) / h[i]^2
      }
      if (k > 1) for (i in 2:k) for (j in 1:(i - 1)) {
        ei <- replace(numeric(k), i, h[i])
        ej <- replace(numeric(k), j, h[j])
        hessian[i, j] <- hessian[j, i] <- (devfun(theta + ei + ej) - 2 * value + devfun(theta - ei - ej)
          - hessian[i, i] * h[i]^2 - hessian[j, j] * h[j]^2) / (2 * h[i] * h[j])
      }
      if (any(eigen(hessian, symmetric=TRUE, only.values=TRUE)$values <= 0)) break
      step <- -solve(hessian, gradient)
      # The criterion is even in each element of theta, so the step is unbounded.
      candidate <- abs(theta + step)
      new <- devfun(candidate)
      if (new > value + 1e-9 * abs(value)) break
      theta <- candidate
      value <- new
      if (max(abs(step)) < 1e-10) break
    }
    list(par=theta, fval=value)
  }
  make <- function(formula, data, REML=TRUE, weights=NULL, offset=NULL, na.action=na.omit) {
    data$.w <- if (is.null(weights)) rep(1, nrow(data)) else weights
    data$.off <- if (is.null(offset)) rep(0, nrow(data)) else offset
    default <- lmer(formula, data, REML=REML, weights=.w, offset=.off, na.action=na.action)
    lmod <- lFormula(formula, data, REML=REML, weights=.w, offset=.off, na.action=na.action)
    devfun <- do.call(mkLmerDevfun, lmod)
    opt <- optimizeLmer(devfun, optimizer="bobyqa", control=list(rhoend=1e-12, maxfun=1e5))
    polished <- polish(devfun, opt$par)
    opt$par <- polished$par
    opt$fval <- polished$fval
    devfun(polished$par)
    tight <- mkMerMod(environment(devfun), opt, lmod$reTrms, fr=lmod$fr)
    list(default=default, tight=tight)
  }
  fits <- list(
    intercept=make(Reaction ~ Days + (1 | Subject), s),
    slopes=make(Reaction ~ Days + (Days || Subject), s),
    slope_only=make(Reaction ~ Days + (0 + Days | Subject), s),
    ml=make(Reaction ~ Days + (Days || Subject), s, REML=FALSE),
    weighted=make(Reaction ~ Days + (Days || Subject), s, weights=s$w),
    weighted_ml=make(Reaction ~ Days + (Days || Subject), s, weights=s$w, REML=FALSE),
    offset=make(Reaction ~ Days + offset(0.5 * Days) + (1 | Subject), s),
    offset_arg=make(Reaction ~ Days + (1 | Subject), s, offset=0.5 * s$Days),
    no_intercept=make(Reaction ~ 0 + Days + (1 | Subject), s),
    transformed=make(log(Reaction) ~ I(Days^2) + (1 | Subject), s),
    factors=make(body_mass_g ~ flipper_length_mm + sex + (1 | species), p),
    interaction=make(body_mass_g ~ flipper_length_mm * sex + (1 | species), p),
    weighted_factors=make(body_mass_g ~ flipper_length_mm + sex + (1 | species), p, weights=p$w),
    crossed=make(body_mass_g ~ flipper_length_mm + (1 | species) + (1 | island), p),
    interaction_group=make(body_mass_g ~ flipper_length_mm + sex + (1 | species:island), p),
    nested=make(body_mass_g ~ flipper_length_mm + sex + (1 | species/sex), p),
    island_slopes=make(body_mass_g ~ sex + (bill_depth_mm || island), p),
    species_slope=make(bill_length_mm ~ sex + (0 + bill_depth_mm | species), p)
  )
  tidy_fit <- function(f) {
    co <- summary(f)$coefficients
    vc <- as.data.frame(VarCorr(f))
    z <- qnorm(.975)
    fixed <- data.frame(effect="fixed", group=NA_character_, term=rownames(co),
      estimate=unname(co[, 1]), std_error=unname(co[, 2]), statistic=unname(co[, 3]),
      p_value=NA_real_, conf_low=unname(co[, 1] - z * co[, 2]), conf_high=unname(co[, 1] + z * co[, 2]))
    random <- data.frame(effect="ran_pars", group=as.character(vc$grp),
      term=paste0("sd__", ifelse(is.na(vc$var1), "Observation", vc$var1)),
      estimate=vc$sdcor, std_error=NA_real_, statistic=NA_real_, p_value=NA_real_,
      conf_low=NA_real_, conf_high=NA_real_)
    rbind(fixed, random)
  }
  glance_fit <- function(f) data.frame(nobs=nobs(f), sigma=sigma(f), logLik=as.numeric(logLik(f)),
    AIC=AIC(f), BIC=BIC(f), REMLcrit=if (isREML(f)) REMLcrit(f) else NA_real_,
    deviance=deviance(f, REML=FALSE), df_residual=df.residual(f))
  augment_fit <- function(f) data.frame(.fitted=unname(fitted(f)), .resid=unname(residuals(f, type="response")),
    .hat=unname(hatvalues(f)), .cooksd=unname(cooks.distance(f)), .fixed=unname(predict(f, re.form=NA)),
    check.names=FALSE)
  ranef_fit <- function(f) {
    r <- as.data.frame(ranef(f, condVar=TRUE))
    data.frame(group=as.character(r$grpvar), term=as.character(r$term), level=as.character(r$grp),
               estimate=r$condval, std_error=r$condsd)
  }
  vcov_fit <- function(f) {
    v <- as.matrix(vcov(f))
    out <- data.frame(term=rownames(v))
    for (name in colnames(v)) out[[name]] <- unname(v[, name])
    out
  }
  anova_fit <- function(f) {
    a <- anova(f)
    data.frame(term=rownames(a), df=a$npar, sumsq=a$`Sum Sq`, meansq=a$`Mean Sq`, statistic=a$`F value`)
  }
  theta_fit <- function(f) data.frame(theta=unname(getME(f, "theta")), sigma=sigma(f))
  tables <- list(tidy=tidy_fit, glance=glance_fit, augment=augment_fit, ranef=ranef_fit, vcov=vcov_fit,
                 anova=anova_fit, theta=theta_fit)
  out <- list()
  for (name in names(fits)) {
    for (table in names(tables)) {
      out[[paste0(name, "__", table)]] <- tables[[table]](fits[[name]]$default)
      out[[paste0(name, "__", table, "__tight")]] <- tables[[table]](fits[[name]]$tight)
    }
  }
  f <- fits$slopes$tight
  known <- data.frame(Days=c(0, 5, 9), Subject=c(308L, 372L, 334L))
  new <- data.frame(Days=c(0, 5, 9), Subject=c(308L, 372L, 999L))
  out$predict_population <- data.frame(fit=unname(predict(f, new, re.form=NA)))
  out$predict_random <- data.frame(fit=unname(predict(f, known)))
  out$predict_new_levels <- data.frame(fit=unname(predict(f, new, allow.new.levels=TRUE)))
  rows <- p[c(1, 2, 3, 153, 277), ]
  out$predict_factors <- data.frame(fit=unname(predict(fits$factors$tight, rows)))
  out$predict_crossed <- data.frame(fit=unname(predict(fits$crossed$tight, rows)))
  out$predict_nested <- data.frame(fit=unname(predict(fits$nested$tight, rows)))
  compare <- function(a, labels) data.frame(term=labels, npar=a$npar, AIC=a$AIC, BIC=a$BIC,
    logLik=a$logLik, deviance=a$`-2*log(L)`, statistic=a$Chisq, df=a$Df, p_value=a$`Pr(>Chisq)`)
  # lme4 refits REML models with ML before comparing, and sorts by parameter count.
  out$compare <- compare(suppressMessages(anova(fits$slopes$default, fits$intercept$default)),
                         c("Reaction ~ Days + (1 | Subject)", "Reaction ~ Days + (Days || Subject)"))
  ml1 <- make(Reaction ~ Days + (1 | Subject), s, REML=FALSE)$tight
  out$compare_ml <- compare(anova(ml1, fits$ml$tight),
                            c("Reaction ~ Days + (1 | Subject)", "Reaction ~ Days + (Days || Subject)"))
  out$compare_three <- compare(suppressMessages(anova(fits$intercept$default, fits$slopes$default, fits$slope_only$default)),
    c("Reaction ~ Days + (1 | Subject)", "Reaction ~ Days + (0 + Days | Subject)", "Reaction ~ Days + (Days || Subject)"))
  out$grouped <- p |> group_by(sex) |>
    group_modify(function(d, k) tidy_fit(make(body_mass_g ~ flipper_length_mm + (1 | species), as.data.frame(d))$tight)) |>
    ungroup() |> as.data.frame()
  ex <- make(body_mass_g ~ flipper_length_mm + sex + (1 | species), p, na.action=na.exclude)$tight
  out$exclude <- data.frame(.fitted=unname(fitted(ex)), .resid=unname(residuals(ex)), check.names=FALSE)
  out
}
