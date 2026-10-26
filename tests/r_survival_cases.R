survival_cases <- function() {
  suppressPackageStartupMessages({library(survival); library(broom)})
  snake <- function(d) {
    d <- as.data.frame(d)
    names(d) <- gsub(".", "_", names(d), fixed=TRUE)
    rownames(d) <- NULL
    d
  }
  lung <- read.csv("tests/data/lung.csv", na.strings="NA")
  names(lung)[names(lung) == "ph.ecog"] <- "ecog"
  i <- seq_len(nrow(lung))
  lung$entry <- pmin(lung$time - 1, (i %% 5) * 20)
  lung$w <- (i %% 3) + 1
  lung$wf <- 0.5 + (i %% 4) / 4
  lung$off <- ((i %% 7) - 3) / 10
  # Interval-censored deaths: each known only to a 30-day interval, early
  # ones only to be before its end; censored times stay right-censored.
  dead <- lung$status == 2
  lung$lo <- ifelse(dead, floor(lung$time / 30) * 30, lung$time)
  lung$hi <- ifelse(dead, lung$lo + 30, NA)
  lung$lo[dead & (lung$time < 60 | lung$lo == 0)] <- NA
  lung$ev <- ifelse(is.na(lung$hi), 0, ifelse(is.na(lung$lo), 2, 3))
  lung$t1 <- ifelse(is.na(lung$lo), lung$hi, lung$lo)
  lung$t2 <- ifelse(is.na(lung$hi), lung$t1, lung$hi)
  vet <- read.csv("tests/data/veteran.csv")
  vet$celltype <- factor(vet$celltype, levels=c("squamous", "smallcell", "adeno", "large"))
  out <- list()

  km_glance <- function(f) {
    t <- summary(f)$table
    if (is.matrix(t)) {
      d <- data.frame(strata=rownames(t), t, check.names=FALSE)
    } else d <- as.data.frame(t(t), check.names=FALSE)
    names(d)[names(d) %in% c("records", "n.max", "n.start", "events", "rmean", "se(rmean)", "median")] <-
      c("records", "n_max", "n_start", "events", "rmean", "rmean_std_error", "median")
    k <- ncol(d)
    names(d)[c(k - 1, k)] <- c("conf_low", "conf_high")
    d$nobs <- d$records
    rownames(d) <- NULL
    d
  }
  km <- function(f) list(tidy=snake(tidy(f)), glance=km_glance(f))
  out$km_sex <- km(survfit(Surv(time, status) ~ sex, lung))
  out$km_one <- km(survfit(Surv(time, status) ~ 1, vet))
  out$km_two <- km(survfit(Surv(time, status) ~ sex + ecog, lung))
  out$km_weights <- km(survfit(Surv(time, status) ~ sex, lung, weights=w))
  out$km_counting <- km(survfit(Surv(entry, time, status) ~ sex, lung))
  out$km_factor <- km(survfit(Surv(time, status) ~ celltype, vet))
  for (type in c("log-log", "plain", "logit", "arcsin"))
    out[[paste0("km_", type)]] <- km(survfit(Surv(time, status) ~ sex, lung, conf.type=type, conf.int=0.9))
  # broom's tidy.survfit fails without intervals; the same columns, less the bounds.
  k <- survfit(Surv(time, status) ~ sex, lung, conf.type="none")
  out$km_none <- list(tidy=data.frame(time=k$time, n_risk=k$n.risk, n_event=k$n.event, n_censor=k$n.censor,
                                      estimate=k$surv, std_error=k$std.err, strata=rep(names(k$strata), k$strata)))
  grouped <- lapply(c(1, 2), function(g) {
    f <- survfit(Surv(time, status) ~ celltype, vet[vet$trt == g, ])
    list(tidy=cbind(trt=g, snake(tidy(f))), glance=cbind(trt=g, km_glance(f)))
  })
  out$km_grouped <- list(tidy=do.call(rbind, lapply(grouped, `[[`, "tidy")),
                         glance=do.call(rbind, lapply(grouped, `[[`, "glance")))

  sdiff <- function(f, vars) {
    t <- data.frame(tidy(f))
    for (v in vars) t[[v]] <- if (is.numeric(vet[[v]]) || is.numeric(lung[[v]])) as.numeric(t[[v]]) else t[[v]]
    list(tidy=snake(t), glance=snake(glance(f)))
  }
  out$diff_sex <- sdiff(survdiff(Surv(time, status) ~ sex, lung), "sex")
  out$diff_rho <- sdiff(survdiff(Surv(time, status) ~ sex, lung, rho=1), "sex")
  out$diff_strata <- sdiff(survdiff(Surv(time, status) ~ sex + strata(ecog), lung), "sex")
  out$diff_cell <- sdiff(survdiff(Surv(time, status) ~ celltype, vet), "celltype")
  out$diff_two <- sdiff(survdiff(Surv(time, status) ~ sex + ecog, lung, rho=0.5), c("sex", "ecog"))
  gd <- lapply(c(1, 2), function(g) {
    f <- survdiff(Surv(time, status) ~ celltype, vet[vet$trt == g, ])
    s <- sdiff(f, "celltype")
    list(tidy=cbind(trt=g, s$tidy), glance=cbind(trt=g, s$glance))
  })
  out$diff_grouped <- list(tidy=do.call(rbind, lapply(gd, `[[`, "tidy")),
                           glance=do.call(rbind, lapply(gd, `[[`, "glance")))

  # summary.coxph rounds the Wald statistic to two decimals; use R's unrounded test.
  cox_glance <- function(f) {
    g <- snake(glance(f))
    g$statistic_wald <- f$wald.test
    g$p_value_wald <- pchisq(f$wald.test, sum(!is.na(coef(f))), lower.tail=FALSE)
    g
  }
  cox_tables <- function(f, new=NULL) {
    v <- vcov(f)
    res <- list(tidy=snake(tidy(f, conf.int=TRUE)),
                glance=cox_glance(f),
                vcov=data.frame(term=rownames(v), unname(v), check.names=FALSE),
                lp=snake(lapply(predict(f, type="lp", se.fit=TRUE), unname)),
                risk=snake(lapply(predict(f, type="risk", se.fit=TRUE), unname)),
                expected=data.frame(fit=unname(predict(f, type="expected"))))
    names(res$vcov) <- c("term", rownames(v))
    if (!is.null(new)) {
      res$new_lp <- data.frame(fit=unname(predict(f, new, type="lp")),
                               se_fit=unname(predict(f, new, type="lp", se.fit=TRUE)$se.fit))
      res$new_risk <- data.frame(fit=unname(predict(f, new, type="risk")))
      res$new_expected <- data.frame(fit=unname(predict(f, new, type="expected")))
    }
    res
  }
  zph <- function(f, ...) {
    t <- cox.zph(f, ...)$table
    data.frame(term=rownames(t), statistic=t[, 1], df=t[, 2], p_value=t[, 3], row.names=NULL)
  }
  lc <- lung[complete.cases(lung[, c("age", "sex", "ecog")]), ]
  newl <- head(lc, 5)
  out$cox_efron <- cox_tables(f1 <- coxph(Surv(time, status) ~ age + sex + ecog, lung), newl)
  out$cox_efron$zph <- zph(f1)
  out$cox_efron$zph_rank <- zph(f1, transform="rank")
  out$cox_efron$zph_identity <- zph(f1, transform="identity", global=FALSE)
  out$cox_breslow <- cox_tables(f2 <- coxph(Surv(time, status) ~ age + sex, lung, ties="breslow"), newl)
  out$cox_breslow$zph <- zph(f2)
  out$cox_exact <- cox_tables(coxph(Surv(time, status) ~ karno + trt, vet, ties="exact"))
  out$cox_strata <- cox_tables(f3 <- coxph(Surv(time, status) ~ age + ecog + strata(sex), lung), newl)
  out$cox_strata$zph <- zph(f3)
  out$cox_cluster <- cox_tables(coxph(Surv(time, status) ~ age + sex + cluster(inst), lung))
  out$cox_robust <- cox_tables(coxph(Surv(time, status) ~ age + sex, lung, robust=TRUE))
  out$cox_weights <- cox_tables(coxph(Surv(time, status) ~ age + sex, lung, weights=w), newl)
  out$cox_fweights <- cox_tables(coxph(Surv(time, status) ~ age + sex, lung, weights=wf))
  out$cox_offset <- cox_tables(coxph(Surv(time, status) ~ age + sex + offset(off), lung))
  out$cox_counting <- cox_tables(f4 <- coxph(Surv(entry, time, status) ~ age + sex, lung))
  out$cox_counting$zph <- zph(f4)
  out$cox_factor <- cox_tables(f5 <- coxph(Surv(time, status) ~ trt + celltype + karno, vet), head(vet, 5))
  out$cox_factor$zph <- zph(f5)
  out$cox_factor$zph_terms <- zph(f5, terms=FALSE)
  a <- anova(f1)
  out$cox_anova <- data.frame(term=rownames(a), logLik=a$loglik, statistic=a$Chisq, df=a$Df,
                              p_value=a$`Pr(>|Chi|)`, row.names=NULL)
  small <- coxph(Surv(time, status) ~ age, lc)
  big <- coxph(Surv(time, status) ~ age + sex + ecog, lc)
  a <- anova(small, big)
  out$cox_compare <- data.frame(term=c("Surv(time, status) ~ age", "Surv(time, status) ~ age + sex + ecog"),
                                logLik=a$loglik, statistic=a$Chisq, df=a$Df, p_value=a$`Pr(>|Chi|)`)
  ex <- coxph(Surv(time, status) ~ age + ecog, lung, na.action=na.exclude)
  out$cox_exclude <- data.frame(fit=unname(predict(ex, type="lp")), expected=unname(predict(ex, type="expected")))
  gc <- lapply(c(1, 2), function(g) {
    f <- coxph(Surv(time, status) ~ celltype + karno, vet[vet$trt == g, ])
    list(tidy=cbind(trt=g, snake(tidy(f))), glance=cbind(trt=g, cox_glance(f)))
  })
  out$cox_grouped <- list(tidy=do.call(rbind, lapply(gc, `[[`, "tidy")),
                          glance=do.call(rbind, lapply(gc, `[[`, "glance")))
  curves <- function(s) list(tidy=snake(tidy(s)), glance=km_glance(s))
  out$cox_curve <- curves(survfit(f1))
  out$cox_curve_strata <- curves(survfit(f3))
  out$cox_curve_new <- curves(survfit(f1, newdata=head(lc, 2)))
  names(out$cox_curve_new$glance)[1] <- "curve"
  out$cox_curve_new$glance$curve <- as.numeric(out$cox_curve_new$glance$curve)
  out$cox_curve_breslow <- curves(survfit(f2, newdata=head(lc, 1), conf.type="log-log"))
  out$cox_curve_counting <- curves(survfit(f4))
  out$cox_curve_weights <- curves(survfit(coxph(Surv(time, status) ~ age + sex, lung, weights=w)))
  newl$curve <- NULL

  reg_tables <- function(f, new) {
    p <- predict(f, new, type="quantile", p=c(.25, .5), se.fit=TRUE)
    list(tidy=snake(tidy(f, conf.int=TRUE)), glance=snake(glance(f)),
         response=data.frame(fit=unname(predict(f))),
         new_response=data.frame(fit=unname(predict(f, new, se.fit=TRUE)$fit),
                                 se_fit=unname(predict(f, new, se.fit=TRUE)$se.fit)),
         new_lp=data.frame(fit=unname(predict(f, new, type="lp", se.fit=TRUE)$fit),
                           se_fit=unname(predict(f, new, type="lp", se.fit=TRUE)$se.fit)),
         new_quantile=data.frame(fit_0.25=unname(p$fit[, 1]), se_fit_0.25=unname(p$se.fit[, 1]),
                                 fit_0.5=unname(p$fit[, 2]), se_fit_0.5=unname(p$se.fit[, 2]), check.names=FALSE),
         new_uquantile=data.frame(fit=unname(predict(f, new, type="uquantile", p=.5))))
  }
  for (d in c("weibull", "exponential", "lognormal", "loglogistic", "gaussian"))
    out[[paste0("reg_", d)]] <- reg_tables(survreg(Surv(time, status) ~ age + sex, lung, dist=d), newl)
  out$reg_factor <- reg_tables(survreg(Surv(time, status) ~ trt + celltype + karno, vet), head(vet, 5))
  out$reg_weights <- reg_tables(survreg(Surv(time, status) ~ age + sex, lung, weights=w), newl)
  out$reg_offset <- reg_tables(survreg(Surv(time, status) ~ age + sex + offset(off), lung), newl)
  out$reg_scale <- reg_tables(survreg(Surv(time, status) ~ age + sex, lung, scale=1.5), newl)
  out$reg_intercept <- reg_tables(survreg(Surv(time, status) ~ 1, lung, dist="lognormal"), newl)
  for (d in c("weibull", "lognormal", "loglogistic"))
    out[[paste0("reg_interval_", d)]] <- reg_tables(survreg(Surv(lo, hi, type="interval2") ~ age + sex, lung, dist=d), newl)
  out$reg_interval_scale <- reg_tables(survreg(Surv(lo, hi, type="interval2") ~ age, lung, dist="gaussian", scale=200), newl)
  out$reg_interval_codes <- reg_tables(survreg(Surv(t1, t2, ev, type="interval") ~ age + sex, lung), newl)
  out$reg_left <- reg_tables(survreg(Surv(time, status == 2, type="left") ~ age + sex, lung), newl)
  out$km_interval <- km(survfit(Surv(lo, hi, type="interval2") ~ sex, lung))
  out$km_left <- km(survfit(Surv(time, status == 2, type="left") ~ 1, lung))
  out$km_fweights <- km(survfit(Surv(time, status) ~ sex, lung, weights=wf))
  gr <- lapply(c(1, 2), function(g) {
    f <- survreg(Surv(time, status) ~ age, lung[lung$sex == g, ])
    list(tidy=cbind(sex=g, snake(tidy(f))), glance=cbind(sex=g, snake(glance(f))))
  })
  out$reg_grouped <- list(tidy=do.call(rbind, lapply(gr, `[[`, "tidy")),
                          glance=do.call(rbind, lapply(gr, `[[`, "glance")))
  out
}
