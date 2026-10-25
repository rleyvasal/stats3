# mgcv supplies the fits, summaries, predictions and comparisons.
#
# Every fit runs mgcv with tightened stopping rules (TIGHT), so that its
# smoothing parameters are the optima to near machine precision and stats3,
# which optimizes the same criteria to full precision, can be compared to
# 1e-7. The "default__" fits use mgcv's own controls: those agree only to
# the accuracy mgcv's optimizers stop at.
gam_cases <- function() {
  suppressPackageStartupMessages(library(mgcv))
  d <- read.csv("tests/data/gam.csv", stringsAsFactors=TRUE)
  TIGHT <- list(mgcv.tol=1e-13, epsilon=1e-13, newton=list(conv.tol=1e-13))
  # A tolerance this tight can end mgcv's Newton method with a "step
  # failure": it stops at the optimum to rounding error, which is the point.
  fit <- function(formula, family=gaussian(), method="GCV.Cp", control=TIGHT, data=d) {
    suppressWarnings(gam(formula, family=family, data=data, method=method, control=control))
  }
  fits <- list(
    gaussian=fit(y ~ s(x0) + s(x1) + s(x2) + f),
    gaussian_reml=fit(y ~ s(x0) + s(x3) + f, method="REML"),
    cr=fit(y ~ s(x0, bs="cr") + s(x1, bs="cr", k=15) + f, method="REML"),
    cs=fit(y ~ s(x0, bs="cs", k=8) + s(x2, bs="cs", k=10), method="REML"),
    ps=fit(y ~ s(x0, bs="ps") + s(x2, bs="ps", m=c(2, 1), k=20)),
    re=fit(y ~ s(x2) + s(subject, bs="re"), method="REML"),
    re_slope=fit(y ~ s(x2) + s(subject, x0, bs="re") + s(f, bs="re"), method="REML"),
    by_factor=fit(y ~ f + s(x2, by=f), method="REML"),
    by_numeric=fit(y ~ s(x2, by=x0) + s(x1)),
    te=fit(y ~ te(x0, x1) + s(x2), method="REML"),
    te_mixed=fit(y ~ te(x0, x2, bs=c("tp", "ps"), k=c(4, 6))),
    te_re=fit(y ~ te(x0, subject, bs=c("cr", "re")), method="REML"),
    tp2=fit(y ~ s(x0, x1) + s(x2, k=20), method="REML"),
    tp3=fit(y ~ s(x0, x1, x2, k=40), method="REML"),
    fixed=fit(y ~ s(x0, fx=TRUE, k=5) + s(x2), method="REML"),
    m1_cr_by=fit(y ~ s(x0, m=1) + s(x2, bs="cr", by=f) + f, method="REML"),
    weighted=suppressWarnings(gam(y ~ s(x0) + s(x2), data=d, weights=w, control=TIGHT)),
    poisson=fit(count ~ s(x0) + s(x2) + offset(log(exposure)), poisson()),
    poisson_offset_arg=suppressWarnings(gam(count ~ s(x0) + s(x2), poisson(), data=d, method="REML",
                                            offset=log(exposure), control=TIGHT)),
    binomial=fit(binary ~ s(x0) + s(x3), binomial(), method="REML"),
    probit=fit(cbind(succ, fail) ~ s(x0) + s(x2), binomial(link="probit"), method="REML"),
    gamma=fit(pos ~ s(x0) + s(x1) + f, Gamma(link="log")),
    gamma_reml=fit(pos ~ s(x0) + s(x1), Gamma(link="log"), method="REML"),
    quasipoisson=fit(count ~ s(x0) + s(x2), quasipoisson(), method="REML"),
    inverse_gaussian=fit(pos ~ s(x0) + s(x2), inverse.gaussian(), method="REML"),
    default_gaussian=fit(y ~ s(x0) + s(x1) + s(x2) + f, control=list()),
    default_reml=fit(pos ~ s(x0) + s(x1), Gamma(link="log"), method="REML", control=list())
  )
  tidy_fit <- function(b) {
    p <- summary(b)$p.table
    z <- qnorm(0.975)
    data.frame(term=rownames(p), estimate=p[, 1], std_error=p[, 2], statistic=p[, 3], p_value=p[, 4],
               conf_low=p[, 1] - z * p[, 2], conf_high=p[, 1] + z * p[, 2], row.names=NULL)
  }
  smooth_fit <- function(b) {
    s <- summary(b)$s.table
    data.frame(term=rownames(s), edf=s[, 1], ref_df=s[, 2], statistic=s[, 3], p_value=s[, 4], row.names=NULL)
  }
  glance_fit <- function(b) {
    s <- summary(b)
    data.frame(df=sum(b$edf), logLik=as.numeric(logLik(b)), AIC=AIC(b), BIC=BIC(b), deviance=deviance(b),
      df_residual=b$df.residual, nobs=nobs(b), adj_r_squared=s$r.sq, npar=length(coef(b)),
      dev_expl=s$dev.expl, method=b$method, sp_criterion=unname(b$gcv.ubre), scale=b$sig2)
  }
  augment_fit <- function(b) {
    p <- predict(b, se.fit=TRUE)
    data.frame(.fitted=unname(p$fit), .se_fit=unname(p$se.fit), .resid=unname(residuals(b)),
               .hat=unname(b$hat), .cooksd=unname(cooks.distance(b)), check.names=FALSE)
  }
  anova_fit <- function(b) {
    a <- anova(b)
    out <- data.frame(term=character(0), type=character(0), df=numeric(0), ref_df=numeric(0),
                      statistic=numeric(0), p_value=numeric(0))
    if (!is.null(a$pTerms.table)) out <- rbind(out, data.frame(term=rownames(a$pTerms.table), type="parametric",
      df=a$pTerms.table[, 1], ref_df=NA_real_, statistic=a$pTerms.table[, 2], p_value=a$pTerms.table[, 3]))
    out <- rbind(out, data.frame(term=rownames(a$s.table), type="smooth", df=a$s.table[, 1],
      ref_df=a$s.table[, 2], statistic=a$s.table[, 3], p_value=a$s.table[, 4]))
    rownames(out) <- NULL
    out
  }
  out <- list()
  for (name in names(fits)) {
    b <- fits[[name]]
    out[[paste0(name, "__tidy")]] <- tidy_fit(b)
    out[[paste0(name, "__smooth")]] <- smooth_fit(b)
    out[[paste0(name, "__glance")]] <- glance_fit(b)
    out[[paste0(name, "__augment")]] <- augment_fit(b)
    out[[paste0(name, "__anova")]] <- anova_fit(b)
    out[[paste0(name, "__sp")]] <- data.frame(term=names(b$sp), sp=unname(b$sp))
  }
  # coefficients and covariances are basis dependent; cr, ps and re bases
  # have no sign ambiguity
  for (name in c("cr", "ps", "re", "by_factor")) {
    b <- fits[[name]]
    out[[paste0(name, "__coef")]] <- data.frame(term=names(coef(b)), estimate=unname(coef(b)))
    for (kind in c("Vp", "Ve", "Vc")) {
      v <- switch(kind, Vp=vcov(b), Ve=vcov(b, freq=TRUE), Vc=vcov(b, unconditional=TRUE))
      out[[paste0(name, "__", kind)]] <- data.frame(term=rownames(v), v, check.names=FALSE, row.names=NULL)
    }
  }
  # the cs penalty: its null space shrinkage depends on LAPACK's choice of
  # eigenvectors for a repeated zero eigenvalue
  for (spec in list(list("x0", 8), list("x2", 10))) {
    sm <- smooth.construct(s(x, bs="cs", k=spec[[2]]), data.frame(x=d[[spec[[1]]]]), NULL)
    out[[paste0("cs_penalty_", spec[[1]])]] <- as.data.frame(sm$S[[1]])
  }
  new <- d[c(1, 7, 50, 120, 200), ]
  new$x0[2] <- 1.05  # extrapolation
  for (name in c("gaussian", "cr", "ps", "by_factor", "te", "gamma", "poisson_offset_arg")) {
    b <- fits[[name]]
    for (type in c("link", "response")) {
      p <- predict(b, newdata=new, type=type, se.fit=TRUE)
      out[[paste0(name, "__predict_", type)]] <- data.frame(fit=unname(p$fit), se_fit=unname(p$se.fit))
    }
    p <- predict(b, newdata=new, type="terms", se.fit=TRUE)
    terms <- data.frame(p$fit, check.names=FALSE, row.names=NULL)
    se <- data.frame(p$se.fit, check.names=FALSE, row.names=NULL)
    names(se) <- paste0("se_", names(se))
    out[[paste0(name, "__predict_terms")]] <- cbind(terms, se)
  }
  small <- fit(y ~ s(x0) + f)
  big <- fit(y ~ s(x0) + s(x1) + f)
  a <- anova(small, big, test="F")
  out$compare_f <- data.frame(term=c("y ~ s(x0) + f", "y ~ s(x0) + s(x1) + f"), df_residual=a$`Resid. Df`,
    residual_deviance=a$`Resid. Dev`, df=a$Df, deviance=a$Deviance, statistic=a$F, p_value=a$`Pr(>F)`)
  small <- fit(count ~ s(x0), poisson(), method="REML")
  big <- fit(count ~ s(x0) + s(x2), poisson(), method="REML")
  a <- anova(small, big, test="Chisq")
  out$compare_chisq <- data.frame(term=c("count ~ s(x0)", "count ~ s(x0) + s(x2)"), df_residual=a$`Resid. Df`,
    residual_deviance=a$`Resid. Dev`, df=a$Df, deviance=a$Deviance, p_value=a$`Pr(>Chi)`)
  out$grouped <- do.call(rbind, lapply(split(d, d$g), function(dd) {
    cbind(g=as.character(dd$g[1]), smooth_fit(fit(y ~ s(x0) + s(x2), data=dd, method="REML")))
  }))
  rownames(out$grouped) <- NULL
  out$grouped_glance <- do.call(rbind, lapply(split(d, d$g), function(dd) {
    cbind(g=as.character(dd$g[1]), glance_fit(fit(y ~ s(x0) + s(x2), data=dd, method="REML")))
  }))
  large <- read.csv("tests/data/gam_large.csv")
  for (form in c("y ~ s(x)", "y ~ s(x, z)")) {
    b <- fit(as.formula(form), data=large, method="REML")
    out[[paste0("large__", gsub("[^a-z]", "", form))]] <- glance_fit(b)
  }
  out
}
