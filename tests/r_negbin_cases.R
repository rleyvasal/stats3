# MASS supplies the fits, theta estimates, likelihoods and inference.
negbin_cases <- function() {
  suppressPackageStartupMessages(library(MASS))
  d <- read.csv("tests/data/negbin.csv")
  d$w0 <- d$w
  d$w0[seq(1, nrow(d), by=10)] <- 0
  fits <- list(
    basic=glm.nb(y ~ x + group, d),
    interaction=glm.nb(y ~ x * group, d),
    offset=glm.nb(y ~ x + group + offset(log(exposure)), d),
    offset_arg=glm.nb(y ~ x + group, d, offset=d$exposure, control=glm.control()),
    weighted=glm.nb(y ~ x + group, d, weights=w),
    zero_weights=glm.nb(y ~ x + group, d, weights=w0),
    initial=glm.nb(y ~ x + group, d, init.theta=1),
    identity=glm.nb(y ~ group, d, link=identity),
    sqrt=glm.nb(y ~ group, d, link=sqrt),
    no_intercept=glm.nb(y ~ 0 + group + x, d),
    aliased=glm.nb(y ~ x + I(2*x) + group, d),
    fixed=glm(y ~ x + group, family=negative.binomial(2), d),
    fixed_sqrt=glm(y ~ group, family=negative.binomial(2, "sqrt"), d)
  )
  tidy_fit <- function(f, wald=FALSE) {
    s <- summary(f)$coefficients
    i <- match(names(coef(f)), rownames(s))
    ci <- if (wald) confint.default(f) else suppressMessages(confint(f))
    data.frame(term=names(coef(f)), estimate=unname(coef(f)), std_error=unname(s[i, 2]),
               statistic=unname(s[i, 3]), p_value=unname(s[i, 4]),
               conf_low=unname(ci[, 1]), conf_high=unname(ci[, 2]))
  }
  glance_fit <- function(f) data.frame(null_deviance=f$null.deviance, df_null=f$df.null,
    logLik=as.numeric(logLik(f)), AIC=AIC(f), BIC=BIC(f), deviance=deviance(f),
    df_residual=f$df.residual, nobs=nobs(f))
  augment_fit <- function(f) {
    # R omits zero-weight rows from influence vectors; align them to fitted rows.
    # Zero-weight rows have zero leverage/influence; deletion sigma is undefined.
    pad <- function(x, zero=0) {
      out <- rep(zero, length(f$fitted.values))
      out[f$prior.weights != 0] <- x
      out
    }
    data.frame(.fitted=unname(predict(f)), .resid=unname(residuals(f)),
      .hat=pad(hatvalues(f)), .sigma=pad(influence(f)$sigma, NA_real_),
      .cooksd=pad(cooks.distance(f)), .std_resid=pad(rstandard(f)), check.names=FALSE)
  }
  anova_fit <- function(f) {
    # Explicit test matches stats3's usual GLM defaults; MASS warns that
    # its single-model term tests hold theta fixed.
    a <- suppressWarnings(anova(f, test=if (inherits(f, "negbin")) "Chisq" else "F"))
    out <- data.frame(term=rownames(a), df=a$Df, deviance=a$Deviance,
                      df_residual=a$`Resid. Df`, residual_deviance=a$`Resid. Dev`)
    if (!is.null(a$F)) out$statistic <- a$F
    out$p_value <- a[[ncol(a)]]
    out
  }
  out <- list()
  for (name in names(fits)) {
    f <- fits[[name]]
    out[[paste0(name, "__tidy")]] <- tidy_fit(f)
    out[[paste0(name, "__wald")]] <- tidy_fit(f, TRUE)
    out[[paste0(name, "__glance")]] <- glance_fit(f)
    out[[paste0(name, "__augment")]] <- augment_fit(f)
    out[[paste0(name, "__anova")]] <- anova_fit(f)
    if (inherits(f, "negbin")) out[[paste0(name, "__theta")]] <- data.frame(theta=f$theta, se_theta=f$SE.theta)
  }
  f <- fits$basic
  for (type in c("link", "response")) {
    p <- predict(f, newdata=d[1:5, ], type=type, se.fit=TRUE)
    out[[paste0("predict_", type)]] <- data.frame(fit=unname(p$fit), se_fit=unname(p$se.fit))
  }
  small <- glm.nb(y ~ x, d)
  a <- anova(f, small) # MASS sorts by residual df, even in reverse argument order.
  out$compare <- data.frame(term=c("y ~ x", "y ~ x + group"), theta=a$theta,
    df_residual=a$`Resid. df`, logLik=a$`   2 x log-lik.`/2, df=a$`   df`,
    statistic=a$`LR stat.`, p_value=a$`Pr(Chi)`)
  out$grouped <- d |> group_by(batch) |>
    group_modify(function(dd, k) tidy_fit(glm.nb(y ~ x + group, dd))) |> ungroup() |> as.data.frame()
  v <- sandwich::vcovHC(f, type="HC3")
  co <- lmtest::coeftest(f, vcov.=v)
  ci <- lmtest::coefci(f, vcov.=v)
  out$robust <- data.frame(term=rownames(co), estimate=co[, 1], std_error=co[, 2],
    statistic=co[, 3], p_value=co[, 4], conf_low=ci[, 1], conf_high=ci[, 2], row.names=NULL)
  suppressPackageStartupMessages(library(emmeans))
  e <- as.data.frame(summary(emmeans(f, "group", type="response")))
  out$emmeans <- data.frame(group=e$group, estimate=e$response, std_error=e$SE, df=e$df,
                            conf_low=e$asymp.LCL, conf_high=e$asymp.UCL)
  out
}
