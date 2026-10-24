# MASS and quantreg provide reference fits; adaptations only name/align tables.
resistant_cases <- function(kind) {
  m <- read.csv("tests/data/mtcars.csv")
  m$w <- 1 + seq_len(nrow(m)) %% 3
  m$w0 <- m$w
  m$w0[c(2, 10, 20)] <- 0
  if (kind == "rlm") {
    suppressPackageStartupMessages(library(MASS))
    fits <- list(
      basic=rlm(mpg ~ wt + hp, m),
      factors=rlm(mpg ~ wt + factor(cyl), m),
      bisquare=rlm(mpg ~ wt + hp, m, psi=psi.bisquare),
      bisquare_custom=rlm(mpg ~ wt + hp, m, psi=psi.bisquare, c=6),
      hampel=rlm(mpg ~ wt + hp, m, psi=psi.hampel),
      hampel_custom=rlm(mpg ~ wt + hp, m, psi=psi.hampel, a=1.5, b=3, c=6),
      tuning=rlm(mpg ~ wt + hp, m, k=2),
      proposal=rlm(mpg ~ wt + hp, m, scale.est="Huber"),
      proposal_k2=rlm(mpg ~ wt + hp, m, scale.est="Huber", k2=2),
      weighted=rlm(mpg ~ wt + hp, m, weights=w),
      zero_weights=rlm(mpg ~ wt + hp, m, weights=w0),
      case=rlm(mpg ~ wt + hp, m, weights=w, wt.method="case"),
      case_proposal=rlm(mpg ~ wt + hp, m, weights=w, wt.method="case", scale.est="proposal 2"),
      offset=rlm(mpg ~ wt + offset(hp), m),
      no_intercept=rlm(mpg ~ 0 + wt + hp, m),
      xtwx=rlm(mpg ~ wt + hp, m),
      case_xtwx=rlm(mpg ~ wt + hp, m, weights=w, wt.method="case")
    )
  } else {
    suppressPackageStartupMessages(library(quantreg))
    fits <- list(
      median=rq(mpg ~ wt + hp, data=m),
      lower=rq(mpg ~ wt + hp, tau=.25, data=m),
      upper=rq(mpg ~ wt + hp, tau=.75, data=m),
      factors=rq(mpg ~ wt + factor(cyl), data=m),
      weighted=rq(mpg ~ wt + hp, data=m, weights=w),
      zero_weights=rq(mpg ~ wt + hp, data=m, weights=w0),
      no_intercept=rq(mpg ~ 0 + wt + hp, data=m),
      iid=rq(mpg ~ wt + hp, data=m),
      weighted_iid=rq(mpg ~ wt + hp, data=m, weights=w)
    )
  }
  vc <- function(f, name) {
    if (kind == "rlm") {
      # vcov.rlm ignores method=; XtWX is an explicit summary.rlm option.
      s <- summary(f, method=if (grepl("xtwx", name)) "XtWX" else "XtX")
      return(s$cov.unscaled * s$stddev^2)
    }
    v <- summary(f, se=if (grepl("iid", name)) "iid" else "nid", covariance=TRUE)$cov
    dimnames(v) <- list(names(coef(f)), names(coef(f)))
    v
  }
  tidy_fit <- function(f, name) {
    if (kind == "rlm") {
      co <- summary(f, method=if (grepl("xtwx", name)) "XtWX" else "XtX")$coefficients
      p <- rep(NA_real_, nrow(co)) # summary.rlm does not report p-values.
      q <- qnorm(.975)
    } else {
      co <- summary(f, se=if (grepl("iid", name)) "iid" else "nid")$coefficients
      p <- co[, 4]
      q <- qt(.975, length(residuals(f)) - length(coef(f)))
    }
    data.frame(term=names(coef(f)), estimate=unname(coef(f)), std_error=unname(co[, 2]),
      statistic=unname(co[, 3]), p_value=unname(p),
      conf_low=unname(co[, 1] - q * co[, 2]), conf_high=unname(co[, 1] + q * co[, 2]))
  }
  out <- list()
  for (name in names(fits)) {
    f <- fits[[name]]
    out[[paste0(name, "__tidy")]] <- tidy_fit(f, name)
    v <- vc(f, name)
    out[[paste0(name, "__vcov")]] <- data.frame(term=names(coef(f)), v, check.names=FALSE, row.names=NULL)
    out[[paste0(name, "__augment")]] <- data.frame(.fitted=unname(fitted(f)), .resid=unname(residuals(f)))
    if (kind == "rlm") {
      out[[paste0(name, "__glance")]] <- data.frame(sigma=f$s, df_residual=NA_real_,
        nobs=nobs(f), converged=f$converged, iterations=length(f$conv))
      out[[paste0(name, "__weights")]] <- data.frame(weight=f$w, wresid=f$wresid)
      pred <- predict(f, newdata=m[1:5, ], interval="confidence", se.fit=TRUE)
      pr <- data.frame(fit=unname(pred$fit[,1]), lwr=unname(pred$fit[,2]),
                       upr=unname(pred$fit[,3]), se_fit=unname(pred$se.fit))
    } else {
      n <- length(residuals(f)); p <- length(coef(f))
      out[[paste0(name, "__glance")]] <- data.frame(tau=f$tau, rho=f$rho, logLik=as.numeric(logLik(f)),
        AIC=AIC(f), BIC=BIC(f), df_residual=n-p, nobs=n)
      pred <- predict(f, newdata=m[1:5, ], interval="confidence", se=if (grepl("iid", name)) "iid" else "nid")
      x <- model.matrix(delete.response(terms(f)), m[1:5, ])
      pr <- data.frame(fit=unname(pred[,1]), lwr=unname(pred[,2]), upr=unname(pred[,3]),
                       se_fit=sqrt(rowSums((x %*% v) * x)))
    }
    out[[paste0(name, "__predict")]] <- pr
  }
  out$grouped <- m |> group_by(am) |>
    group_modify(function(d, k) tidy_fit(if (kind == "rlm") rlm(mpg ~ wt + hp, d) else rq(mpg ~ wt + hp, data=d), "basic")) |>
    ungroup() |> as.data.frame()
  if (kind == "rq") {
    f <- fits$median
    a <- anova(f, rq(mpg ~ wt, data=m))$table
    out$compare <- data.frame(term="mpg ~ wt", df=a$ndf, df_residual=a$ddf, statistic=a$Tn, p_value=a$pvalue)
  }
  suppressPackageStartupMessages(library(emmeans))
  f <- fits$factors
  e <- as.data.frame(emmeans(f, "cyl"))
  lower <- intersect(c("lower.CL", "asymp.LCL"), names(e))[1]
  upper <- intersect(c("upper.CL", "asymp.UCL"), names(e))[1]
  out$emmeans <- data.frame(cyl=e$cyl, estimate=e$emmean, std_error=e$SE, df=e$df,
                            conf_low=e[[lower]], conf_high=e[[upper]])
  out
}
