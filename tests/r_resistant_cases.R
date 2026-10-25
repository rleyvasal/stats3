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
      case_xtwx=rlm(mpg ~ wt + hp, m, weights=w, wt.method="case"),
      # C(32, 3) < 5000 subsets, so lqs searches all of them: no seed needed.
      mm=rlm(mpg ~ wt + hp, m, method="MM"),
      mm_factors={set.seed(3); rlm(mpg ~ wt + factor(cyl), m, method="MM")},
      mm_weighted=rlm(mpg ~ wt + hp, m, method="MM", weights=w),
      mm_case=rlm(mpg ~ wt + hp, m, method="MM", weights=w, wt.method="case"),
      mm_tuning=rlm(mpg ~ wt + hp, m, method="MM", c=3),
      mm_offset=rlm(mpg ~ wt + offset(hp / 10), m, method="MM"),
      lts={set.seed(3); rlm(mpg ~ wt + hp, m, init="lts")},
      lts_exact=rlm(mpg ~ wt + hp, m, init="lts", lqs.control=list(nsamp="exact")),
      lts_bisquare={set.seed(4); rlm(mpg ~ wt + hp, m, init="lts", psi=psi.bisquare)}
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
      weighted_iid=rq(mpg ~ wt + hp, data=m, weights=w),
      rank=rq(mpg ~ wt + hp, data=m),
      rank_lower=rq(mpg ~ wt + hp, tau=.25, data=m),
      rank_factors=rq(mpg ~ wt + factor(cyl), tau=.6, data=m),
      rank_nid=rq(mpg ~ wt + hp, data=m),
      rank_nid_lower=rq(mpg ~ wt + hp, tau=.25, data=m),
      rank_weighted=rq(mpg ~ wt + hp, data=m, weights=w),
      rank_zero_weights=rq(mpg ~ wt + hp, data=m, weights=w0),
      # quantreg 6.1 ignores offset() terms; stats3 fits the response less
      # the offset, which is this fit with the offset added back below.
      # Rank cases use taus where rqbr reports a unique solution: with n * tau
      # whole, its vertex path (and so its intervals) can depend on the pivoting.
      offset=rq(I(mpg - hp / 10) ~ wt, tau=.6, data=m),
      boot=rq(mpg ~ wt + hp, data=m),
      boot_weighted=rq(mpg ~ wt + hp, tau=.75, data=m, weights=w)
    )
    seeds <- c(boot=1, boot_weighted=2)
    rq_se <- function(name) {
      if (grepl("^rank|^offset|^grouped", name)) "rank"
      else if (grepl("^boot", name)) "boot"
      else if (grepl("iid", name)) "iid"
      else "nid"
    }
    offsets <- function(name, d) if (name == "offset") d$hp / 10 else 0
    # summary.rq(se="boot") draws from R's stream; one draw serves tidy and vcov.
    boot_summary <- function(f, name) { set.seed(seeds[[name]]); summary(f, se="boot", covariance=TRUE) }
    big <- function(v) ifelse(abs(v) >= .Machine$double.xmax, sign(v) * Inf, v) # rq.fit.br's unbounded sides
  }
  vc <- function(f, name) {
    if (kind == "rlm") {
      # vcov.rlm ignores method=; XtWX is an explicit summary.rlm option.
      s <- summary(f, method=if (grepl("xtwx", name)) "XtWX" else "XtX")
      return(s$cov.unscaled * s$stddev^2)
    }
    se <- rq_se(name)
    # summary.rq gives rank fits no covariance; covariance=TRUE uses nid.
    v <- if (se == "boot") boot_summary(f, name)$cov else summary(f, se=if (se == "iid") "iid" else "nid", covariance=TRUE)$cov
    dimnames(v) <- list(names(coef(f)), names(coef(f)))
    v
  }
  tidy_fit <- function(f, name) {
    if (kind == "rlm") {
      co <- summary(f, method=if (grepl("xtwx", name)) "XtWX" else "XtX")$coefficients
      p <- rep(NA_real_, nrow(co)) # summary.rlm does not report p-values.
      q <- qnorm(.975)
    } else {
      se <- rq_se(name)
      if (se == "rank") {
        co <- summary(f, se="rank", alpha=.05, iid=!grepl("nid", name))$coefficients
        return(data.frame(term=names(coef(f)), estimate=unname(coef(f)), std_error=NA_real_,
          statistic=NA_real_, p_value=NA_real_, conf_low=big(unname(co[, 2])), conf_high=big(unname(co[, 3]))))
      }
      co <- if (se == "boot") boot_summary(f, name)$coefficients else summary(f, se=se)$coefficients
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
    off <- if (kind == "rq") offsets(name, m) else 0
    out[[paste0(name, "__augment")]] <- data.frame(.fitted=unname(fitted(f)) + off, .resid=unname(residuals(f)))
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
      x <- model.matrix(delete.response(terms(f)), m[1:5, ])
      se_fit <- sqrt(rowSums((x %*% v) * x))
      if (rq_se(name) == "boot") {
        # predict.rq(se="boot") would draw again; this is its interval from the same draw.
        fit <- drop(x %*% coef(f))
        pred <- cbind(fit, fit - qt(.975, n - p) * se_fit, fit + qt(.975, n - p) * se_fit)
      } else {
        pred <- predict(f, newdata=m[1:5, ], interval="confidence", se=if (rq_se(name) == "iid") "iid" else "nid")
      }
      pr <- data.frame(fit=unname(pred[,1]) + offsets(name, m[1:5, ]), lwr=unname(pred[,2]) + offsets(name, m[1:5, ]),
                       upr=unname(pred[,3]) + offsets(name, m[1:5, ]), se_fit=se_fit)
    }
    out[[paste0(name, "__predict")]] <- pr
  }
  out$grouped <- m |> group_by(am) |>
    group_modify(function(d, k) tidy_fit(if (kind == "rlm") rlm(mpg ~ wt + hp, d) else rq(mpg ~ wt + hp, data=d), "grouped")) |>
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
