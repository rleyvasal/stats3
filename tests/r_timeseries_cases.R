timeseries_cases <- function() {
  d <- read.csv("tests/data/timeseries.csv")
  s <- function(name) d$value[d$dataset == name]
  lh <- s("lh"); lk <- s("LakeHuron"); us <- s("USAccDeaths"); ap <- s("AirPassengers")
  with_na <- function(x, at) { x[at] <- NA; x }
  lk_na <- with_na(lk, c(10, 40, 41))
  datasets <- c("AirPassengers", "LakeHuron", "USAccDeaths", "lh")  # dplyr's order
  bounds <- function(a, ci_type="white", conf=.95) {
    v <- drop(a$acf); w <- qnorm((1 + conf)/2)/sqrt(a$n.used)
    b <- if (a$type == "covariance") rep(NA_real_, length(v))
         else if (ci_type == "white") rep(w, length(v))
         else c(NA, w*sqrt(cumsum(c(1, 2*v[-c(1, length(v))]^2))))
    b
  }
  at <- function(a, ci_type="white", freq=1) {
    b <- bounds(a, ci_type)
    data.frame(lag=round(drop(a$lag)*freq), acf=drop(a$acf), conf_low=-b, conf_high=b)
  }
  grouped <- function(f) do.call(rbind, lapply(datasets, function(g) cbind(dataset=g, f(s(g)))))
  acfs <- list(
    lh_cor=at(acf(lh, plot=FALSE)),
    lh_cov=at(acf(lh, lag.max=5, type="covariance", plot=FALSE)),
    lh_partial=at(pacf(lh, plot=FALSE)),
    lh_partial_type=at(acf(lh, lag.max=8, type="partial", plot=FALSE)),
    lk_ma=at(acf(lk, plot=FALSE), "ma"),
    us_nodemean=at(acf(us, lag.max=30, demean=FALSE, plot=FALSE)),
    lk_na_pass=at(acf(lk_na, na.action=na.pass, plot=FALSE)),
    lk_na_pass_partial=at(pacf(lk_na, na.action=na.pass, plot=FALSE)),
    lk_na_omit=at(acf(with_na(lk, c(1, 2, 98)), na.action=na.omit, plot=FALSE)),
    lk_contiguous=at(acf(with_na(lk, c(10, 50)), na.action=na.contiguous, plot=FALSE)),
    grouped=grouped(function(x) at(acf(x, plot=FALSE))),
    grouped_partial=grouped(function(x) at(pacf(x, lag.max=6, plot=FALSE))),
    ccf=at(ccf(us, ap[1:72], plot=FALSE)),
    ccf_cov=at(ccf(us, ap[1:72], lag.max=6, type="covariance", plot=FALSE)),
    ccf_grouped=rbind(cbind(g="a", at(ccf(us, ap[1:72], lag.max=4, plot=FALSE))),
                      cbind(g="b", at(ccf(us, ap[73:144], lag.max=4, plot=FALSE)))))
  box <- function(b) data.frame(statistic=unname(b$statistic), df=unname(b$parameter),
                                p_value=b$p.value, method=b$method)
  lh3 <- arima(lh, c(3, 0, 0))
  boxes <- list(
    lh_bp=box(Box.test(lh)),
    lk_lb=box(Box.test(lk, lag=10, type="Ljung-Box", fitdf=2)),
    lk_na=box(Box.test(lk_na, lag=5, type="Ljung-Box")),
    grouped=grouped(function(x) box(Box.test(x, lag=6, type="Ljung-Box"))),
    residuals=box(Box.test(residuals(lh3), lag=8, type="Ljung-Box", fitdf=3)))
  tables <- function(f, n_ahead=3, newxreg=NULL) {
    b <- coef(f); se <- rep(NA_real_, length(b)); se[f$mask] <- sqrt(diag(f$var.coef))
    k <- sum(f$mask) + 1; n <- f$nobs; css <- is.na(f$aic)
    v <- f$var.coef; covariance <- data.frame(term=rownames(v), v, check.names=FALSE)
    p <- predict(f, n.ahead=n_ahead, newxreg=newxreg)
    list(tidy=data.frame(term=names(b), estimate=unname(b), std_error=se, statistic=unname(b)/se,
                         p_value=NA_real_, conf_low=unname(b) - qnorm(.975)*se,
                         conf_high=unname(b) + qnorm(.975)*se),
         glance=data.frame(sigma2=f$sigma2, logLik=f$loglik, AIC=if (css) NA_real_ else f$aic,
                           AICc=if (css) NA_real_ else f$aic + 2*k*(k + 1)/(n - k - 1),
                           BIC=if (css) NA_real_ else -2*f$loglik + log(n)*k, nobs=n, converged=f$code == 0),
         vcov=covariance,
         predict=data.frame(horizon=seq_len(n_ahead), fit=as.numeric(p$pred), se_fit=as.numeric(p$se)),
         resid=data.frame(.resid=as.numeric(residuals(f))))
  }
  yr <- seq_along(lk) + 1874 - 1920
  fits <- list(
    lh100=tables(arima(lh, c(1, 0, 0))),
    lh300=tables(lh3, 12),
    lh101_ml=tables(arima(lh, c(1, 0, 1), method="ML")),
    lh300_css=tables(arima(lh, c(3, 0, 0), method="CSS")),
    lh_fixed=tables(arima(lh, c(3, 0, 0), fixed=c(NA, 0, NA, NA), transform.pars=FALSE)),
    lh_notrans=tables(arima(lh, c(1, 0, 1), method="ML", transform.pars=FALSE)),
    lh_nomean=tables(arima(lh, c(1, 0, 0), include.mean=FALSE)),
    lh_ncond=tables(arima(lh, c(1, 0, 0), method="CSS", n.cond=3)),
    lk_xreg=tables(arima(lk, c(2, 0, 0), xreg=yr), 5, max(yr) + 1:5),
    lk_101=tables(arima(lk, c(1, 0, 1))),
    lk_na=tables(arima(lk_na, c(1, 0, 1))),
    lk_110=tables(arima(lk, c(1, 1, 0))),
    us=tables(arima(us, c(0, 1, 1), seasonal=list(order=c(0, 1, 1), period=12)), 12),
    us_sar=tables(arima(us, c(1, 1, 0), seasonal=list(order=c(1, 1, 0), period=12)), 12),
    ap=tables(arima(log(ap), c(0, 1, 1), seasonal=list(order=c(0, 1, 1), period=12)), 12))
  per_group <- lapply(datasets, function(g) tables(arima(s(g), c(0, 1, 1))))
  stack <- function(part) do.call(rbind, lapply(seq_along(datasets), function(i)
    cbind(dataset=datasets[i], per_group[[i]][[part]])))
  fits$grouped <- list(tidy=stack("tidy"), glance=stack("glance"), predict=stack("predict"))
  list(acf=acfs, box=boxes, arima=fits)
}
