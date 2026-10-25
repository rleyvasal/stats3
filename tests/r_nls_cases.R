# stats's nls, profile confint and anova on R's Puromycin, DNase and ChickWeight.
#
# R's nls takes forward-difference derivatives unless the model returns a
# "gradient" attribute, as its selfStart models do. stats3 differentiates
# the expression symbolically, so these cases give R the exact gradient too,
# from deriv(): both then run the same Gauss-Newton iterations. Most cases
# fit to tol = 1e-7 (near the rounding floor of the criterion), where both stop at the optimum whatever the start
# (selfStart initial values come from inner plinear fits with
# forward differences, which move them by about 1e-8 relative).
nls_cases <- function() {
  P <- read.csv("tests/data/Puromycin.csv")
  P$treated <- as.numeric(P$state == "treated")
  P$w <- rep_len(c(1, 2, 0.5, 1.5), nrow(P)); P$w[5] <- 0
  T <- P[P$state == "treated", ]
  D <- read.csv("tests/data/DNase.csv")
  D1 <- D[D$Run == 1, ]
  CW <- read.csv("tests/data/ChickWeight.csv")
  C1 <- CW[CW$Chick == 1, ]
  C6 <- CW[CW$Chick == 6 & CW$Time > 0, ]
  NA_T <- T; NA_T$rate[3] <- NA
  tight <- nls.control(tol = 1e-7)

  # The model as a function returning deriv()'s exact gradient; a cbind()
  # right-hand side gives plinear's n x columns x parameters array.
  exact <- function(formula, pars) {
    rhs <- formula[[3L]]
    cbound <- is.call(rhs) && identical(rhs[[1L]], as.name("cbind"))
    cols <- if (cbound) as.list(rhs)[-1L] else list(rhs)
    vars <- unique(unlist(lapply(cols, all.vars)))
    model <- function(...) {
      args <- list(...)
      n <- max(lengths(args))
      parts <- lapply(cols, function(col) {
        v <- eval(deriv(col, pars), args)
        g <- attr(v, "gradient")
        list(value = rep_len(as.vector(v), n), gradient = g[rep_len(seq_len(nrow(g)), n), , drop = FALSE])
      })
      if (!cbound) {
        out <- parts[[1L]]$value
        attr(out, "gradient") <- parts[[1L]]$gradient
        return(out)
      }
      out <- sapply(parts, `[[`, "value")
      grad <- array(0, c(n, length(parts), length(pars)))
      for (k in seq_along(parts)) grad[, k, ] <- parts[[k]]$gradient
      attr(out, "gradient") <- grad
      out
    }
    e <- new.env(parent = environment(formula))
    e$.model <- model
    f <- formula
    f[[3L]] <- as.call(c(as.name(".model"), setNames(lapply(vars, as.name), vars)))
    environment(f) <- e
    f
  }

  fits <- list(
    micmen = nls(exact(rate ~ Vm * conc / (K + conc), c("Vm", "K")), T,
                 start = list(Vm = 200, K = 0.1), control = tight),
    rise = nls(exact(density ~ a * (1 - exp(-b * conc)), c("a", "b")), D1,
               start = list(a = 2, b = 0.3), control = tight),
    functions = nls(exact(density ~ a * log1p(conc^p) / (1 + sqrt(conc)), c("a", "p")), D1,
                    start = list(a = 1, p = 1), control = tight),
    weighted = nls(exact(rate ~ Vm * conc / (K + conc), c("Vm", "K")), P,
                   start = list(Vm = 200, K = 0.1), weights = w, control = tight),
    exclude = nls(exact(rate ~ Vm * conc / (K + conc), c("Vm", "K")), NA_T,
                  start = list(Vm = 200, K = 0.1), na.action = na.exclude, control = tight),
    plinear = nls(exact(rate ~ conc / (K + conc), "K"), T, start = list(K = 0.1),
                  algorithm = "plinear", control = tight),
    plinear_cbind = nls(exact(density ~ cbind(1, 1 - exp(-exp(lrc) * conc)), "lrc"), D1,
                        start = list(lrc = -1), algorithm = "plinear", control = tight),
    port = nls(exact(rate ~ Vm * conc / (K + conc), c("Vm", "K")), T, start = list(Vm = 200, K = 0.1),
               algorithm = "port", control = tight),
    port_bounded = nls(exact(rate ~ Vm * conc / (K + conc), c("Vm", "K")), T, start = list(Vm = 200, K = 0.1),
                       algorithm = "port", lower = c(0, 0.07), upper = c(250, 1), control = tight),
    logis = nls(density ~ SSlogis(log(conc), Asym, xmid, scal), D1, control = tight),
    asymp = nls(density ~ SSasymp(conc, Asym, R0, lrc), D1, control = tight),
    micmen_ss = nls(rate ~ SSmicmen(conc, Vm, K), T, control = tight),
    gompertz = nls(density ~ SSgompertz(log(conc), Asym, b2, b3), D1, control = tight),
    fpl = nls(weight ~ SSfpl(Time, A, B, xmid, scal), C1, control = tight),
    weibull = nls(weight ~ SSweibull(Time, Asym, Drop, lrc, pwr), C6, control = tight),
    logis_default = nls(density ~ SSlogis(log(conc), Asym, xmid, scal), D1),
    micmen_default = nls(exact(rate ~ Vm * conc / (K + conc), c("Vm", "K")), T, start = list(Vm = 200, K = 0.1))
  )
  newdata <- list(micmen = T, rise = D1, functions = D1, weighted = P, exclude = T, plinear = T,
                  plinear_cbind = D1, port = T, port_bounded = T, logis = D1, asymp = D1, micmen_ss = T,
                  gompertz = D1, fpl = C1, weibull = C6, logis_default = D1, micmen_default = T)

  tables <- function(f, new) {
    s <- summary(f)$coefficients
    # profile.nls fails on a cbind() plinear model: its gradCall repeats the
    # row index for the n x columns x parameters gradient.
    ci <- tryCatch(suppressMessages(confint(f)), error = function(e) NULL)
    b <- coef(f)
    if (!is.null(ci)) {
      ci <- matrix(ci, ncol = 2)
      if (nrow(ci) < length(b)) ci <- rbind(ci, matrix(NA_real_, length(b) - nrow(ci), 2))  # plinear's .lin
    }
    v <- vcov(f)
    covariance <- data.frame(term = names(b), v, check.names = FALSE)
    names(covariance) <- c("term", names(b))
    port <- identical(f$call$algorithm, "port")
    coefs <- data.frame(term = names(b), estimate = unname(b), std_error = s[, 2], statistic = s[, 3],
                        p_value = s[, 4], row.names = NULL)
    if (!is.null(ci)) coefs <- cbind(coefs, conf_low = ci[, 1], conf_high = ci[, 2])
    list(tidy = coefs,
         vcov = covariance,
         # finTol for port is PORT's internal NREDUC; stats3 does not run PORT.
         glance = data.frame(sigma = summary(f)$sigma, is_conv = f$convInfo$isConv,
                             fin_tol = if (port) NA_real_ else f$convInfo$finTol,
                             logLik = as.numeric(logLik(f)), AIC = AIC(f), BIC = BIC(f),
                             deviance = deviance(f), df_residual = df.residual(f), nobs = nobs(f)),
         augment = data.frame(.fitted = fitted(f), .resid = residuals(f)),
         predict = data.frame(fit = predict(f, head(new, 5))))
  }
  out <- lapply(names(fits), function(name) tables(fits[[name]], newdata[[name]]))
  names(out) <- names(fits)

  grouped <- lapply(c("treated", "untreated"), function(state) {
    t <- tables(nls(rate ~ SSmicmen(conc, Vm, K), P[P$state == state, ], control = tight), P)
    list(tidy = cbind(state = state, t$tidy), glance = cbind(state = state, t$glance))
  })
  out$grouped <- list(tidy = do.call(rbind, lapply(grouped, `[[`, "tidy")),
                      glance = do.call(rbind, lapply(grouped, `[[`, "glance")))

  common <- nls(rate ~ SSmicmen(conc, Vm, K), P, control = tight)
  shifted <- nls(exact(rate ~ (Vm + dV * treated) * conc / (K + conc), c("Vm", "dV", "K")), P,
                 start = list(Vm = 160, dV = 50, K = 0.05), control = tight)
  both <- nls(exact(rate ~ (Vm + dV * treated) * conc / (K + dK * treated + conc), c("Vm", "dV", "K", "dK")), P,
              start = list(Vm = 160, dV = 50, K = 0.05, dK = 0.01), control = tight)
  a <- anova(common, shifted, both)
  out$compare <- data.frame(term = c("rate ~ SSmicmen(conc, Vm, K)",
                                     "rate ~ (Vm + dV * treated) * conc / (K + conc)",
                                     "rate ~ (Vm + dV * treated) * conc / (K + dK * treated + conc)"),
                            df_residual = a[[1]], rss = a[[2]], df = a[[3]], sumsq = a[[4]],
                            statistic = a[[5]], p_value = a[[6]])
  b <- anova(both, common)
  out$compare_down <- data.frame(term = c("rate ~ (Vm + dV * treated) * conc / (K + dK * treated + conc)",
                                          "rate ~ SSmicmen(conc, Vm, K)"),
                                 df_residual = b[[1]], rss = b[[2]], df = b[[3]], sumsq = b[[4]],
                                 statistic = b[[5]], p_value = b[[6]])

  errors <- function(expr) tryCatch({ expr; "none" }, error = function(e) conditionMessage(e))
  out$errors <- data.frame(
    case = c("maxiter", "min_factor", "singular"),
    message = c(errors(nls(exact(rate ~ Vm * conc / (K + conc), c("Vm", "K")), T,
                           start = list(Vm = 200, K = 0.1), control = nls.control(maxiter = 2))),
                errors(nls(exact(density ~ a * (1 - exp(-b * conc)), c("a", "b")), D1,
                           start = list(a = 2, b = 0.3), control = nls.control(minFactor = 0.9, tol = 1e-12))),
                errors(nls(exact(rate ~ (a + b) * conc / (K + conc), c("a", "b", "K")), T,
                           start = list(a = 100, b = 100, K = 0.1)))))
  out
}
