multinom_cases <- function() {
  suppressPackageStartupMessages({library(nnet); library(broom)})
  p <- read.csv("tests/data/penguins.csv", na.strings="NA")
  i <- seq_len(nrow(p))
  p$species <- factor(p$species)
  p$sex <- factor(p$sex)
  p$mass <- p$body_mass_g/1000; p$flipper <- p$flipper_length_mm/10; p$mass2 <- p$mass
  p$w <- 1 + i %% 3; p$w[c(2, 30, 160, 250)] <- 0
  p$off <- (i %% 5)/10; p$oa <- 0; p$ob <- p$off; p$oc <- -p$off/2
  gentoo <- p; gentoo$species <- relevel(gentoo$species, "Gentoo")
  specs <- list(
    main=list(formula=species ~ mass + sex),
    flipper=list(formula=species ~ flipper),
    intercept=list(formula=species ~ 1),
    no_intercept=list(formula=species ~ 0 + mass + sex),
    weighted=list(formula=species ~ mass + sex, weights=quote(w)),
    sum_contrasts=list(formula=species ~ mass + sex + factor(year),
                       contrasts=list(sex="contr.sum", `factor(year)`="contr.sum")),
    aliased=list(formula=species ~ mass + mass2 + sex),
    reference=list(formula=species ~ mass + sex, data=gentoo),
    offset_matrix=list(formula=species ~ mass + offset(cbind(oa, ob, oc))),
    binary=list(formula=sex ~ mass + species),
    binary_offset=list(formula=sex ~ mass + species + offset(off)))
  # nnet's BFGS stops at reltol 1e-8 after at most 100 iterations by default;
  # tighten both so the oracle is as close to the MLE as nnet can get.
  fit_one <- function(spec, data=p) {
    if (!is.null(spec$data)) { data <- spec$data; spec$data <- NULL }
    # Hess=TRUE keeps the fit's own model matrix (vcov() alone rebuilds it with
    # default contrasts); nnet's multinomHess cannot take the offset columns.
    hess <- is.null(attr(terms(spec$formula), "offset"))
    do.call(multinom, c(spec, list(data=data, Hess=hess, trace=FALSE, maxit=10000, reltol=1e-15)))
  }
  new <- p[c(1, 2, 3, 5, 6, 200, 300), ]
  # predict.multinom hands nnet's C code the new model matrix without the
  # offset columns the net was fitted with, so for offset fits the new-data
  # probabilities come from the fitted coefficients, as multinom models them.
  predict_new <- function(f, type) {
    if (is.null(attr(f$terms, "offset"))) return(predict(f, new, type=type))
    tt <- delete.response(f$terms); mf <- model.frame(tt, new)
    x <- model.matrix(tt, mf, contrasts.arg=f$contrasts)
    b <- coef(f); if (is.vector(b)) b <- t(b)
    off <- as.matrix(model.offset(mf)); if (ncol(off) == 1) off <- cbind(0, off)
    eta <- cbind(0, x %*% t(b)) + off
    y <- exp(eta - apply(eta, 1, max)); y <- y/rowSums(y)
    if (type == "class") return(factor(f$lev[max.col(y)], levels=f$lev))
    if (length(f$lev) == 2) y[, 2] else y
  }
  tables <- function(f) {
    lev <- f$lev; binary <- length(lev) == 2
    t <- as.data.frame(tidy(f, conf.int=TRUE))
    if (binary) t$y.level <- lev[2]
    names(t) <- gsub(".", "_", names(t), fixed=TRUE)
    v <- vcov(f)
    covariance <- data.frame(term=rownames(v), v, check.names=FALSE, row.names=NULL)
    probs <- function(y) { y <- as.matrix(y); if (binary) y <- cbind(1 - y, y); colnames(y) <- lev; as.data.frame(y, row.names=NULL) }
    classes <- function(y) data.frame(fit=as.character(y))
    list(tidy=t, vcov=covariance,
         probs=probs(predict(f, type="probs")), class=classes(predict(f)),
         new_probs=probs(predict_new(f, "probs")), new_class=classes(predict_new(f, "class")),
         glance=as.data.frame(glance(f)), converged=f$convergence == 0)
  }
  out <- lapply(specs, function(spec) tables(fit_one(spec)))
  grouped <- lapply(sort(unique(p$year)), function(year) {
    t <- tables(fit_one(list(formula=species ~ mass), p[p$year == year, ]))
    list(tidy=cbind(year=year, t$tidy), glance=cbind(year=year, t$glance))
  })
  out$grouped <- list(tidy=do.call(rbind, lapply(grouped, `[[`, "tidy")),
                      glance=do.call(rbind, lapply(grouped, `[[`, "glance")))
  complete <- p[complete.cases(p[c("mass", "sex")]), ]
  small <- fit_one(list(formula=species ~ 1), complete)
  middle <- fit_one(list(formula=species ~ mass), complete)
  big <- fit_one(list(formula=species ~ mass + sex), complete)
  comparison <- anova(big, small, middle)
  out$compare <- data.frame(term=paste("species ~", comparison$Model),
      df_residual=comparison[[2]], residual_deviance=comparison[[3]], df=comparison[[5]],
      statistic=comparison[[6]], p_value=comparison[[7]])
  out
}
