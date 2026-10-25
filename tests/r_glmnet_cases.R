# glmnet supplies the paths and cross-validation; tables are built as
# broom's tidy.glmnet and tidy.cv.glmnet build them, with stats3's names.
glmnet_cases <- function() {
  suppressPackageStartupMessages({library(glmnet); library(survival)})
  m <- read.csv("tests/data/mtcars.csv"); m$model <- NULL
  m$w <- (seq_len(nrow(m)) %% 3) + 1
  m$off <- seq(-.2, .2, length.out=nrow(m))
  m$fold <- rep(1:4, length.out=nrow(m))
  p <- read.csv("tests/data/penguins.csv", na.strings="NA")
  l <- read.csv("tests/data/lung_glmnet.csv")
  # glmnet stops coordinate descent at thresh; the oracle tightens it as far
  # as each family's iterations still converge, so R is near its exact solution.
  tight <- function(family) list(thresh=switch(family, cox=1e-16, multinomial=1e-16, 1e-20), maxit=1e8)
  design <- function(formula, data, weights=NULL) {
    mf <- model.frame(formula, data, na.action=na.omit)
    x <- model.matrix(attr(mf, "terms"), mf)[, -1, drop=FALSE]
    keep <- as.integer(rownames(mf))
    list(x=x, y=model.response(mf), offset=model.offset(mf), keep=keep,
         weights=if (is.null(weights)) NULL else data[[weights]][match(keep, seq_len(nrow(data)))])
  }
  long <- function(b) {
    b <- as.matrix(b)
    data.frame(term=rep(rownames(b), each=ncol(b)), step=rep(seq_len(ncol(b)), times=nrow(b)),
               estimate=as.vector(t(b)))
  }
  tidy_fit <- function(f, zeros=FALSE) {
    b <- coef(f)
    out <- if (is.list(b)) do.call(rbind, lapply(names(b), function(k) cbind(class=k, long(b[[k]]))))
           else long(b)
    out$lambda <- f$lambda[out$step]; out$dev_ratio <- f$dev.ratio[out$step]
    if (!zeros) out <- out[out$estimate != 0, ]
    rownames(out) <- NULL
    out
  }
  coef_table <- function(f, s) {
    b <- coef(f, s=s)
    one <- function(b) { b <- as.matrix(b); data.frame(term=rownames(b), b, check.names=FALSE, row.names=NULL) }
    if (is.list(b)) do.call(rbind, lapply(names(b), function(k) cbind(class=k, one(b[[k]])))) else one(b)
  }
  nonzero <- function(f, s) {
    nz <- predict(f, type="nonzero", s=s)
    names(nz) <- if (inherits(f, "coxnet")) seq_along(s) else paste0("s=", format(s))
    rows <- lapply(names(nz), function(k) if (length(nz[[k]])) data.frame(s=k, term=rownames(f$beta)[nz[[k]]]))
    do.call(rbind, rows)
  }
  run <- function(spec) {
    d <- design(spec$formula, spec$data, spec$weights)
    args <- c(list(x=d$x, y=d$y, weights=d$weights, offset=d$offset, family=spec$family,
                   control=if (isTRUE(spec$default)) list() else tight(spec$family)), spec$args)
    if (spec$family == "cox") args$cox.ties <- if (is.null(spec$ties)) "breslow" else spec$ties
    f <- suppressWarnings(do.call(glmnet, args))
    s <- spec$s
    new <- d$x[1:5, , drop=FALSE]
    newoff <- if (is.null(d$offset)) NULL else d$offset[1:5]
    pr <- function(type) {
      v <- if (is.null(newoff)) predict(f, new, s=s, type=type) else predict(f, new, s=s, type=type, newoffset=newoff)
      if (length(dim(v)) == 3) {
        v <- v[, , 1]
        as.data.frame(v)
      } else {
        as.data.frame(v)
      }
    }
    out <- list(tidy=tidy_fit(f), tidy_zeros=tidy_fit(f, TRUE),
                glance=data.frame(nulldev=f$nulldev, nobs=f$nobs),
                coef=coef_table(f, s), link=pr("link"))
    if (spec$family != "gaussian") out$response <- pr("response")
    if (spec$family %in% c("binomial", "multinomial")) out$class <- pr("class")
    if (spec$family != "multinomial") out$nonzero <- nonzero(f, s)
    out
  }
  cars <- mpg ~ cyl + disp + hp + drat + wt + qsec + vs + am + gear + carb
  pen <- species ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g
  lungf <- Surv(time, status) ~ age + sex + ph_ecog + ph_karno + wt_loss
  specs <- list(
    gaussian=list(formula=cars, data=m, family="gaussian", s=c(1, .1)),
    elastic=list(formula=cars, data=m, family="gaussian", args=list(alpha=.5), s=.3),
    ridge=list(formula=cars, data=m, family="gaussian", args=list(alpha=0), s=2),
    weighted=list(formula=mpg ~ wt + hp + disp + qsec + offset(off), data=m, family="gaussian", weights="w", s=.2),
    options=list(formula=cars, data=m, family="gaussian", s=.5,
                 args=list(standardize=FALSE, intercept=FALSE, penalty.factor=c(0, rep(1, 9)),
                           lower.limits=-1, upper.limits=2, nlambda=30, lambda.min.ratio=.001)),
    nonnegative=list(formula=cars, data=m, family="gaussian", args=list(lower.limits=0), s=.5),
    lambda=list(formula=cars, data=m, family="gaussian", args=list(lambda=c(.01, 2, .5, .1)), s=.3),
    factors=list(formula=body_mass_g ~ species + island + sex + bill_length_mm + flipper_length_mm,
                 data=p, family="gaussian", s=10),
    binomial=list(formula=am ~ wt + hp + disp + qsec + drat, data=m, family="binomial", s=.05),
    binomial_offset=list(formula=vs ~ wt + hp + disp + qsec + offset(off), data=m, family="binomial",
                         weights="w", args=list(alpha=.7), s=.02),
    poisson=list(formula=carb ~ cyl + disp + hp + drat + wt + qsec + offset(off), data=m, family="poisson",
                 weights="w", s=.05),
    multinomial=list(formula=pen, data=p, family="multinomial", s=.01),
    cox=list(formula=lungf, data=l, family="cox", s=.05),
    efron=list(formula=lungf, data=l, family="cox", ties="efron", args=list(alpha=.5), s=.05),
    default_gaussian=list(formula=cars, data=m, family="gaussian", default=TRUE, s=.1),
    default_binomial=list(formula=am ~ wt + hp + disp + qsec + drat, data=m, family="binomial", default=TRUE, s=.05)
  )
  out <- lapply(specs, run)
  grouped <- lapply(c(0, 1), function(g) {
    r <- run(list(formula=mpg ~ wt + hp + disp + qsec + drat, data=m[m$am == g, ], family="gaussian", s=.5))
    list(tidy=cbind(am=g, r$tidy), glance=cbind(am=g, r$glance))
  })
  out$grouped <- list(tidy=do.call(rbind, lapply(grouped, `[[`, "tidy")),
                      glance=do.call(rbind, lapply(grouped, `[[`, "glance")))
  cv <- function(spec) {
    d <- design(spec$formula, spec$data, spec$weights)
    fold <- spec$data$fold[d$keep]
    args <- c(list(x=d$x, y=d$y, weights=d$weights, offset=d$offset, family=spec$family, foldid=fold,
                   type.measure=spec$measure, control=tight(spec$family)), spec$args)
    if (spec$family == "cox") args$cox.ties <- "breslow"
    f <- suppressWarnings(do.call(cv.glmnet, args))
    list(tidy=data.frame(lambda=f$lambda, estimate=f$cvm, std_error=f$cvsd, conf_low=f$cvlo,
                         conf_high=f$cvup, nzero=unname(f$nzero)),
         glance=data.frame(lambda_min=f$lambda.min, lambda_1se=f$lambda.1se, nobs=f$glmnet.fit$nobs),
         coef=coef_table(f$glmnet.fit, c(lambda.1se=f$lambda.1se)))
  }
  p$fold <- rep(1:5, length.out=nrow(p))
  l$fold <- rep(1:5, length.out=nrow(l))
  cvspecs <- list(
    mse=list(formula=cars, data=m, family="gaussian", measure="mse"),
    mae=list(formula=cars, data=m, family="gaussian", measure="mae", args=list(alpha=.5)),
    ungrouped=list(formula=cars, data=m, family="gaussian", measure="mse", args=list(grouped=FALSE)),
    deviance=list(formula=am ~ wt + hp + disp + qsec + drat, data=m, family="binomial", measure="deviance"),
    auc=list(formula=sex ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g, data=p,
             family="binomial", measure="auc"),
    class=list(formula=sex ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g, data=p,
               family="binomial", measure="class"),
    poisson=list(formula=carb ~ cyl + disp + hp + drat + wt + qsec + offset(off), data=m, family="poisson",
                 measure="deviance", weights="w"),
    multinomial=list(formula=pen, data=p, family="multinomial", measure="deviance"),
    multinomial_class=list(formula=pen, data=p, family="multinomial", measure="class"),
    cox=list(formula=lungf, data=l, family="cox", measure="deviance"),
    cindex=list(formula=lungf, data=l, family="cox", measure="C")
  )
  out$cv <- lapply(cvspecs, cv)
  out
}
