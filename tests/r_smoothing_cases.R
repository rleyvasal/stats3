# Base R's smoothers; adaptations only name and align their results.
smoothing_cases <- function() {
  cars <- read.csv("tests/data/cars.csv")
  cars$w <- 1 + seq_len(nrow(cars)) %% 3
  m <- read.csv("tests/data/mtcars.csv")
  p <- read.csv("tests/data/penguins.csv")
  p2 <- p[!is.na(p$flipper_length_mm) & !is.na(p$body_mass_g), ]
  pb <- p2[!is.na(p2$bill_length_mm), ]
  out <- list()

  fits <- list(
    cars_default = loess(dist ~ speed, cars),
    cars_degree1 = loess(dist ~ speed, cars, span=.5, degree=1),
    cars_degree0 = loess(dist ~ speed, cars, span=.6, degree=0),
    cars_direct = loess(dist ~ speed, cars, control=loess.control(surface="direct")),
    cars_symmetric = loess(dist ~ speed, cars, family="symmetric"),
    cars_exact = loess(dist ~ speed, cars, control=loess.control(statistics="exact")),
    cars_trace_approx = loess(dist ~ speed, cars, control=loess.control(trace.hat="approximate")),
    cars_direct_symmetric = loess(dist ~ speed, cars, family="symmetric", control=loess.control(surface="direct")),
    cars_weighted = loess(dist ~ speed, cars, weights=w),
    cars_cell = loess(dist ~ speed, cars, span=.9, control=loess.control(cell=.1, iterations=2), family="symmetric"),
    cars_enp = loess(dist ~ speed, cars, enp.target=6),
    mtcars_2d = loess(mpg ~ wt + hp, m),
    mtcars_2d_symmetric = loess(mpg ~ wt + hp, m, family="symmetric"),
    mtcars_2d_raw = loess(mpg ~ wt * hp, m, normalize=FALSE, span=.9),
    mtcars_2d_direct = loess(mpg ~ wt + hp, m, control=loess.control(surface="direct")),
    mtcars_2d_exact = loess(mpg ~ wt + hp, m, degree=1, control=loess.control(statistics="exact")),
    mtcars_3d = suppressWarnings(loess(mpg ~ wt + hp + disp, m, span=1)),
    penguins_1d = loess(body_mass_g ~ flipper_length_mm, p),
    penguins_2d = loess(body_mass_g ~ flipper_length_mm + bill_length_mm, p, span=.3)
  )
  new <- list(cars=data.frame(speed=c(3, 4, 10.5, 25, 26)),
              mtcars=data.frame(wt=c(1.5, 2.62, 3.44, 5.4, 6), hp=c(60, 110, 175, 230, 100), disp=c(100, 160, 360, 400, 300)),
              penguins=data.frame(flipper_length_mm=c(171, 190.5, 210, 231, 240), bill_length_mm=c(40, 39.1, 45, 50, 59.6)))
  for (name in names(fits)) {
    f <- fits[[name]]
    out[[paste0(name, "__glance")]] <- data.frame(nobs=f$n, enp=f$enp, sigma=f$s, trace_hat=f$trace.hat,
                                                  one_delta=f$one.delta, two_delta=f$two.delta)
    out[[paste0(name, "__augment")]] <- data.frame(.fitted=unname(fitted(f)), .resid=unname(residuals(f)))
    out[[paste0(name, "__robust")]] <- data.frame(weight=f$robust)
    nd <- new[[sub("_.*", "", name)]]
    pr <- predict(f, nd, se=TRUE)
    out[[paste0(name, "__predict")]] <- data.frame(fit=unname(pr$fit), se_fit=unname(pr$se.fit),
                                                   residual_scale=pr$residual.scale, df=pr$df)
    out[[paste0(name, "__se")]] <- data.frame(se_fit=unname(predict(f, se=TRUE)$se.fit))
  }
  big <- fits$cars_default
  small <- loess(dist ~ speed, cars, span=.4)
  a <- anova(small, big)
  dfnum <- abs(diff(c(small$one.delta, big$one.delta)))^2 / abs(diff(c(small$two.delta, big$two.delta)))
  out$anova <- data.frame(term=c("dist ~ speed", "dist ~ speed"), enp=c(small$enp, big$enp), rss=a$RSS,
                          statistic=a$`F-value`, p_value=a$`Pr(>F)`, df=c(NA, dfnum),
                          df_residual=small$one.delta^2 / small$two.delta)
  out$loess_grouped <- p2 |> group_by(species) |> group_modify(function(d, k) {
    f <- loess(body_mass_g ~ flipper_length_mm, d)
    data.frame(nobs=f$n, enp=f$enp, sigma=f$s, trace_hat=f$trace.hat, one_delta=f$one.delta, two_delta=f$two.delta)
  }) |> ungroup() |> as.data.frame()

  lw <- function(l) data.frame(x=l$x, y=l$y)
  out$lowess_default <- lw(lowess(cars$speed, cars$dist))
  out$lowess_tight <- lw(lowess(cars$speed, cars$dist, f=.3, iter=0))
  out$lowess_delta <- lw(lowess(cars$speed, cars$dist, f=.5, delta=0))
  out$lowess_mtcars <- lw(lowess(m$wt, m$mpg, f=.4, iter=5, delta=.3))
  out$lowess_penguins <- lw(lowess(p2$flipper_length_mm, p2$body_mass_g))
  out$lowess_grouped <- p2 |> group_by(species) |>
    group_modify(function(d, k) lw(lowess(d$flipper_length_mm, d$body_mass_g, f=.5))) |> ungroup() |> as.data.frame()

  out$ksmooth_box <- lw(ksmooth(cars$speed, cars$dist))
  out$ksmooth_normal <- lw(ksmooth(cars$speed, cars$dist, "normal", bandwidth=2))
  out$ksmooth_points <- lw(ksmooth(cars$speed, cars$dist, "box", bandwidth=3, x.points=c(30, 2, 10.5, 15, 1)))
  out$ksmooth_range <- lw(ksmooth(m$wt, m$mpg, "normal", bandwidth=.5, range.x=c(1, 6), n.points=40))
  out$ksmooth_grouped <- p2 |> group_by(species) |>
    group_modify(function(d, k) lw(ksmooth(d$flipper_length_mm, d$body_mass_g, "normal", bandwidth=5, n.points=20))) |>
    ungroup() |> as.data.frame()

  splines <- list(
    cars_gcv = smooth.spline(cars$speed, cars$dist),
    cars_cv = suppressWarnings(smooth.spline(cars$speed, cars$dist, cv=TRUE, spar=.6)),
    cars_df = smooth.spline(cars$speed, cars$dist, df=5),
    cars_spar = smooth.spline(cars$speed, cars$dist, spar=.5),
    cars_lambda = smooth.spline(cars$speed, cars$dist, lambda=1e-3),
    cars_none = smooth.spline(cars$speed, cars$dist, spar=.7, cv=NA),
    cars_weighted = smooth.spline(cars$speed, cars$dist, w=cars$w),
    cars_all_knots = smooth.spline(cars$speed, cars$dist, all.knots=TRUE, df=8),
    cars_nknots = smooth.spline(cars$speed, cars$dist, nknots=10, spar=.3),
    cars_penalty = smooth.spline(cars$speed, cars$dist, penalty=1.4, df.offset=1),
    mtcars_gcv = smooth.spline(m$wt, m$mpg),
    mtcars_cv = suppressWarnings(smooth.spline(m$hp, m$mpg, cv=TRUE)),
    penguins_gcv = smooth.spline(p2$flipper_length_mm, p2$body_mass_g),
    penguins_df = smooth.spline(p2$bill_length_mm[!is.na(p2$bill_length_mm)], p2$body_mass_g[!is.na(p2$bill_length_mm)], df=6)
  )
  at <- list(cars=c(2, 4, 10.5, 17, 25, 28), mtcars=c(1, 1.513, 2.5, 3.44, 5.424, 6), penguins=c(150, 172, 190.5, 205, 231, 240))
  for (name in names(splines)) {
    s <- splines[[name]]
    out[[paste0("ss_", name, "__glance")]] <- data.frame(df=s$df, spar=s$spar, lambda=s$lambda, cv_crit=s$cv.crit,
                                                  pen_crit=s$pen.crit, crit=if (is.na(s$cv)) NA else s$crit, nobs=s$n)
    out[[paste0("ss_", name, "__augment")]] <- data.frame(.fitted=fitted(s), .resid=residuals(s))
    out[[paste0("ss_", name, "__fit")]] <- data.frame(x=s$x, y=s$y)
    xs <- if (name == "penguins_df") c(30, 32.1, 45, 59.6, 62) else at[[sub("_.*", "", name)]]
    if (name == "mtcars_cv") xs <- c(40, 52, 150, 335, 360)
    for (d in 0:3) {
      pr <- predict(s, xs, deriv=d)
      out[[paste0("ss_", name, "__deriv", d)]] <- data.frame(x=pr$x, y=pr$y)
    }
  }
  out$spline_grouped <- p2 |> group_by(species) |> group_modify(function(d, k) {
    s <- smooth.spline(d$flipper_length_mm, d$body_mass_g, df=4)
    data.frame(df=s$df, spar=s$spar, lambda=s$lambda, cv_crit=s$cv.crit, pen_crit=s$pen.crit, crit=s$crit, nobs=s$n)
  }) |> ungroup() |> as.data.frame()
  out
}
