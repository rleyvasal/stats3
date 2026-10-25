# Multivariate linear models: lm with a matrix response, summary.manova,
# anova.mlm (and its model comparisons) and mauchly.test.
mlm_cases <- function() {
  p <- read.csv("tests/data/penguins.csv", na.strings="NA")
  i <- read.csv("tests/data/iris.csv")
  names(i) <- c("sepal_length", "sepal_width", "petal_length", "petal_width", "species")
  p$w <- (seq_len(nrow(p)) %% 5) + 1
  formulas <- list(
    iris=list(cbind(sepal_length, sepal_width, petal_length, petal_width) ~ species, i),
    numeric=list(cbind(petal_length, petal_width) ~ sepal_length + sepal_width, i),
    penguins=list(cbind(bill_length_mm, bill_depth_mm, flipper_length_mm) ~ species + sex + body_mass_g, p),
    interaction=list(cbind(bill_length_mm, bill_depth_mm) ~ species * sex, p),
    expressions=list(cbind(log(bill_length_mm), bill_depth_mm / 10) ~ island + year, p),
    no_intercept=list(cbind(bill_length_mm, bill_depth_mm, flipper_length_mm) ~ 0 + species, p),
    aliased=list(cbind(bill_length_mm, bill_depth_mm) ~ body_mass_g + I(2 * body_mass_g) + species, p)
  )
  tests <- c(pillai="Pillai", wilks="Wilks", hl="Hotelling-Lawley", roy="Roy")
  test_table <- function(stats, name, terms) {
    out <- data.frame(term=terms, df=stats[, 1])
    out[[name]] <- stats[, 2]
    out$statistic <- stats[, 3]; out$num_df <- stats[, 4]; out$den_df <- stats[, 5]; out$p_value <- stats[, 6]
    out
  }
  manova_table <- function(f, test, intercept=FALSE) {
    s <- summary(f, test=tests[[test]], intercept=intercept)$stats
    test_table(unclass(s), test, rownames(s))
  }
  anova_table <- function(a, test) test_table(as.matrix(a), test, rownames(a))
  spherical <- function(a, sph) data.frame(term=rownames(a), df=a$Df, statistic=a$F, num_df=a$`num Df`,
    den_df=a$`den Df`, p_value=a$`Pr(>F)`, gg_p_value=a$`G-G Pr`, hf_p_value=a$`H-F Pr`,
    gg_epsilon=sph$GG.eps, hf_epsilon=sph$HF.eps)
  tidy_fit <- function(f) {
    s <- summary(f)
    do.call(rbind, lapply(seq_along(s), function(k) {
      co <- coef(s[[k]])
      r <- sub("Response ", "", names(s)[k])
      terms <- rownames(coef(f))
      j <- match(terms, rownames(co))
      # Intervals from the t quantile: R's confint.mlm repeats the first
      # response when the responses are unnamed expressions.
      q <- qt(0.975, f$df.residual)
      data.frame(response=r, term=terms, estimate=coef(f)[, k], std_error=co[j, 2],
                 statistic=co[j, 3], p_value=co[j, 4], conf_low=coef(f)[, k] - q * co[j, 2],
                 conf_high=coef(f)[, k] + q * co[j, 2], row.names=NULL)
    }))
  }
  out <- list()
  fits <- list()
  for (name in names(formulas)) {
    fm <- formulas[[name]][[1]]; d <- formulas[[name]][[2]]
    f <- lm(fm, d)
    fits[[name]] <- f
    m <- manova(fm, d)
    out[[paste0(name, "__tidy")]] <- tidy_fit(f)
    for (test in names(tests)) {
      out[[paste0(name, "__manova_", test)]] <- manova_table(m, test)
      out[[paste0(name, "__anova_", test)]] <- anova_table(anova(f, test=tests[[test]]), test)
    }
  }
  fi <- fits$iris
  out$manova_intercept <- manova_table(manova(formulas$iris[[1]], i), "wilks", intercept=TRUE)
  out$manova_weighted <- manova_table(manova(formulas$penguins[[1]], p, weights=w), "pillai")
  out$by_year_manova <- p |> group_by(year) |> group_modify(function(d, k)
    manova_table(manova(cbind(bill_length_mm, bill_depth_mm, flipper_length_mm) ~ species + sex, d), "pillai")) |>
    ungroup() |> as.data.frame()
  by_year_fit <- function(d) lm(cbind(bill_length_mm, bill_depth_mm, flipper_length_mm) ~ species + sex, d)
  out$by_year_tidy <- p |> group_by(year) |> group_modify(function(d, k) tidy_fit(by_year_fit(d))) |>
    ungroup() |> as.data.frame()
  out$by_year_anova <- p |> group_by(year) |> group_modify(function(d, k)
    anova_table(anova(by_year_fit(d), test="Wilks"), "wilks")) |> ungroup() |> as.data.frame()
  out$by_year_mauchly <- p |> group_by(year) |> group_modify(function(d, k) {
    t <- mauchly.test(by_year_fit(d), X=~1)
    data.frame(statistic=unname(t$statistic), p_value=t$p.value, method=t$method[1])
  }) |> ungroup() |> as.data.frame()
  out$by_year_spherical <- p |> group_by(year) |> group_modify(function(d, k) {
    f <- by_year_fit(d)
    spherical(anova(f, test="Spherical", X=~1), stats:::sphericity(SSD(f), X=~1))
  }) |> ungroup() |> as.data.frame()
  # Sepal and petal, length and width: a 2 x 2 within-flower design.
  idata <- data.frame(part=factor(c("sepal", "sepal", "petal", "petal")),
                      dim=factor(c("length", "width", "length", "width")))
  designs <- list(
    identity=list(),
    within=list(X=~1),
    part=list(X=~part, idata=idata),
    dim=list(M=~part * dim, X=~dim, idata=idata),
    interaction=list(M=~part * dim, X=~part + dim, idata=idata),
    one_row=list(M=~part + dim, X=~part, idata=idata),
    matrix=list(M=cbind(1, c(1, 1, -1, -1), c(1, -1, 1, -1)), X=cbind(rep(1, 4)))
  )
  for (name in names(designs)) {
    args <- designs[[name]]
    t <- do.call(mauchly.test, c(list(fi), args))
    out[[paste0("mauchly__", name)]] <- data.frame(statistic=unname(t$statistic), p_value=t$p.value,
                                                   method=t$method[1])
    sph <- do.call(stats:::sphericity, c(list(SSD(fi)), args))
    out[[paste0("spherical__", name)]] <- spherical(do.call(anova, c(list(fi, test="Spherical"), args)), sph)
    # With a one-row contrast R's scaling step fails (sapply drops to a
    # vector), so those designs have no multivariate oracle.
    if (!name %in% c("interaction", "one_row"))
      out[[paste0("transformed__", name)]] <- anova_table(do.call(anova, c(list(fi, test="Pillai"), args)), "pillai")
  }
  small <- lm(cbind(sepal_length, sepal_width, petal_length, petal_width) ~ 1, i)
  model_text <- c("cbind(sepal_length, sepal_width, petal_length, petal_width) ~ 1",
                  "cbind(sepal_length, sepal_width, petal_length, petal_width) ~ species")
  for (test in names(tests)) {
    a <- anova(small, fi, test=tests[[test]])
    tab <- data.frame(term=model_text, df_residual=a$Res.Df, df=a$Df, gen_var=a$Gen.var.)
    tab[[test]] <- a[[tests[[test]]]]
    tab$statistic <- a$`approx F`; tab$num_df <- a$`num Df`; tab$den_df <- a$`den Df`; tab$p_value <- a$`Pr(>F)`
    out[[paste0("compare_", test)]] <- tab
  }
  a <- anova(fi, small)
  out$compare_reversed <- data.frame(term=rev(model_text), df_residual=a$Res.Df, df=a$Df, gen_var=a$Gen.var.,
    pillai=a$Pillai, statistic=a$`approx F`, num_df=a$`num Df`, den_df=a$`den Df`, p_value=a$`Pr(>F)`)
  a <- anova(small, fi, test="Spherical", X=~1)
  sph <- stats:::sphericity(SSD(fi), X=~1)
  out$compare_spherical <- data.frame(term=model_text, df_residual=a$Res.Df, df=a$Df, gen_var=a$Gen.var.,
    statistic=a$F, num_df=a$`num Df`, den_df=a$`den Df`, p_value=a$`Pr(>F)`, gg_p_value=a$`G-G Pr`,
    hf_p_value=a$`H-F Pr`, gg_epsilon=sph$GG.eps, hf_epsilon=sph$HF.eps)
  fp <- fits$penguins
  new <- p[c(1, 2, 3, 5, 6), ]
  pr <- predict(fp, new)
  out$predict <- data.frame(pr, check.names=FALSE, row.names=NULL)
  v <- vcov(fits$aliased)
  out$vcov <- data.frame(term=rownames(v), v, check.names=FALSE, row.names=NULL)
  out
}
