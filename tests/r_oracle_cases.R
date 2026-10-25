# R oracle for stats3: each case runs R's own tests and prints the results
# as JSON for tests/oracle.py. Run from the repository root:
#   Rscript tests/r_oracle_cases.R stats_tier1
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(jsonlite))

column_type <- function(x) {
  if (is.factor(x)) return("factor")
  if (inherits(x, "Date")) return("date")
  if (inherits(x, "POSIXt")) return("datetime")
  if (is.list(x)) return("list")
  if (is.logical(x)) return("logical")
  if (is.integer(x)) return("integer")
  if (is.numeric(x)) return("double")
  if (is.character(x)) return("character")
  class(x)[[1]]
}

normalise_value <- function(x) {
  if (is.data.frame(x)) return(normalise(x))
  if (inherits(x, "Date")) return(format(x, "%Y-%m-%d"))
  if (inherits(x, "POSIXt")) return(format(x, "%Y-%m-%dT%H:%M:%S", tz="UTC"))
  if (is.factor(x)) return(as.character(x))
  if (is.list(x)) return(lapply(x, normalise_value))
  x
}

normalise <- function(x) {
  if (is.data.frame(x)) {
    return(list(
      kind="dataframe",
      columns=unname(names(x)),
      types=unname(vapply(x, column_type, character(1))),
      data=unname(lapply(x, function(column) {
        cells <- lapply(as.list(column), normalise_value)
        # Keep list-column cells as arrays: auto_unbox would turn list(3)
        # into a bare 3.
        if (is.list(column)) {
          cells <- lapply(cells, function(v) if (is.atomic(v) && !is.null(v)) I(v) else v)
        }
        unname(cells)
      }))
    ))
  }
  if (is.list(x)) return(unname_or_names(x))
  if (length(x) == 1) return(normalise_value(x))
  list(kind="vector", data=unname(lapply(as.list(x), normalise_value)))
}

unname_or_names <- function(x) {
  values <- lapply(x, normalise)
  if (is.null(names(x))) unname(values) else values
}

case_name <- commandArgs(trailingOnly=TRUE)[[1]]

result <- switch(
  case_name,
  t_test = {
    p <- read.csv("tests/data/penguins.csv", na.strings="NA")
    # One row per test, in tidy3's column names. jsonlite writes the
    # infinite bound of a one-sided interval as null.
    row <- function(h) {
      estimates <- unname(h$estimate)
      out <- data.frame(estimate=if (length(estimates) == 2) estimates[1] - estimates[2] else estimates)
      if (length(estimates) == 2) {
        out$estimate1 <- estimates[1]
        out$estimate2 <- estimates[2]
      }
      out$statistic <- unname(h$statistic)
      out$df <- unname(h$parameter)
      out$p_value <- h$p.value
      out$conf_low <- h$conf.int[1]
      out$conf_high <- h$conf.int[2]
      out$method <- trimws(h$method)
      out$alternative <- h$alternative
      out
    }
    by <- function(data, groups, test) {
      data |> group_by(across(all_of(groups))) |>
        group_modify(function(d, k) row(test(d))) |> ungroup() |> as.data.frame()
    }
    welch <- function(d) t.test(body_mass_g ~ sex, data=d)
    student <- function(d) t.test(body_mass_g ~ sex, data=d, var.equal=TRUE)
    paired <- function(d) t.test(d$bill_length_mm, d$bill_depth_mm, paired=TRUE)
    one <- function(d) t.test(d$flipper_length_mm, mu=200)
    list(
      welch=row(welch(p)),
      student=row(student(p)),
      paired=row(paired(p)),
      one_sample=row(one(p)),
      one_sample_formula=row(t.test(flipper_length_mm ~ 1, data=p, mu=200)),
      two_columns=row(t.test(p$bill_length_mm, p$bill_depth_mm)),
      welch_mu_less=row(t.test(body_mass_g ~ sex, data=p, mu=-500, alternative="less", conf.level=0.9)),
      student_greater=row(t.test(body_mass_g ~ sex, data=p, var.equal=TRUE, alternative="greater", conf.level=0.99)),
      paired_less=row(t.test(p$bill_length_mm, p$bill_depth_mm, paired=TRUE, mu=27, alternative="less")),
      one_sample_greater=row(t.test(p$flipper_length_mm, mu=200, alternative="greater")),
      welch_by_species=by(p, "species", welch),
      student_by_species=by(p, "species", student),
      paired_by_species=by(p, "species", paired),
      one_sample_by_species=by(p, "species", one),
      welch_by_species_island=by(p, c("species", "island"), welch),
      student_by_island_less=by(p, "island", function(d) t.test(body_mass_g ~ sex, data=d, var.equal=TRUE, alternative="less")),
      welch_by_sex_species_levels=by(p, "sex", function(d) t.test(flipper_length_mm ~ species, data=d[d$species != "Chinstrap", ]))
    )
  },
  stats_tier1 = {
    p <- read.csv("tests/data/penguins.csv", na.strings="NA")
    p$dream <- p$island == "Dream"
    p$first_year <- p$year == 2007
    s <- read.csv("tests/data/ranks.csv")
    # One row per test in tidy3's column names; jsonlite writes infinite
    # interval bounds as null.
    row <- function(h, parameter="df", conf=FALSE) {
      out <- list()
      est <- unname(h$estimate)
      if (length(est) == 1) out$estimate <- est
      if (length(est) > 1) for (i in seq_along(est)) out[[paste0("estimate", i)]] <- est[i]
      if (!is.null(h$statistic)) out$statistic <- unname(h$statistic)
      if (!is.null(h$parameter)) out[[parameter]] <- unname(h$parameter)
      out$p_value <- h$p.value
      if (!is.null(h$conf.int) || conf) {
        out$conf_low <- if (is.null(h$conf.int)) NA_real_ else h$conf.int[1]
        out$conf_high <- if (is.null(h$conf.int)) NA_real_ else h$conf.int[2]
      }
      out$method <- trimws(h$method)
      if (!is.null(h$alternative) && !grepl("Chi-squared|Kruskal", h$method)) out$alternative <- h$alternative
      as.data.frame(out, check.names=FALSE)
    }
    # r x c Fisher tests have no estimate or interval; stats3 keeps the columns.
    rxc <- function(h) data.frame(estimate=NA_real_, p_value=h$p.value, conf_low=NA_real_,
                                  conf_high=NA_real_, method=trimws(h$method), alternative=h$alternative)
    aov_table <- function(fit) {
      t <- summary(fit)[[1]]
      data.frame(term=trimws(rownames(t)), df=t$Df, sumsq=t$`Sum Sq`, meansq=t$`Mean Sq`,
                 statistic=t$`F value`, p_value=t$`Pr(>F)`)
    }
    tukey_table <- function(fit, ...) {
      t <- TukeyHSD(fit, ...)
      m <- t[[1]]
      data.frame(term=names(t)[1], contrast=rownames(m), estimate=m[, "diff"],
                 conf_low=m[, "lwr"], conf_high=m[, "upr"], adj_p_value=m[, "p adj"])
    }
    by <- function(data, groups, f) {
      data |> group_by(across(all_of(groups))) |>
        group_modify(function(d, k) f(d)) |> ungroup() |> as.data.frame()
    }
    props <- function(d) {
      d <- d[!is.na(d$sex), ]
      list(x=as.vector(tapply(d$sex == "male", d$island, sum)), n=as.vector(table(d$island)))
    }
    q <- props(p)
    two <- props(p[p$island != "Biscoe", ])
    # Every term's table, stacked: TukeyHSD gives a list of them.
    tukey_terms <- function(fit, ...) {
      t <- TukeyHSD(fit, ...)
      do.call(rbind, lapply(names(t), function(nm) {
        m <- t[[nm]]
        data.frame(term=nm, contrast=rownames(m), estimate=m[, "diff"], conf_low=m[, "lwr"],
                   conf_high=m[, "upr"], adj_p_value=m[, "p adj"], row.names=NULL)
      }))
    }
    sexed <- p[!is.na(p$sex), ]
    # Balanced: the first 10 penguins of each species and sex.
    balanced <- sexed |> group_by(species, sex) |> filter(row_number() <= 10) |> ungroup() |> as.data.frame()
    # An empty cell: no Chinstrap males.
    no_cell <- p[!(p$species == "Chinstrap" & p$sex %in% "male"), ]
    suppressWarnings(list(
      w_two=row(wilcox.test(body_mass_g ~ sex, data=p)),
      w_two_no_correct=row(wilcox.test(body_mass_g ~ sex, data=p, correct=FALSE)),
      w_two_exact=row(wilcox.test(a ~ g, data=s)),
      w_two_exact_less_mu=row(wilcox.test(a ~ g, data=s, mu=0.3, alternative="less")),
      w_two_edgeworth=row(wilcox.test(a ~ g, data=s, exact=FALSE, correct=2)),
      w_two_exact_ties_by=by(p, c("species", "island"), function(d) row(wilcox.test(body_mass_g ~ sex, data=d))),
      w_columns=row(wilcox.test(p$bill_length_mm, p$bill_depth_mm)),
      w_one=row(wilcox.test(p$flipper_length_mm, mu=200)),
      w_one_exact=row(wilcox.test(s$a, mu=0.1)),
      w_one_exact_zeros=row(wilcox.test(s$c, alternative="greater")),
      w_one_zeros_asymp=row(wilcox.test(s$c, exact=FALSE)),
      w_one_edgeworth=row(wilcox.test(s$a, exact=FALSE, correct=3)),
      w_paired=row(wilcox.test(p$bill_length_mm, p$bill_depth_mm, paired=TRUE)),
      w_paired_exact=row(wilcox.test(s$e, s$a, paired=TRUE, alternative="greater")),
      w_by_species=by(p, "species", function(d) row(wilcox.test(body_mass_g ~ sex, data=d))),
      k_species=row(kruskal.test(body_mass_g ~ species, data=p)),
      k_by_sex=by(p, "sex", function(d) row(kruskal.test(flipper_length_mm ~ island, data=d))),
      k_small=row(kruskal.test(a ~ h, data=s)),
      c_pearson=row(cor.test(p$bill_length_mm, p$bill_depth_mm)),
      c_pearson_by=by(p, "species", function(d) row(cor.test(d$bill_length_mm, d$bill_depth_mm))),
      c_pearson_less=row(cor.test(p$bill_length_mm, p$bill_depth_mm, alternative="less", conf.level=0.9)),
      c_pearson_four=row(cor.test(s$a[1:4], s$b[1:4], alternative="greater")),
      c_pearson_three=row(cor.test(s$a[1:3], s$b[1:3]), conf=TRUE),
      c_spearman=row(cor.test(p$bill_length_mm, p$bill_depth_mm, method="spearman")),
      c_spearman_by_h=by(s, "h", function(d) row(cor.test(d$a, d$b, method="spearman"))),
      c_spearman_all=row(cor.test(s$a, s$b, method="spearman")),
      c_spearman_greater_by_h=by(s, "h", function(d) row(cor.test(d$a, d$e, method="spearman", alternative="greater"))),
      c_spearman_less=row(cor.test(s$a, s$b, method="spearman", alternative="less")),
      c_kendall=row(cor.test(p$bill_length_mm, p$bill_depth_mm, method="kendall")),
      c_kendall_continuity=row(cor.test(p$bill_length_mm, p$bill_depth_mm, method="kendall", continuity=TRUE)),
      c_kendall_exact=row(cor.test(s$a, s$b, method="kendall")),
      c_kendall_by_h=by(s, "h", function(d) row(cor.test(d$a, d$e, method="kendall", alternative="less"))),
      a_species=aov_table(aov(body_mass_g ~ species, data=p)),
      a_by_sex=by(p, "sex", function(d) aov_table(aov(flipper_length_mm ~ island, data=d))),
      a_small=aov_table(aov(a ~ h, data=s)),
      t_species=tukey_table(aov(body_mass_g ~ species, data=p)),
      t_by_sex=by(p, "sex", function(d) tukey_table(aov(flipper_length_mm ~ island, data=d))),
      t_small_99=tukey_table(aov(a ~ h, data=s), conf.level=0.99),
      a_two_way=aov_table(aov(body_mass_g ~ species + sex, data=p)),
      a_two_way_reversed=aov_table(aov(body_mass_g ~ sex + species, data=p)),
      a_interaction=aov_table(aov(body_mass_g ~ species * sex, data=p)),
      a_balanced=aov_table(aov(body_mass_g ~ species * sex, data=balanced)),
      a_covariate=aov_table(aov(body_mass_g ~ flipper_length_mm + species, data=p)),
      a_covariate_interaction=aov_table(aov(body_mass_g ~ flipper_length_mm * species, data=p)),
      a_aliased=aov_table(aov(body_mass_g ~ species * island, data=p)),
      a_empty_cell=aov_table(aov(body_mass_g ~ species * sex, data=no_cell)),
      a_three_way=aov_table(aov(bill_length_mm ~ species * sex * factor(year), data=p)),
      a_interaction_by_year=by(p, "year", function(d) aov_table(aov(bill_depth_mm ~ species * sex, data=d))),
      t_two_way=tukey_terms(aov(body_mass_g ~ species + sex, data=p)),
      t_interaction=tukey_terms(aov(body_mass_g ~ species * sex, data=p)),
      t_balanced_90=tukey_terms(aov(body_mass_g ~ species * sex, data=balanced), conf.level=0.9),
      t_covariate=tukey_terms(aov(body_mass_g ~ flipper_length_mm + species, data=p), which="species"),
      t_empty_cell=tukey_terms(aov(body_mass_g ~ species * sex, data=no_cell), which="species:sex"),
      t_ordered=tukey_terms(aov(body_mass_g ~ sex * species, data=p), which=c("species", "sex:species"), ordered=TRUE),
      t_one_way_ordered=tukey_table(aov(flipper_length_mm ~ island, data=p), ordered=TRUE),
      t_three_way_year=tukey_terms(aov(bill_length_mm ~ species * sex * factor(year), data=p), which="sex:factor(year)"),
      t_interaction_by_year=by(p, "year", function(d) tukey_terms(aov(bill_depth_mm ~ species * sex, data=d))),
      x_species_island=row(chisq.test(p$species, p$island)),
      x_yates=row(chisq.test(p$sex, p$dream)),
      x_no_yates=row(chisq.test(p$sex, p$dream, correct=FALSE)),
      x_gof=row(chisq.test(table(p$species))),
      x_gof_p=row(chisq.test(table(p$species), p=c(0.5, 0.2, 0.3))),
      x_gof_rescale=row(chisq.test(table(p$species), p=c(5, 2, 3), rescale.p=TRUE)),
      x_by_year=by(p, "year", function(d) row(chisq.test(d$species, d$sex))),
      f_sex_dream=row(fisher.test(p$sex, p$dream)),
      f_by_species=by(p, "species", function(d) row(fisher.test(d$sex, d$first_year))),
      f_less=row(fisher.test(p$sex, p$dream, alternative="less", conf.level=0.9)),
      f_greater=row(fisher.test(p$sex, p$first_year, alternative="greater")),
      f_rxc_sex_island=rxc(fisher.test(p$sex, p$island)),
      f_rxc_by_sex=by(p, "sex", function(d) rxc(fisher.test(d$species, d$island, workspace=2e8))),
      f_rxc_island_year=rxc(fisher.test(p$island, p$year, workspace=2e8)),
      p_three=row(prop.test(q$x, q$n), conf=TRUE),
      p_three_given=row(prop.test(q$x, q$n, p=c(0.5, 0.4, 0.6)), conf=TRUE),
      p_two=row(prop.test(two$x, two$n)),
      p_two_less=row(prop.test(two$x, two$n, alternative="less", conf.level=0.9)),
      p_two_no_correct=row(prop.test(two$x, two$n, correct=FALSE)),
      p_one=row(prop.test(q$x[1], q$n[1], p=0.4)),
      p_one_greater=row(prop.test(q$x[2], q$n[2], alternative="greater")),
      b_male=row(binom.test(sum(p$sex == "male", na.rm=TRUE), sum(!is.na(p$sex))), parameter="trials"),
      b_counts=row(binom.test(7, 20, p=0.4, alternative="greater"), parameter="trials"),
      b_zero=row(binom.test(0, 12, p=0.1), parameter="trials"),
      b_less=row(binom.test(9, 30, p=0.5, alternative="less", conf.level=0.9), parameter="trials"),
      b_by_species=by(p, "species", function(d) row(binom.test(sum(d$sex == "male", na.rm=TRUE), sum(!is.na(d$sex)), p=0.45), parameter="trials"))
    ))
  },
  stats_tier2 = {
    p <- read.csv("tests/data/penguins.csv", na.strings="NA")
    s <- read.csv("tests/data/ranks.csv")
    row <- function(h, alternative=NULL, drop=character()) {
      out <- list()
      est <- unname(h$estimate)
      if (length(est) == 1) out$estimate <- est
      if (!is.null(h$parameter) && length(h$parameter) == 2) {
        out$num_df <- unname(h$parameter[1]); out$den_df <- unname(h$parameter[2])
      }
      out$statistic <- unname(h$statistic)
      if (!is.null(h$parameter) && length(h$parameter) == 1) out$df <- unname(h$parameter)
      out$p_value <- h$p.value
      if (!is.null(h$conf.int)) { out$conf_low <- h$conf.int[1]; out$conf_high <- h$conf.int[2] }
      out$method <- trimws(h$method)
      if (!is.null(alternative)) out$alternative <- alternative
      else if (!is.null(h$alternative)) out$alternative <- h$alternative
      as.data.frame(out[setdiff(names(out), drop)], check.names=FALSE)
    }
    by <- function(data, groups, f) {
      data |> group_by(across(all_of(groups))) |>
        group_modify(function(d, k) f(d)) |> ungroup() |> as.data.frame()
    }
    levene <- function(formula, d, center="median") {
      y <- d[[all.vars(formula)[1]]]; g <- d[[all.vars(formula)[2]]]
      ok <- !is.na(y) & !is.na(g); y <- y[ok]; g <- factor(g[ok])
      if (requireNamespace("car", quietly=TRUE)) {
        a <- car::leveneTest(y, g, center=center)
        return(data.frame(statistic=a$`F value`[1], df=a$Df[1], df_residual=a$Df[2], p_value=a$`Pr(>F)`[1],
          method=sprintf("Levene's Test for Homogeneity of Variance (center = %s)", center)))
      }
      # car's definition: a one-way ANOVA on absolute deviations from each level's centre.
      dev <- abs(y - ave(y, g, FUN=if (center == "median") median else mean))
      a <- anova(lm(dev ~ g))
      data.frame(statistic=a$`F value`[1], df=a$Df[1], df_residual=a$Df[2], p_value=a$`Pr(>F)`[1],
        method=sprintf("Levene's Test for Homogeneity of Variance (center = %s)", center))
    }
    ks <- function(..., alternative="two.sided") row(ks.test(..., alternative=alternative), alternative=alternative)
    adjust <- function(pv, method, n=NULL) data.frame(adj=if (is.null(n)) p.adjust(pv, method) else p.adjust(pv, method, n))
    pvalues <- c(0.01, 0.04, NA, 0.03, 0.2, 0.0005, 0.04, 0.6, 0.012)
    # Effect sizes: effectsize's noncentral-t and noncentral-F intervals.
    ncp_ci <- function(t, df, conf=0.95) {
      f <- function(prob) uniroot(function(ncp) pt(t, df, ncp) - prob, c(t - 60, t + 60), tol=1e-13)$root
      c(f((1 + conf) / 2), f((1 - conf) / 2))
    }
    hedges_j <- function(df) exp(lgamma(df / 2) - log(sqrt(df / 2)) - lgamma((df - 1) / 2))
    d_two <- function(y, g, pooled=TRUE, hedges=FALSE) {
      ok <- !is.na(y) & !is.na(g); y <- y[ok]; g <- factor(g[ok])
      a <- y[g == levels(g)[1]]; b <- y[g == levels(g)[2]]
      n1 <- length(a); n2 <- length(b); v1 <- var(a); v2 <- var(b)
      if (pooled) { df <- n1 + n2 - 2; sd <- sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df); scale <- sqrt(1/n1 + 1/n2) }
      else { sd <- sqrt((v1 + v2) / 2); se <- sqrt(v1/n1 + v2/n2); df <- se^4 / ((v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1)); scale <- se / sd }
      d <- (mean(a) - mean(b)) / sd
      j <- if (hedges) hedges_j(df) else 1
      ci <- ncp_ci(d / scale, df) * scale * j
      data.frame(estimate=d * j, conf_low=ci[1], conf_high=ci[2])
    }
    d_one <- function(x, mu=0) {
      x <- x[!is.na(x)]; n <- length(x); d <- (mean(x) - mu) / sd(x)
      ci <- ncp_ci(d * sqrt(n), n - 1) / sqrt(n)
      data.frame(estimate=d, conf_low=ci[1], conf_high=ci[2])
    }
    eta2 <- function(formula, d, conf=0.95) {
      a <- summary(aov(formula, data=d))[[1]]
      f <- a$`F value`[1]; df1 <- a$Df[1]; df2 <- a$Df[2]
      ncp <- if (pf(f, df1, df2) < conf) 0 else
        uniroot(function(nc) pf(f, df1, df2, nc) - conf, c(0, f * df1 * 50), tol=1e-13)$root
      data.frame(term=all.vars(formula)[2], estimate=a$`Sum Sq`[1] / sum(a$`Sum Sq`),
                 conf_low=ncp / (ncp + df1 + df2 + 1), conf_high=1)
    }
    power_row <- function(h, values) {
      out <- as.data.frame(h[values])
      names(out) <- sub(".", "_", names(out), fixed=TRUE)
      out$power <- h$power; out$alternative <- h$alternative; out$method <- h$method
      out
    }
    suppressWarnings(list(
      s_mass=row(shapiro.test(p$body_mass_g)),
      s_by_species=by(p, "species", function(d) row(shapiro.test(d$flipper_length_mm))),
      s_small=row(shapiro.test(s$a)),
      b_species=row(bartlett.test(body_mass_g ~ species, data=p)),
      b_by_year=by(p, "year", function(d) row(bartlett.test(flipper_length_mm ~ species, data=d))),
      l_species=levene(body_mass_g ~ species, p),
      l_mean=levene(body_mass_g ~ species, p, center="mean"),
      l_by_sex=by(p, "sex", function(d) levene(flipper_length_mm ~ island, d)),
      v_sex=row(var.test(body_mass_g ~ sex, data=p)),
      v_columns_less=row(var.test(s$a, s$b, ratio=2, alternative="less", conf.level=0.9)),
      v_greater=row(var.test(s$a, s$e, alternative="greater")),
      v_by_species=by(p, "species", function(d) row(var.test(body_mass_g ~ sex, data=d))),
      k1_pnorm=ks(p$flipper_length_mm, "pnorm", 200, 14),
      k1_exact=ks(s$a, "pnorm"),
      k1_greater=ks(s$a, "pnorm", 0.4, alternative="greater"),
      k1_less=ks(s$a, "pnorm", 0.4, alternative="less"),
      k1_punif=ks(s$a, "punif", -3, 3),
      k1_asymp=ks(s$a, "pnorm", exact=FALSE),
      k1_asymp_less=ks(s$a, "pnorm", exact=FALSE, alternative="less"),
      k2_sex=ks(body_mass_g ~ sex, data=p),
      k2_exact=ks(s$a, s$b),
      k2_ties=ks(s$c, s$b),
      k2_greater=ks(s$a, s$e, alternative="greater"),
      k2_less_ties=ks(s$c, s$e, alternative="less"),
      k2_by_species=by(p, "species", function(d) ks(flipper_length_mm ~ sex, data=d)),
      pa_holm=adjust(pvalues, "holm"),
      pa_hochberg=adjust(pvalues, "hochberg"),
      pa_hommel=adjust(pvalues, "hommel"),
      pa_bonferroni=adjust(pvalues, "bonferroni"),
      pa_BH=adjust(pvalues, "BH"),
      pa_BY=adjust(pvalues, "BY"),
      pa_fdr=adjust(pvalues, "fdr"),
      pa_none=adjust(pvalues, "none"),
      pa_hommel_n=adjust(pvalues, "hommel", n=12),
      pa_BH_n=adjust(pvalues, "BH", n=12),
      d_pooled=d_two(p$body_mass_g, p$sex),
      d_unpooled=d_two(p$body_mass_g, p$sex, pooled=FALSE),
      g_pooled=d_two(p$body_mass_g, p$sex, hedges=TRUE),
      d_one=d_one(p$flipper_length_mm, mu=200),
      d_paired=d_one(p$bill_length_mm - p$bill_depth_mm),
      d_by_species=by(p, "species", function(d) d_two(d$body_mass_g, d$sex)),
      e_species=eta2(body_mass_g ~ species, p),
      e_by_sex=by(p, "sex", function(d) eta2(flipper_length_mm ~ island, d)),
      e_small=eta2(a ~ h, s),
      pw_t_power=power_row(power.t.test(n=20, delta=1), c("n", "delta", "sd", "sig.level")),
      pw_t_n=power_row(power.t.test(power=0.9, delta=1), c("n", "delta", "sd", "sig.level")),
      pw_t_delta=power_row(power.t.test(n=30, power=0.8, type="paired", alternative="one.sided"), c("n", "delta", "sd", "sig.level")),
      pw_t_sd=power_row(power.t.test(n=25, delta=2, sd=NULL, power=0.85, type="one.sample"), c("n", "delta", "sd", "sig.level")),
      pw_t_sig=power_row(power.t.test(n=20, delta=1, sig.level=NULL, power=0.9), c("n", "delta", "sd", "sig.level")),
      pw_t_strict=power_row(power.t.test(n=10, delta=0.5, strict=TRUE), c("n", "delta", "sd", "sig.level")),
      pw_p_power=power_row(power.prop.test(n=50, p1=0.5, p2=0.75), c("n", "p1", "p2", "sig.level")),
      pw_p_n=power_row(power.prop.test(p1=0.5, p2=0.75, power=0.9), c("n", "p1", "p2", "sig.level")),
      pw_p_p2=power_row(power.prop.test(n=50, p1=0.5, power=0.9), c("n", "p1", "p2", "sig.level")),
      pw_p_strict=power_row(power.prop.test(n=30, p1=0.4, p2=0.6, strict=TRUE, alternative="two.sided"), c("n", "p1", "p2", "sig.level"))
    ))
  },
  models = {
    p <- read.csv("tests/data/penguins.csv", na.strings="NA")
    m <- read.csv("tests/data/mtcars.csv")
    tidy_fit <- function(f) {
      co <- coef(f)
      s <- summary(f)$coefficients
      i <- match(names(co), rownames(s))
      ci <- if (inherits(f, "glm")) suppressMessages(confint(f)) else confint(f)
      data.frame(term=names(co), estimate=unname(co), std_error=unname(s[i, 2]),
                 statistic=unname(s[i, 3]), p_value=unname(s[i, 4]),
                 conf_low=unname(ci[, 1]), conf_high=unname(ci[, 2]))
    }
    glance_fit <- function(f) {
      if (inherits(f, "glm")) {
        return(data.frame(null_deviance=f$null.deviance, df_null=f$df.null,
          logLik=as.numeric(logLik(f)), AIC=AIC(f), BIC=BIC(f), deviance=deviance(f),
          df_residual=f$df.residual, nobs=nobs(f)))
      }
      s <- summary(f)
      fs <- s$fstatistic
      data.frame(r_squared=s$r.squared, adj_r_squared=s$adj.r.squared, sigma=s$sigma,
        statistic=unname(fs[1]), p_value=unname(pf(fs[1], fs[2], fs[3], lower.tail=FALSE)),
        df=unname(fs[2]), logLik=as.numeric(logLik(f)), AIC=AIC(f), BIC=BIC(f),
        deviance=deviance(f), df_residual=f$df.residual, nobs=nobs(f))
    }
    augment_fit <- function(f) {
      data.frame(.fitted=unname(predict(f)), .resid=unname(residuals(f)), .hat=unname(hatvalues(f)),
        .sigma=unname(influence(f)$sigma), .cooksd=unname(cooks.distance(f)),
        .std_resid=unname(rstandard(f)), check.names=FALSE)
    }
    anova_fit <- function(f) {
      a <- anova(f)
      if (inherits(f, "glm")) {
        out <- data.frame(term=rownames(a), df=a$Df, deviance=a$Deviance,
          df_residual=a$`Resid. Df`, residual_deviance=a$`Resid. Dev`)
        if (!is.null(a$F)) out$statistic <- a$F
        out$p_value <- a[[ncol(a)]]
        return(out)
      }
      data.frame(term=trimws(rownames(a)), df=a$Df, sumsq=a$`Sum Sq`, meansq=a$`Mean Sq`,
        statistic=a$`F value`, p_value=a$`Pr(>F)`)
    }
    compare <- function(...) {
      fits <- list(...)
      a <- do.call(anova, fits)
      terms <- vapply(fits, function(f) paste(deparse(formula(f)), collapse=" "), "")
      if (inherits(fits[[1]], "glm")) {
        out <- data.frame(term=terms, df_residual=a$`Resid. Df`, residual_deviance=a$`Resid. Dev`,
          df=a$Df, deviance=a$Deviance)
        if (!is.null(a$F)) out$statistic <- a$F
        out$p_value <- a[[ncol(a)]]
        return(out)
      }
      data.frame(term=terms, df_residual=a$Res.Df, rss=a$RSS, df=a$Df, sumsq=a$`Sum of Sq`,
        statistic=a$F, p_value=a$`Pr(>F)`)
    }
    by <- function(data, groups, f) {
      data |> group_by(across(all_of(groups))) |>
        group_modify(function(d, k) f(d)) |> ungroup() |> as.data.frame()
    }
    fits <- list(
      lm_simple=lm(body_mass_g ~ flipper_length_mm, p),
      lm_factor=lm(body_mass_g ~ flipper_length_mm + species, p),
      lm_interaction=lm(body_mass_g ~ flipper_length_mm * species, p),
      lm_two_factors=lm(body_mass_g ~ species * sex, p),
      lm_no_intercept=lm(body_mass_g ~ 0 + species, p),
      lm_margins=lm(body_mass_g ~ species + species:sex, p),
      lm_transform=lm(log(body_mass_g) ~ log(flipper_length_mm) + I(bill_length_mm^2), p),
      lm_weights=lm(mpg ~ wt + hp, m, weights=cyl),
      lm_factor_numeric=lm(mpg ~ wt + factor(cyl), m),
      lm_aliased=lm(mpg ~ wt + I(2 * wt) + hp, m),
      lm_dot=lm(mpg ~ ., m[c("mpg", "wt", "hp", "qsec")]),
      glm_binomial=glm(factor(sex) ~ body_mass_g + species, binomial, p),
      glm_binomial_01=glm(am ~ wt + hp, binomial, m),
      glm_probit=glm(am ~ wt, binomial(link="probit"), m),
      glm_cloglog=glm(am ~ wt, binomial(link="cloglog"), m),
      glm_cbind=glm(cbind(carb, gear) ~ wt + am, binomial, m),
      glm_poisson=glm(carb ~ hp + factor(cyl), poisson, m),
      glm_poisson_offset=glm(carb ~ hp + offset(log(gear)), poisson, m),
      glm_quasipoisson=glm(carb ~ hp, quasipoisson, m),
      glm_gamma_log=glm(mpg ~ wt + hp, Gamma(link="log"), m),
      glm_gamma_inverse=glm(mpg ~ wt, Gamma, m),
      glm_inverse_gaussian=glm(mpg ~ wt, inverse.gaussian, m),
      glm_gaussian_log=glm(mpg ~ wt, gaussian(link="log"), m)
    )
    out <- list()
    for (name in names(fits)) {
      f <- fits[[name]]
      out[[paste0(name, "__tidy")]] <- tidy_fit(f)
      out[[paste0(name, "__glance")]] <- glance_fit(f)
      out[[paste0(name, "__augment")]] <- augment_fit(f)
      out[[paste0(name, "__anova")]] <- anova_fit(f)
    }
    wald <- fits$glm_binomial
    out$glm_binomial__wald <- transform(tidy_fit(wald), conf_low=unname(confint.default(wald)[, 1]),
                                        conf_high=unname(confint.default(wald)[, 2]))
    out$lm_by_species__tidy <- by(p, "species", function(d) tidy_fit(lm(body_mass_g ~ flipper_length_mm, d)))
    out$lm_by_species__glance <- by(p, "species", function(d) glance_fit(lm(body_mass_g ~ flipper_length_mm, d)))
    out$glm_by_species__tidy <- by(p, "species", function(d) tidy_fit(glm(factor(sex) ~ body_mass_g, binomial, d)))
    out$compare_lm <- compare(lm(mpg ~ wt, m), lm(mpg ~ wt + hp, m), lm(mpg ~ wt + hp + qsec, m))
    out$compare_glm <- compare(glm(am ~ wt, binomial, m), glm(am ~ wt + hp, binomial, m))
    out$compare_gamma <- compare(glm(mpg ~ wt, Gamma(link="log"), m), glm(mpg ~ wt + hp, Gamma(link="log"), m))
    new <- data.frame(flipper_length_mm=c(190, 210), species=c("Adelie", "Gentoo"))
    out$predict_confidence <- as.data.frame(predict(fits$lm_factor, new, interval="confidence"))
    out$predict_prediction <- as.data.frame(predict(fits$lm_factor, new, interval="prediction", level=0.9))
    newcars <- data.frame(wt=c(2.5, 3.5), hp=c(100, 200))
    pr <- predict(fits$glm_binomial_01, newcars, type="response", se.fit=TRUE)
    out$predict_glm <- data.frame(fit=unname(pr$fit), se_fit=unname(pr$se.fit))
    # Offsets given as an argument are added to predictions on new rows.
    m$log_gear <- log(m$gear)
    lm_off <- lm(mpg ~ wt, m, offset=qsec)
    out$predict_offset_lm <- data.frame(fit=unname(predict(lm_off, newcars2 <- data.frame(wt=c(2.5, 3.5), qsec=c(16, 19)))))
    glm_off <- glm(carb ~ hp, poisson, m, offset=log_gear)
    pg <- predict(glm_off, data.frame(hp=c(100, 200), log_gear=log(c(3, 5))), type="response", se.fit=TRUE)
    out$predict_offset_glm <- data.frame(fit=unname(pg$fit), se_fit=unname(pg$se.fit))
    # interval and se.fit apply to the rows the model was fitted to as well.
    pc <- predict(fits$lm_factor, interval="confidence", se.fit=TRUE)
    out$predict_current_confidence <- data.frame(fit=unname(pc$fit[, 1]), lwr=unname(pc$fit[, 2]),
                                                 upr=unname(pc$fit[, 3]), se_fit=unname(pc$se.fit))
    out$predict_current_prediction <- suppressWarnings(as.data.frame(predict(fits$lm_simple, interval="prediction")))
    out$predict_current_weighted <- suppressWarnings(as.data.frame(predict(fits$lm_weights, interval="prediction")))
    pgc <- predict(fits$glm_poisson_offset, type="response", se.fit=TRUE)
    out$predict_current_glm <- data.frame(fit=unname(pgc$fit), se_fit=unname(pgc$se.fit))
    # Type II and III tests by car's definitions, in base R (car is not
    # needed): drop a term's columns (type III), or a term and its
    # higher-order relatives against the relatives alone (type II), from
    # the full model matrix, and compare the fits.
    anova_type <- function(f, type) {
      X <- model.matrix(f)
      asg <- attr(X, "assign")
      tt <- terms(f)
      labels <- attr(tt, "term.labels")
      fac <- attr(tt, "factors")
      is_glm <- inherits(f, "glm")
      n <- nrow(X)
      y <- if (is_glm) f$y else model.response(model.frame(f))
      w <- if (is_glm) f$prior.weights else if (is.null(weights(f))) rep(1, n) else weights(f)
      off <- if (is_glm && !is.null(f$offset)) f$offset else rep(0, n)
      refit <- function(drop) {
        keep <- setdiff(seq_len(ncol(X)), drop)
        if (is_glm) {
          g <- glm.fit(X[, keep, drop=FALSE], y, weights=w, offset=off, family=family(f))
          c(g$deviance, g$rank)
        } else {
          l <- lm.wfit(X[, keep, drop=FALSE], y - off, w)
          c(sum(w * l$residuals^2), l$rank)
        }
      }
      hyp <- list()
      if (type == 3 && attr(tt, "intercept") == 1) hyp[["(Intercept)"]] <- list(which(asg == 0), integer())
      for (k in seq_along(labels)) {
        own <- which(asg == k)
        if (type == 3) { hyp[[labels[k]]] <- list(own, integer()); next }
        vars <- fac[, k] > 0
        rel <- which(vapply(seq_along(labels), function(j)
          j != k && all(fac[vars, j] > 0) && sum(fac[, j] > 0) > sum(vars), NA))
        relc <- which(asg %in% rel)
        hyp[[labels[k]]] <- list(c(relc, own), relc)
      }
      disp <- if (is_glm) summary(f)$dispersion else 1
      fixed <- is_glm && family(f)$family %in% c("binomial", "poisson")
      rows <- lapply(names(hyp), function(term) {
        a <- refit(hyp[[term]][[1]])
        b <- refit(hyp[[term]][[2]])
        df <- b[2] - a[2]
        diff <- a[1] - b[1]
        if (!is_glm) {
          ms <- deviance(f) / f$df.residual
          return(data.frame(term=term, df=df, sumsq=diff, meansq=diff / df,
            statistic=(diff / df) / ms, p_value=pf((diff / df) / ms, df, f$df.residual, lower.tail=FALSE)))
        }
        if (fixed) return(data.frame(term=term, df=df, statistic=diff / disp,
                                     p_value=pchisq(diff / disp, df, lower.tail=FALSE)))
        fv <- (diff / df) / disp
        data.frame(term=term, df=df, statistic=fv, p_value=pf(fv, df, f$df.residual, lower.tail=FALSE))
      })
      table <- do.call(rbind, rows)
      if (!is_glm) {
        table <- rbind(table, data.frame(term="Residuals", df=f$df.residual, sumsq=deviance(f),
          meansq=deviance(f) / f$df.residual, statistic=NA, p_value=NA))
      }
      table
    }
    sum_contrasts <- list(species="contr.sum", sex="contr.sum")
    two <- lm(body_mass_g ~ species * sex, p)
    two_sum <- lm(body_mass_g ~ species * sex, p, contrasts=sum_contrasts)
    mixed <- lm(body_mass_g ~ flipper_length_mm * species + sex, p)
    logit <- glm(factor(sex) ~ body_mass_g * species, binomial, p)
    logit_sum <- glm(factor(sex) ~ body_mass_g * species, binomial, p, contrasts=list(species="contr.sum"))
    gamma <- glm(mpg ~ wt * factor(am) + hp, Gamma(link="log"), m)
    out$type2_lm <- anova_type(two, 2)
    out$type3_lm_sum <- anova_type(two_sum, 3)
    out$tidy_lm_sum <- tidy_fit(two_sum)
    out$type2_mixed <- anova_type(mixed, 2)
    out$type3_mixed <- anova_type(mixed, 3)
    out$type2_glm <- anova_type(logit, 2)
    out$type3_glm_sum <- anova_type(logit_sum, 3)
    out$type2_gamma <- anova_type(gamma, 2)
    out
  },
  stats_tier3 = {
    p <- read.csv("tests/data/penguins.csv", na.strings="NA")
    p$dream <- p$island == "Dream"
    p$first_year <- p$year == 2007
    p$heavy <- p$body_mass_g > 4000
    s <- read.csv("tests/data/ranks.csv")
    m <- read.csv("tests/data/mtcars.csv")
    by <- function(data, groups, f) {
      data |> group_by(across(all_of(groups))) |>
        group_modify(function(d, k) f(d)) |> ungroup() |> as.data.frame()
    }
    test <- function(h) {
      out <- list(statistic=unname(h$statistic))
      if (length(h$parameter) == 2) { out <- list(statistic=unname(h$statistic),
        num_df=unname(h$parameter[1]), den_df=unname(h$parameter[2])) }
      else out$df <- unname(h$parameter)
      out$p_value <- h$p.value
      out$method <- trimws(h$method)
      as.data.frame(out, check.names=FALSE)
    }
    mh <- function(h) {
      if (is.null(h$estimate)) return(test(h))
      data.frame(estimate=unname(h$estimate), statistic=unname(h$statistic),
        df=if (is.null(h$parameter)) NA_real_ else unname(h$parameter), p_value=h$p.value,
        conf_low=h$conf.int[1], conf_high=h$conf.int[2], method=h$method, alternative=h$alternative)
    }
    pois <- function(h) {
      two <- grepl("Comparison", h$method)
      data.frame(estimate=unname(h$estimate), statistic=unname(h$statistic),
        time_base=if (two) NA_real_ else unname(h$parameter),
        expected_count1=if (two) unname(h$parameter) else NA_real_,
        p_value=h$p.value, conf_low=h$conf.int[1], conf_high=h$conf.int[2],
        method=h$method, alternative=h$alternative)
    }
    pw <- function(h) {
      x <- h$p.value
      idx <- which(lower.tri(x, diag=TRUE), arr.ind=TRUE)
      idx <- idx[order(idx[, 1], idx[, 2]), , drop=FALSE]
      data.frame(group1=type.convert(rownames(x)[idx[, 1]], as.is=TRUE),
        group2=type.convert(colnames(x)[idx[, 2]], as.is=TRUE),
        p_value=x[idx], method=h$method, p_adjust_method=h$p.adjust.method)
    }
    power_row <- function(h) data.frame(groups=h$groups, n=h$n, between_var=h$between.var,
      within_var=h$within.var, sig_level=h$sig.level, power=h$power, method=h$method)
    long <- data.frame(id=rep(s$id, 3), k=rep(c("a", "b", "e"), each=nrow(s)), v=c(s$a, s$b, s$e),
                       g=rep(s$g, 3))
    cars <- data.frame(id=rep(m$model, 3), k=rep(c("cyl", "gear", "carb"), each=nrow(m)),
                       v=c(m$cyl, m$gear, m$carb))
    counts <- m |> group_by(gear, am) |> summarise(events=sum(carb), time=n(), .groups="drop") |> as.data.frame()
    props <- p[!is.na(p$sex), ] |> group_by(year, island) |>
      summarise(male=sum(sex == "male"), n=n(), .groups="drop") |> as.data.frame()
    named <- function(d) setNames(d$male, d$island)
    scale_row <- function(h) {
      out <- list()
      if (!is.null(h$estimate)) out$estimate <- unname(h$estimate)
      out$statistic <- unname(h$statistic)
      out$p_value <- h$p.value
      if (!is.null(h$conf.int)) {
        out$conf_low <- h$conf.int[1]
        out$conf_high <- h$conf.int[2]
      }
      out$method <- h$method
      out$alternative <- h$alternative
      as.data.frame(out)
    }
    na <- data.frame(x=c(1.5, -2.1, Inf, NA, 0.4, 3.3, -0.7, 2.2),
                     y=c(0.2, NA, -1.1, 2.5, -Inf, 0.9, 1.7, NaN))
    perfect <- data.frame(id=rep(1:4, each=3), k=rep(c("a", "b", "c"), 4),
                          v=rep(1:3, 4) + rep(c(0, 10, 20, 30), each=3))
    smokers <- data.frame(x=c(83, 90, 129, 70), n=c(86, 93, 136, 82), dose=c(0, 1, 2, 4))
    suppressWarnings(list(
      an_exact=scale_row(ansari.test(s$a, s$b)),
      an_less=scale_row(ansari.test(s$a, s$b, alternative="less")),
      an_greater=scale_row(ansari.test(s$e, s$b, alternative="greater")),
      an_ci=scale_row(ansari.test(s$a, s$b, conf.int=TRUE)),
      an_ci_less=scale_row(ansari.test(s$a, s$e, conf.int=TRUE, alternative="less", conf.level=0.9)),
      an_ci_greater=scale_row(ansari.test(s$e, s$b, conf.int=TRUE, alternative="greater")),
      an_formula=scale_row(ansari.test(a ~ g, s)),
      an_by_h=by(s, "h", function(d) scale_row(ansari.test(d$a, d$b, conf.int=TRUE))),
      an_ties=scale_row(ansari.test(c ~ g, s)),
      an_approx=scale_row(ansari.test(s$a, s$b, exact=FALSE, conf.int=TRUE)),
      an_approx_less=scale_row(ansari.test(s$a, s$e, exact=FALSE, conf.int=TRUE, alternative="less")),
      an_approx_greater=scale_row(ansari.test(s$e, s$b, exact=FALSE, conf.int=TRUE, alternative="greater")),
      an_approx_ties=scale_row(ansari.test(s$c, s$b, exact=FALSE, conf.int=TRUE)),
      an_penguins=scale_row(ansari.test(body_mass_g ~ sex, p)),
      an_by_species=by(p, "species", function(d) scale_row(
        ansari.test(bill_depth_mm ~ sex, d, exact=FALSE, conf.int=TRUE))),
      an_missing=scale_row(ansari.test(na$x, na$y)),
      mo_ranks=scale_row(mood.test(s$a, s$b)),
      mo_less=scale_row(mood.test(s$a, s$e, alternative="less")),
      mo_greater=scale_row(mood.test(s$e, s$b, alternative="greater")),
      mo_formula=scale_row(mood.test(a ~ g, s)),
      mo_ties=scale_row(mood.test(c ~ g, s)),
      mo_penguins=scale_row(mood.test(body_mass_g ~ sex, p)),
      mo_by_species=by(p, "species", function(d) scale_row(mood.test(flipper_length_mm ~ sex, d))),
      mo_missing=scale_row(mood.test(na$x, na$y)),
      qu_ranks=test(quade.test(v ~ k | id, long)),
      qu_ties=test(quade.test(v ~ k | id, cars)),
      qu_by_g=by(long, "g", function(d) test(quade.test(v ~ k | id, d))),
      qu_perfect=test(quade.test(v ~ k | id, perfect)),
      ptr_smokers=test(prop.trend.test(smokers$x, smokers$n)),
      ptr_score=test(prop.trend.test(smokers$x, smokers$n, c(0, 0, 0, 1))),
      ptr_score_column=test(prop.trend.test(smokers$x, smokers$n, smokers$dose)),
      ptr_two=test(prop.trend.test(smokers$x[1:2], smokers$n[1:2])),
      ptr_by_year=by(props, "year", function(d) test(prop.trend.test(d$male, d$n))),
      ptr_by_island=by(props, "island", function(d) test(prop.trend.test(d$male, d$n, d$year))),
      o_welch=test(oneway.test(body_mass_g ~ species, p)),
      o_equal=test(oneway.test(body_mass_g ~ species, p, var.equal=TRUE)),
      o_by_year=by(p, "year", function(d) test(oneway.test(flipper_length_mm ~ island, d))),
      fl_species=test(fligner.test(body_mass_g ~ species, p)),
      fl_by_sex=by(p, "sex", function(d) test(fligner.test(flipper_length_mm ~ island, d))),
      fr_ranks=test(friedman.test(v ~ k | id, long)),
      fr_ties=test(friedman.test(v ~ k | id, cars)),
      fr_by_g=by(long, "g", function(d) test(friedman.test(v ~ k | id, d))),
      mc_cars=test(mcnemar.test(m$vs, m$am)),
      mc_three=test(mcnemar.test(m$cyl, m$gear)),
      mc_logical=test(mcnemar.test(p$dream, p$first_year)),
      mc_no_correct=test(mcnemar.test(p$dream, p$first_year, correct=FALSE)),
      mc_by_species=by(p, "species", function(d) test(mcnemar.test(d$heavy, d$first_year))),
      mh_species=mh(mantelhaen.test(p$sex, p$heavy, p$species)),
      mh_less=mh(mantelhaen.test(p$sex, p$heavy, p$species, alternative="less", correct=FALSE, conf.level=0.9)),
      mh_greater=mh(mantelhaen.test(p$sex, p$dream, p$year, alternative="greater")),
      mh_exact=mh(mantelhaen.test(p$sex, p$heavy, p$species, exact=TRUE)),
      mh_exact_less=mh(mantelhaen.test(p$sex, p$dream, p$year, exact=TRUE, alternative="less")),
      mh_general=mh(mantelhaen.test(p$island, p$sex, p$year)),
      mh_by_species=by(p, "species", function(d) mh(mantelhaen.test(d$sex, d$heavy, d$year))),
      po_one=pois(poisson.test(137, 24.19893)),
      po_one_r=pois(poisson.test(137, 24.19893, r=5, alternative="greater")),
      po_small=pois(poisson.test(2, 1, r=6)),
      po_zero=pois(poisson.test(0, 2, alternative="less", conf.level=0.9)),
      po_two=pois(poisson.test(counts$events[counts$gear == 4], counts$time[counts$gear == 4])),
      po_two_less=pois(poisson.test(counts$events[counts$gear == 4], counts$time[counts$gear == 4],
                                    r=1.5, alternative="less", conf.level=0.9)),
      po_by_gear=by(counts, "gear", function(d) pois(poisson.test(d$events, d$time))),
      pt_pooled=pw(pairwise.t.test(p$body_mass_g, p$species)),
      pt_welch=pw(pairwise.t.test(p$body_mass_g, p$species, pool.sd=FALSE, p.adjust.method="BH")),
      pt_less=pw(pairwise.t.test(p$flipper_length_mm, p$island, alternative="less", p.adjust.method="none")),
      pt_paired=pw(pairwise.t.test(long$v, long$k, paired=TRUE, p.adjust.method="bonferroni")),
      pt_by_year=by(p, "year", function(d) pw(pairwise.t.test(d$body_mass_g, d$species))),
      pt_numeric=pw(pairwise.t.test(m$mpg, m$cyl, p.adjust.method="hommel")),
      pw_species=pw(pairwise.wilcox.test(p$flipper_length_mm, p$species)),
      pw_exact=pw(pairwise.wilcox.test(s$a, s$h, p.adjust.method="BH")),
      pw_paired=pw(pairwise.wilcox.test(long$v, long$k, paired=TRUE)),
      pw_by_year=by(p, "year", function(d) pw(pairwise.wilcox.test(d$body_mass_g, d$species))),
      pp_islands=pw(pairwise.prop.test(named(props[props$year == 2008, ]), props$n[props$year == 2008])),
      pp_numbered=pw(pairwise.prop.test(props$male[props$year == 2009], props$n[props$year == 2009],
                                        p.adjust.method="BY", correct=FALSE)),
      pp_by_year=by(props, "year", function(d) pw(pairwise.prop.test(named(d), d$n))),
      pa_power=power_row(power.anova.test(groups=4, n=5, between.var=1, within.var=3)),
      pa_n=power_row(power.anova.test(groups=4, between.var=1, within.var=3, power=0.8)),
      pa_groups=power_row(power.anova.test(n=10, between.var=1, within.var=3, power=0.8)),
      pa_within=power_row(power.anova.test(groups=3, n=10, between.var=1, power=0.9)),
      pa_between=power_row(power.anova.test(groups=3, n=10, within.var=2, power=0.9)),
      pa_sig=power_row(power.anova.test(groups=3, n=10, between.var=1, within.var=2, power=0.9, sig.level=NULL))
    ))
  },
  ordinal = {
    source("tests/r_ordinal_cases.R")
    ordinal_cases()
  },
  bayes = {
    source("tests/r_bayes_cases.R")
    bayes_cases()
  },
  rlm = {
    source("tests/r_resistant_cases.R")
    resistant_cases("rlm")
  },
  rq = {
    source("tests/r_resistant_cases.R")
    resistant_cases("rq")
  },
  negbin = {
    source("tests/r_negbin_cases.R")
    negbin_cases()
  },
  emmeans = {
    source("tests/r_emm_cases.R")
    emm_cases()
  },
  robust = {
    suppressPackageStartupMessages({library(sandwich); library(lmtest)})
    p <- read.csv("tests/data/penguins.csv", na.strings="NA")
    m <- read.csv("tests/data/mtcars.csv")
    by <- function(data, groups, f) {
      data |> group_by(across(all_of(groups))) |>
        group_modify(function(d, k) f(d)) |> ungroup() |> as.data.frame()
    }
    ct <- function(fit, v) {
      t <- coeftest(fit, vcov.=v); ci <- coefci(fit, vcov.=v)
      data.frame(term=rownames(t), estimate=t[, 1], std_error=t[, 2], statistic=t[, 3],
                 p_value=t[, 4], conf_low=ci[, 1], conf_high=ci[, 2], row.names=NULL)
    }
    vc <- function(v) data.frame(term=rownames(v), v, check.names=FALSE, row.names=NULL)
    base <- lm(body_mass_g ~ flipper_length_mm + species, p)
    weighted <- lm(body_mass_g ~ flipper_length_mm + species, p, weights=bill_depth_mm)
    d <- p[complete.cases(p[, c("body_mass_g", "flipper_length_mm", "species")]), ]
    logit <- glm(am ~ wt + hp, binomial, m)
    gam <- glm(mpg ~ wt + hp, Gamma(link="log"), m)
    pois <- glm(carb ~ hp + offset(log(gear)), poisson, m)
    out <- list()
    for (t in c("const", "HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5")) {
      out[[paste0("hc_", t)]] <- ct(base, vcovHC(base, type=t))
    }
    out$hc_weighted <- ct(weighted, vcovHC(weighted, type="HC3"))
    out$hc_weighted_hc1 <- ct(weighted, vcovHC(weighted, type="HC1"))
    out$cl_island <- ct(base, vcovCL(base, cluster=~island))
    out$cl_two_way <- ct(base, vcovCL(base, cluster=~island + year))
    out$cl_year_hc2 <- ct(base, vcovCL(base, cluster=~year, type="HC2"))
    out$cl_year_hc3 <- ct(base, vcovCL(base, cluster=~year, type="HC3"))
    out$cl_weighted_hc2 <- ct(weighted, vcovCL(weighted, cluster=~year, type="HC2"))
    out$cl_no_adjust <- ct(base, vcovCL(base, cluster=~island, type="HC0", cadjust=FALSE))
    out$nw_default <- ct(base, NeweyWest(base))
    out$nw_lag <- ct(base, NeweyWest(base, lag=4, prewhite=FALSE, adjust=TRUE))
    out$nw_order <- ct(base, NeweyWest(base, order.by=~flipper_length_mm, data=d))
    out$glm_hc3 <- ct(logit, vcovHC(logit, type="HC3"))
    out$glm_hc0 <- ct(logit, vcovHC(logit, type="HC0"))
    out$glm_cl <- ct(logit, vcovCL(logit, cluster=~cyl))
    out$gamma_hc1 <- ct(gam, vcovHC(gam, type="HC1"))
    out$gamma_nw <- ct(gam, NeweyWest(gam))
    out$pois_cl <- ct(pois, vcovCL(pois, cluster=~cyl + am))
    out$by_species <- by(p, "species", function(d) ct(lm(body_mass_g ~ flipper_length_mm + sex, d),
      vcovCL(lm(body_mass_g ~ flipper_length_mm + sex, d), cluster=~year)))
    w <- waldtest(base, vcov=vcovHC(base, type="HC1"))
    out$wald_hc1 <- data.frame(statistic=w$F[2], p_value=w$`Pr(>F)`[2])
    out$vcov_hc3 <- vc(vcovHC(base, type="HC3"))
    out$vcov_classical <- vc(vcov(base))
    ex <- lm(body_mass_g ~ flipper_length_mm + sex, p, na.action=na.exclude)
    pr <- predict(ex, interval="confidence")
    out$exclude_lm <- data.frame(fit=pr[, 1], lwr=pr[, 2], upr=pr[, 3])
    out$exclude_resid <- data.frame(.fitted=unname(fitted(ex)), .resid=unname(residuals(ex)))
    exg <- glm(factor(sex) ~ body_mass_g + species, binomial, p, na.action=na.exclude)
    pg <- predict(exg, type="response", se.fit=TRUE)
    out$exclude_glm <- data.frame(fit=unname(pg$fit), se_fit=unname(pg$se.fit))
    out
  },
  stop(sprintf("unknown oracle case: %s", case_name))
)

cat(toJSON(normalise(result), auto_unbox=TRUE, na="null", null="null", digits=15))
