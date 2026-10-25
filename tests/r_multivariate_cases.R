multivariate_cases <- function() {
  us <- read.csv("tests/data/usarrests.csv")
  rownames(us) <- us$state
  usx <- us[, c("Murder", "Assault", "UrbanPop", "Rape")]
  ir <- read.csv("tests/data/iris.csv")
  irx <- ir[, 1:4]
  pg <- read.csv("tests/data/penguins.csv", na.strings="NA")
  pcols <- c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")
  pgx <- na.omit(pg[, pcols])
  mt <- read.csv("tests/data/mtcars.csv")
  rownames(mt) <- mt$model
  mtx <- mt[, c("mpg", "disp", "hp", "drat", "wt", "qsec")]
  # Row labels: the labels column, else positions in the input (R's row names).
  labels_of <- function(x, named) if (named) rownames(x) else as.integer(rownames(x))
  long <- function(m, name, labels) {
    out <- data.frame(a=rep(labels, each=ncol(m)), PC=rep(seq_len(ncol(m)), nrow(m)),
                      value=as.vector(t(m)))
    names(out)[1] <- name
    out
  }
  pca_tables <- function(fit, x, named, prefix) {
    load <- if (inherits(fit, "prcomp")) fit$rotation else unclass(fit$loadings)
    scores <- if (inherits(fit, "prcomp")) fit$x else fit$scores
    k <- ncol(load)
    vars <- fit$sdev^2 / sum(fit$sdev^2)
    aug <- as.data.frame(unname(scores))
    names(aug) <- paste0(".fitted", prefix, seq_len(ncol(scores)))
    list(x=long(scores, "row", labels_of(x, named)),
         rotation=long(load, "column", rownames(load)),
         pcs=data.frame(PC=seq_len(k), std_dev=unname(fit$sdev[1:k]),
                        percent=round(vars, 5)[1:k], cumulative=round(cumsum(vars), 5)[1:k]),
         augment=aug)
  }
  out <- list()
  out$prcomp <- list(
    usarrests=pca_tables(prcomp(usx, scale.=TRUE), usx, TRUE, "PC"),
    iris=pca_tables(prcomp(irx), irx, FALSE, "PC"),
    penguins=pca_tables(prcomp(pgx, scale.=TRUE, rank.=2), pgx, FALSE, "PC"),
    tol=pca_tables(prcomp(irx, tol=0.2), irx, FALSE, "PC"),
    uncentred=pca_tables(prcomp(usx, center=FALSE, scale.=TRUE), usx, TRUE, "PC"))
  p <- prcomp(usx, scale.=TRUE)
  out$prcomp$predict <- as.data.frame(unname(predict(p, head(usx, 5))))
  names(out$prcomp$predict) <- paste0("PC", 1:4)
  out$princomp <- list(
    usarrests=pca_tables(princomp(usx, cor=TRUE), usx, TRUE, "Comp."),
    iris=pca_tables(princomp(irx), irx, FALSE, "Comp."),
    penguins=pca_tables(princomp(pgx, cor=TRUE), pgx, FALSE, "Comp."))
  # factanal: tighter L-BFGS-B than R's default, so both reach the optimum.
  opt <- list(opt=list(factr=1e3, pgtol=0))
  fa_tables <- function(f) {
    l <- unclass(f$loadings)
    tidy <- data.frame(variable=rownames(l), uniqueness=unname(f$uniquenesses))
    for (j in seq_len(ncol(l))) tidy[[paste0("fl", j)]] <- unname(l[, j])
    glance <- data.frame(n_factors=f$factors, total_variance=sum(colSums(l^2) / nrow(l)),
      statistic=if (is.null(f$STATISTIC)) NA_real_ else unname(f$STATISTIC),
      p_value=if (is.null(f$PVAL)) NA_real_ else unname(f$PVAL), df=f$dof, n=f$n.obs,
      method=f$method, converged=f$converged, nobs=f$n.obs)
    res <- list(tidy=tidy, glance=glance)
    if (!is.null(f$scores)) {
      s <- as.data.frame(unname(f$scores)); names(s) <- paste0(".fs", seq_len(ncol(s)))
      res$scores <- s
    }
    res
  }
  out$factanal <- list(
    mtcars2=fa_tables(factanal(mtx, 2, control=opt)),
    mtcars2_none=fa_tables(factanal(mtx, 2, rotation="none", scores="regression", control=opt)),
    mtcars2_promax=fa_tables(factanal(mtx, 2, rotation="promax", scores="Bartlett", control=opt)),
    mtcars3=fa_tables(factanal(mtx, 3, control=opt)),
    usarrests=fa_tables(factanal(usx, 1, control=opt)),
    iris=fa_tables(factanal(irx, 1, scores="regression", control=opt)),
    penguins=fa_tables(factanal(pgx, 1, control=opt)))
  dist_table <- function(d, labels) {
    m <- as.matrix(d); n <- nrow(m)
    i <- unlist(lapply(2:n, function(a) rep(a, a - 1))); j <- unlist(lapply(2:n, function(a) seq_len(a - 1)))
    data.frame(item1=labels[i], item2=labels[j], distance=m[cbind(i, j)])
  }
  methods <- c("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski")
  out$dist <- lapply(setNames(methods, methods), function(m) dist_table(dist(usx, m, p=3), rownames(usx)))
  na <- head(irx, 6); na[2, 3] <- NA; na[4, 1] <- NA; na[5, 2] <- 0
  out$dist$missing <- dist_table(dist(na), 1:6)
  out$dist$missing_canberra <- dist_table(dist(na, "canberra"), 1:6)
  hc_tables <- function(h, labels, k=4) {
    list(merge=data.frame(step=seq_along(h$height), merge1=h$merge[, 1], merge2=h$merge[, 2], height=h$height),
         order=data.frame(position=seq_along(h$order), order=h$order, label=labels[h$order]),
         cutree=data.frame(item=labels, cluster=unname(cutree(h, k))))
  }
  hmethods <- c("complete", "single", "average", "mcquitty", "ward.D", "ward.D2", "centroid", "median")
  ud <- dist(usx)
  out$hclust <- lapply(setNames(hmethods, hmethods), function(m) {
    d <- if (m %in% c("centroid", "median")) ud^2 else ud
    hc_tables(hclust(d, m), rownames(usx))
  })
  # iris has duplicated rows and many tied distances.
  idist <- dist(irx)
  for (m in c("complete", "single", "average", "ward.D2")) {
    out$hclust[[paste0("iris_", m)]] <- hc_tables(hclust(idist, m), 1:150, 3)
  }
  h <- hclust(ud, "average")
  out$hclust$cut_h <- data.frame(item=rownames(usx), cluster=unname(cutree(h, h=50)))
  mds <- function(d, k, labels) {
    f <- cmdscale(d, k, eig=TRUE)
    pts <- data.frame(item=labels, unname(f$points)); names(pts) <- c("item", paste0("dim", 1:k))
    list(points=pts, eig=data.frame(dim=seq_along(f$eig), eigenvalue=f$eig),
         glance=data.frame(k=k, gof1=f$GOF[1], gof2=f$GOF[2], ac=f$ac))
  }
  out$cmdscale <- list(usarrests=mds(ud, 2, rownames(usx)), iris=mds(idist, 3, 1:150),
                       manhattan=mds(dist(usx, "manhattan"), 3, rownames(usx)))
  km_tables <- function(f, names) {
    tidy <- as.data.frame(unname(f$centers)); names(tidy) <- names
    tidy$size <- f$size; tidy$withinss <- f$withinss; tidy$cluster <- as.character(seq_len(nrow(f$centers)))
    list(tidy=tidy, glance=data.frame(totss=f$totss, tot_withinss=f$tot.withinss, betweenss=f$betweenss,
                                      iter=f$iter),
         augment=data.frame(.cluster=as.character(unname(f$cluster))))
  }
  out$kmeans <- list()
  for (a in c("Hartigan-Wong", "Lloyd", "MacQueen")) {
    out$kmeans[[paste0("iris_", a)]] <- km_tables(kmeans(irx, irx[c(1, 51, 101), ], algorithm=a), names(irx))
    out$kmeans[[paste0("usarrests_", a)]] <- km_tables(kmeans(usx, usx[1:4, ], algorithm=a), names(usx))
    out$kmeans[[paste0("penguins_", a)]] <- km_tables(kmeans(pgx, pgx[c(1, 50, 200), ], algorithm=a, iter.max=30),
                                                     names(pgx))
  }
  out$kmeans$k2 <- km_tables(kmeans(usx, usx[c(2, 9), ]), names(usx))
  set.seed(1)
  out$kmeans$random <- data.frame(tot_withinss=kmeans(irx, 3, nstart=25)$tot.withinss)
  # Grouped: one fit per species.
  sp <- split(ir, ir$Species)
  cbind_key <- function(key, d) cbind(Species=key, d)
  stack <- function(f) do.call(rbind, unname(Map(function(key, d) cbind_key(key, f(d)), names(sp), sp)))
  out$grouped <- list(
    prcomp=stack(function(d) pca_tables(prcomp(d[, 1:4], scale.=TRUE), d[, 1:4], FALSE, "PC")$pcs),
    factanal=stack(function(d) fa_tables(factanal(d[, 1:4], 1, control=opt))$tidy),
    hclust=stack(function(d) hc_tables(hclust(dist(d[, 1:4]), "average"), as.integer(rownames(d)), 3)$merge),
    cutree=stack(function(d) hc_tables(hclust(dist(d[, 1:4]), "average"), as.integer(rownames(d)), 3)$cutree),
    kmeans=stack(function(d) km_tables(kmeans(d[, 1:4], d[1:2, 1:4]), names(irx))$tidy))
  out
}
