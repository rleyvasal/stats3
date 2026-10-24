# Independent closed-form reference: proper normal-inverse-gamma regression.
bayes_cases <- function() {
  d <- read.csv("tests/data/mtcars.csv")
  d$w <- 1 + seq_len(nrow(d)) %% 3
  reference <- function(formula, weights=rep(1,nrow(d)), offset=rep(0,nrow(d))) {
    x <- model.matrix(formula, d)
    y <- d$mpg-offset
    p <- ncol(x)
    m0 <- rep(.3,p); v0 <- diag(9,p); a0 <- 2.5; b0 <- 1.7
    precision <- solve(v0)+crossprod(x,weights*x)
    vn <- solve(precision)
    mn <- as.vector(vn %*% (solve(v0)%*%m0+crossprod(x,weights*y)))
    an <- a0+length(y)/2
    bn <- b0+sum(weights*(y-x%*%mn)^2)/2+as.numeric(t(mn-m0)%*%solve(v0)%*%(mn-m0))/2
    covariance <- vn*bn/(an-1)
    scale <- sqrt(diag(vn)*bn/an)
    tidy <- data.frame(term=colnames(x),estimate=mn,std_error=sqrt(diag(covariance)),
      conf_low=mn+qt(.025,2*an)*scale,conf_high=mn+qt(.975,2*an)*scale,
      prob_positive=pt(mn/scale,2*an),statistic=NA_real_,p_value=NA_real_)
    vcov <- data.frame(term=colnames(x),covariance,check.names=FALSE)
    colnames(vcov) <- c("term",colnames(x))
    fit <- as.vector(x%*%mn)+offset
    se <- sqrt(rowSums((x%*%vn)*x)*bn/an)
    width <- qt(.975,2*an)*se
    pred <- data.frame(fit=fit,lwr=fit-width,upr=fit+width)
    width <- qt(.975,2*an)*sqrt((rowSums((x%*%vn)*x)+1/weights)*bn/an)
    predictive <- data.frame(fit=fit,lwr=fit-width,upr=fit+width)
    logml <- -length(y)/2*log(2*pi)+sum(log(weights))/2-log(det(v0))/2-
      as.numeric(determinant(precision,logarithm=TRUE)$modulus)/2+
      a0*log(b0)-an*log(bn)+lgamma(an)-lgamma(a0)
    list(tidy=tidy,vcov=vcov,predict=pred,predictive=predictive,
         evidence=data.frame(log_marginal_likelihood=logml))
  }
  list(basic=reference(mpg~wt+hp),factors=reference(mpg~wt+factor(cyl)),
       weighted=reference(mpg~wt+hp,d$w),offset=reference(mpg~wt+hp,offset=d$hp),
       no_intercept=reference(mpg~0+wt+hp),intercept=reference(mpg~1))
}
