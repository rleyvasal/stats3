ordinal_cases <- function() {
  suppressPackageStartupMessages(library(MASS))
  d <- housing
  d$off <- seq(-.3,.3,length.out=nrow(d))
  d$x <- (seq_len(nrow(d)) %% 7)/4
  d$w0 <- d$Freq; d$w0[c(2,11,28)] <- 0
  specs <- list(
    logistic=list(formula=Sat~Infl+Type+Cont),
    probit=list(formula=Sat~Infl+Type+Cont,method="probit"),
    loglog=list(formula=Sat~Infl+Type+Cont,method="loglog"),
    cloglog=list(formula=Sat~Infl+Type+Cont,method="cloglog"),
    intercept=list(formula=Sat~1),
    numeric=list(formula=Sat~x+Cont),
    no_intercept=list(formula=Sat~0+x),
    offset=list(formula=Sat~Infl+Type+offset(off)),
    offset_arg=list(formula=Sat~Infl+Type+offset(off)),
    zero_weights=list(formula=Sat~Infl+Type+Cont,weights=d$w0),
    sum_contrasts=list(formula=Sat~Infl+Type+Cont,contrasts=list(Infl="contr.sum",Type="contr.sum",Cont="contr.sum")))
  fit_one <- function(spec,data=d) {
    if(is.null(spec$weights)) spec$weights <- data$Freq
    if(!is.null(spec$contrasts)) {
      for(name in names(spec$contrasts)) contrasts(data[[name]]) <- contr.sum(nlevels(data[[name]]))
      spec$contrasts <- NULL
    }
    do.call(polr,c(spec,list(data=data,Hess=TRUE,control=list(reltol=1e-13,maxit=1000))))
  }
  tables <- function(f,extra_offset=FALSE) {
    b <- c(coef(f),f$zeta); v <- vcov(f); se <- sqrt(diag(v))
    probs <- f$fitted.values
    classes <- function(p) levels(f$model[[1]])[max.col(p,ties.method="first")]
    new <- head(d,5)
    tt <- delete.response(terms(f))
    mf <- model.frame(tt,new)
    mx <- model.matrix(tt,mf,contrasts.arg=f$contrasts)
    eta <- if(length(coef(f))) drop(mx[,names(coef(f)),drop=FALSE]%*%coef(f)) else rep(0,nrow(new))
    off <- model.offset(mf)
    if(!is.null(off)) eta <- eta+off
    if(extra_offset) eta <- eta+new$off
    cdf <- switch(f$method,logistic=plogis,probit=pnorm,loglog=MASS:::pgumbel,cloglog=MASS:::pGumbel)
    probability <- t(vapply(eta,function(e) diff(c(0,cdf(f$zeta-e),1)),numeric(length(f$lev))))
    colnames(probability) <- f$lev
    covariance <- data.frame(term=names(b),v,check.names=FALSE)
    names(covariance) <- c("term",names(b))
    list(tidy=data.frame(term=names(b),estimate=unname(b),std_error=unname(se),statistic=unname(b/se),
          p_value=NA_real_,conf_low=unname(b-qnorm(.975)*se),conf_high=unname(b+qnorm(.975)*se)),
      vcov=covariance,probs=as.data.frame(probs),class=data.frame(fit=classes(probs)),
      new_probs=as.data.frame(probability),new_class=data.frame(fit=classes(probability)),
      augment=data.frame(.fitted=classes(probs),.linear_predictor=unname(f$lp)),
      glance=data.frame(logLik=as.numeric(logLik(f)),AIC=AIC(f),BIC=BIC(f),deviance=deviance(f),
                        df_residual=f$df.residual,nobs=f$nobs,converged=f$convergence==0))
  }
  out <- lapply(names(specs),function(name) tables(fit_one(specs[[name]])))
  names(out) <- names(specs)
  grouped <- lapply(levels(d$Cont),function(group) {
    f <- fit_one(list(formula=Sat~Infl+Type),d[d$Cont==group,])
    t <- tables(f)
    list(tidy=cbind(Cont=group,t$tidy),glance=cbind(Cont=group,t$glance))
  })
  out$grouped <- list(tidy=do.call(rbind,lapply(grouped,`[[`,"tidy")),
                       glance=do.call(rbind,lapply(grouped,`[[`,"glance")))
  small <- fit_one(list(formula=Sat~1)); big <- fit_one(specs$logistic)
  comparison <- anova(small,big)
  out$compare <- data.frame(term=c("Sat ~ 1","Sat ~ Infl + Type + Cont"),
      df_residual=comparison[[2]],residual_deviance=comparison[[3]],df=comparison[[5]],
      statistic=comparison[[6]],p_value=comparison[[7]])
  out
}
