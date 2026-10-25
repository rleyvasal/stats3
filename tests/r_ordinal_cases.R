ordinal_cases <- function() {
  suppressPackageStartupMessages({library(MASS); library(sandwich)})
  d <- housing
  d$off <- seq(-.3,.3,length.out=nrow(d))
  d$x <- (seq_len(nrow(d)) %% 7)/4
  d$w0 <- d$Freq; d$w0[c(2,11,28)] <- 0
  specs <- list(
    logistic=list(formula=Sat~Infl+Type+Cont),
    probit=list(formula=Sat~Infl+Type+Cont,method="probit"),
    loglog=list(formula=Sat~Infl+Type+Cont,method="loglog"),
    cloglog=list(formula=Sat~Infl+Type+Cont,method="cloglog"),
    # MASS's default start fails for cauchit here; start from the logistic fit.
    cauchit=list(formula=Sat~Infl+Type+Cont,method="cauchit",
                 start=c(-0.4,-1.2,0.6,0.4,1.1,-0.4,-0.5,0.7)*c(-1,-1,-1,-1,-1,-1,1,1)),
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
  # confint.polr profiles the slopes only; the thresholds get no interval.
  profile_table <- function(f,tidy,level=.95) {
    ci <- matrix(NA_real_,nrow(tidy),2)
    p <- length(coef(f))
    # profile.polr drops the model matrix's first column as the intercept,
    # so it fails for 0 + x; stats3 profiles the same model as Sat ~ x.
    if(attr(terms(f),"intercept")==0) f <- update(f,.~.+1,data=f$model,weights=`(weights)`)
    if(p) ci[seq_len(p),] <- matrix(suppressMessages(confint(f,level=level)),p,2)
    tidy$conf_low <- ci[,1]; tidy$conf_high <- ci[,2]
    tidy
  }
  # lmtest's coeftest/coefci (t on df.residual) with sandwich's vcovCL, for
  # every parameter; coeftest itself lists the slopes only.
  robust_table <- function(f,v) {
    b <- c(coef(f),f$zeta); se <- sqrt(diag(v)); t <- b/se; q <- qt(.975,f$df.residual)
    data.frame(term=names(b),estimate=unname(b),std_error=unname(se),statistic=unname(t),
               p_value=unname(2*pt(-abs(t),f$df.residual)),conf_low=unname(b-q*se),conf_high=unname(b+q*se))
  }
  robust_vcov <- function(f,v) {
    out <- data.frame(term=rownames(v),v,check.names=FALSE)
    names(out) <- c("term",rownames(v))
    out
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
    cdf <- switch(f$method,logistic=plogis,probit=pnorm,loglog=MASS:::pgumbel,cloglog=MASS:::pGumbel,cauchit=pcauchy)
    probability <- t(vapply(eta,function(e) diff(c(0,cdf(f$zeta-e),1)),numeric(length(f$lev))))
    colnames(probability) <- f$lev
    covariance <- data.frame(term=names(b),v,check.names=FALSE)
    names(covariance) <- c("term",names(b))
    tidy <- data.frame(term=names(b),estimate=unname(b),std_error=unname(se),statistic=unname(b/se),
          p_value=NA_real_,conf_low=unname(b-qnorm(.975)*se),conf_high=unname(b+qnorm(.975)*se))
    list(tidy=tidy,profile=profile_table(f,tidy),profile90=profile_table(f,tidy,.9),
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
  robust <- list(
    cluster=list(spec="logistic",args=list(cluster=~Type)),
    hc1=list(spec="logistic",args=list(cluster=~Type,type="HC1")),
    no_cadjust=list(spec="logistic",args=list(cluster=~Type,cadjust=FALSE)),
    two_way=list(spec="logistic",args=list(cluster=~Type+Cont)),
    probit=list(spec="probit",args=list(cluster=~Infl)),
    cauchit=list(spec="cauchit",args=list(cluster=~Type)),
    offset=list(spec="offset",args=list(cluster=~Type)),
    zero_weights=list(spec="zero_weights",args=list(cluster=~Type)),
    unweighted=list(spec="unweighted",args=list(cluster=~Type)))
  specs$unweighted <- list(formula=Sat~x+Cont,weights=rep(1,nrow(d)))
  out$robust <- lapply(robust,function(r) {
    f <- fit_one(specs[[r$spec]])
    v <- do.call(vcovCL,c(list(f),r$args))
    list(tidy=robust_table(f,v),vcov=robust_vcov(f,v),hessian=robust_vcov(f,vcov(f)))
  })
  out
}
