#####################################################################
## R Code for Ridge/Lasso/Penalized Regression
## - See lecture: penalized.pdf
## 
## Michael D. Porter
## Created: Mar 2019
## For: Data Mining (SYS-6018) at University of Virginia
## https://mdporter.github.io/SYS6018/
#####################################################################

#-- Install Required Packages
library(broom)
library(glmnet)
library(tidyverse)   


#---------------------------------------------------------------------------#
#-- Advertising Data
#---------------------------------------------------------------------------#

#-- read in advertising data
#   first column is row numbers, so remove
library(readr)
advert = read_csv("https://www.statlearning.com/s/Advertising.csv") %>% 
  select(-1) 

#-- Fit OLS
advert_lm = lm(sales ~ TV + radio + newspaper, data=advert)
broom::tidy(advert_lm)


#---------------------------------------------------------------------------#
#-- Prostate Data
#---------------------------------------------------------------------------#

#-- read in prostate data
#   tab separated so use read_tsv()
#   first column is row numbers, so remove
library(readr)
data.url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
prostate = readr::read_tsv(data.url) %>% select(-1) # remove row numbers
train = prostate$train
test = !train
prostate.train = filter(prostate, train) %>% select(-train)
prostate.test = filter(prostate, test) %>% select(-train)

#-- Fit OLS
prostate_lm = lm(lpsa~., data=prostate.train)
broom::tidy(prostate_lm)

#---------------------------------------------------------------------------#
#-- Best Subsets
#---------------------------------------------------------------------------#

#-- Fit best subsets
library(leaps)
prostate_best = regsubsets(lpsa~., data=prostate.train)
plot(prostate_best)
broom::tidy(prostate_best) %>% arrange(BIC)  # minimize BIC (two predictors)
beta_best = coef(prostate_best, which.min(tidy(prostate_best)$BIC))  # coefficients of best subsets model


#-- Bootstrap Results
set.seed(55)
data.boot = dplyr::slice_sample(prostate.train, prop=1, replace=TRUE)
tmp = regsubsets(lpsa~., data=data.boot)
broom::tidy(tmp) %>% arrange(BIC)  # minimize BIC (three predictors)
beta_boot = coef(tmp, which.min(tidy(tmp)$BIC))  # coefficients of best subsets model

#-- Table of coefficients
tidy(prostate_lm) %>% select(term, OLS = estimate) %>% 
  full_join(enframe(beta_boot, name="term", value = "bootstrap"), by="term") %>% 
  full_join(enframe(beta_best, name="term", value = "best subset"), by="term") %>% 
  replace_na(list(bootstrap = 0, `best subset`=0))
  
#-- predictors
preds = c("lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45")
response = "lpsa"
p = length(preds)

#-- fit best subset and evaluate on train and test sets
out = NULL
for(k in 1:p){
  var.mat = combn(preds, k)
  nk = ncol(var.mat)
  for(i in 1:nk){
    vars = c(response, var.mat[,i])
    m = lm(lpsa~., data=prostate[train, vars])
    MSE.train = mean(m$residuals^2)
    yhat.test = predict(m, newdata=prostate[test,  ])
    MSE.test = mean((prostate[test, response, drop=TRUE] - yhat.test)^2)
    out = rbind(out, data.frame(k, MSE.train, MSE.test))
  }
}


best.k = out %>% group_by(k) %>% summarize(MSE.train = min(MSE.train), MSE.test = min(MSE.test))

#-- Plot training MSE
out %>% 
  ggplot(aes(k, MSE.train)) + 
  geom_point(color="gray") + theme_bw() + scale_x_continuous(breaks=1:8) + 
  geom_line(data = best.k, color="red") + geom_point(data = best.k, color="red")  
  
  
yrng = range(c(out$MSE.train, out$MSE.test))
plot(out$k, out$MSE.train, 
     pch=19, col="gray", 
     xlab='k', ylab='MSE.train', 
     las=1, ylim=yrng)
lines(best.k$k, best.k$MSE.train, col='red', type='b')


#-- Plot test MSE
out %>% 
  ggplot(aes(k, MSE.test)) + 
  geom_point(color="gray") + theme_bw() + scale_x_continuous(breaks=1:8) + 
  geom_line(data = best.k, color="red") + geom_point(data = best.k, color="red")  


plot(out$k, out$MSE.test, 
     pch=19, col="gray", 
     xlab='k', ylab='MSE.test', 
     las=1, ylim=yrng)
lines(best.k$k, best.k$MSE.test, col='red', type='b')




#---------------------------------------------------------------------------#
#-- Create new glance() function for glmnet objects
#---------------------------------------------------------------------------#
glance.glmnet <- function(enet, X, Y){
  #- get coefficient estimates
  beta = tidy(enet, return_zeros=TRUE)
  
  #- get training mse
  yhat = predict(enet, s = enet$lambda, newx=X)
  mse = apply(yhat, 2, function(est) mean((Y - est)^2))
  
  #- get ridge edf estimate 
  ridge_edf <- function(lambda, X) {
    X = scale(X)                       # scale X (divide by std.dev)
    XTX = crossprod(cbind(1,X))        # t(X) %*% X  (ignoring intercept)
    trace <- function(X) sum(diag(X))  # function for matrix trace     
    n = nrow(X)                        # because glmnet uses 1/n in loss
    sapply(lambda, function(lam)
      trace(solve(XTX + diag(c(0, rep(n*lam, ncol(X))))) %*% XTX) )
  }
  #- add all stats and return
  beta %>% 
    filter(term != "(Intercept)") %>%   # don't penalize intercept
    group_by(lambda) %>% 
    summarize(alpha = enet$call$alpha, 
              L1 = sum(abs(estimate)), L2 = sum(estimate^2), 
              nonzero = sum(abs(estimate)>0)) %>% 
    mutate(penalty = (alpha*L1 + (1-alpha)*L2/2), 
           edf = ifelse(alpha==0, ridge_edf(lambda, X), 1+nonzero),
           mse = mse[match(lambda, enet$lambda)], 
           step = min_rank(-lambda))
}





#---------------------------------------------------------------------------#
#-- Ridge Regression: Prostate Data
#---------------------------------------------------------------------------#
library(glmnet) # An Introduction to glmnet: https://glmnet.stanford.edu/articles/glmnet.html      
library(glmnetUtils)  # adds formula interface
## Note: glmnet() requires matrix inputs, not formulas
#        But you can use glmnetUtils package to provide a formula interface!  
#  https://cran.r-project.org/web/packages/glmnetUtils/vignettes/intro.html


#-- get matrix inputs (if not using formula interface)
X.train = prostate.train %>% select(-lpsa) %>% glmnet::makeX()
Y.train = prostate.train %>% select(lpsa) %>% as.matrix()


#-- Fit ridge model (alpha=0) with 10 fold cross-validation (nfolds=10)
set.seed(123)
#cvfit = cv.glmnet(lpsa~., data=prostate.train, nfolds=10, alpha=0)
cvfit = cv.glmnet(X.train, Y.train, nfolds=10, alpha=0) 
cvfit$lambda.min   # best lambda according to cross-validation
cvfit$lambda.1se   # largest lambda within 1SE of best
plot(cvfit)        # plot of CV error and lambda penalty

#-- Evaluate Performance on the Test Set
X.test = prostate.test %>% select(-lpsa) %>% glmnet::makeX() 

yhat_min = predict(cvfit, newx = X.test, s="lambda.min", type="response")
mean((yhat_min - prostate.test$lpsa)^2)  # MSE for lambda min

yhat_1se = predict(cvfit, newx = X.test, s="lambda.1se", type="response")
mean((yhat_1se - prostate.test$lpsa)^2)  # MSE for lambda min

lm_fit = lm(lpsa~., data=prostate.train)
yhat_lm = predict(lm_fit, prostate.test)
mean((yhat_lm - prostate.test$lpsa)^2)  # MSE for OLS




#-- Fit Ridge Model on all training data
#fit.ridge = glmnet(lpsa~., data=prostate.train, alpha=0)
fit.ridge = glmnet(X.train, Y.train, alpha=0, lambda.min.ratio=1e-15)

B.ridge = fit.ridge %>% broom::tidy(return_zeros = TRUE)

(stats_ridge = glance(fit.ridge, X.train, Y.train))

B.ridge = B.ridge %>% filter(term != "(Intercept)") %>% 
  left_join(stats_ridge, by="lambda") %>% 
  mutate(log_lambda = log(lambda))

#-- Plot: beta vs log(lambda) 
B.ridge %>% 
  ggplot(aes(log_lambda, estimate, color=term)) + geom_line()

#-- Plot: beta vs. L2 norm (penalty)
B.ridge %>% 
  ggplot(aes(penalty, estimate, color=term)) + geom_line()


#-- Plot: beta vs. edf
B.ridge %>% 
  ggplot(aes(edf, estimate, color=term)) + geom_line() + 
  scale_x_continuous(breaks = 1:9)





####### SAME AS ABOVE BUT WITH BASE R

#-- get matrix inputs
X.train = prostate.train %>% select(-lpsa) %>% as.matrix()
Y.train = prostate.train %>% select(lpsa) %>% as.matrix()

#- Center and Scale X 
X = scale(X.train)            # Center and scale predictors
Y = Y.train 
# Note: you won't normally do the scaling. I just did this here
#  so I can easily calculate the edf (which is based on the scaled predictors)


#-- Fit ridge model (alpha=0) with 10 fold cross-validation (nfolds=10)
set.seed(123)
cvfit = cv.glmnet(X, Y, nfolds=10, alpha=0) 
cvfit$lambda.min   # best lambda according to cross-validation
cvfit$lambda.1se   # largest lambda within 1SE of best
plot(cvfit)        # plot of CV error and lambda penalty

#-- Fit Ridge Model on all training data
fit.ridge = glmnet(X.train, Y.train, alpha=0)
B.ridge = t(predict(fit.ridge, type="coef"))
fit.ridge$penalty = rowSums(B.ridge[,-1]^2)  
XTX = crossprod(X)
fit.ridge$df = sapply(fit.ridge$lambda, 
                      function(l) 1+sum(diag(solve(XTX + diag(l,ncol(X))) %*% XTX)))

#-- Plot: beta vs. L2 norm
xrng = range(fit.ridge$penalty) + c(0,.09)
yrng = range(B.ridge[,-1])
plot(xrng, yrng, typ='n', lty=1, las=1,
     xlab='L2 norm: sum of squared betas', ylab='beta (standardized)')
abline(h=0,col='grey80')
abline(v=axTicks(1),col='grey80')
matlines(fit.ridge$pen, B.ridge[,-1],lty=1)
text(max(fit.ridge$pen), tail(B.ridge[,-1],1),labels=colnames(X),pos=4,cex=.8)
axis(3,axTicks(1),round(approx(fit.ridge$pen,fit.ridge$df,axTicks(1))$y,1))
mtext('Ridge',side=3,line=2.5)

#-- Plot: beta vs log(lambda) 
xrng = range(log(fit.ridge$lambda)) + c(-.090,0)
yrng = range(B.ridge[,-1])
plot(xrng, yrng, typ='n', lty=1, las=1,
     xlab='log(lambda)', ylab='beta (standardized)')
abline(h=0,col='grey80')
abline(v=axTicks(1),col='grey80')
matlines(log(fit.ridge$lambda), B.ridge[,-1],lty=1)
text(min(log(fit.ridge$lambda)), tail(B.ridge[,-1],1),labels=colnames(X),pos=2,cex=.8)
axis(3,axTicks(1),round(approx(log(fit.ridge$lambda),fit.ridge$df,axTicks(1))$y,1))
mtext('Ridge',side=3,line=2.5)
abline(v = log(cvfit$lambda.min), col="lightblue")
abline(v = log(cvfit$lambda.1se), col="lightblue")












