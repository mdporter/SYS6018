#####################################################################
## R Code for Anomaly Detection
## - See lecture: 08-shrinkage.pdf
## 
## Michael D. Porter
## Created: Mar 2019
## For: Data Mining (SYS-6018/SYS-4582) at University of Virginia
## https://mdporter.github.io/SYS6018/
#####################################################################

#-- Install Required Packages
library(glmnet)
library(tidyverse)   



#---------------------------------------------------------------------------#
#-- Prostate Data
#---------------------------------------------------------------------------#

#-- read in prostate data
#   tab separated so use read_tsv()
#   first column is row numbers, so remove
library(readr)
data.url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
prostate = readr::read_tsv(data.url) %>% select(-1) # remove row numbers
prostate.train = filter(prostate, train) %>% select(-train)
prostate.test = filter(prostate, !train) %>% select(-train)



#-- get matrix inputs
X.train = prostate.train %>% select(-lpsa) %>% as.matrix()
Y.train = prostate.train %>% select(lpsa) %>% as.matrix()


#- Center and Scale X
X = scale(X.train)            # Center and scale predictors
Y = Y.train
# Note: you won't normally do the scaling. I just did this here
#  so I can easily calculate the edf and penalties 
#  (which are based on the scaled predictors). 


#---------------------------------------------------------------------------#
#-- Fit Models
#---------------------------------------------------------------------------#

#-- Fit OLS model
fit.ls = lm(Y~X)
B.ls = coef(fit.ls)
fit.ls$df = length(B.ls)  # number of estimated parameters

#-- Fit Ridge Model
library(glmnet)
fit.ridge = glmnet(X, Y, alpha=0)
B.ridge = t(coef(fit.ridge))   # coefficients
fit.ridge$penalty = rowSums((B.ridge[,-1])^2)
XTX = crossprod(X)
n = nrow(X)
fit.ridge$df = sapply(n*fit.ridge$lambda, 
                     function(l) sum(diag(solve(XTX + diag(c(0, rep(l, ncol(X)-1)))) %*% XTX)))


# #-- Using MASS::lm.ridge() for Ridge Regression
# lamseq = c(0,exp(seq(log(.01),log(2000),length=1000))) # exponential sequence
# fit.ridge = MASS::lm.ridge(Y~X, lambda=lamseq )
# B.ridge = coef(fit.ridge)             # coefficients


#-- Fit Lasso Model
library(glmnet)
fit.lasso = glmnet(X, Y, alpha=1)
B.lasso = t(coef(fit.lasso))   # coefficients
fit.lasso$penalty = rowSums(abs(B.lasso[,-1]))
fit.lasso$df = apply(B.lasso, 1, function(beta) sum( abs(beta)>0 ))


#---------------------------------------------------------------------------#
#-- Plots
#---------------------------------------------------------------------------#

#-- Path Plots

#- Lasso Fit
xrng = range(fit.lasso$penalty) + c(0,.22)
yrng = range(B.lasso[,-1])
plot(xrng, yrng,typ='n', lty=1, las=1,
     xlab='L1 norm: sum of absolute betas', ylab='beta (standardized)')
abline(h=0, col='grey80')
abline(v=axTicks(1), col='grey80')
matlines(fit.lasso$penalty, B.lasso[,-1], lty=1)
text(max(fit.lasso$penalty), tail(B.lasso[,-1],1), labels=colnames(X),pos=4,cex=.8)
axis(3,axTicks(1), round(approx(fit.lasso$pen, fit.lasso$df, axTicks(1))$y,1))
mtext('Lasso',side=3,line=2.5)

#- Ridge Fit
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



#-- MSE vs. EDF

yhat.lasso = predict(fit.lasso, newx = X)
r = -sweep(yhat.lasso, 1, Y, '-')
mse.lasso = colMeans(r^2)
plot(fit.lasso$df, mse.lasso, xlab="edf", ylab="mse (training)", las=1,
     main='MSE vs. EDF')

yhat.ridge = cbind(1,X) %*% t(B.ridge)
#yhat.ridge= predict(fit.ridge, newx=X)
r = -sweep(yhat.ridge, 1, Y, '-')
mse.ridge = colMeans(r^2)
lines(fit.ridge$df, mse.ridge, col="red")

yhat.ols = fit.ls$fitted.values
mse.ols = mean( (yhat.ols - Y)^2)
abline(h = mse.ols, col=3, lty=3)

legend("topright",c("lasso","ridge", "ols"),
       col=c(1,2,3),lty=c(NA,1,3),pch=c(1,NA,NA))




#-------------------------------------------------------------------------------
#-- Prostate Data: Cross validation
## Notice here we don't do any scaling, but leave it up to the functions.
## This is the normal use; above was scaled for pedagogical reasons
#-------------------------------------------------------------------------------

X.train = prostate.train %>% select(-lpsa) %>% as.matrix()
Y.train = prostate.train %>% select(lpsa) %>% as.matrix()

X.test = prostate.test %>% select(-lpsa) %>% as.matrix()
Y.test = prostate.test %>% select(lpsa) %>% as.matrix()


#- Get K-fold partition (so consistent to all models)
set.seed(721)                      # set seed for replicability
n.folds = 10                       # number of folds for cross-validation
fold = sample(rep(1:n.folds, length=nrow(X.train)))  
# vector of fold labels
# notice how this is different than:  sample(1:K,n,replace=TRUE), 
#  which won't give equal group sizes


#-- OLS
fit.ls = lm(Y.train~X.train)
beta.ls = coef(fit.ls)
yhat.ls = cbind(1, X.test) %*% coef(fit.ls)

#-- Ridge
fit.ridge = cv.glmnet(X.train, Y.train, alpha=0, foldid=fold)
beta.ridge = coef(fit.ridge, s="lambda.min")
yhat.ridge = predict(fit.ridge, newx = X.test, s="lambda.min")

#-- Lasso
fit.lasso = cv.glmnet(X.train, Y.train, alpha=1, foldid=fold)
beta.lasso = coef(fit.lasso, s="lambda.min")
yhat.lasso = predict(fit.lasso, newx = X.test, s="lambda.min")

#-- Elastic Net
a = .8  # set alpha for elastic net
fit.enet = cv.glmnet(X.train, Y.train, alpha=a, foldid=fold)
beta.enet = coef(fit.enet, s="lambda.min")
yhat.enet = predict(fit.enet, newx = X.test, s="lambda.min")


#-- MSE
YHAT = list(ols = yhat.ls, ridge = yhat.ridge, 
              lasso=yhat.lasso, enet=yhat.enet)

sapply(YHAT, function(yhat) mean((Y.test - yhat)^2))


#-- Coefficients

tibble(variable=c("(Intercept)", colnames(X.train)), 
       ols = beta.ls, 
       ridge=beta.ridge[,1], 
       lasso=beta.lasso[,1],
       enet = beta.enet[,1])








