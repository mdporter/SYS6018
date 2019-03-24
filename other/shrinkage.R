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
library(MASS)
library(glmnet)
library(tidyverse)   


#---------------------------------------------------------------------------#
#-- Advertisting Data
#---------------------------------------------------------------------------#

#-- read in advertising data
#   tab separated so use read_tsv()
#   first column is row numbers, so remove
library(readr)
advert = read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv") %>% 
  select(-1)        # remove first column of rownames

#-- Fit OLS
advert.lm = lm(sales ~ TV + radio + newspaper, data=advert)



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

#-- Fit OLS
prostate.lm = lm(lpsa~., data=prostate.train)


#---------------------------------------------------------------------------#
#-- Best Subsets
#---------------------------------------------------------------------------#

#-- Fit best subsets
library(leaps)
prostate.best = regsubsets(lpsa~., data=prostate.train)
plot(prostate.best)

which.min(summary(prostate.best)$bic)  # minimize BIC
coef(prostate.best, 2)



#-- Bootstrap Results
set.seed(55)
data.boot = sample_n(prostate.train, size=nrow(prostate.train), replace=TRUE)
tmp = regsubsets(lpsa~., data=data.boot)
plot(tmp)
which.min(summary(tmp)$bic)  # 4 predictors


#-- Table of coefficients
full_join(
  data.frame(lm=coef(prostate.lm)) %>% rownames_to_column(var="predictor")
  ,data.frame(best_subset=coef(prostate.best, 2)) %>% rownames_to_column(var="predictor"),
  by="predictor") %>% 
  full_join(data.frame(bootstrap=coef(prostate.best, 4)) %>% rownames_to_column(var="predictor"),
            by="predictor") %>% 
  replace_na(list(best_subset=0L, bootstrap=0L)) 


#-- predictors
preds = c("lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45")
response = "lpsa"
p = length(preds)


#-- fit best subset and eval on train and test sets
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


#-- Plot training MSE
yrng = range(c(out$MSE.train, out$MSE.test))
best.k = out %>% group_by(k) %>% summarize(MSE = min(MSE.train))
plot(out$k, out$MSE.train, 
     pch=19, col="gray", 
     xlab='k', ylab='MSE.train', 
     las=1, ylim=yrng)
lines(best.k$k, best.k$MSE, col='red', type='b')


#-- Plot test MSE
best.k = out %>% group_by(k) %>% summarize(MSE = min(MSE.test))
plot(out$k, out$MSE.test, 
     pch=19, col="gray", 
     xlab='k', ylab='MSE.test', 
     las=1, ylim=yrng)
lines(best.k$k, best.k$MSE, col='red', type='b')







#---------------------------------------------------------------------------#
#-- Correlated Predictors
#---------------------------------------------------------------------------#
#-- Generate Data
set.seed(10)
n = 125
x1 = rnorm(n)
x2 = rnorm(n, mean=x1, sd=.01)
cor(x1,x2)                            # strong correlation
y = rnorm(n, mean=1+1*x1+2*x2, sd=2)  # f(x) = 1 + 1x_1 +2x_2


#-- Pairs Plot
pairs(cbind(y, x1, x2))

#-- OLS estimation
fit.lm = lm(y~x1 + x2)
coef(fit.lm)

#-- Ridge estimation (using lm.ridge() )
library(MASS)
fit.ridge = MASS::lm.ridge(y~x1+x2, lambda=.05)
coef(fit.ridge)


#-- Fit ridge regression model for sequence of lambdas
lam.seq = exp(seq(log(500),log(1e-5),length=200))
m = lm.ridge(y~x1+x2, lambda=lam.seq)   # ridge regression model
beta = coef(m)                   # matrix of estimated coefficients (for each lambda)
penalty = rowSums(beta[,-1]^2)   # total penalty P(\beta) (sum of squared betas)

ridge = tibble(lam = lam.seq, intercept=beta[,1], x1=beta[,2],
               x2=beta[,3], penalty=penalty, CV=m$GCV)


beta.true = c(1,1,2)
#-- ridge path for sequence of lambdas
matplot(log(m$lambda),coef(m)[,2:3],typ='l',lty=1,xlab="log(lambda)",ylab="beta",las=1)
legend("topright",c("beta_1","beta_2"),col=1:2,lty=1)
abline(h=beta.true[-1],lty=1,col="lightblue")
abline(h=0, col="grey")

#-- ridge path for sequence of lambdas
matplot(penalty,coef(m)[,2:3],typ='l',lty=1,xlab="penalty",ylab="beta",las=1)
legend("topright",c("beta_1","beta_2"),col=1:2,lty=1)
abline(h=beta.true[-1],lty=1,col="lightblue")
abline(h=0, col="grey")

#-- penalty for sequence of lambdas
matplot(log(m$lambda),penalty,typ='l',lty=1,xlab="log(lambda)",ylab="Pen(beta)",las=1)

#-- GCV for sequence of lambdas
matplot(log(m$lambda),m$GCV,typ='l',lty=1,xlab="log(lambda)",ylab="GCV",las=1)
ind = which.min(m$GCV)
points(log(m$lambda[ind]), m$GCV[ind], col="red", pch=19)
m$lambda[ind]  # lambda that minimizes Cross-Validation Error

#-- CV-best ridge model
m.ridge = lm.ridge(y~x1+x2, lambda=m$lambda[ind])
coef(m.ridge)






#---------------------------------------------------------------------------#
#-- Ridge Regression: Prostate Data
#---------------------------------------------------------------------------#
library(glmnet)
## Note: glmnet() requires matrix inputs, not formulas

#-- get matrix inputs
X.train = prostate.train %>% select(-lpsa) %>% as.matrix()
Y.train = prostate.train %>% select(lpsa) %>% as.matrix()


#- Center and Scale X 
X = scale(X.train)            # Center and scale predictors
Y = Y.train 
# Note: you won't normally do the scaling. I just did this here
#  so I can easily calculate the edf (which is based on the scaled predictors)


#-- Fit ridge model with 10 fold cross-validation
set.seed(123)
cvfit = cv.glmnet(X, Y, nfolds=10, alpha=0)
cvfit$lambda.min   # best lambda according to cross-validation
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












