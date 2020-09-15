#####################################################################
## R Code for Lasso
## - See lecture: 03-penalized.pdf
## 
## Michael D. Porter
## Created: Mar 2019; Updated Sept 2020
## For: Data Mining (SYS-6018) at University of Virginia
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
#
# Fit OLS, Ridge, Lasso
# - Ridge and Lasso are a set of models indexed by the tuning parameter lambda
# - Add the penalty, edf, training mse to each model
#---------------------------------------------------------------------------#
library(broom)  # for tidy(), glance(), and augment()

#-- Fit OLS model
fit.ls = lm(Y~X)
B.ls = tidy(fit.ls)                     # Base R: coef(fit.ls)
stats.ls = tibble(edf = nrow(B.ls),      # number of estimated parameters
                  mse = mean(fit.ls$residuals^2))  # training MSE


#-- Create new glance() function for glmnet objects
#   note: assumes coefficients are standardized
glance.glmnet <- function(enet, X, Y){
  #- get coefficient estimates
  beta = tidy(enet, return_zeros=TRUE)
  
  #- get training mse
  yhat = predict(enet, s = enet$lambda, newx=X)
  mse = apply(yhat, 2, function(est) mean((Y - est)^2))
  
  #- get ridge edf estimate (assumes X is standardized)
  ridge_edf <- function(lambda, X) {
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

#-- Fit Ridge Model
library(glmnet)
fit.ridge = glmnet(X, Y, alpha=0, lambda.min.ratio = 1E-9)  # alpha=0 gives ridge

B.ridge = tidy(fit.ridge) # long format. Base R: t(coef(fit.ridge))      

stats.ridge = glance(fit.ridge, X, Y)


# #-- Using MASS::lm.ridge() for Ridge Regression
# lamseq = c(0,exp(seq(log(.01),log(2000),length=1000))) # exponential sequence
# fit.ridge = MASS::lm.ridge(Y~X, lambda=lamseq )
# B.ridge = coef(fit.ridge)             # coefficients


#-- Fit Lasso Model
library(glmnet)
fit.lasso = glmnet(X, Y, alpha=1) # alpha=1 gives lasso

B.lasso = tidy(fit.lasso, return_zeros=TRUE) # long format. Base R: t(coef(fit.ridge))      

stats.lasso = glance(fit.lasso, X, Y)



#---------------------------------------------------------------------------#
#-- Plots
#---------------------------------------------------------------------------#

#-- Path Plots


#- Lasso Model
B.lasso_plot = B.lasso %>% 
  select(-lambda) %>% left_join(stats.lasso, by="step") %>% # add stats info to coefficients
  filter(term != "(Intercept)")     # remove intercept from plots

B.lasso_plot %>% 
  ggplot(aes(penalty, estimate, color=term)) +
  geom_line() + 
  labs(y="coefficient", title="Lasso Path: penalty")

B.lasso_plot %>% 
  ggplot(aes(lambda, estimate, color=term)) +
  geom_line() + 
  labs(y="coefficient", title="Lasso Path: lambda")

B.lasso_plot %>% 
  ggplot(aes(log(lambda), estimate, color=term)) +
  geom_line() + 
  labs(y="coefficient", title="Lasso Path: log lambda")

# one plot with facets
B.lasso_plot %>% 
  pivot_longer(c(lambda, penalty, edf), names_to="metric") %>% 
  ggplot(aes(value, estimate, color=term)) + geom_line() + 
  facet_wrap(~metric, scales="free_x")


#- Ridge Model
B.ridge_plot = B.ridge %>% 
  select(-lambda) %>% left_join(stats.ridge, by="step") %>% # add stats info to coefficients
  filter(term != "(Intercept)") %>%      # remove intercept from plots
  mutate(log.lambda = log(lambda))
  
B.ridge_plot %>% 
  ggplot(aes(penalty, estimate, color=term)) +
  geom_line() + 
  labs(y="coefficient", title="Lasso Path: penalty")

B.ridge_plot %>% 
  ggplot(aes(lambda, estimate, color=term)) +
  geom_line() + 
  labs(y="coefficient", title="Lasso Path: lambda")

B.ridge_plot %>% 
  ggplot(aes(log(lambda), estimate, color=term)) +
  geom_line() + 
  labs(y="coefficient", title="Lasso Path: log lambda")

# one plot with facets
B.ridge_plot %>% 
  pivot_longer(c(log.lambda, penalty, edf), names_to="metric") %>% 
  ggplot(aes(value, estimate, color=term)) + geom_line() + 
  facet_wrap(~metric, scales="free_x")


#-- MSE vs. EDF
ggplot(data=tibble(), aes(edf, mse)) + 
  geom_point(data=stats.lasso, aes(color="lasso")) + 
  geom_line(data=stats.ridge, aes(color="ridge")) + 
  geom_point(data=stats.ls, aes(color="ols")) + 
  scale_x_continuous(breaks=0:9) + 
  scale_colour_discrete(name="")





#-------------------------------------------------------------------------------
#-- Prostate Data: Cross-validation
## Notice here we don't do any scaling, but leave it up to the functions.
## This is the normal use; above was scaled for pedagogical reasons
## - use the foldid argument in cv.glmnet() to ensure same folds used for all
##   models
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








