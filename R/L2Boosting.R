#####################################################################
## R Code for L2 Boosting
## - See lecture: boosting.pdf
## 
## Michael D. Porter
## Created: April 2019
## For: Data Mining (SYS-6018/SYS-4582) at University of Virginia
## https://mdporter.github.io/SYS6018/
#####################################################################


#-- L2 Boost Algorithm
library(rpart)

# L2boost()
#------------------------------------------------------------------------------#
# L2 boosted trees (boosted regression trees)
# Inputs:
#  x,y: training data. x should be data frame or matrix, y a vector
#  x.test optional test data (data frame or matrix)
#  M: number of iterations
#  depth: tree depth. depth = 2 gives 4 leaf nodes.
#  nu: shrinkage parameter
# Outputs:
#  YHAT: matrix of in-sample predictions (predicting x)
#  R: matrix of residuals
#  YHAT.test: matrix of predictions for x.test
#  TREE: list of rpart trees  
#------------------------------------------------------------------------------#

L2boost <- function(x, y, x.test=NULL, M=100, depth=2, nu=.1){

  #- use training data if test data is not specified
  if(is.null(x.test)) {
    x.test = x
  } 
  
  #- storage
  n = length(y)  
  R = YHAT =  matrix(NA_real_, n, M) 
  YHAT.test = matrix(NA_real_, nrow(x.test), M) 
  colnames(YHAT) = colnames(YHAT.test) = colnames(R) = paste0("iter = ", 1:M)
  TREE = vector("list", M)
  names(TREE) = paste0("iter = ", 1:M)
  
  #-- 1) initialize model
  yhat = rep(mean(y), n)
  yhat.test = yhat
  
  for(m in 1:M){
    
    #-- 2a) Calculate Residuals
    r = y - yhat
    R[,m]=r
    
    #-- 2b) Fit regression tree
    tree = rpart(r ~ ., data=x,
                 maxdepth = depth,  # control tree depth
                 cp = -1,           # no pruning
                 minsplit = 0,      # allow all splits   
                 minbucket = 1,     # no minimum on leaf size
                 method = "anova",  # least-squares loss function
                 xval = 0)          # no cross-validation
    
    TREE[[m]] = tree
    
    #-- 2c) Update model
    yhat = yhat + nu*predict(tree, x)
    YHAT[, m] = yhat
    
    yhat.test = yhat.test + nu*predict(tree, x.test)
    YHAT.test[, m] = yhat.test 
    
  }
  
  #-- 3) Output 
  return(list(YHAT=YHAT, R=R, YHAT.test=YHAT.test, TREE=TREE))
}


#-- Data Generation
n = 100                                 # number of observations
generate_x <- function(n) runif(n)      # U[0,1]
f <- function(x) 1 + 2*x + 5*sin(5*x)   # true mean function
sd = 2                                  # stdev for error

set.seed(825)                           # set seed for reproducibility
x = generate_x(n)                       # get x values
y = f(x) + rnorm(n, sd=sd)              # get y values
data_train = data.frame(x, y)           # training data
x_eval = seq(0, 1, length=500)          # evaluation points

#-- L2 boosting
L2 = L2boost(data.frame(x), y, x.test=data.frame(x=x_eval),  # data
             depth = 1, M = 100, nu = .1)             # tuning parameters

#-- Plotting 
library(tidyverse)   # for ggplot2 package
library(rpart.plot)  # for prp()

# set iteration
i = 1  

# Residual Plot
ggplot(data_train, aes(x)) + 
  geom_point(aes(y = L2$R[,i]), col="black") + 
  geom_hline(yintercept=0, col="black")  + 
  scale_x_continuous(breaks=seq(0, 1, by=.20)) +
  coord_cartesian(ylim=c(-8, 8)) +
  labs(y="residual", title=colnames(L2$R)[i])

# Tree
prp(L2$TREE[[i]], type=1, extra=1, branch=1, roundint=FALSE)

# Model prediction
ggplot(data_train, aes(x, y)) + 
  geom_point() + 
  annotate("line", x=x_eval, y=f(x_eval), color = "black") + 
  geom_line(data=tibble(x=x_eval, y=L2$YHAT.test[,i]), col=2) + 
  scale_x_continuous(breaks=seq(0, 1, by=.20)) +
  labs(title=colnames(L2$R)[i])


