#####################################################################
## R Code for supervised learning.
## - See lecture: 06-supervised.pdf
## 
## Michael D. Porter
## Created: Oct 2019
## For: Data Mining (SYS-6018) at University of Virginia
## Website: https://mdporter.github.io/SYS6018/
#####################################################################

#-- Load Required Packages
library(tidyverse)
library(FNN)


#--------------------------------------------------------------------#
#-- Generate Data
#--------------------------------------------------------------------#

#-- Settings
n = 100                                 # number of observations
sim_x <- function(n) runif(n)           # U[0,1]
f <- function(x) 1 + 2*x + 5*sin(5*x)   # true mean function
sd = 2                                  # stdev for error
sim_y <- function(x, sd){               # generate Y|X from N{f(x),sd}
  n = length(x)
  f(x) + rnorm(n, sd=sd)             
}

#-- Generate Data
set.seed(825)                           # set seed for reproducibility
x = sim_x(n)                            # get x values
y = sim_y(x, sd=sd)                     # get y values


#-- Scatter plots
plot(x, y, las=1)
grid()
abline(v=c(0, .40, .62), col="lightblue")

ggplot(tibble(x,y), aes(x,y)) + 
  geom_point() + 
  geom_vline(xintercept=c(0, .4, .62), col="lightblue") + 
  scale_x_continuous(breaks=seq(0, 1, by=.20))


#--------------------------------------------------------------------#
#-- Simple Linear Model
#--------------------------------------------------------------------#

#-- Fitting
data_train = tibble(x,y)
m1 = lm(y~x, data=data_train) # fit simple OLS
summary(m1)                   # summary of model

#-- Prediction
xseq = seq(0, 1, length=200)        # sequence of equally spaced values from 0 to 1
xeval = tibble(x = xseq)            # make into a tibble object
yhat1 = predict(m1, newdata=xeval)  # vector of yhat's (predictions)

#-- Plotting
plot(x, y, las=1)                 # plot data
lines(xseq, yhat1, col="black")   # add fitted line

ggplot(data_train, aes(x,y)) + 
  geom_point() + 
  geom_line(data=tibble(x=xseq, y=yhat1), col="black") + 
  # geom_smooth(method="lm") +                # equivalent
  geom_vline(xintercept=c(0, .4, .62), col="lightblue")  + 
  scale_x_continuous(breaks=seq(0, 1, by=.20))
  

#--------------------------------------------------------------------#
#-- Polynomial Models
#--------------------------------------------------------------------#

#-- Fit Quadratic Model
m2 = lm(y~poly(x, degree=2), data=data_train) 
yhat2 = predict(m2, newdata=xeval)  


#- base R plot
plot(x, y, las=1)                 
lines(xseq, yhat1, col="black")   
lines(xseq, yhat2, col="red")
legend("topright", 
       c("linear", "quadratic"), 
       col=c("black", "red"), 
       lty=1, cex=.8)

#- ggplot2 plot
poly.data = tibble(x = xseq, linear=yhat1, quadratic=yhat2) %>% 
  gather(model, y, -x)
ggplot(tibble(x,y), aes(x,y)) + 
  geom_point() + 
  geom_line(data=poly.data, aes(color=model)) + 
  geom_vline(xintercept=c(0, .4, .62), col="lightblue")  + 
  scale_x_continuous(breaks=seq(0, 1, by=.20))

#- using geom_smooth() to fit automatically  
ggplot(tibble(x,y), aes(x,y)) + 
    geom_point() + 
    geom_smooth(method="lm", se=FALSE, color="red") + 
    geom_smooth(method="lm", formula="y~poly(x,2)", se=FALSE, color="blue") + 
    geom_vline(xintercept=c(0, .4, .62), col="lightblue")  + 
    scale_x_continuous(breaks=seq(0, 1, by=.20))  


#--------------------------------------------------------------------#
#-- kNN regression
#--------------------------------------------------------------------#
library(FNN)                   # For kNN regression

#-- fit a k=20 knn regression
knn.20 = knn.reg(tibble(x), test=xeval, y=y, k=20)

#-- base R plot
plot(x, y, las=1) 
lines(xseq, knn.20$pred)

#-- ggplot2 plot
ggplot(tibble(x,y), aes(x,y)) + 
  geom_point() + 
  geom_line(data=tibble(x=xseq, y=knn.20$pred)) + 
  geom_vline(xintercept=c(0, .4, .62), col="lightblue")  + 
  scale_x_continuous(breaks=seq(0, 1, by=.20)) + 
  ggtitle("kNN k=20")



#--------------------------------------------------------------------#
#-- Evaluate Models
#--------------------------------------------------------------------#

#-- Generate Test Data
ntest = 50000                           # Number of test samples
set.seed(235)                           # set *different* seed 
xtest = sim_x(ntest)                    # generate test X's
ytest = sim_y(xtest, sd=sd)             # generate test Y's

#-- Function to evaluate polynomials
poly_eval <- function(deg, data_train, data_test){
  if(deg==0) m = lm(y~1, data=data_train)  # intercept only model
  else m = lm(y~poly(x, degree=deg), data=data_train) # polynomial
  p = length(coef(m))                      # number of parameters
  mse.train = mean(m$residuals^2)          # training MSE
  yhat = predict(m, newdata=data_test)     # predictions at test X's
  mse.test = mean( (data_test$y - yhat)^2 )# test MSE
  tibble(degree=deg, edf=p, mse.train, mse.test)
}

#-- Evaluate polynomials
Deg = c(1:6, seq(8, 20, by=2))   # sequence of Degrees
poly.eval = tibble()
for(deg in Deg){
  tmp = poly_eval(deg, data_train=tibble(x,y), data_test=tibble(x=xtest, y=ytest))
  poly.eval = rbind(poly.eval, tmp)
}

#-- base R plot
with(poly.eval, plot(edf, mse.train, type='l', las=1))
with(poly.eval, lines(edf, mse.test, col=2))

#-- ggplot2
poly.eval %>% gather(data, mse, mse.train, mse.test) %>% 
  mutate(data = str_remove(data, "mse\\.")) %>% 
  ggplot(aes(degree, mse, color=data)) + geom_line() + geom_point() + 
  labs(title="Polynomial Regression", 
        y="MSE") + 
  scale_x_continuous(breaks=seq(2, 30, by=2))

#-- best poly
filter(poly.eval, mse.test == min(mse.test))


#-- Function to evaluate kNN
knn_eval <- function(k, data_train, data_test){
  knn = knn.reg(data_train[,'x', drop=FALSE], 
                y = data_train$y, 
                test=data_train[,'x', drop=FALSE], 
                k=k)
  edf = nrow(data_train)/k        # effective dof (edof)
  r = data_train$y-knn$pred        # residuals on training data  
  mse.train = mean(r^2)            # training MSE
  knn.test = knn.reg(data_train[,'x', drop=FALSE], 
                     y = data_train$y, 
                     test=data_test[,'x', drop=FALSE], 
                     k=k)
  r.test = data_test$y-knn.test$pred # residuals on test data
  mse.test = mean(r.test^2)          # test MSE
  tibble(k=k, edof=edf, mse.train, mse.test)
}

#-- Evaluate kNN
K = c(100, 50, 35, 25, 20, 15, 12, 10, 8, 7, 6, 5, 3 )
knn.eval = tibble()
for(k in K){
  tmp = knn_eval(k, data_train=tibble(x,y), data_test=tibble(x=xtest, y=ytest))
  knn.eval = rbind(knn.eval, tmp)
}

#-- base R plot
with(knn.eval, plot(edof, mse.train, type='l', las=1))
with(knn.eval, lines(edof, mse.test, col=2))

#-- ggplot2
knn.eval %>% gather(data, mse, mse.train, mse.test) %>% 
  mutate(data = str_remove(data, "mse\\.")) %>% 
  ggplot(aes(edof, mse, color=data)) + geom_line() + geom_point() + 
  labs(title="kNN Regression", 
       x="Effective degrees of freedom (edf)", y="MSE") + 
  scale_x_continuous(breaks=seq(1, 100, by=2))

#-- best knn
filter(knn.eval, mse.test == min(mse.test))
filter(knn.eval, min_rank(mse.test) == 1)
top_n(knn.eval, n=1, -mse.test)

#-- Optimal MSE
# theoretically, the optimal MSE is sd^2 
# but there will be some error when estimating with a finite test set
mean((f(xtest) - ytest)^2)



#--------------------------------------------------------------------#
#-- Ensemble Evaluation
#--------------------------------------------------------------------#
library(FNN)
library(tidyverse)


#-- Create Ensemble weights from class input
ensemble.data = tribble(
  ~model, ~tuning, ~n,
  'poly', 1, 1,
  'poly', 2, 1, 
  'poly', 3, 14,
  'poly', 5, 8,
  'poly', 10, 5,
  'poly', 20, 4, 
  'knn', 50, 1, 
  'knn', 30, 1, 
  'knn', 20, 2, 
  'knn', 15, 1, 
  'knn', 10, 6,
  'knn', 5, 2
) %>% mutate(w = n/sum(n))


#-- Create helper functions to make predictions
pred_knn <- function(k, data_train, data_test){
  m = knn.reg(data_train[,'x', drop=FALSE], 
              y = data_train$y, 
              test = data_test[,'x', drop=FALSE], 
              k = k)
  m$pred
}

pred_poly <- function(deg, data_train, data_test){
  m = lm(y~poly(x, degree=deg), data=data_train)
  predict(m, newdata=data_test)
}


#-- make training and test data sets. 
#   Ensure they have same column names
data_train = tibble(x, y) 
data_test = tibble(x=xtest, y=ytest)

#-- Create matrix of predictions from each model
YHAT = matrix(NA, nrow=nrow(data_test), ncol=nrow(ensemble.data))
for(i in 1:nrow(ensemble.data)){
  if(ensemble.data$model[i] == "poly"){
    deg = ensemble.data$tuning[i]
    YHAT[,i] = pred_poly(deg=deg, data_train, data_test)
  }
  if(ensemble.data$model[i] == "knn"){
    k = ensemble.data$tuning[i]
    YHAT[,i] = pred_knn(k=k, data_train, data_test)
  }
}

#-- Ensemble prediction is weighted sum of individual predictions
yhat = YHAT %*% ensemble.data$w    

#-- MSE of ensemble
mean( (data_test$y - yhat)^2 )


#-- Add results to the list
ensemble.data %>%
  # Add the MSE for each model individually
  mutate(mse=apply(YHAT, 2, function(yhat) mean((data_test$y - yhat)^2))) %>% 
  # Add row for ensemble
  bind_rows(tibble(model="ensemble", mse=mean( (data_test$y - yhat)^2 ))) %>% 
  # order/arrange results by mse
  arrange(mse)



#--------------------------------------------------------------------#
#-- Two training sets
#--------------------------------------------------------------------#
#-- Settings
n = 100                                 # number of observations
sim_x <- function(n) runif(n)           # U[0,1]
f <- function(x) 1 + 2*x + 5*sin(5*x)   # true mean function
sd = 2                                  # stdev for error
sim_y <- function(x, sd){               # generate Y|X from N{f(x),sd}
  n = length(x)
  f(x) + rnorm(n, sd=sd)             
}

#-- Generate 1st Training Data (same as above)
set.seed(825)                           # set seed for reproducibility
x = sim_x(n)                            # get x values
y = sim_y(x, sd=sd)                     # get y values
data1 = tibble(x,y)

#-- Generate 2nd training data
set.seed(826)
data2 = tibble(x=sim_x(n), y=sim_y(x,sd=sd))  # use tibble() so y uses x=sim_x

#-- Fit model for each training data
poly1 = lm(y~poly(x, degree=4), data=data1)
poly2 = lm(y~poly(x, degree=4), data=data2)

#-- Plots (base R)
plot(y~x, data=data1, pch=19,col="#E5720050", las=1)
points(data2$x, data2$y, pch=19, col="#232D4B50")
lines(xseq, predict(poly1, newdata=tibble(x=xseq)), col="#E57200")
lines(xseq, predict(poly2, newdata=tibble(x=xseq)), col="#232D4B")

#-- Plots (ggplot2)
data.points = bind_rows(`train_set=1` = data1, `train_set=2`=data2, .id="data")
data.pred = tibble(x=xseq, 
                   `train_set=1` = predict(poly1, newdata=tibble(x=xseq)), 
                   `train_set=2` = predict(poly2, newdata=tibble(x=xseq))) %>% 
  gather(data, y, -x)

ggplot(data.points, aes(x,y, color=data)) +
  geom_point() + 
  geom_line(data=data.pred) + 
  scale_color_manual(values=c("#232D4B", "#E57200"), name="data set") + 
  ggtitle("polynomial, degree=4")


#--------------------------------------------------------------------#
#-- Bias-Variance Evaluation
#--------------------------------------------------------------------#
#- Run M simulations; each time draw training data from P(X,Y)


#-- Distributions
sim_x <- function(n) runif(n)           # generate n obs from U[0,1]
f <- function(x) 1 + 2*x +5*sin(5*x)    # true mean function
sim_y <- function(x){                   # generate Y|X from N{f(x),sd}
  n = length(x)
  f(x) + rnorm(n, sd=2)
}

xseq = seq(0, 1, length=200)            # sequence of equally spaced values from 0 to 1



#-- Single Realization
set.seed(825)                           # set seed for exact replication
n = 100                                 # number of obs
x = sim_x(n)
y = sim_y(x)

plot(x,y,pch=19,col="#00000080", las=1)
curve(f,add=TRUE,lwd=2,col="#808080C0")



#-- function to generate prediction function for polynomial regression
poly_predict <- function(deg, x, y, xseq){
  xeval = tibble(x=xseq)
  if(deg == 0) return( rep(mean(y), length(xseq)))
  m = lm(y~poly(x, degree=deg))
  predict(m, newdata=xeval)
}


#-- Settings
set.seed(2015)                          # set seed
M = 2000                                # number of simulations

#-- Initialization
Degree = c(0,1,2,4,8)   # degree sequence for polynomial
YHAT = array(0, dim=c(length(xseq), M, length(Degree)), 
             dimnames=list(NULL, 1:M, paste0('deg=',Degree))) 

#-- Run Simulation: Generate sequence of yhat's
for(m in 1:M){
  x = sim_x(n)
  y = sim_y(x)
  for(j in 1:length(Degree)){
    YHAT[, m, j] = poly_predict(deg=Degree[j], x, y, xseq)  # yhat
  }
}



#-- Function to evaluate simulation results
eval_metrics <- function(Yhat, mu, irr.error){
  bias = rowMeans(Yhat) - mu        # E[f_D(X)] - mu.true
  bias.sq = bias^2
  var = apply(Yhat, 1, var)         # V[f_D(X)] 
  mse = var + bias.sq + irr.error
  df = tibble(x=xseq, mse, bias, bias.sq, var)
  return(df)  
}

#-- Evaluate results
perf = apply(YHAT, 3, eval_metrics, mu=f(xseq), irr.error=sd^2)  # calculate metrics for each model

# Note: apply() is is a shortcut for the loop
# perf = list()
# for(j in 1:dim(YHAT)[3]){
#   yhat = YHAT[,,j]
#   perf[[j]] = eval_metrics(yhat, mu=f(xseq))
# }

#-- Plots: Mean Squared Error
with(perf$`deg=0`, plot(x, mse, type='l', col="#569BBD", ylab="MSE", ylim=c(4, 19) ))
with(perf$`deg=1`, lines(x, mse, col="#F05133" ))
with(perf$`deg=2`, lines(x, mse, col="#4C721D" ))
with(perf$`deg=4`, lines(x, mse, col="#F4DC00" ))
with(perf$`deg=8`, lines(x, mse, col="#FF6600" ))
title("Mean Square Error")
legend("top",c("intercept",'linear',"quadratic","poly4", 'poly8'),
       col=c("#569BBD","#F05133","#4C721D","#F4DC00", "#FF6600"),lwd=2,cex=.7)


#-- Interated MSE 
IMSE = sapply(perf,  colMeans)
IMSE[-c(1,3),] %>% knitr::kable(digits=2)


#-- get the estimated MSE for every simulation and every model
est_mse <- function(yhat, sd) apply(yhat, 2, function(x) sd^2+mean((x-f(xseq))^2)) 
MSE.matrix = apply(YHAT, 3, est_mse, sd=sd )           # matrix of estimate MSE

#-- Count of best model
best.model = apply(MSE.matrix, 1, which.min)   # vector of best model
table(colnames(MSE.matrix)[best.model])        # counts of best model

#-- same, but using dplyr
MSE = MSE.matrix %>% 
  as_tibble()  %>%                    # convert to data frame
  mutate(sim_id = row_number()) %>%   # add simulation number
  gather(model, mse, -sim_id)         # convert to long format


MSE %>% 
  group_by(sim_id) %>% filter(min_rank(mse) == 1) %>% # keep best model  
  ungroup() %>% count(model)                          # get count of best models 

