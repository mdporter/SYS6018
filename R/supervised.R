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





