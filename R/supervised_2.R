#####################################################################
## R Code for supervised learning.
## - See lecture: supervised_2.pdf
## 
## Michael D. Porter
## Created: Oct 2019; updated Aug 2020
## For: Data Mining (SYS-6018) at University of Virginia
## Website: https://mdporter.github.io/SYS6018
#####################################################################

#-- Load Required Packages
library(tidyverse)
library(FNN)


#--------------------------------------------------------------------#
#-- Training and Test Data
#--------------------------------------------------------------------#

#-- Functions
sim_x <- function(n) runif(n)           # U[0,1]
f <- function(x) 1 + 2*x + 5*sin(5*x)   # true mean function
sim_y <- function(x, sd){               # generate Y|X from N{f(x),sd}
  n = length(x)
  f(x) + rnorm(n, sd=sd)             
}

#-- Settings
n = 100                                 # number of observations
sd = 2                                  # stdev for error

#-- Generate Training Data 
set.seed(825)                           # set seed for reproducibility
x = sim_x(n)                            # get x values
y = sim_y(x, sd=sd)                     # get y values
data_train = tibble(x, y)               # training data tibble

#-- Generate Test Data
ntest = 50000                           # Number of test samples
set.seed(235)                           # set *different* seed 
xtest = sim_x(ntest)                    # generate test X's
ytest = sim_y(xtest, sd=sd)             # generate test Y's
data_test = tibble(x=xtest, y=ytest)    # test data


#--------------------------------------------------------------------#
#-- Evaluate Polynomial Models
#--------------------------------------------------------------------#

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

# Using purrr:map_df()
data_poly = map_df(Deg, poly_eval, data_train=data_train, data_test=data_test)

# Using a loop
data_poly = tibble()
for(deg in Deg){
  tmp = poly_eval(deg, data_train=data_train, data_test=data_test)
  data_poly = bind_rows(data_poly, tmp)
}

#-- Plot Results
data_poly %>% 
  # make long:
  pivot_longer(starts_with("mse"), names_to="data", values_to="mse") %>% 
  # remove 'mse.' from values:
  mutate(data = str_remove(data, "mse\\.")) %>%   
  # plot:
  ggplot(aes(edf, mse, color=data)) + geom_line() + geom_point() + 
  labs(title="Polynomial Regression", x = 'edf (degree + 1)', y="MSE") + 
  scale_x_continuous(breaks=seq(2, 30, by=2)) 

#- base R plot
with(data_poly, plot(edf, mse.train, type='l', las=1))
with(data_poly, lines(edf, mse.test, col=2))


#-- best poly
data_poly %>% filter(mse.test == min(mse.test))


#--------------------------------------------------------------------#
#-- Evaluate kNN Models
#--------------------------------------------------------------------#

#-- Function to evaluate kNN
knn_eval <- function(k, data_train, data_test){
  knn = knn.reg(data_train[,'x', drop=FALSE], 
                y = data_train$y, 
                test=data_train[,'x', drop=FALSE], 
                k=k)
  edf = nrow(data_train)/k         # effective dof (edof)
  r = data_train$y-knn$pred        # residuals on training data  
  mse.train = mean(r^2)            # training MSE
  knn.test = knn.reg(data_train[,'x', drop=FALSE], 
                     y = data_train$y, 
                     test=data_test[,'x', drop=FALSE], 
                     k=k)
  r.test = data_test$y-knn.test$pred # residuals on test data
  mse.test = mean(r.test^2)          # test MSE
  tibble(k=k, edf=edf, mse.train, mse.test)
}

#-- Evaluate kNN
K = c(50, 35, 25, 20, 18, 15, 12, 10, 8, 7, 6, 5)

# Using purrr:map_df()
data_knn = map_df(K, knn_eval, data_train=data_train, data_test=data_test)

# Using a loop
data_knn = tibble()
for(k in K){
  tmp = knn_eval(k, data_train=data_train, data_test=data_test)
  data_knn = bind_rows(data_knn, tmp)
}

#-- Plot Results
data_knn %>% 
  # make long:
  pivot_longer(starts_with("mse"), names_to="data", values_to="mse") %>% 
  # remove 'mse.' from values:
  mutate(data = str_remove(data, "mse\\.")) %>%   
  # plot:
  ggplot(aes(edf, mse, color=data)) + geom_line() + geom_point() + 
  labs(title="kNN Regression", x='edf (n/k)', y="MSE") + 
  scale_x_continuous(breaks=seq(2, 30, by=2)) 

#- base R plot
with(data_knn, plot(edf, mse.train, type='l', las=1))
with(data_knn, lines(edf, mse.test, col=2))
grid()


#-- best knn
data_knn %>% slice_min(mse.test)


#--------------------------------------------------------------------#
#-- Optimal Results
#--------------------------------------------------------------------#
#-- Optimal MSE
# theoretically, the optimal MSE is sd^2 
# but there will be some error when estimating with a finite test set

mean((sim_y(xtest, sd=0) - ytest)^2)



#--------------------------------------------------------------------#
#-- Ensemble Evaluation
#--------------------------------------------------------------------#
library(FNN)
library(tidyverse)


#-- Create Ensemble weights from class input
ensemble.data = tribble(
  ~model, ~tuning, ~n,
  'knn',  5, 2, 
  'knn', 10, 2, 
  'knn', 20, 2, 
  'poly',  3, 2, 
  'knn',  9, 1, 
  'knn', 50, 1
) %>% mutate(w = n/sum(n)) %>% 
  arrange(-n)


#-- Create helper functions to make predictions
pred_knn <- function(k, data_train, data_test){
  m = knn.reg(select(data_train, x), 
              y = data_train$y, 
              test = select(data_test, x), 
              k = k)
  m$pred
}

pred_poly <- function(deg, data_train, data_test){
  if(deg == 0) m = lm(y~1, data=data_train)
  else m = lm(y~poly(x, degree=deg), data=data_train)
  predict(m, newdata=data_test)
}


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
yhat.ensemble = YHAT %*% ensemble.data$w    

#-- MSE of ensemble
mean( (data_test$y - yhat.ensemble)^2 )


#-- Add results to the list
ensemble.data %>%
  # Add the MSE for each model individually
  mutate(mse=apply(YHAT, 2, function(yhat) mean((data_test$y - yhat)^2))) %>% 
  # Add row for ensemble
  bind_rows(tibble(model="ensemble", mse=mean( (data_test$y - yhat.ensemble)^2 ))) %>% 
  # order/arrange results by mse
  arrange(mse)



#--------------------------------------------------------------------#
#-- Two training sets
#--------------------------------------------------------------------#
#-- Settings
n = 100                                 # number of observations
sd = 2                                  # stdev for error

#-- Generate 1st Training Data (same as above)
set.seed(825)                           # set seed for reproducibility
x = sim_x(n)                            # get x values
y = sim_y(x, sd=sd)                     # get y values
data1 = tibble(x, y)

#-- Generate 2nd training data
set.seed(826)
data2 = tibble(x=sim_x(n), y=sim_y(x, sd=sd))  # use tibble() so y uses x=sim_x

#-- Fit model for each training data
poly1 = lm(y~poly(x, degree=4), data=data1)
poly2 = lm(y~poly(x, degree=4), data=data2)
data_eval = tibble(x=seq(0, 1, length=200))

#-- Plots (ggplot2 #1)
data.points = bind_rows(`train_set=1` = data1, `train_set=2`=data2, .id="data")
data.pred = tibble(x=seq(0, 1, length=200), 
                   `train_set=1` = predict(poly1, newdata=data_eval), 
                   `train_set=2` = predict(poly2, newdata=data_eval)) %>% 
  pivot_longer(-x, names_to="data", values_to="y")

ggplot(data.points, aes(x,y, color=data)) +
  geom_point() + 
  geom_line(data=data.pred) + 
  scale_color_manual(values=c("#232D4B50", "#E5720050"), name="data set") + 
  ggtitle("polynomial, degree=4")

#-- Plots (ggplot2 #2)
ggplot(tibble(), aes(x, y)) + 
  geom_point(data=data1, aes(color="#232D4B50")) + 
  geom_point(data=data2, aes(color="#E5720050")) + 
  geom_line(data=data_eval %>% mutate(y=predict(poly1, newdata=.)), 
            aes(color="#232D4B")) +
  geom_line(data=data_eval %>% mutate(y=predict(poly2, newdata=.)), 
            aes(color="#E57200")) +
  scale_color_identity() + 
  ggtitle("polynomial, degree=4")


#-- Plots (base R)
plot(y~x, data=data1, pch=19,col="#E5720050", las=1)
points(data2$x, data2$y, pch=19, col="#232D4B50")
xseq = data_eval$x
lines(xseq, predict(poly1, newdata=tibble(x=xseq)), col="#E57200")
lines(xseq, predict(poly2, newdata=tibble(x=xseq)), col="#232D4B")

#--------------------------------------------------------------------#
#-- Bias-Variance Evaluation
#--------------------------------------------------------------------#
#- Run M simulations; each time draw training data from P(X,Y)


#-- Function to fit and predict from polynomial model
pred_poly <- function(deg, data_train, data_test){
  if(deg == 0) m = lm(y~1, data=data_train)
  else m = lm(y~poly(x, degree=deg), data=data_train)
  data_test %>% mutate(deg, y=predict(m, newdata=data_test))
}

#-- Settings
set.seed(2015)                          # set seed
M = 1000                                # number of simulations

#-- Initialization
Degree = c(0,1,2,4,8)   # degree sequence for polynomial
data_eval = tibble(x = seq(0, 1, length=200))

#-- Run Simulation: Generate sequence of yhat's
Yhat = tibble()
for(m in 1:M){
  data = tibble(x=sim_x(n), y=sim_y(x, sd=sd))
  res = map_df(Degree, pred_poly, data_train=data, data_test=data_eval)
  Yhat = bind_rows(Yhat, res %>% mutate(iter=m))
}


#-- evaluate simulation results
eval_metrics = Yhat %>% 
  mutate(f = f(x)) %>%   # add true mean
  group_by(x, deg) %>%   # group by x-values
  summarize(bias = mean(y-f), bias.sq = bias^2, var=var(y), 
            mse = var + bias.sq + sd^2 )


eval_metrics %>% select(-bias) %>% 
  mutate(model = ifelse(deg==0, "Intercept Only", str_c('poly(deg=', deg, ')'))) %>% 
  pivot_longer(c(-x, -deg, -model), names_to="metric", values_to="y") %>% 
  ggplot(aes(x, y, color=model)) + geom_line() + 
  facet_wrap(~metric, ncol=1, scales="free_y")







