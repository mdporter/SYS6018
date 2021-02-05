#####################################################################
## R Code for supervised learning.
## - See lecture: supervised_1.pdf
## 
## Michael D. Porter
## Created: Oct 2019
## For: Data Mining (SYS-6018) at University of Virginia
## Website: https://mdporter.github.io/SYS6018/
#####################################################################

#-- Load Required Packages
library(tidyverse)
library(FNN)
library(broom)


#--------------------------------------------------------------------#
#-- Generate Data
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

#-- Generate Data
set.seed(825)                           # set seed for reproducibility
x = sim_x(n)                            # get x values
y = sim_y(x, sd=sd)                     # get y values
data_train = tibble(x, y)               # training data tibble

#-- Scatter plot
gg_example = ggplot(data_train, aes(x,y)) +   # data and global aesthetics
  geom_point() +                              # add points
  geom_vline(xintercept=c(0, .4, .62), col="orange") + # add vertical lines
  scale_x_continuous(breaks=seq(0, 1, by=.20)) # change x axis labels

gg_example                              # saved base plot for later use

# plot(x, y, las=1)
# grid()
# abline(v=c(0, .40, .62), col="orange")

#-- Add true mean line
gg_example + 
  geom_function(fun=f, color="black")

#--------------------------------------------------------------------#
#-- Simple Linear Model
#--------------------------------------------------------------------#

#-- Fitting
m1 = lm(y~x, data=data_train) # fit simple OLS
summary(m1)                   # summary of model

#- tidy output using broom
broom::tidy(m1)               # model coefficients (as a data frame)
broom::glance(m1)             # model properties

#-- Prediction
xseq = seq(0, 1, length=200)        # sequence of 200 equally spaced values from 0 to 1
xeval = tibble(x = xseq)            # make into a tibble object
yhat1 = predict(m1, newdata=xeval)  # vector of yhat's (predictions)

#- tidy predictions using broom::augment()
broom::augment(m1, newdata=xeval)

#-- Plotting
gg_example +                        # re-use base plot
  geom_line(data=tibble(x=xseq, y=yhat1), col="black")  
  # geom_smooth(method="lm")                 # equivalent method

# plot(x, y, las=1)                 # plot data
# lines(xseq, yhat1, col="black")   # add fitted line


#--------------------------------------------------------------------#
#-- Polynomial Models
#--------------------------------------------------------------------#

#-- Fit Quadratic Model
m2 = lm(y~poly(x, degree=2), data=data_train) 
yhat2 = predict(m2, newdata=xeval)  

#- ggplot2 plot
poly.data = tibble(x = xseq, linear=yhat1, quadratic=yhat2) %>%  # long data
  pivot_longer(-x, names_to="model", values_to="y")
gg_example + 
  geom_line(data=poly.data, aes(color=model)) 

#- using geom_smooth() to fit automatically  
gg_example + 
  geom_smooth(method="lm", se=FALSE, aes(color="linear")) + 
  geom_smooth(method="lm", formula="y~poly(x,2)", se=FALSE, aes(color="quadratic")) + 
  scale_color_discrete(name="model")
    
# #- base R plot
# plot(x, y, las=1)                 
# lines(xseq, yhat1, col="black")   
# lines(xseq, yhat2, col="red")
# legend("topright", 
#        c("linear", "quadratic"), 
#        col=c("black", "red"), 
#        lty=1, cex=.8)

#--------------------------------------------------------------------#
#-- kNN regression
#--------------------------------------------------------------------#
library(FNN)                   # For kNN regression

#-- fit a k=20 knn regression
knn.20 = knn.reg(select(data_train, x), test=xeval, y=data_train$y, k=20)

#-- ggplot2 plot
gg_example + 
  geom_line(data=tibble(x=xseq, y=knn.20$pred)) + 
  ggtitle("kNN k=20")

#-- base R plot
plot(x, y, las=1) 
lines(xseq, knn.20$pred)


