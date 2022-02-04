#####################################################################
## R Code for bootstrap
## - See lecture: bootstrap.pdf
## 
## Michael D. Porter
## Created: Oct 2019
## For: Data Mining (SYS6018) at University of Virginia
## https://mdporter.github.io/SYS6018/
#####################################################################

#-- Install Required Packages
library(broom)
library(splines)
library(tidyverse)


#-----------------------------------------------------------------------------
#-- Simulate Data
#-----------------------------------------------------------------------------
n = 100                                 # number of observations
sim_x <- function(n) runif(n)           # U[0,1]
f <- function(x) 1 + 2*x + 5*sin(5*x)   # true mean function
sd = 2                                  # stdev for error

set.seed(825)                           # set seed for reproducibility
x = sim_x(n)                            # get x values
y = f(x) + rnorm(n, sd=sd)              # get y values
data_train = tibble(x,y)                # create a data frame/tibble

#-----------------------------------------------------------------------------
#-- Fit Linear Model
#-----------------------------------------------------------------------------
m1 = lm(y~x, data=data_train)  # fit simple OLS
broom::tidy(m1, conf.int=TRUE) # OLS estimated coefficients
vcov(m1)                       # variance matrix

#-----------------------------------------------------------------------------
#-- Bootstrap distribution
#-----------------------------------------------------------------------------
M = 2000                           # number of bootstrap samples
beta = list()                      # initialize list for test statistics
set.seed(201910)                   # set random seed
for(m in 1:M){
  #- sample from empirical distribution
  ind = sample(n, replace=TRUE)    # sample indices with replacement
  data.boot = data_train[ind,]     # bootstrap sample
  #- fit regression model
  m.boot = lm(y~x, data=data.boot) # fit simple OLS
  #- save test statistics
  beta[[m]] = broom::tidy(m.boot) %>% select(term, estimate)
}
#- convert to tibble (and add column names)
beta = bind_rows(beta, .id = "iteration") %>% 
  pivot_wider(names_from = term, values_from=estimate) %>% 
  select(intercept = "(Intercept)", slope = "x", -iteration)

#- Plot
ggplot(beta, aes(intercept, slope)) + geom_point() + 
  geom_point(data=tibble(intercept=coef(m1)[1], 
                         slope = coef(m1)[2]), color="red", size=4)

#- bootstrap estimate
var(beta)            # variance matrix
apply(beta, 2, sd)   # standard errors (sqrt of diagonal)

#-----------------------------------------------------------------------------
#-- Fit a B-spline model
#-----------------------------------------------------------------------------
#- fit a 5 df B-spline
# Note: don't need to include an intercept in the lm()
# Note: the boundary.knots are set just a bit outside the range of the data
#       so prediction is possible outside the range (see below for usage). 
#       You probably won't need to set this in practice, unless you need 
#       predictions for outside the range of your training data. 
kts.bdry = c(-.2, 1.2)          
model_bs = lm(y~bs(x, df=5, deg=3, Boundary.knots = kts.bdry)-1,
              data=data_train)
tidy(model_bs)
ggplot(data_train, aes(x,y)) + geom_point() + 
  geom_smooth(method='lm', formula='y~bs(x, df=5, deg=3, Boundary.knots = kts.bdry)-1')

#-- Evaluate the B-spline Basis
B = bs(x, df=7, deg=3, Boundary.knots = kts.bdry)
matplot(x, B, type='p')

#-----------------------------------------------------------------------------
#-- Bootstrap Uncertainty in B-spline Fit
#-----------------------------------------------------------------------------
M = 100                                     # number of bootstrap samples
data_eval = tibble(x=seq(0, 1, length=300)) # evaluation points
YHAT = matrix(NA, nrow(data_eval), M)       # initialize matrix for fitted values

#-- Spline Settings
for(m in 1:M){
  #- sample from empirical distribution
  ind = sample(n, replace=TRUE)               # sample indices with replacement
  #- fit bspline model
  m_boot = lm(y~bs(x, df=5, Boundary.knots=kts.bdry)-1, 
              data=data_train[ind,])   # fit bootstrap data
  #- predict from bootstrap model
  YHAT[,m] = predict(m_boot, data_eval)
}

#-- Convert to tibble and plot
data_fit = as_tibble(YHAT) %>%  # convert matrix to tibble
  bind_cols(data_eval) %>%      # add the eval points
  pivot_longer(-x, names_to="simulation", values_to="y") # convert to long format

ggplot(data_train, aes(x,y)) + 
  geom_smooth(method='lm', 
              formula='y~bs(x, df=5, deg=3, Boundary.knots = kts.bdry)-1') + 
  geom_line(data=data_fit, color="red", alpha=.10, aes(group=simulation)) +   
  geom_point() 

#-----------------------------------------------------------------------------
#-- Out-of-bag performance evaluation
#-----------------------------------------------------------------------------
M = 500                   # number of bootstrap samples
DF = seq(4, 15, by=1)     # edfs for spline
results = list()          # initialize results list
set.seed(2019)            # set seed so reproducible

#-- Spline Settings
for(m in 1:M){
  #- sample from empirical distribution
  ind = sample(n, replace=TRUE)       # sample indices with replacement
  oob.ind = setdiff(1:n, ind)         # out-of-bag samples      
  #- fit bspline models
  for(df in DF){
    if(length(oob.ind) < 1) next
    #- fit with bootstrap data
    m_boot = lm(y~bs(x, df=df, Boundary.knots=kts.bdry)-1, 
                data=data_train[ind,])        
    #- predict on oob data
    yhat.oob = predict(m_boot, data_train[oob.ind, ]) 
    #- get errors
    sse = sum( (data_train$y[oob.ind] - yhat.oob)^2 )
    n.oob = length(oob.ind)
    #- save results
    results = c(results, list(tibble(m, df, sse, n.oob)))
  }
}
results = bind_rows(results)    # convert from list to tibble

results %>% 
  group_by(df) %>% summarize(mse = sum(sse)/sum(n.oob)) %>% 
  ggplot(aes(df, mse)) + geom_point() + geom_line() + 
  geom_point(data=. %>% slice_min(mse), color="red", size=3) + 
  scale_x_continuous(breaks=1:20)
