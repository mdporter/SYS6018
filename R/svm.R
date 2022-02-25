#####################################################################
## R Code for support vector machines (SVM).
## - See lecture: svm.pdf
## 
## Michael D. Porter
## Created: Feb 2022
## For: Data Mining (SYS-6018) at University of Virginia
## Website: https://mdporter.github.io/SYS6018
#####################################################################

#: Load Required Packages
library(tidyverse)
library(e1071)
library(yardstick)

#--------------------------------------------------------------------#
# Create Data
#--------------------------------------------------------------------#
library(mvtnorm)

prior = c(.60, .40)
mu1 = c(2, 2)
mu2 = c(0, 0)
mu2.B = c(4, 4)
sigma1 = .5*matrix(c(2,-1, -1, 2), nrow=2)
sigma2 = .5*matrix(c(1,0,0,1), nrow=2)


set.seed(2020)
n = 200
n1 = rbinom(1, size=n, prob=prior[1])
n2 = n-n1
n2 = round(c(n2/2, n2/2))
X1 = rmvnorm(n1, mean=mu1, sigma = sigma1)
X2 = rmvnorm(n2[1], mean=mu2, sigma = sigma2)
X2 = rbind(X2, rmvnorm(n2[2], mean=mu2.B, sigma = sigma2))
labels = c("+1", "-1")


data_mix = bind_rows(
  !!labels[1] := as_tibble(X1, .name_repair = ~str_c("X", seq_along(.))),
  !!labels[2] := as_tibble(X2, .name_repair = ~str_c("X", seq_along(.))),
  .id = "class"
)


data_mix %>% 
  ggplot(aes(X1, X2, color = class)) + 
  geom_point()


#--------------------------------------------------------------------#
# Create Cross-Validation Folds
#--------------------------------------------------------------------#
set.seed(2022)
n.folds = 10                       
fold = sample(rep(1:n.folds, length=nrow(data_mix)))  



#--------------------------------------------------------------------#
# Polynomial SVM Example
#--------------------------------------------------------------------#
library(e1071)
fit = svm(factor(class) ~ X1 + X2, 
    data = data_mix[fold != 1, ],   # hold out fold 1
    #: tuning parameters
    kernel = "poly",
    degree = 2, 
    cost = 1
    ) 

# Notes:
# - svm() requires that the outcome variable be a "factor" for classification
#   problems. 
# - the polynomial kernel only has cost and degree parameters


#: basic model summary
summary(fit)

#: predictions
pred = predict(fit, data_mix[fold == 1,], decision.values = TRUE)

eval_data = tibble(
  outcome = data_mix$class[fold == 1], 
  pred_hard = c(pred), 
  pred_soft = as.numeric(attr(pred, "decision.values"))
)

# Notes:
# - using the `decision.values = TRUE` argument to get the scores. But set
#   to default of FALSE to just get the hard classifications
# - the scores can be used to create rank or threshold-based metrics like ROC
#   curves
# - There is also a way to get probability estimatates by setting
#   svm(..., probability = TRUE) and predict(..., probability = TRUE).
#   This attempts to convert scores to probabilities 

#: evaluation
library(yardstick)
levs = c("+1", "-1")  # set outcome of interest at first level

eval_data %>% 
  yardstick::roc_curve(truth = factor(outcome, levels=levs), estimate = pred_soft)

eval_data %>% 
  yardstick::accuracy(truth = factor(outcome, levels=levs), 
                      estimate = factor(pred_hard, levels=levs)

# Notes:
# - the yardstick packages wants outcome variables and hard predictions be
#   "factors" with the outcome of interest as the first level
                      
                      
#--------------------------------------------------------------------#
# Radial basis SVM tuning parameter selection
#--------------------------------------------------------------------#
#: create function to fit, predict, and evaluate
eval_svm <- function(data_train, data_test, cost, gamma){
  
  #: fit
  fit = svm(factor(class) ~ X1 + X2, 
            data = data_train,
            #: tuning parameters
            kernel = "radial",
            gamma = gamma,
            cost = cost
            )
  
  #: predict
  pred = predict(fit, data_test, decision.values = TRUE)
  
  #: evaluate
  eval_data = tibble(
    outcome = data_test$class, 
    pred_hard = c(pred), 
    pred_soft = as.numeric(attr(pred, "decision.values"))
  )
  
  levs = c("+1", "-1")  # set outcome of interest at first level
  
  auroc = eval_data %>% 
    yardstick::roc_auc(truth = factor(outcome, levels=levs), estimate = pred_soft)
  
  accuracy = eval_data %>% 
      yardstick::accuracy(truth = factor(outcome, levels=levs), 
                          estimate = factor(pred_hard, levels=levs))
  
  bind_rows(auroc, accuracy) %>% 
    mutate(cost, gamma)  # add tuning parameters

}

#: test it out
eval_svm(
  data_train = data_mix[fold != 1,],
  data_test = data_mix[fold == 1,],
  cost = 1, 
  gamma = .1
)



#: create grid of tuning parameter values
tune_grid = expand_grid(
  cost = 10^(-2:2),
  gamma = c(.1, .5, 1)
)


#: loop over folds and grids
out = tibble()

for(k in unique(folds)){
  
  for(i in 1:nrow(tune_grid)){
    
    metrics = eval_svm(
      data_train = data_mix[fold != k,],
      data_test = data_mix[fold == k,],
      cost = tune_grid$cost[i], 
      gamma = tune_grid$gamma[i]
    )
    
    out = bind_rows(out, metrics %>% mutate(fold = k) )
  }
}

#: average over folds

out %>% 
  # calculate average performance over folds 
  group_by(.metric, cost, gamma) %>% 
  summarize(mu = mean(.estimate), .groups = "drop") %>% 
  # spread data wider to see both metrics 
  pivot_wider(names_from = .metric, values_from = mu) %>% 
  # arrange by auc and then accuracy (if ties)
  arrange(desc(roc_auc), desc(accuracy))


# Notes:
# - can go back and modify grid to focus on best tuning region








