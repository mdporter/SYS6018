#####################################################################
## R Code for Classification
## - See lecture: classification.pdf
## 
## Michael D. Porter
## Created: Mar 2019
## For: Data Mining (SYS-6018/SYS-4582) at University of Virginia
## https://mdporter.github.io/SYS6018/
#####################################################################

#-- Install Required Packages
library(ISLR)
library(caret)
library(FNN)
library(broom)
library(yardstick)
library(tidyverse)   


#---------------------------------------------------------------------------#
#-- Default Data
#   From the ISLR package
#   The response variable is `default`
#---------------------------------------------------------------------------#
library(ISLR)
data(Default, package="ISLR")    # load the Default Data

#-- Create binary column (y)
Default = Default %>% mutate(y = ifelse(default == "Yes", 1L, 0L)) 

#-- Summary Stats (Notice only 333 (3.3%) have defaulted)
summary(Default)

#-- Plots
plot_cols = c(Yes="orange", No="blue")  # set colors 

Default %>% group_by(default) %>% slice(1:3000) %>%  # choose max of 3000 from each group
ggplot( aes(balance, income, color=default, shape=default)) + 
  geom_point(alpha=.5) + 
  scale_color_manual(values=plot_cols) + 
  scale_shape_manual(values=c(Yes=19, No=1)) 

ggplot(Default, aes(default, balance, fill=default)) + 
  geom_violin(draw_quantiles=.5) +  #alternative: geom_boxplot() + 
  scale_fill_manual(values=plot_cols, guide=FALSE)

ggplot(Default, aes(default, income, fill=default)) + 
  geom_violin(draw_quantiles=.5) +
  scale_fill_manual(values=plot_cols, guide=FALSE)

count(Default, default, student) %>% 
  group_by(student) %>% mutate(p=n/sum(n)) %>% 
  filter(default == "Yes") %>% 
  ggplot(aes(student, p, fill=default)) + 
  geom_col() + 
  geom_hline(yintercept=mean(Default$default == "Yes")) + 
  scale_fill_manual(values=plot_cols, guide=FALSE) + 
  labs(title="Proportion of Defaults by Student status",
       y="proportion default")


#---------------------------------------------------------------------------#
#-- Linear Regression for binary response variable
#---------------------------------------------------------------------------#
library(broom) # to extract good stuff from models

#-- Create binary column (y)
Default = Default %>% mutate(y = ifelse(default == "Yes", 1L, 0L)) 

#-- Fit Linear Regression Model
fit.lm = lm(y~student + balance + income, data=Default)

#-- Extract coefficients
coef(fit.lm)           # generic coef function to get coefficients
broom::tidy(fit.lm)    # tidy way to get coefficients


#---------------------------------------------------------------------------#
#-- kNN for binary response variable
#---------------------------------------------------------------------------#
library(FNN)     # for knn.reg() function
library(caret)   # for preProcess() function

#-- get matrix of predictors (balance and income)
X = Default %>% select(balance, income)        # select only two predictors
  
#-- center and scale predictors so Euclidean distance makes more sense
transform = caret::preProcess(X, method = c("center", "scale"))
X.scale = predict(transform, newdata=X)

#-- Evaluation Points
eval.pts = expand.grid(balance = seq(min(Default$balance), 
                                     max(Default$balance), 
                                     length=50), 
                       income = seq(min(Default$income), 
                                    max(Default$income), 
                                    length=50)) 

X.eval = predict(transform, eval.pts)  # scale eval pts too
#  Note: this uses the same center and scale from the *training data*. This is important!
#        don't rescale the hold-out data

#-- fit knn model
knn5 = knn.reg(X.scale, y=Default$y, test=X.eval, k=5)



#---------------------------------------------------------------------------#
#-- Logistic Regression
#   More details in ISL 4.6.2
#---------------------------------------------------------------------------#

#-- Fit logistic regression model
fit.lr = glm(y~student + balance + income, data=Default, 
             family="binomial")

## Alternatively, you can replace the binary y with a factor/character column
# fit.lr = glm(default~student + balance + income, data=Default, 
#              family="binomial")

#-- Get coefficients
tidy(fit.lr)

#-- Get predictions (for training data)
prob.lr = predict(fit.lr, type="response") # probabilities
link.lr = predict(fit.lr, type="link")     # logit (linear part)


#-- Interpret
# Notice that Student=Yes has a negative coefficient, but the plot of
#  defaults by student status suggests otherwise. 
# Reason is because students have more balance on average than non-students, 
#  and they get over-estimated once balance is in the model

plot_cols = c(Yes="orange", No="blue")  # set colors 

ggplot(Default, aes(balance, fill=student)) + 
  geom_density(alpha=.75) + 
  facet_wrap(~default, labeller=label_both) + 
  scale_fill_manual(values=plot_cols)
  
ggplot(Default, aes(student, balance, fill=student)) + 
  geom_violin(draw_quantiles = .5) + 
  facet_wrap(~default, labeller=label_both) + 
  scale_fill_manual(values=plot_cols)


#-- probability at certain values
eval.pts = tibble(student=c("Yes", "No"), 
                  balance=c(1000, 1000),    # balance = 1000
                  income = c(40000, 40000)) # income set to 40K

predict(fit.lr, eval.pts, type="link")
predict(fit.lr, eval.pts, type="response")

#-- Simpson's Paradox

# Students have higher default rate
Default %>% 
  group_by(student) %>% summarize(n=n(), p_default = mean(default == 'Yes'))

# People with higher balances have higher default rate    
ggplot(Default, aes(balance, default, fill=default)) + 
  geom_violin(draw_quantiles=.5) +  #alternative: geom_boxplot() + 
  scale_fill_manual(values=plot_cols, guide=FALSE)

Default %>% 
  group_by(default) %>% summarize(avg_balance = mean(balance))

# Students have higher balances on average, so they appear to be more likely
#  to default if the balance is not taken into account
Default %>% 
  group_by(student) %>% summarize(avg_balance = mean(balance))


# This is why the logistic regression model correctly adjusts the student status
#  negative. 

Default %>% 
  mutate(p_hat = prob.lr) %>% 
  group_by(student) %>% 
  summarize(n=n(), default_rate = mean(default == 'Yes'), avg_p = mean(p_hat))
  



#---------------------------------------------------------------------------#
#-- Penalized Logistic Regression
#---------------------------------------------------------------------------#
library(glmnet)
library(glmnetUtils)  # to allow formula interface in glmnet()

#-- Elastic net with alpha = .5. Use CV to select lambda.
set.seed(2020)
fit.enet = cv.glmnet(y~student + balance + income, data=Default, 
                     alpha=.5,
                     family="binomial")

#-- CV performance plot
plot(fit.enet, las=1)

#-- probability at certain values
predict(fit.enet, eval.pts, s="lambda.min", type="response")
predict(fit.enet, eval.pts, s="lambda.1se", type="response")

#-- Compare with intercept only model. Set large penalty (s large)
predict(fit.enet, eval.pts, s=1000, type="response") # intercept only (effectively)
mean(Default$y)                                      # actual intercept only

#-- Compare with unpenalized logistic regression. Set penalty s=0. 
predict(fit.enet, eval.pts, s=0, type="response") # unpenalized (effectively)
predict(fit.lr, eval.pts, type="response")

#---------------------------------------------------------------------------#
#-- Performance Metrics and Curves
#---------------------------------------------------------------------------#
#-- train/test split
set.seed(2019)
test = sample(nrow(Default), size=1000)
train = -test

#-- fit model on training data
fit.lm = glm(y~student + balance + income, family='binomial', 
             data=Default[train, ])

#-- Get predictions (of p(x) and gamma(x)) on test data
p.hat = predict(fit.lm, Default[test, ], type='response')  
gamma = predict(fit.lm, Default[test, ], type='link')  

#-- Make Hard classification (use .10 as cut-off)
G.hat = ifelse(p.hat >= .10, 1, 0)

#-- Make Confusion Table (base R)
G.test = Default$y[test]  # true values
table(predicted=G.hat, truth = G.test) %>% addmargins()

#-- Make Confusion Table (yardstick)
library(yardstick)
# Note: the yardstick package functions, like conf_mat(), require that hard
#  classifications require factor inputs (instead of characters)
cm = tibble(G.test, G.hat) %>% 
  mutate_all(~factor(.x, levels=c("1", "0"))) %>% # conf_mat() requires factors
  conf_mat(truth = G.test, estimate = G.hat) 
cm
autoplot(cm, type = "heatmap")


#-- Visualize Performance by score
Default[test,] %>% 
  mutate(gamma = gamma) %>% 
  ggplot(aes(gamma, fill=default)) + 
  geom_density(alpha=.70) +                     # add kernel density estimates
  geom_rug(data=. %>% filter(default == 'Yes'), # add rug to top
           aes(color=default), sides='t') + 
  geom_rug(data=. %>% filter(default == 'No'),  # add rug to bottom
           aes(color=default), sides='b') + 
  scale_fill_manual(values=c(Yes="orange", No="blue")) + # modify fill colors
  scale_color_manual(values=c(Yes="orange", No="blue"), guide=FALSE) # modify colors

#-- Get performance data (by threshold)
#   This table has one row for every threshold. The columns are the elements
#   of the confusion table plus FPR, TPR
perf = tibble(truth = G.test, gamma, p.hat) %>% 
  #- group_by() + summarize() in case of ties
  group_by(gamma, p.hat) %>%     
  summarize(n=n(), n.1=sum(truth), n.0=n-sum(truth)) %>% ungroup() %>% 
  #- calculate metrics
  arrange(gamma) %>% 
  mutate(FN = cumsum(n.1),    # false negatives 
         TN = cumsum(n.0),    # true negatives
         TP = sum(n.1) - FN,  # true positives
         FP = sum(n.0) - TN,  # false positives
         N = cumsum(n),       # number of cases predicted to be 1
         TPR = TP/sum(n.1), FPR = FP/sum(n.0)) %>% 
  #- only keep relevant metrics
  select(-n, -n.1, -n.0, gamma, p.hat)


## Note: gamma = log(p.hat) - log(1-p.hat) = log(p.hat / (1-p.hat))

#-- Make performance curves
col_lines = c(TP = "blue", FP="orange", FN="green", TN="brown",
              TPR = "blue", FPR="orange", FNR="green", TNR="brown")

#-- Make performance curves
perf %>% select(threshold=gamma, FN, TP) %>% 
  gather(metric, n, -threshold) %>% 
  ggplot(aes(threshold, n, color=metric)) + geom_line() + 
  labs(x= "threshold (gamma)", y="count") + 
  scale_color_manual(values=col_lines)

perf %>% select(threshold=gamma, FN, FP) %>% 
  gather(metric, n, -threshold) %>% 
  ggplot(aes(threshold, n, color=metric)) + geom_line() + 
  labs(x= "threshold (gamma)", y="count") + 
  scale_color_manual(values=col_lines)

perf %>% select(threshold=p.hat, FN, TP) %>% 
  gather(metric, n, -threshold) %>% 
  ggplot(aes(threshold, n, color=metric)) + geom_line() + 
  labs(x= "threshold (p.hat)", y="count") + 
  scale_color_manual(values=col_lines)

perf %>% select(threshold=p.hat, FN, FP) %>% 
  gather(metric, n, -threshold) %>% 
  ggplot(aes(threshold, n, color=metric)) + geom_line() + 
  labs(x= "threshold (p.hat)", y="count") + 
  scale_color_manual(values=col_lines)



#-- Make Cost curves
perf %>% mutate(cost = 1*FP + 10*FN) %>%   # use 1:10 costs
  ggplot(aes(p.hat, cost)) + geom_line() + 
  geom_point(data=. %>% filter(cost==min(cost)), size=3, color='orange') + # # optimal from test data
  geom_vline(xintercept = 1/11, color='purple') +  # theoretical optimal
  ggtitle('Cost of FP = 1; Cost of FN=10') + 
  labs(x="threshold (p.hat)")

perf %>% mutate(cost = 10*FP + 1*FN) %>%   # use 10:1 costs
  ggplot(aes(p.hat, cost)) + geom_line() + 
  geom_point(data=. %>% filter(cost==min(cost)), size=3, color='orange') + # optimal from test data
  geom_vline(xintercept = 10/11, color='purple') + # theoretical optimal
  ggtitle('Cost of FP = 10; Cost of FN=1') +   labs(x="threshold (p.hat)")


#-- Make ROC curve
perf %>% 
  ggplot(aes(FPR, TPR)) + geom_path() + 
  labs(x='FPR (1-specificity)', y='TPR (sensitivity)') + 
  geom_segment(x=0, xend=1, y=0, yend=1, lty=3, color='grey50') + 
  scale_x_continuous(breaks = seq(0, 1, by=.20)) + 
  scale_y_continuous(breaks = seq(0, 1, by=.20)) + 
  ggtitle("ROC Curve")


## Using yardstick package
library(yardstick)  # for evaluation functions

#-- ROC plots
ROC = tibble(truth = factor(G.test, levels=c(1,0)), gamma) %>% 
  yardstick::roc_curve(truth, gamma)

autoplot(ROC)  # autoplot() method

ROC %>%        # same as autoplot()
  ggplot(aes(1-specificity, sensitivity)) + geom_line() + 
  geom_abline(lty=3) + 
  coord_equal()


#-- Area under ROC (AUROC)
tibble(truth = factor(G.test, levels=c(1,0)), gamma) %>% 
  roc_auc(truth, gamma)

yardstick::roc_auc_vec(factor(G.test, 1:0), gamma)



#-- Log Loss Metric
yardstick::mn_log_loss_vec(factor(G.test, 1:0), gamma)

tibble(truth = factor(G.test, levels=c(1,0)), gamma) %>% 
  yardstick::mn_log_loss(truth, gamma)



#-- Precision-Recall
perf %>% mutate(threshold = p.hat, precision = TP/(TP + FP)) %>% 
  select(threshold, TPR, precision) %>% 
  gather(metric, n, -threshold) %>% 
  ggplot(aes(threshold, n, color=metric)) + geom_line() +
  scale_x_continuous(breaks = seq(0, 1, by=.20)) + 
  scale_y_continuous(breaks = seq(0, 1, by=.20)) + 
  scale_color_manual(values = c(TPR="blue", precision="brown")) +
  labs(x="thredhold (p.hat)", y="score") 

perf %>% 
  mutate(threshold=p.hat, precision = TP/(TP + FP)) %>% 
  ggplot(aes(TPR, precision)) + geom_line() + 
  scale_x_continuous(breaks = seq(0, 1, by=.20)) + 
  scale_y_continuous(breaks = seq(0, 1, by=.20)) + 
  labs(x='Recall (TPR)', y='Precision', # (TP/(TP+FP))
       title="Precision-Recall Curve")

