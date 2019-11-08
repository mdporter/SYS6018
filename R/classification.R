#####################################################################
## R Code for Classification
## - See lecture: 09-classification.pdf
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
Default %>% group_by(default) %>% slice(1:3000) %>%  
ggplot( aes(balance, income, color=default, shape=default)) + 
  geom_point(alpha=.5) + 
  scale_color_manual(values=c(Yes="orange", No="blue")) + 
  scale_shape_manual(values=c(Yes=19, No=1)) 

ggplot(Default, aes(default, balance, fill=default)) + 
  #geom_boxplot() + 
  geom_violin(draw_quantiles=.5) +  
  scale_fill_manual(values=c(Yes="orange", No="blue"), guide=FALSE)

ggplot(Default, aes(default, income, fill=default)) + 
  geom_violin(draw_quantiles=.5) +
  scale_fill_manual(values=c(Yes="orange", No="blue"), guide=FALSE)

count(Default, default, student) %>% 
  group_by(student) %>% mutate(p=n/sum(n)) %>% 
  filter(default == "Yes") %>% 
  ggplot(aes(student, p, fill=default)) + 
  geom_col() + 
  scale_fill_manual(values=c(Yes="orange", No="blue"), guide=FALSE) + 
  labs(title="Proportion of Defaults by Student status")


#---------------------------------------------------------------------------#
#-- Linear Regression for binary response variable
#---------------------------------------------------------------------------#
library(broom) # to extract good stuff from models

#-- Create binary column (y)
Default = Default %>% mutate(y = ifelse(default == "Yes", 1L, 0L)) 

#-- Fit Linear Rergression Model
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
X = select(Default, balance, income) %>%  # select only two predictors
  as.matrix()                             # convert to matrix
  
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

X.eval = predict(transform, newdata=eval.pts)  # scale eval pts too

#-- fit knn model
knn5 = knn.reg(X.scale, test=X.eval, y=Default$y, k=5)



#---------------------------------------------------------------------------#
#-- Logistic Regression
#   More details in ISL 4.6.2
#---------------------------------------------------------------------------#
library(glmnet)

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

ggplot(Default, aes(balance, fill=student)) + 
  geom_density(alpha=.5) + 
  facet_wrap(~default, labeller=label_both)
  
ggplot(Default, aes(student, balance, fill=student)) + 
  geom_violin(draw_quantiles = .5) + 
  facet_wrap(~default, labeller=label_both)



#---------------------------------------------------------------------------#
#-- Performance Metrics and Curves
#---------------------------------------------------------------------------#
#-- train/test split
set.seed(2019)
Default = Default %>% 
  mutate(group = sample(c('train', 'test'), size=nrow(.), 
                        replace=TRUE, prob=c(.75, .25) )) # ~75% train

#-- fit model on training data
fit.lm = glm(y~student + balance + income, 
             family='binomial', 
             data=filter(Default, group=='train'))

#-- Get predictions (of gamma(x)) on test data
gamma = predict(fit.lm, 
                newdata=filter(Default, group=='test'), 
                type='link')  
Gtest = filter(Default, group=='test') %>% pull(y) # true values

#-- Visualize Performance by score
filter(Default, group=='test') %>% 
  mutate(gamma) %>% 
  ggplot(aes(gamma, fill=default)) + geom_density(alpha=.70) + 
  geom_rug(data=. %>% filter(default == 'Yes'), 
           aes(color=default), sides='t') + 
  geom_rug(data=. %>% filter(default == 'No'), 
           aes(color=default), sides='b') + 
  scale_fill_manual(values=c(Yes="orange", No="blue")) + 
  scale_color_manual(values=c(Yes="orange", No="blue"), guide=FALSE)

#-- Get performance data (by threshold)
perf = tibble(truth = Gtest, prediction = gamma) %>% 
  #- group_by() + summarize() in case of ties
  group_by(prediction) %>%     
  summarize(n=n(), n.1=sum(truth), n.0=n-sum(truth)) %>% 
  #- calculate metrics
  arrange(prediction) %>% 
  mutate(FN = cumsum(n.1),    # false negatives 
         TN = cumsum(n.0),    # true positives
         TP = sum(n.1) - FN,  # true positives
         FP = sum(n.0) - TN,  # false positives
         N = cumsum(n),       # number of cases predicted to be 1
         TPR = TP/sum(n.1), FPR = FP/sum(n.0)) %>% 
  #- only keep relevant metrics
  select(-n, -n.1, -n.0, threshold=prediction)

#-- Make performance curves
perf %>% select(threshold, FN, TP) %>% 
  gather(metric, n, -threshold) %>% 
  ggplot(aes(threshold, n, color=metric)) + geom_line()

#-- Make Cost curves
perf %>% mutate(cost = 1*FP + 10*FN) %>%   # use 1:10 costs
  ggplot(aes(threshold, cost)) + geom_line() + 
  geom_point(data=. %>% filter(cost==min(cost)), size=3, color='orange') + # # optimal from test data
  geom_vline(xintercept = log(1/11), color='purple') +  # theoretical optimal
  ggtitle('Cost of FP = 1; Cost of FN=10')

#-- Make ROC curve
perf %>% 
  ggplot(aes(FPR, TPR)) + geom_path() + 
  labs(x='FPR (1-specificity)', y='TPR (sensitivity)') + 
  #geom_abline(lty=3, color='grey') + 
  ggtitle("ROC Curve")

#-- Area under the ROC curve (AUC)
yardstick::roc_auc_vec(truth=Gtest %>% as.factor(), # truth must be a factor!
                       estimate=gamma)


#-- Precision-Recall
perf %>% 
  mutate(precision = TP/(TP + FP)) %>% 
  ggplot(aes(TPR, precision)) + geom_line() + 
  labs(x='Recall (TPR)', y='Precision', 
       title="Precision-Recall Curve")
