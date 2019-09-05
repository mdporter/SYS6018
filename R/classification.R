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
coef(fit.lm)

library(broom)
tidy(fit.lm)    # tidy way to get coefficients


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






