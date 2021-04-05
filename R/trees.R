#####################################################################
## R Code for Classification and Regression Trees
## - See lecture: trees.pdf and trees-demo.pdf
## 
## Michael D. Porter
## Created: April 2019
## For: Data Mining (SYS-6018/SYS-4582) at University of Virginia
## https://mdporter.github.io/SYS6018/
#####################################################################



#-- Load Required Packages
library(ISLR)         # for Hitters data
library(rpart)        # for classification and regression trees (CART)
library(rpart.plot)   # for prp() which allows more plotting control
library(randomForest) # for randomForest() function
library(tidyverse)    # for data manipulation

#-- Make Baseball Data
#   Goal is to predict the log Salary

library(ISLR)
data(Hitters, package="ISLR")        # load the Hitters data

Hitters = Hitters %>% 
  filter(!is.na(Salary)) %>%         # remove missing Salary
  mutate(Salary = log(Salary)) %>%   # convert to log Salary
  rename(Y = Salary)                 # rename Salary to Y

set.seed(2019)
train = sample(c(rep(TRUE, 200), rep(FALSE, nrow(Hitters)-200)))
bball = Hitters[train, ]

#- test data
X.test = Hitters[!train, ] %>% select(-Y)
Y.test = Hitters[!train, ] %>% pull(Y)



################################################################
#-- Regression Trees in R
# trees are in many packages: rpart, tree, party, ...
# there are also many packages to display tree results
#
# Formulas: you don't need to specify interactions as the tree does this
#  naturally. 
################################################################
#-- Build Tree
library(rpart)   
tree = rpart(Y~., data=bball)
summary(tree, cp=1)
length(unique(tree$where))       # number of leaf nodes

#-- Plot Tree
library(rpart.plot)   # for prp() which allows more plotting control
prp(tree, type=1, extra=1, branch=1)

# rpart() functions can also plot (just not as good):
#   plot(tree, uniform=TRUE)
#   text(tree, use.n=TRUE, xpd=TRUE)




#-- Evaluate Tree

#- mean squared error function
mse <- function(yhat, y){
  yhat = as.matrix(yhat)
  apply(yhat, 2, function(f) mean((f-y)^2))
}


mse(predict(tree), bball$Y)            # training error
mse(predict(tree, X.test), Y.test)     # testing error


#-- More complex tree
# see ?rpart.control() for details
# xval: number of cross-validations
# minsplit: min obs to still allow a split
# cp: complexity parameter

tree2 = rpart(Y~., data=bball, xval=0, minsplit=5, cp=0.005)
summary(tree2, cp=1)
length(unique(tree2$where)) 

prp(tree2, type=1, extra=1, branch=1)

mse(predict(tree2), bball$Y)            # training error
mse(predict(tree2, X.test), Y.test)     # testing error


cp = seq(.05,0,length=100)  # cp is like a penalty on the tree size
for(i in 1:length(cp)){
  if(i == 1){train.error = test.error = nleafs = numeric(length(cp))}
  tree.fit = rpart(Y~.,data=bball, xval=0, minsplit=5, cp=cp[i])
  train.error[i] = mse(predict(tree.fit),bball$Y)            # training error
  test.error[i] = mse(predict(tree.fit,X.test),Y.test)   # testing error
  nleafs[i] = length(unique(tree.fit$where))
}

plot(range(cp),range(train.error,test.error),typ='n',xlab="cp",ylab="mse",las=1)
lines(cp,train.error,col="black",lwd=2)
lines(cp,test.error,col="red",lwd=2)
legend("topleft",c('train error','test error'),col=c("black","red"),lwd=2)
axis(3,at=cp,labels=nleafs)
mtext("number of leaf nodes",3,line=2.5)




################################################################
#-- Regression Tree Examples for 2D
################################################################
library(ggplot2)

ggplot(bball) + geom_histogram(aes(x=Y), bins=30)   


#-- 2D plot (using only Years and Hits)
p2D = ggplot(bball) + scale_size_area() + 
            scale_colour_gradient2(midpoint=mean(bball$Y),mid="lightyellow",low="red",high="blue")
p2D + geom_point(aes(x=Years,y=Hits,color=Y,size=Y),alpha=.8) 


#-- Fit tree to only Years and Hits
tree3 = rpart(Y~Years+Hits, data=bball)
summary(tree3,cp=1)
length(unique(tree3$where))             # number of leaf nodes
prp(tree3, type=1, extra=1, branch=1)
mse(predict(tree3), bball$Y)            # training error
mse(predict(tree3,X.test),Y.test)       # testing error


#-- Plot Results
grid = expand.grid(Years = seq(min(bball$Years),max(bball$Years),length=90),
           Hits = seq(min(bball$Hits),max(bball$Hits),length=90))
grid$yhat3 = predict(tree3,newdata = grid)

p2D + geom_point(data=grid,aes(x=Years,y=Hits,color=yhat3),alpha=.9) + 
  geom_point(aes(x=Years,y=Hits,color=Y,size=Y),alpha=.8) 




#-- Fit more complex tree to only Years and Hits
tree4 = rpart(Y~Years+Hits,data=bball,xval=0,minsplit=5,cp=0.001)
length(unique(tree4$where))            # number of leaf nodes
prp(tree4, type=1, extra=1, branch=1)
mse(predict(tree4), bball$Y)            # training error
mse(predict(tree4,X.test), Y.test)      # testing error


#-- Plot Results
grid$yhat4 = predict(tree4,newdata = grid)

p2D + geom_point(data=grid,aes(x=Years,y=Hits,color=yhat4),alpha=.9) +
      geom_point(aes(x=Years,y=Hits,color=Y,size=Y),alpha=.8) 




################################################################
#-- Find best split points (regression trees)
################################################################

#-- Function to find optimal splits in regression tree (minimizing MSE)
split_info <- function(x, y){
  a = data.frame(x,y) %>% group_by(x) %>% 
    summarize(sum.y=sum(y), sum.ysq=sum(y^2), n=n()) 
  SSE.0 = sum((y - mean(y))^2)  # SSE for no split
  b = a %>% mutate(csum=cumsum(sum.y), csum.ysq=cumsum(sum.ysq), 
                   cn=cumsum(n), split.pt=x+(lead(x)-x)/2, 
                   est.L=csum/cn, est.R=(sum(y)-csum)/(sum(n)-cn),
                   SSE=sum(y^2)-cn*est.L^2-(sum(n)-cn)*est.R^2,
                   SSE.L=csum.ysq-cn*est.L^2, SSE.R=SSE-SSE.L,
                   n.L = cn, n.R=sum(n)-n.L,
                   gain=SSE.0-SSE )
  b = select(b,split.pt,n.L,n.R,est.L,est.R,SSE.L,SSE.R,SSE,gain)
  return(b)
}

#-- Function to find summary of leaves
split_metrics <- function(x,y,split.pt){
  data.frame(x,y) %>% 
    mutate(region=ifelse(x<split.pt,"LEFT","RIGHT")) %>%
    group_by(region) %>% summarize(SSE=sum((y-mean(y))^2),n=n())
}




## Split by Years
years = split_info(x=bball$Years, y=bball$Y)
head(years)

ggplot(years,aes(x=split.pt,y=SSE)) + geom_line() + geom_point()

filter(years, min_rank(SSE) == 1)  # optimal split point for Years

ggplot(years,aes(x=split.pt)) + 
  geom_line(aes(y=est.L,color="left")) +             # mean left of split pt
  geom_line(aes(y=est.R,color="right")) +            # mean right of split pt
  geom_hline(yintercept=mean(bball$Y))+              # overall mean
  scale_color_manual("mean",values=c('left'='red','right'='blue')) + 
  geom_point(data=bball,aes(x=Years,y=Y))            # add points
  
  

## Split by Hits
hits = split_info(x=bball$Hits,y=bball$Y)
head(hits)

ggplot(hits,aes(x=split.pt,y=SSE)) + geom_line() + geom_point()

filter(hits, min_rank(SSE)==1)    # optimal split point for Hits

ggplot(hits,aes(x=split.pt)) + 
  geom_line(aes(y=est.L,color="left")) +             # mean left of split pt
  geom_line(aes(y=est.R,color="right")) +            # mean right of split pt
  geom_hline(yintercept=mean(bball$Y))+         # overall mean
  scale_color_manual("mean",values=c('left'='red','right'='blue')) +
  geom_point(data=bball,aes(x=Hits,y=Y))            # add points  

  
## No splits
sum((bball$Y-mean(bball$Y))^2)   # SSE if no splits are made     
# (nrow(bball)-1)*var(bball$Y)



## Results (see function split_metrics at top of file)
#  splitting on Years gives the best reduction in SSE, so we would split on
#  Years (at a value of 4.5).
sum((bball$Y-mean(bball$Y))^2)       # no split
filter(years, min_rank(SSE)==1)      # split on years
filter(hits, min_rank(SSE)==1)       # split on hits
split_metrics(bball$Years,bball$Y, 4.5)

## Comparison of splitting on both variables
bind_rows(hits=hits, years=years, .id="split.var") %>% 
  ggplot(aes(x=split.pt, y=SSE)) + geom_line() + geom_point() + 
  facet_wrap(~split.var, scales="free_x")


#-- 2nd Split
# now we have to compare 4 possibilities. We can split on Years or Hits, but
#  use data that has Years < 4.5 or Years > 4.5

left = (bball$Years<=4.5)                                # split point from previous step
years2.L = split_info(x=bball$Years[left],y=bball$Y[left])
years2.R = split_info(x=bball$Years[!left],y=bball$Y[!left])
hits2.L = split_info(x=bball$Hits[left],y=bball$Y[left])
hits2.R = split_info(x=bball$Hits[!left],y=bball$Y[!left])

#-- Find best region to split on
max(years2.L$gain,na.rm=TRUE)
max(years2.R$gain,na.rm=TRUE)
max(hits2.L$gain,na.rm=TRUE)
max(hits2.R$gain,na.rm=TRUE)

hits2.R[which.max(hits2.R$gain),]

# 2nd split on Hits <= 117.5 in region 2.



#-- Summary of Splits
# Rule 1: Years < 4.5
# Rule 2: Years >= 4.5 & Hits < 117.5
# ...
prp(tree3, type=1, extra=1, branch=1)





################################################################
#-- Bagging Trees
################################################################

B = 500             # number of bootstrap samples
n = nrow(bball)     # number of observations

yhat.train = matrix(NA,n,B)
yhat.test = matrix(NA,nrow(X.test),B)
OOB = matrix(FALSE,n,B)

#- fit bootstrap trees
set.seed(10)
for(b in 1:B) {
  boot.ind = sample.int(n,replace=TRUE)      # bootstrap indices
  tree.boot = rpart(Y~.,data=bball[boot.ind,],xval=0,minsplit=5,cp=0.005)
  
  yhat.train[,b] = predict(tree.boot,bball)   # fit to training data
  yhat.test[,b]  = predict(tree.boot,X.test)  # fit to test data
  
  oob = setdiff(1:n,boot.ind)       # out-of-bag observations     
  OOB[oob,b] = TRUE 
}


#- Evaluate each bootstrap tree
mse.test = mse(yhat.test,Y.test)                # error for each bootstrap tree

#- Evaluate bagged trees
yhat.bag = t(apply(yhat.test,1,cumsum)/(1:B))   # prediction from bagged trees
mse.bag = mse(yhat.bag,Y.test)                  # error from bagging

#- Evaluate oob predictions
a = t(apply(yhat.train * OOB,1,cumsum))
b = t(apply(OOB,1,cumsum))
yhat.oob = a/b                                  # out of bag prediction
mse.oob = mse(yhat.oob,bball$Y)

#-- Evalute regular tree (fit to all data, no bootstrap)
tree.all = rpart(Y~.,data=bball)
mse.train.all = mse(predict(tree.all),bball$Y)
mse.test.all = mse(predict(tree.all,X.test),Y.test)


#-- Plot Results
plot(1:B,mse.test,type='p',las=1,xlab="bootstrap iteration",ylab="mse",pch=19)
abline(h=mse.test.all,col="blue",lwd=2)
title("performance for individual bootstrap trees")
legend("topright",c("bootstrap tree","tree with all data"),col=c("black","blue"),
       lwd=c(NA,2),pch=c(19,NA))


yrng = range(c(mse.bag,mse.oob,mse.test.all),na.rm=TRUE)
plot(c(1,B),yrng,type='n',las=1,xlab="# of trees",ylab="mse")
points(mse.test,col="black",pch=19,cex=.8)
lines(mse.bag,col="red",lwd=2)
lines(mse.oob,col="green",lwd=2)
abline(h=mse.test.all,col="blue",lwd=2)
title("performance of bagging")
legend("topright",c("bagged trees","oob estimate","tree with all data"),
       col=c("red","green","blue"),lwd=2)



#-- Training Data Analysis and Bumping
mse.train = mse(yhat.train,bball$Y)
plot(mse.train,las=1,xlab="bootstrap iteration",ylab="mse",pch=19)
title("training error")
abline(h=mse.train.all,col="purple")
abline(h=mse.test.all,col="blue")

#  bumping picks bootstrap model that minimizes trainging data
ind = which.min(mse.train)
ind

mse.test[ind]  # how good does bumping perform? Not so good for this data! 



################################################################
#-- Random Forest
################################################################
library(randomForest)

set.seed(11)
rf = randomForest(Y~.,data=bball)
print(rf)
varImpPlot(rf)


mse(predict(rf,bball),bball$Y)   # training error
mse(predict(rf),bball$Y)         # based on out-of-bag
mse(predict(rf,X.test),Y.test)   # on test set
mse.rf = mse(predict(rf,X.test), Y.test) # performance on test set


yrng = range(c(mse.rf,mse.bag,mse.test.all),na.rm=TRUE)
plot(c(1,B),yrng,type='n',las=1,xlab="# of trees",ylab="mse")
lines(mse.bag,col="red",lwd=2)
abline(h=mse.test.all,col="blue",lwd=2)
abline(h=mse.rf,col="black",lwd=2)
title("comparison of random forest and bagging")
legend("topright",c("bagged trees","random forest","tree with all data"),
       col=c("red","black","blue"),lwd=2)


