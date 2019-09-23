#####################################################################
## R Code for clustering
## - See lecture: 04-clustering.pdf
## 
## Michael D. Porter
## Created: Feb 2019
## For: Data Mining (SYS-6018/SYS-4582) at University of Virginia
## https://mdporter.github.io/SYS6018/
#####################################################################

#-- Install Required Packages
# library(MASS)       # only needed for the crabs data
library(tidyverse)    # load after MASS so dplyr::select() is not overwritten
library(mclust)

#---------------------------------------------------------------------------#
#-- Load crabs data
#---------------------------------------------------------------------------#
crabs = MASS::crabs                               # get crabs data
crabsX = dplyr::select(crabs, FL, RW, CL, CW, BD) # extract features
crabsY = paste(crabs$sp, crabs$sex, sep=":")      # get true labels


#---------------------------------------------------------------------------#
#-- Hierarchical Clustering
#---------------------------------------------------------------------------#

#-- Calculate Distance (dissimilarity matrix)
dX = dist(crabsX, method="euclidean")  # calculate distance

#-- Run hierarchcial clustering
hc = hclust(dX, method="average")      # average linkage

#-- Plot
plot(hc)     # basic plot

plot(as.dendrogram(hc), las=1, leaflab="none")
ord = hc$order                # order of x-axis
labels = crabsY[ord]
colors = ifelse(str_detect(labels, "B"), "blue", "orange")
shapes = ifelse(str_detect(labels, "M"), 17, 15)
n = nrow(crabsX)
points(1:n, rep(0, n), col=colors, pch=shapes, cex=.8)
legend("topright", c("Blue:Male", "Blue:Female", "Orange:Male", "Orange:Female"), 
       pch=c(17, 15, 17, 15), col=c("blue", "blue", "orange", "orange"), cex=.8)

#-- Guess number of clusters by height plot
n = length(hc$height)     # get number of merges
plot(n:1, hc$height, type='o', xlab="K", ylab="height", las=1, 
     xlim=c(1, 50))
points(5, hc$height[n-4], col="red", pch=19) # K=5

#-- Extract cluster membership
yhat = cutree(hc, k=5)     # cut so K=5
# yhat = cutree(hc, h=8.2) # cut at height of h=8.2

#-- Confusion Matrix
table(true=crabsY, est=yhat)



#-- Scaling
X2 = scale(crabsX)        # each column has mean=0, sd=1
apply(X2, 2, sd)
apply(X2, 2, mean)

dX2 = dist(scale(X2)) # new distance
hc2 = hclust(dX2, method="complete") # new hclust

n = length(hc2$height)     # get number of merges
plot(n:1, hc2$height, type='o', xlab="K", ylab="height", las=1, 
     xlim=c(1, 50))
points(4, hc2$height[n-3], col="red", pch=19) # K=4

table(true=crabsY, est=cutree(hc2, k=4))



#-- PCA
X3 = prcomp(crabsX, scale=TRUE)$x[, 1:2]  # first 2 PC
plot(X3, 
     col = ifelse(str_detect(crabsY, "B"), "blue", "orange"),
     pch = ifelse(str_detect(crabsY, "M"), 17, 15))

dX3 = dist(scale(X3))                # new distance
hc3 = hclust(dX3, method="complete") # new hclust

n = length(hc3$height)     # get number of merges
plot(n:1, hc3$height, type='o', xlab="K", ylab="height", las=1, 
     xlim=c(1, 50))
points(9, hc3$height[n-8], col="red", pch=19) # K=4

table(true=crabsY, est=cutree(hc2, k=9))



#---------------------------------------------------------------------------#
#-- K-means clustering: crabs
#---------------------------------------------------------------------------#

X = scale(crabsX)                         # scale crabs data

#-- Run kmeans for multiple K
Kmax = 20                                 # maximum K
SSE = numeric(Kmax)                       # initiate SSE vector
for(k in 1:Kmax){
  km = kmeans(X, centers=k, nstart=25)    # use 25 starts
  SSE[k] = km$tot.withinss                # get SSE
}

#-- Plot results
plot(1:Kmax, SSE, type='o', las=1, xlab="K")

#-- Plot 1-step differences
dif1 = SSE - lead(SSE)  # SSE(K) - SSE(K-1)
plot(dif1, ylab="difference", xlab="K", las=1, 
     ylim=c(0, 50))

#-- Plot linear (2nd) differences
dif2 = SSE - 2*lead(SSE) + lead(SSE, 2)  # SSE(K) -2SSE(K-1) +SSE(K)
plot(dif2, ylab="2nd difference", xlab="K", las=1, 
     ylim=c(0, 50))

#-- Evaluate to truth (both indicate K=5 isn't bad)
km = kmeans(X, centers=5, nstart=25)  # choose K=6
table(true=crabsY, est=km$cluster)

#---------------------------------------------------------------------------#
#-- K-means clustering: Old Faithful
#---------------------------------------------------------------------------#

oldf = datasets::faithful 
plot(oldf, pch=19, las=1, main="Old Faithful"); grid()
plot(oldf, pch=19, las=1, main="Old Faithful: same aspect", asp=1); grid()


#-- Unscaled Solution
X = datasets::faithful
km = kmeans(X, centers=2, nstart=25)
plot(X, col=km$cluster, las=1, main="unscaled")
points(km$centers, col=1:2, pch=15, cex=2)

plot(X, col=km$cluster, las=1, main="unscaled: same aspect", asp=1)
points(km$centers, col=1:2, pch=15, cex=2)


#-- Scaled Solution
X2 = scale(datasets::faithful)
km2 = kmeans(X2, centers=2, nstart=25)
plot(X, col=km2$cluster, las=1, main="scaled")
points(km$centers, col=1:2, pch=15, cex=2)

plot(X2, col=km2$cluster, las=1, main="scaled: raw values", asp=1)
points(km2$centers, col=1:2, pch=15, cex=2)


#---------------------------------------------------------------------------#
#-- GMM Example: Old Faithful (Waiting Time)
# See interactive shiny example at: https://pasda.shinyapps.io/Old_Faithful/
#---------------------------------------------------------------------------#
#-- Load the Old Faithful data
oldf = datasets::faithful

#-- Make a ggplot object
pp = ggplot(oldf, aes(x=waiting)) + xlab("waiting time (min)")

#-- Function to calculate Gaussian mixture pdf
dnmix <- function(theta1, theta2, w=.5, x.seq=seq(-4, 4, length=100)){
  f1 = dnorm(x.seq, mean=theta1[1], sd=theta1[2])
  f2 = dnorm(x.seq, mean=theta2[1], sd=theta2[2])
  fmix = f1*w + f2*(1-w)
  return(fmix)
}

#-- Set parameters
theta1 = c(mu=50, sigma=10)       # parameters for component 1
theta2 = c(mu=90, sigma=5)        # parameters for component 2
w = .5                            # mixture weight

#-- Make data for plotting
x.seq = seq(40, 100, length=200)     # make sequence of x values
f = dnmix(theta1, theta2, w, x.seq)  # calculate the density at those values
data.mix = tibble(x.seq, f)          # make into a data frame/tibble

#-- Make plot
pp + geom_histogram(binwidth = 1, aes(y=stat(density)), alpha=.5) + 
  geom_line(data=data.mix, aes(x=x.seq, y=f), color="blue", size=1.25) 


#---------------------------------------------------------------------------#
#-- Sampling from a two component univariate GMM
#---------------------------------------------------------------------------#

#-- Set parameters
theta1 = c(mu=50, sigma=10)       # parameters for component 1
theta2 = c(mu=90, sigma=5)        # parameters for component 2
w = c(.4, .6)                     # mixture weights (must sum to one)
n = 300                           # number of samples to draw
set.seed(2019)                    # set the random seed for replication

#-- (1) Draw the group labels
g = sample(c(1,2), size=n, replace=TRUE, prob=w)


#-- (2) Sample from the component densities
#   To avoid loops, generate n observations from each density and pick the
#   one according to group label. 
X = ifelse(g == 1, 
           rnorm(n, mean=theta1[1], sd=theta1[2]), 
           rnorm(n, mean=theta2[1], sd=theta2[2]))

hist(X, breaks=50)

#---------------------------------------------------------------------------#
#-- GMM clustering: Old Faithful
#---------------------------------------------------------------------------#

X = datasets::faithful 

#-- Fit K=2 component mixture model
GMM =  mvnormalmixEM(X, k=2)  # use mvnormalmixEM()
(w = GMM$lambda)           # estimated weights
(mu = GMM$mu)              # estimate means
(Sigma = GMM$sigma)        # estimated covariance matrix
sapply(Sigma, det)         # determinant

#-- Plot 95% contour
plot(GMM, whichplots = 2, alpha=.05) # 95% is 1-alpha


#---------------------------------------------------------------------------#
#-- MBC clustering: Old Faithful
#---------------------------------------------------------------------------#
# Use the mclust package to help search across all K's
library(mclust)
mix = Mclust(X)
summary(mix)   # finds 3 clusters

plot(mix, what="BIC")  
plot(mix, what="classification")
plot(mix, what="uncertainty")  
plot(mix, what="density")  

#-- get parameters
summary(mix, parameters=TRUE)


#-- More detailed analysis: see https://www.stat.washington.edu/sites/default/files/files/reports/2012/tr597.pdf
faithfulBIC = mclustBIC(X)
faithfulSummary  = summary(faithfulBIC, data=X)
faithfulSummary

plot(faithfulBIC, G=1:7, 
     ylim=c(-2500,-2300), legendArgs=list(x="bottomright",ncol=5))


