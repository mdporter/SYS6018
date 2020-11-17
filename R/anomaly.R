#####################################################################
## R Code for Anomaly Detection
## - See lecture: 05-anomaly.pdf
## 
## Michael D. Porter
## Created: Feb 2019
## For: Data Mining (SYS-6018) at University of Virginia
## https://mdporter.github.io/SYS6018/
#####################################################################

#-- Install Required Packages
library(mclust)
library(tidyverse)   
library(readxl)    # for reading excel data (part of tidyverse, but not loaded)



#---------------------------------------------------------------------------#
#-- Benford' Distribution Functions
#---------------------------------------------------------------------------#

#-- Extract First Digit

## extract_first_digit() 
#  Extracts the first digit (natural number [1-9])
#  ignores negative signs, decimals, zeros, etc
#  x: vector of numbers (could contain characters)
#  type: either "factor" or "integer". Returns vector of this type.
#  Returns a vector of first digits. The default is to return a factor with 
#   levels 1:9 to ensure further analysis is not affected by missing levels.
extract_first_digit <- function(x, type="factor"){
  d = stringr::str_extract(x, "[1-9]")           # extract first natural number
  if(type == "factor") d = factor(d, levels=1:9) # convert to factor with levels 1:9
  else if (type == "integer") d = as.integer(d)  # convert to an integer
  return(d)
}

#-- dbenford()
# PMF for Benford's distrubtion
# x: vector of natural numbers (positive integers)
# for 1 digit: x in 1,2, ... 9
# for 2 digits: x in 10, 11, ... 99
dbenford <- function(x) log10(1+1/x)

#-- rbenford()
# simulate data from Benford's distribution
# n: number of observations
# digits: number of digits
# seed: use number to ensure exact replication
rbenford <- function(n, digits=1, seed=NULL){
  if(!is.null(seed)) set.seed(seed)
  x = 10^(digits-1):(10^digits - 1)
  p = dbenford(x)
  sample(x, size=n, replace=TRUE, prob=p)
}


#-- ddunif()
# PMF for discrete uniform distribution
# x: vector of values
# levels: vector of all possible values
ddunif <- function(x, levels){
  K = length(levels)  # number of values possible
  ifelse(x %in% levels, 1/K, 0)
}

#-- rdunif()
# simulate data from a discrete uniform distribution
# n: number of observations
# levels: vector giving the set of possible values
# seed: use number to ensure exact replication
rdunif <- function(n, levels, seed=NULL){
  if(!is.null(seed)) set.seed(seed)
  x = levels
  p = rep(1/length(levels), length(x))
  sample(x, size=n, replace=TRUE, prob=p)
}

#---------------------------------------------------------------------------#
#-- Benford' Distribution Analysis
#---------------------------------------------------------------------------#

#-- first digit
dbenford(1:9)
plot(dbenford(1:9), las=1, 
     xlab="first digit", ylab='probability')


#-- distribution of first two digits
two = expand.grid(first=1:9, second=0:9) %>% 
  mutate(two = paste0(first, second) %>% as.integer) %>% 
  mutate(f = dbenford(two)) %>% 
  select(first, second, f) %>% 
  spread(second, f) 


rowSums(two[,-1])  # distribution of first digit
colSums(two[,-1])  # distribution of second digit




#---------------------------------------------------------------------------#
#-- Load corporate payments data
# Mark Nigrini provides a dataset of the 2010 payments 
#  from a division of a West Coast utility company 
#  https://www.nigrini.com/BenfordsLaw/CorporatePaymentsData.xlsx
#---------------------------------------------------------------------------#
## Download xlsx file to local machine, then load into R

#-- Set your local director
data.dir = 'topics/anomaly/data/'   # set your directory here


#-- Load into R
## Notice that the data is in the "Data" sheet
library(readxl)
corp_payments = read_excel(file.path(data.dir, "CorporatePaymentsData.xlsx"), 
                           sheet="Data") %>% 
  filter(Amount > 0)  # only consider positive payments

#-- Extract the first digit from the *Amount* column
first = extract_first_digit(corp_payments$Amount)


#---------------------------------------------------------------------------#
#-- Calculate Test Statistics
#---------------------------------------------------------------------------#

#-- Get counts
Y = table(first) %>% as.integer   # ensure first is factor with properly ordered
n = length(first)                 # number of observations

#-- chi-squared 
n = length(first)    # number of observations
E = n*dbenford(1:9)  # expected count vector
chi = (Y-E)/sqrt(E)  # vector of deviations
(chisq = sum(chi^2)) # chi-squared test statistic

#-- Alternative R function chisq.test()
csq = chisq.test(Y, p=dbenford(1:9))
csq
csq$statistic    # test statistic value
csq$p.value      # estimated p-value
csq$residuals    # residuals are the chi from above


#-- log likelihood ratio test statistic

# MLE
L.mle = dmultinom(Y, prob=Y/sum(Y), log=TRUE)
L.null = dmultinom(Y, prob=dbenford(1:9), log=TRUE)
llr.mle = L.mle - L.null   # log-likelihood ratio

# discrete uniform
L.dunif = dmultinom(Y, prob=rep(1/9, 9), log=TRUE)
llr.dunif = L.dunif - L.null

## 2*llr should be close to chi-squared
c(llr = 2*llr.mle, chi.sq=chisq)


#---------------------------------------------------------------------------#
#-- Significance Testing
#---------------------------------------------------------------------------#

#-- chi-squared p-value
1-pchisq(chisq, df=8, lower.tail=TRUE )

#-- llr p-value
1-pchisq(2*llr.mle, df=8, lower.tail=TRUE )


#-- Monte Carlo Testing
M = 1000                           # number of simulations
stat.chisq = stat.llr = numeric(M) # initialize statistics
for(m in 1:M){
  #- generate observation under the null of Benford
  y.sim = rmultinom(1, size=n, prob=dbenford(1:9)) 
  #- calculate statistics
  stat.chisq[m] = chisq.test(y.sim, p=dbenford(1:9))$statistic
  p.mle = y.sim/sum(y.sim)
  stat.llr[m] = dmultinom(y.sim, prob=p.mle, log=TRUE) - 
    dmultinom(y.sim, prob=dbenford(1:9), log=TRUE)
}

#- calculate p-values
(1 + sum(stat.chisq > chisq)) / (M+1)  # chi-square p-value
(1 + sum(stat.llr > llr.mle)) / (M+1)  # LLR p-value

cor(stat.chisq, stat.llr)  # strong correlation between metrics

#- plot
par(mfrow=c(1,2))
plot(density(stat.chisq))   # kde plot
abline(v = chisq, col="red")
lines(0:40, dchisq(0:40, df=8), col="blue")         # overlay the asympotic distribution

plot(ecdf(stat.chisq))      # ecdf plot
abline(v = chisq, col="red")
lines(0:40, pchisq(0:40, df=8), col="blue")         # overlay the asympotic distribution

plot(density(2*stat.llr))
abline(v = llr.mle, col="red")
lines(0:40, dchisq(0:40, df=8), col="blue")         # overlay the asympotic distribution

plot(ecdf(2*stat.llr))      # ecdf plot
abline(v = llr.mle, col="red")
lines(0:40, pchisq(0:40, df=8), col="blue")         # overlay the asympotic distribution


#---------------------------------------------------------------------------#
#-- outlier Detection
#---------------------------------------------------------------------------#
library(mvtnorm)

#-- Simulate Data
set.seed(2019)
N = 200
mu = c(10, 20)
sigma = matrix(c(4,2,2, 2), nrow=2)
X = mvtnorm::rmvnorm(N, mu, sigma)
outliers = matrix(c(5, 19, 10, 15, 18, 25), byrow=TRUE, ncol=2)
cols = c(rep("red", nrow(outliers)), rep("black", nrow(X)))
X = rbind(outliers, X)

plot(X, pch=19, col=cols, 
     xlab="x", ylab="y", las=1, asp=1)
text(X[1:3,], label=1:3, col=2, pos=2)


#-- Mahalanobis Distance
n = nrow(X)
xbar = colMeans(X)  # sample mean
S = var(X)*(n-1)/n
Dsq = apply(X[1:3,], 1, 
            function(x) t(x - xbar) %*% solve(S) %*% (x-xbar))


#-- Fit mixture model with uniform noise
library(mclust)
mc = Mclust(X,
            G = NULL,    # set number of non-noise components
            initialization=list(noise=TRUE)) # allow noise/outliers
summary(mc)

plot(mc, what="classification", asp=1, las=1)
grid()


#---------------------------------------------------------------------------#
#-- Two-Sample Testing: Clinical Trials
#---------------------------------------------------------------------------#

#- observed data
n1 = 600
p1 = 0.40
n2 = 1200
p2 = 0.36
p0 = (n1*p1 + n2*p2)/(n1+n2)  # average cure rate
T.obs = p1 - p2       # Test Statistic: observed difference

#- Simulation Data
n = n1 + n2            # number of patients
x = n1*p1 + n2*p2      # total number cured

#- Run Simulation
set.seed(100)                            # set seed for replication
nsim = 10000                             # of simulations
x1.sim = rhyper(nsim, m=x, n=n-x, k=n1)  # simulated # cured in pop 1
x2.sim = x - x1.sim                      # simulated # cured in pop 2
T.sim = x1.sim/n1 - x2.sim/n2            # simulated test statistics

#- plots
hist(T.sim, breaks=seq(-.1,.1,by=.01), freq=FALSE, las=1) # histogram  
abline(v=T.obs,col="orange",lwd=2)       # add observed test statistic

#- p-value
(sum(T.sim >= T.obs) + 1) / (nsim +1)    # non-parametric p-value



