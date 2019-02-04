#####################################################################
## R Code for parametric density estimation.
## - See lecture: 03-density.pdf
## 
## Michael D. Porter
## Created: Feb 2019
## For: Data Mining (SYS-6018/SYS-4582) at University of Virginia
## https://mdporter.github.io/SYS6018/
#####################################################################


#-- Install Required Packages
library(tidyverse)    # install.packages("tidyverse")
library(fitdistrplus) # install.packages("fitdistrplus")
library(mixtools)     # install.packages("mixtools")

#---------------------------------------------------------------------------#
#-- Load ED counts Data
#---------------------------------------------------------------------------#
#-- Load Data
url = 'https://raw.githubusercontent.com/mdporter/SYS6018/master/data/ED-counts.csv'
x = readr::read_csv(url)$count

#-- empirical pmf
ggplot() + geom_bar(aes(x=x))

#-- Summary
summary(x)
var(x)


#---------------------------------------------------------------------------#
#-- Poisson Model: ED counts Data
#---------------------------------------------------------------------------#

## Grid Search
#-- Get sequence of lambda values
lam.seq = seq(65, 100, length=200)
nlam = length(lam.seq)

#-- Calculate the log-likelihood for those lambda values
loglike = numeric(nlam)
for(i in 1:nlam){
  loglike[i] = sum(dpois(x, lambda=lam.seq[i], log=TRUE))
}

#-- Alternative to loop using sapply()
loglike2 = sapply(lam.seq, function(lam) sum(dpois(x, lambda=lam, log=TRUE)))
all.equal(loglike, loglike2)

#- best lambda via grid search:
lam.seq[which.max(loglike)]

#-- Make log-likelihood plot
lam.data = tibble(lam.seq, loglike)

ggplot(lam.data, aes(lam.seq, loglike)) + 
  geom_line() + 
  geom_point(data=filter(lam.data, loglike == max(loglike)), 
             color="red", size=2) + 
  labs(x = expression(lambda), y="log-likelihood")


#---------------------------------------------------------------------------#
#-- Negative Binomial Model: ED counts Data
#---------------------------------------------------------------------------#

#-- MOM
mean(x)
var(x)

#-- MLE
library(fitdistrplus)
opt = fitdist(data=x, distr="nbinom", method="mle")
nb.pars = opt$estimate


#-- Plots
#-- Make Data
x.seq = 0:162      # sequence of x values
f.nb = dnbinom(x.seq, size=nb.pars[1], mu=nb.pars[2])  # pmf values at x.seq
f.pois = dpois(x.seq, lambda=mean(x))

fit.data = tibble(x.seq, poisson=f.pois, neg.binom=f.nb) %>% 
  gather(model, pmf, -x.seq)

#-- Make PMF curve; overlay histogram/barplot
ggplot() + 
  geom_bar(aes(x, y=stat(prop)), fill="grey70") + 
  geom_point(data=fit.data, 
             aes(x=x.seq, y=pmf, color=model),  alpha=.5) + 
  scale_color_manual(values= c("red", "blue"))


#---------------------------------------------------------------------------#
#-- Normal Model: ED counts Data
#---------------------------------------------------------------------------#
#-- MLE/MOM
mean(x)
sd(x)
n = length(x)   
sd(x)*sqrt((n-1)/n)   # correction for n-1 

library(fitdistrplus)
opt = fitdist(data=x, distr="norm", method="mle")
gauss.pars = opt$estimate


#-- Plots
f.gauss = dnorm(x.seq, mean=gauss.pars[1], sd=gauss.pars[2])  # pmf values at x.seq

# if we wanted to be particular, we could discretize the gaussian
# f.gauss = pnorm(x.seq+.5, mean=gauss.pars[1], sd=gauss.pars[2]) - pnorm(x.seq-.5, mean=gauss.pars[1], sd=gauss.pars[2])

fit.data = tibble(x.seq, poisson=f.pois, neg.binom=f.nb, gaussian=f.gauss) %>% 
  gather(model, pmf, -x.seq)

#-- Make PMF curve; overlay histogram/barplot
ggplot() + 
  geom_bar(aes(x, y=stat(prop)), fill="grey70") + 
  geom_point(data=fit.data, 
             aes(x=x.seq, y=pmf, color=model),  alpha=.5) + 
  scale_color_manual(values= c("green", "red", "blue"))


#- base R version of plot
h = hist(x, breaks=seq(-.5, 163, by=.5), plot=FALSE)
h$counts = h$counts/sum(h$counts)  # make relative frequency
plot(h, ylim=range(f.pois), ylab='prop', col='lightgrey', border=NA)
lines(x.seq, f.pois, col="blue")
lines(x.seq, f.nb, col="red")
lines(x.seq, f.gauss, col="green")

#---------------------------------------------------------------------------#
#-- Old Faithful
#---------------------------------------------------------------------------#

#-- Load the Old Faithful data
wait = datasets::faithful$waiting

#-- Calculate summary stats
length(wait)            # sample size
summary(wait)           # six number summary
mean(wait)              # mean
sd(wait)
median(wait)
quantile(wait, probs=c(.25,.50,.75))  # quantiles

#-- Plots
#-- Put data into a data.frame/tibble for use with ggplot
wait.df = tibble(wait)

#-- Make a ggplot object
pp = ggplot(wait.df, aes(x=wait)) + xlab("waiting time (min)")

#-- Histogram
pp + geom_histogram(binwidth = 1) + ggtitle("histogram")

#-- overlay kernel density plot
pp + geom_histogram(binwidth = 1, aes(y=stat(density))) +  # *density* histogram
  geom_density(bw=2, size=2, color="blue") + ggtitle("kernel density") 



#---------------------------------------------------------------------------#
#-- Two-component Gaussian Mixture Model
#---------------------------------------------------------------------------#
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
x.seq = seq(40, 100, length=200) 
f = dnmix(theta1, theta2, w, x.seq)
data.mix = tibble(x.seq, f)

#-- Make plot
pp + geom_histogram(binwidth = 1, aes(y=stat(density)), alpha=.5) + 
  geom_line(data=data.mix, aes(x=x.seq, y=f), color="blue", size=1.25) 


#---------------------------------------------------------------------------#
#-- Fitting Gaussian Mixture Model with mixtools package
#---------------------------------------------------------------------------#
library(mixtools)
gauss_mix = normalmixEM(wait, k=2)  # 2 component gaussian mixture 

w = gauss_mix$lambda       # prior probabilities (pi)
mu = gauss_mix$mu          # component means
sigma = gauss_mix$sigma    # component standard deviations
r = gauss_mix$posterior    # responsibiliites matrix


#-- Responsibility Plot
data.resp = as_tibble(r) %>% rename(f1=comp.1, f2=comp.2) %>% 
  mutate(x=wait) %>% arrange(x)

#- ggplot2
data.resp %>% 
  gather(model, r, -x) %>%            # convert to long format
  ggplot(aes(x, r, color=model)) +    # make ggplot object
  geom_line() + geom_point() +        # add lines and points
  labs(x="waiting time (min)", y="responsibilities") # change labels

#- base R
plot(data.resp$x, data.resp$f1, type='l', col="red", 
     xlab="waiting time (min)", ylab="responsibilities", las=1)
lines(data.resp$x, data.resp$f2, col="blue")

#-- Component Plot

data.mix = tibble(x = seq(40, 100, length=200),
                  f1 = w[1]*dnorm(x, mean=mu[1], sd=sigma[1]),
                  f2 = w[2]*dnorm(x, mean=mu[2], sd=sigma[2]),
                  fmix = f1 + f2) 

#- ggplot2
ggplot(data.mix) + 
  geom_area(aes(x=x, y=f1, fill="f1"), alpha=.6) +
  geom_area(aes(x=x, y=f2, fill="f2"), alpha=.6) +
  geom_line(aes(x=x, y=fmix), color="black", size=1.25) + 
  labs(x="waiting time (min)", y="density")

#- baseR
plot(data.mix$x, data.mix$f1, col="red", type='l', 
     xlab="waiting time (min)", ylab="density", las=1, 
     ylim=c(range(data.mix$fmix)))
lines(data.mix$x, data.mix$f2, col="blue")
lines(data.mix$x, data.mix$fmix, col="black", lwd=2)
