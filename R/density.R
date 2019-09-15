#####################################################################
## R Code for parametric density estimation.
## - See lecture: 03-density.pdf
## 
## Michael D. Porter
## Created: Feb 2019
## For: Data Mining (SYS-6018) at University of Virginia
## Website: https://mdporter.github.io/SYS6018/
#####################################################################


#-- Install Required Packages
library(tidyverse)    # install.packages("tidyverse")
library(fitdistrplus) # install.packages("fitdistrplus")

#---------------------------------------------------------------------------#
#-- Load ED counts Data
#---------------------------------------------------------------------------#
#-- Load Data
url = 'https://raw.githubusercontent.com/mdporter/SYS6018/master/data/ED-counts.csv'
x = readr::read_csv(url)$count  # extract as a vector

#-- empirical pmf
ggplot() + geom_bar(aes(x=x))

#-- Summary
summary(x)
mean(x)
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


#-- Plot counts with pmf overlaid
lam.opt = 85   # set lambda
fit_pois = tibble(x=seq(min(x), max(x), by=1), y=dpois(x, lambda=lam.opt))

ggplot() + 
  geom_bar(aes(x, y=stat(prop)), fill="grey70") + 
  geom_point(aes(x,y), data=fit_pois, color="blue", alpha=.5) + 
  labs(y = "pmf")
  


#---------------------------------------------------------------------------#
#-- Negative Binomial Model: ED counts Data
#---------------------------------------------------------------------------#

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



