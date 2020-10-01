#####################################################################
## R Code for parametric density estimation.
## - See lecture: 06-density.pdf
## 
## Michael D. Porter
## Created: Feb 2019
## For: Data Mining (SYS-6018) at University of Virginia
## Website: https://mdporter.github.io/SYS6018/
#####################################################################

## Interactive Shiny App: https://pasda.shinyapps.io/Old_Faithful/


#-- Load Required Packages
library(fitdistrplus) # install.packages("fitdistrplus")
library(tidyverse)    # install.packages("tidyverse")

#---------------------------------------------------------------------------#
#-- Load ED counts Data
#---------------------------------------------------------------------------#
#-- Load Data
url = 'https://mdporter.github.io/SYS6018/data/ED-counts.csv'
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
lap.opt = lam.seq[which.max(loglike)]

#-- Make log-likelihood plot
lam.data = tibble(lam.seq, loglike)

ggplot(lam.data, aes(lam.seq, loglike)) + 
  geom_line() + 
  geom_point(data=filter(lam.data, loglike == max(loglike)), 
             color="red", size=2) + 
  labs(x = expression(lambda), y="log-likelihood")


#-- Plot counts with pmf overlaid
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


#---------------------------------------------------------------------------#
#-- Histograms
#---------------------------------------------------------------------------#

#-- Load the Old Faithful data
wait = datasets::faithful$waiting


#-- Histogram settings
bw = 10                     # binwidth parameter
bks = seq(40, 110, by=bw)   # create a sequence of numbers

#-- Frequency Histogram
ggplot() + geom_histogram(aes(x=x), breaks = bks, color="white") +
  labs(title="Frequency Histogram")

hist(wait, breaks=bks, las=1, main="Frequency Histogram")

#-- Relative Frequency Histogram
ggplot() + 
  geom_histogram(aes(x=x, y=after_stat(count/sum(count))), breaks = bks, color="white") +
  labs(title="Relative Frequency Histogram", y="proportion")

h.rf = hist(wait, breaks=bks, plot=FALSE)
h.rf$counts = h.rf$counts/sum(h.rf$counts)   # make relative frequency
plot(h.rf, las=1, main="Relative Frequency Histogram")

#-- Density Histogram
ggplot() + 
  geom_histogram(aes(x=x, y=after_stat(density)), breaks = bks, color="white") +
  labs(title="Density Histogram")

hist(wait, freq=FALSE, breaks=bks, las=1, main="Density Histogram")


#-- Manual histogram calculations
hist.data = tibble(wait) %>% 
  mutate(bin = cut_width(wait, width=bw, boundary=40)) %>% 
  count(bin) %>% 
  mutate(rel.freq = n/sum(n), density=rel.freq/bw)

ggplot(hist.data) + geom_col(aes(bin, rel.freq), width=1)


#---------------------------------------------------------------------------#
#-- KDE 
#---------------------------------------------------------------------------#
library(ks)

#-- Histogram
bw = 5                      # binwidth parameter
bks = seq(40, 100, by=bw)   # create a sequence of numbers
hh = hist(wait,  breaks=bks)# histogram object

#-- KDE
library(ks)
f = kde(wait, h=bw/3)
plot(f)

#-- Plot hist and kde
plot(hh, freq=FALSE, ylim=c(0, max(c(hh$density, f$estimate))), 
     las=1, main='', border='white', col='grey75') 
rug(jitter(wait))
lines(f$eval.points, f$estimate, col=2, lwd=1.25)
# OR: plot(f, add=TRUE, col=2, lwd=1.25)





#---------------------------------------------------------------------------#
#-- Multivariate KDE 
#---------------------------------------------------------------------------#
#-- Load the Old Faithful data
X = datasets::faithful

#-- Plot: Base R
plot(X, las=1); grid()
#-- Plot: ggplot
ggplot(X) + geom_point(aes(eruptions, waiting))

#-- MV KDE: Unconstrained
H1 = Hscv(X)                  # smoothed cross-validation bw estimator
f1 = kde(X, H=H1)             # use H for multivariate data

plot(f1, 
     cont = c(10, 50, 95),                        # set contour levels
     # display = "filled.contour",                # use filled contour
     las=1, xlim = c(1.0, 5.5), ylim=c(35, 100))  # set asthetics
points(X, pch=19, cex=.5, col='grey60')           # add points
grid()                                            # add grid lines



#-- Product kernel
H2 = Hscv.diag(X)
f2 = kde(X, H=H2)             

plot(f2, 
     cont = c(10, 50, 95),                        # set contour levels
     las=1, xlim = c(1.0, 5.5), ylim=c(35, 100))  # set asthetics
points(X, pch=19, cex=.5, col='grey60')           # add points
grid()                                            # add grid lines




#---------------------------------------------------------------------------#
#-- Visualize kernel shapes
#---------------------------------------------------------------------------#
plot(X, las=1); grid()

#-- Add 95% confidence ellipse for *unconstrained* at location (2, 60)
library(mixtools)
points(2, 60, pch="+", col="red", cex=1.5)
mixtools::ellipse(mu=c(2, 60), 
                  sigma=H1, 
                  alpha = .05, col="red") 

#-- Add 95% confidence ellipse for *product kernel* at location (4, 80)
library(mixtools)
points(4, 80, pch="+", col="blue", cex=1.5)
mixtools::ellipse(mu=c(4, 80), 
                  sigma=H2, 
                  alpha = .05, col="blue") 


