#####################################################################
## Function to predict binary node label.
## Analysis on Yeast protein interaction network data
## 
## Michael D. Porter
## Created: Jan 2019; updated Oct 2020
## For: Data Mining (SYS-6018/SYS-4582) at University of Virginia
#####################################################################

#-- Load Required Functions
library(tidyverse)
library(igraph)
library(sand)


#-- node_predict()
#-------------------------------------------------------------------#
# Function to predict binary label using nearest neighbors
#
#  Approach is to find all 1st order neighbors of the nodes without
#   labels, get the labels of the neighbors, and estimate the 
#   probability of being in class 1 as a function of the labels of 
#   the neighbors. 
# 
# Inputs: 
# - g: is an igraph graph object
# - nodes: index or names of vertices in which to predict label
# - labels: binary vector of node labels (must be in {0,1})
# - k: number of pseudo-observations
# - p0: prior probability that label=1
# Outputs:
# - node: node label
# - p: estimated probability label is 1
# - n: number of neighbors with label = 1
# - d: number of neighbors with non-missing labels
# Notes:
# - would be more efficient using the matrix calculations
#-------------------------------------------------------------------#
node_predict <- function(g, nodes, labels, k=0, p0=NULL){
  
  #-- Add the labels to the graph. Name it "y"
  V(g)$y = labels                      # add labels as an attribute
  
  #-- Compute p0 if not given as an argument
  if(is.null(p0)){
    N = vcount(g) - length(nodes)         # number of nodes (with labels)   
    N1 = sum(V(g)$y) - sum(V(g)[nodes]$y) # number of x=1 (for labeled nodes)
    p0 = N1/N                             # prior probability of node with label=1
  }

  #-- Loop over all nodes, get the neighborhood, and record labels
  d = n = numeric(length(nodes))
  for(i in 1:length(nodes)){
    node = V(g)[nodes[i]]$name         # get name of node
    gamma = neighbors(g, node)         # neighboring nodes
    y = V(g)[gamma]$y                  # labels of the neighbors
    y = y[!is.na(y)]                   # remove missing labels
    d[i] = length(y)                   # degree
    n[i] = sum(y)                      # number of neighbors with label = 1
  }
  
  #-- Estimate probability
  p = (n+k*p0)/(d+k)                   
  
  #-- Return the results as a data frame
  score = tibble(node = V(g)[nodes]$name, p, n, d)
return(score)
}


#-- eval_function()
#-------------------------------------------------------------------#
# Function to evaluate the node_predict() algorithm
#
# Uses repeated hold-out to evaluate performance. 
# k is the tuning parameter of the node_predict() model. 
#
# Inputs: 
# - G: is an igraph graph object
# - y: binary vector of node labels (must be in {0,1})
# - k: number of pseudo-observations
# - f: fraction of nodes to remove labels
# - niter: number of times to repeat
# Outputs:
# - node: node label
# - p: estimated probability label is 1
# - n: number of neighbors with label = 1
# - d: number of neighbors with non-missing labels
# - y: the true node label
# Notes:
# - Could use cross-validation instead of repeated hold-out.
#-------------------------------------------------------------------#
eval_function <- function(G, y = V(G)$ICSC, k=0, f=.05, niter=100){
  N = vcount(G)          # total number of nodes
  n.samp = ceiling(f*N)  # number of nodes in sample
  out = tibble()
  for(i in 1:niter){
    ind = sample(N, size=n.samp)    # indices in which to remove labels
    y.true = y[ind]                 # true label values
    phat = node_predict(G, nodes=ind, labels=y, k=k)       # predict label
    out = bind_rows(out, 
                    phat %>% mutate(y = y.true, iter=i))   # add true label
  }
  return(out)
}


#==================================================================#
#-- Evaluate Algorithm 
#
# Evaluate how our nearest neighbor approach can classify nodes
#  using the ICSC attribute: indicates whether the protein is 
#  annotated with the “intracellular signaling cascade” GO term, 
#  zero or one.
#==================================================================#

#-- Load the Yeast Protein Interaction Data
data(ppi.CC, package="sand") 
G = upgrade_graph(ppi.CC)    # it apparently need to be "updated"


#-- Visualize
set.seed(2019)
plot(G,
     layout=layout_with_fr(G),   # determines coordinates of nodes
     vertex.label=NA,
     vertex.size=5,
     vertex.color=ifelse(V(G)$ICSC==1, "red", "blue"))  # color vertices by “intracellular signaling cascade” GO term


#-- Test out our function (using k=0)
node_predict(G, nodes=1:4, labels=V(G)$ICSC, k=0)


#-- Evaluate Performance over k
perf = 
  bind_rows(
    `k=0` = eval_function(G, y = V(G)$ICSC, k=0, f=.05),
    `k=1` = eval_function(G, y = V(G)$ICSC, k=1, f=.05),
    `k=2` = eval_function(G, y = V(G)$ICSC, k=2, f=.05),
  .id = "k")

#-- Boxplot
ggplot(perf, aes(y, p, group=y)) + geom_boxplot() + facet_wrap(~k)

#-- Accuracy
thres = 1/2
perf %>% 
  mutate(y.hat = ifelse(p > thres, 1, 0)) %>%   # hard classification 
  group_by(k) %>% summarize(accuracy = mean( y.hat == y ))

#-- Confusion Tables
perf %>% 
  mutate(y.hat = ifelse(p > thres, 1, 0)) %>%   # hard classification 
  group_by(k) %>% 
  summarize(table = list(table(pred = y.hat, true=y))) %>% 
  mutate(table = set_names(table, k)) %>% 
  pull(table)





