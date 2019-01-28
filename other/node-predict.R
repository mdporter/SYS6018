#####################################################################
## Function to predict binary node label.
## Analysis on Yeast protein interaction network data
## 
## Michael D. Porter
## Created: Jan 2019
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
# Notes:
# - would be more efficient using the matrix calculations
#-------------------------------------------------------------------#
node_predict <- function(g, nodes, labels, k=0, p0=NULL){
  
  #-- Add the labels to the graph. Name it "x"
  V(g)$x = labels                      # add labels as an attribute
  
  #-- Compute p0 if not given as an argument
  if(is.null(p0)){
    N = vcount(g) - length(nodes)      # number of nodes (with labels)   
    N1 = sum(V(g)$x) - sum(V(g)[nodes]$x) # number of x=1 (for labeled nodes)
    p0 = N1/N                          # prior probability of node with label=1
  }

  #-- Loop over all nodes, get the neighborhood, and record labels
  d = n = numeric(length(nodes))
  for(i in 1:length(nodes)){
    node = V(g)[nodes[i]]$name         # get name of node
    gamma = neighbors(g, node)         # neighboring nodes
    y = V(g)[gamma]$x                  # labels of the neighbors
    y = y[!is.na(y)]                   # remove missing labels
    d[i] = length(y)                   # degree
    n[i] = sum(y)                      # number of neighbors with label = 1
  }
  
  #-- Estimate probability
  p = (n+k*p0)/(d+k)                   
  
  #-- Return the results as a data frame
  score = data.frame(node = V(g)[nodes]$name, p, n, d)
return(score)
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
plot(G,
     layout=layout_with_fr(G),   # determines coordinates of nodes
     vertex.label=NA,
     vertex.size=5,
     vertex.color=ifelse(V(G)$ICSC==1, "red", "blue"))  # color of vertices by “intracellular signaling cascade” GO term


#-- Test out our function
node_predict(G, nodes=1:4, labels=V(G)$ICSC, k=0)


#-- Randomly remove labels and evaluate performance
f = .05      # fraction of nodes to remove labels
k = 0        # set shrinkage parameter k
niter = 100  # number of iterations

N = vcount(G)          # total number of nodes
n.samp = ceiling(f*N)  # number of nodes in sample


#-- Run niter times
X = P = numeric(niter*n.samp)
ii = 0
for(i in 1:niter){
  ind = sample(N, size=n.samp) # indices in which to remove labels
  x.true = V(G)$ICSC[ind]         # true label values
  phat = node_predict(G, nodes=ind, labels=V(G)$ICSC, k=k)
  X[1:n.samp + ii] = x.true    
  P[1:n.samp + ii] = phat$p
  ii = ii + n.samp
}

#-- Boxplot
boxplot(P~X, las=1)

#-- Make hard classification 
thres = 1/2            
X.hat = ifelse(P > thres, 1, 0)

#-- Probability of correct classification
mean(X == X.hat)

#-- Confusion Matrix
table(X, X.hat)


