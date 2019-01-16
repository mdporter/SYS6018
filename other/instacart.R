## Analysis of Instacart Data
##
## Instacart Data
## https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2
## https://www.instacart.com/datasets/grocery-shopping-2017
## https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b

#-- Load Required Packages
library(tidyverse) # loads dplyr, readr, ggplot2, etc
library(arules)
library(arulesViz)

#-----------------------------------------------------------------------#
#-- Load Data
#-----------------------------------------------------------------------#

#-- Load the "orders_products__train.csv" (notice two __) and "products.csv"
data.dir = "https://github.com/mdporter/SYS6018/raw/master/data"
orders = read_csv(file.path(data.dir, "order_products__train.csv"))
products = read_csv(file.path(data.dir, "products.csv"))


#- Join orders and products, only keeping columns of interest
Y = left_join(orders %>% select(order_id, product_id), 
              products %>% select(product_id, product_name), 
              by = "product_id") %>% 
  arrange(order_id)


#-----------------------------------------------------------------------#
#-- Examine and Clean Data
#-----------------------------------------------------------------------#
#-- ensure that there is a one-to-one between product_id, product_name
Y %>% 
  group_by(product_id) %>% 
  summarize(ndups = n_distinct(product_name)) %>% 
  filter(ndups>1)

Y %>% 
  group_by(product_name) %>% 
  summarize(ndups = n_distinct(product_id)) %>% 
  filter(ndups>1)
# Looks good, one-to-one matching

#-- No order should have duplicate items. Remove any duplicates.
Y = Y %>% distinct(order_id, product_id, .keep_all = TRUE)

#-- basic stats
NT = n_distinct(Y$order_id)     # Number of transactions
NI = n_distinct(Y$product_name) # Number of items

#-- distribution of itemset length
count(Y, order_id) %>% 
  ggplot(aes(n)) + geom_bar() + xlab("length of itemset")

count(Y, order_id) %>% 
  ggplot(aes(n)) + stat_ecdf() + xlab("length of itemset")

#-----------------------------------------------------------------------#
#-- Convert to arules::transactions class
#-----------------------------------------------------------------------#

#-- get transaction list
tList = split(Y$product_name, Y$order_id)    # get transaction list
# tList = lapply(tList, unique)             # remove duplicates  

#-- get transaction class
trans = as(tList, "transactions")

summary(trans)   # print summary info

#-----------------------------------------------------------------------#
#-- apriori2df(): convert output from apriori() to a tibble/dataframe
#-----------------------------------------------------------------------#

#-- Convert to data frame / tibble
# use this instead of inspect(), which only prints to screen
apriori2df <- function(x){
  if(class(x) == "itemsets"){
    out = data.frame(items=labels(x), x@quality, stringsAsFactors = FALSE)
  }
  else if(class(x) == "rules"){
    out = data.frame(
      lhs = labels(lhs(x)),
      rhs = labels(rhs(x)),
      x@quality, 
      stringsAsFactors = FALSE)
  }
  else stop("Only works with class of itemsets or rules")
  if(require(dplyr)) tbl_df(out) else out
}


#-----------------------------------------------------------------------#
#-- Find Frequent Itemsets
#-----------------------------------------------------------------------#

#-- get item counts and support for single itemsets
itemFreq = count(Y, product_name, sort=TRUE) %>% mutate(support=n/NT)


# plot top 20
itemFreq %>% slice(1:20) %>% 
  ggplot(aes(fct_reorder(product_name, n), n)) + # order bars by n
  geom_col() +         # barplot
  coord_flip() +       # rotate plot 90 deg
  theme(axis.title.y = element_blank()) # remove y axis title

#-- Find all frequent itemsets with support >= .01
fis = apriori(trans, 
              parameter = list(support = .01, target="frequent"))

apriori2df(fis) %>% arrange(-support)  # order by support (largest to smallest)

#-- Find all frequent itemsets (s=.01) of length 2 (minlen=2)
fis2 = apriori(trans, 
               parameter = list(support = .01, minlen=2, target="frequent"))

apriori2df(fis2) %>% arrange(-support)  # order by support (largest to smallest)

#-- Add lift using the interestMeasure() function
apriori2df(fis2) %>% 
  mutate(lift = interestMeasure(fis2, measure="lift", trans)) %>% 
  arrange(-lift)

#-----------------------------------------------------------------------#
#-- Find Association Rules
#-----------------------------------------------------------------------#

#-- Find association rules with support>=.001 and confidence>=.50
rules = apriori(trans, 
             parameter = list(support=.001, confidence=.50, 
                              minlen=2,target="rules"))

apriori2df(rules) %>% arrange(-lift)
apriori2df(rules) %>% arrange(-confidence)

#-- Add other interest measures
apriori2df(rules) %>% 
  mutate(addedValue = interestMeasure(rules, measure="addedValue", trans), 
         PS = interestMeasure(rules, measure="leverage", trans)) %>% 
  arrange(-addedValue)


#-----------------------------------------------------------------------#
#-- Visualizing Association Rules
#-----------------------------------------------------------------------#
library(arulesViz)

#-- Interactive plot. Some nodes are items, others are rules
plot(rules, method="graph", measure="lift", engine="interactive")



#-----------------------------------------------------------------------#
#-- Target specific items
#-----------------------------------------------------------------------#

#-- Find Hass Avocados, Tomatoes, and Onions
filter(itemFreq, str_detect(product_name, "Hass Avocado"))
  # Notice all the variations on "Avocado"
filter(itemFreq, str_detect(product_name, "Tomato"))
filter(itemFreq, str_detect(product_name, "Onions"))


#-- Find all rules with 'Small Hass Avocado' on the lhs
rules2 = apriori(trans, 
              parameter = list(support=.001, confidence=.10, 
                               minlen=2,target="rules"), 
              appearance = list(lhs = "Small Hass Avocado"))

apriori2df(rules2) %>% 
  mutate(addedValue = interestMeasure(rules2, measure="addedValue", trans), 
         PS = interestMeasure(rules2, measure="leverage", trans)) %>% 
  arrange(-lift)

#-- find frequency of the 3 items 
itemset = c("Small Hass Avocado", "Red Vine Tomato", "Yellow Onions")
ap = apriori(trans, 
        parameter = list(support=0, target="frequent"), 
        appearance = list(items = itemset))

apriori2df(ap) %>% 
  mutate(lift = interestMeasure(ap, measure="lift", trans)) %>% 
  arrange(-lift)


  


