# -----------------------------------------------------
# Install the nat tools
# -----------------------------------------------------
if(!require("natmanager")) install.packages("natmanager")
# natmanager::install("natverse") # See http://natverse.org/install/ for details / troubleshooting
if(!require("here")) install.packages("here")
# if(!require("ggpubr")) install.packages("ggpubr")
if(!require("reshape2")) install.packages("reshape2")
if(!require("catnat")) remotes::install_github("jefferislab/catnat")

# -----------------------------------------------------
# Import tools
# -----------------------------------------------------
library(natverse) # load the natverse, which includes packages like nat, nat.nblast, etc.
library(catnat) # additional packages that we will use in these examples
# library(ggpubr)
library(reshape2)
library(Morpho)
library(igraph)
library(rgl)
library(nat)
library(ggplot2)