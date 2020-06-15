# Setup
try(setwd(dirname(rstudioapi::getActiveDocumentContext()$path))) # set wd or RStudio
source('../importDependencies.R')

# ----------------------------------------------
# Quick start
# ----------------------------------------------
"The 'nat' package (imported in importDependencies.R) in contains a number of
default neuronlists. A simple neuronlist to play with is, kcs20, which contains
20 skeletonised Drosophila Kenyon cells as dotprops objects. Original data is due
to Chiang et al. 2011, who have shared their raw data at http://flycircuit.tw"

# Show neuron types
table(with(kcs20, type))

# Show first entries
head(kcs20)

# Plot neurons
clear3d()
plot3d(kcs20, type=='ab', col='red')
plot3d(kcs20, type=='apbp', col='blue')
plot3d(kcs20, subset=type=='gamma', col='green')
