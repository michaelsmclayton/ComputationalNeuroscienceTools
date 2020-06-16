# Setup
try(setwd(dirname(rstudioapi::getActiveDocumentContext()$path))) # set wd or RStudio
source('../importDependencies.R')

"In this script, we will plot some basic nat data types. Specifically, we we will look
at olfactory projection neurons in Drosophila melanogaster. These data originate from
the FlyCircuit.tw project, and we access them using the lhns r package, developed by
Frechter et al. 2019, eLife"

# Import lhns (a data package for lateral horn neurons)
if(!require("lhns")) remotes::install_github("jefferislab/lhns")
library(lhns)

# ----------------------------------------------
# Get a DA1 neuron (light) from Frechter et al. 2019, eLife
# ----------------------------------------------

# Get uniglomerular ACh neurons
cellTypes = unique(lhns::most.lhins[,"anatomy.group"]) # Get all cell types in dataset
AL_mALT_PN1 = subset(lhns::most.lhins, anatomy.group == "AL-mALT-PN1")
AL_mALT_PN1[,"data.type"] = "light" # set data type to light (not sure why?)

# Get only neurons from DA1 glomerulus
"The DA1 PN is one of the most studied neurons in the fly brain. It responds to fly sex pheromones"
allGlomeruli = unique(AL_mALT_PN1[,"glomerulus"]) # Get all glomeruli
da1s = subset(AL_mALT_PN1, glomerulus == "DA1") 
n = da1s[[2]] # get one neuron

# Get points of neuron
getPoints <- function(n){
  points = nat::xyzmatrix(n)
  epoints = nat::xyzmatrix(n)[nat::endpoints(n),]
  bpoints = nat::xyzmatrix(n)[nat::branchpoints(n),]
  rpoints = nat::xyzmatrix(n)[nat::rootpoints(n),]
  return(list("points"=points, "epoints"=epoints, "bpoints"=bpoints, "rpoints"=rpoints))
}

# Plot neuron
plotNeurons <- function(n){
  nopen3d()
  points = getPoints(n)
  plot3d(n, lwd = 3, col = "green")
  spheres3d(points$epoints, col = "blue", radius = 0.5)
  spheres3d(points$bpoints, col = "red", radius = 0.75)
  spheres3d(points$rpoints, col = "darkgreen", radius = 2)
}

# Plot result
plotNeurons(n)


# ----------------------------------------------
# Get a DA1 neuron (EM) from Zheng et al. 2018
# ----------------------------------------------

# Connect to the public FAFB (female adult fly brain) instance (Zheng et al. 2018) hosted publicly by Virtual Fly Brain
adult.conn = catmaid_login(server="https://catmaid-fafb.virtualflybrain.org/")

# Get the neuron in question
da1.em = fetchn_fafb("name:PN glomerulus DA1", mirror = FALSE, reference = FAFB14)[1:5] # Bridge to the same space as our light-level neuron!
n.em = da1.em[[1]] # get one neuron
n = unspike(n, threshold = 100) # correct data abberation fromn mis-registrations

# Plot result
plotNeurons(n)


# ----------------------------------------------
# Plot neurons on atlas
# ----------------------------------------------
clear3d()

# Plot atlas (FCWB FlyCircuit reference brain)
plot3d(FCWB, alpha = 0.1, col = "lightgrey")

# Highlight given regions
regionNames = FCWBNP.surf$RegionList
highlightRegion <- function(region, color){
  plot3d(subset(FCWBNP.surf,region), alpha=0.2, col=color, lwd=2)}
highlightRegion("LH_R", "green") # right lateral horn
highlightRegion("EB", "blue") # ellipsoid body
highlightRegion("FB", "red") # fan-shaped body

# Plot prunned neurons
"We may want only bit of a neuron. If so, we can prune neurons in a given neuropil"
pruned = prune_in_volume(AL_mALT_PN1, brain = FCWBNP.surf, neuropil = "LH.R")
plot3d(pruned, col="black", soma = FALSE)

# Plot full neurons
plot3d(AL_mALT_PN1, soma = TRUE)

# Plot atlas and neurons for FAFB14 atlas
nopen3d()
plot3d(FAFB14, alpha = 0.1, col = "lightgrey")
plot3d(da1.em, soma = TRUE)


# # ----------------------------------------------
# # Plot mammalian neuron (an olfactory projection, mitral cell)
# # ----------------------------------------------

# # Get mammmalian neuron by searching the NeuroMorpho database (not working?)
# library(neuromorphr) # provides R client utilities for interacting with the API for neuromorpho.org
# mitral.df = neuromorpho_search(search_terms = c("brain_region:main olfactory bulb"))
# mitral.cells = neuromorpho_read_neurons(neuron_name = mitral.df$neuron_name, batch.size = 4, nat = TRUE, progress = TRUE, OmitFailures = TRUE)
# mitral.cells = mitral.cells[mitral.cells[,"species"]=="mouse"]
# mitral.cells = mitral.cells[grepl("mitral",mitral.cells[,"cell_type"])]
# mt  = mitral.cells["86520"][[1]]

# # Plot result
# plotNeurons(mt)