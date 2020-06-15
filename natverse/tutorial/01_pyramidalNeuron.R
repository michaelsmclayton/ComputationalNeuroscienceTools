# Setup
try(setwd(dirname(rstudioapi::getActiveDocumentContext()$path))) # set wd or RStudio
source('../importDependencies.R')

# ----------------------------------------------
# Load neuron data
# ----------------------------------------------
"Here, we want to test out some data that the Akerman group in Oxford has derived from confocal
imaging of Layer 5 pyramidal cells, and processed through simple neurite tracer and other data
extraction tools in FIJI"

# Load neuron skeleton and bouton files
neuron = read.neuron("./data/axon_traces_27_06_19_s4_c7_Pom.traces")
neuron = nat::resample(neuron, stepsize=0.1) # downsample skeleton
boutons = read.csv("./data/27_06_19_s4_c7_Pom_boutons.csv")

# ----------------------------------------------
# Plot the neuron (and correct bouton locations)
# ----------------------------------------------
"If soma=True (or a given value), and sphere will be plotted at the root of the skeleon.
However, sometimes the root may not actually be the soma. If not, you can try to use the
correctsoma function in the catnat package to update the soma/root of a skeleton interactively"

# Define plotting function
plotNeuronWithBoutons <- function(neuron,boutons,dims,color) {
  plot3d(as.neuronlist(neuron), col="black", soma=FALSE, lwd=4) # Plot neuron
  points3d(boutons[,dims], col=color, size=5) # Plot the new boutons
}

# Plot neuron with boutons
nopen3d() # new figure...
plotNeuronWithBoutons(neuron,boutons,dims=c("X","Y","Slice"),color="red")

"You will see in this initial plot that the boutons seem to be systematically displaced
from the skeleton. We can align the boutons with the skeeton in two ways:"

# 1. Move the skeleton towards the boutons
nopen3d()
trans = computeTransform(x=points, y=points.new, type = "affine") # compute transform between
xyzmatrix(neuron) <- applyTransform(xyzmatrix(neuron), trans)
boutons = read.csv("./data/27_06_19_s4_c7_Pom_boutons.csv")
plotNeuronWithBoutons(neuron,boutons,dims=c("X","Y","Slice"),color="green")

# 2. Move bouton locations towards skeleton
nopen3d()
points = boutons[,c("X","Y","Slice")] # Get bouton points
near = nabor::knn(data=xyzmatrix(neuron), query=points, k=1)$nn.idx # Find idxs of nearest skeleton elements to each bouton position
points.new = xyzmatrix(neuron)[near,] # Get points of nearest skeleton elements (defined above)
boutons[,c("X","Y","Z")] = points.new # Set bouton positions to their nearest skeleton elements
boutons$treenode = near
plotNeuronWithBoutons(neuron,boutons,dims=c("X","Y","Z"),color="blue")

# ----------------------------------------------
# Assess neuron statistics
# ----------------------------------------------

# Compute tree statistics for all the neurons in a neuronlist object
info = summary(neuron) 
print(info)

# Calculate boutons and branchpoinrs per cable
boutons.per.cable = nrow(boutons)/info$cable.length # boutons per cable length
# -> ~0.05 boutons per micron of cable
bps.per.cable = info$branchpoints/info$cable.length # branchpoints per cable
# -> ~0.006 branchpoints per micron of cable (so about 10x more boutons per cable than branch points)

# Assign the Strahler number of the neurons (a numerical measure of its branching complexity)
"NOTE: for this to work properly, the root of the neuron MUST be the soma"
nopen3d()
neuron = assign_strahler(neuron)
neuron$d$strahler_order
plot3d(as.neuronlist(neuron), col = "black", soma = FALSE, lwd = 1)
points3d(xyzmatrix(neuron), col = neuron$d$strahler_order) # Plot branches coloured by Strahler order

# Assign Strahler numbers to the boutons
boutons$strahler = neuron$d$strahler_order[boutons$treenode]
print(table(boutons$strahler))
# most of the boutons are on Strahler order 1 branches (i.e. the leaf branches of the neuron)