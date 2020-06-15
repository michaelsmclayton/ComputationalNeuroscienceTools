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

# Plot neuron with boutons
nopen3d() # new figure...
plot3d(as.neuronlist(neuron), col = "black", soma = FALSE, lwd = 4) # Plot neuron
points3d(boutons[,c("X","Y","Slice")], col = "red") # Plot boutons

"You will see in this initial plot that the boutons seem to be systematically displaced
from the skeleton. We can align the boutons with the skeeton in two ways:"

# 1. Move bouton locations towards skeleton
points = boutons[,c("X","Y","Slice")] # Get bouton points
near = nabor::knn(data=xyzmatrix(neuron), query=points, k=1)$nn.idx # Find idxs of nearest skeleton elements to each bouton position
points.new = xyzmatrix(neuron)[near,] # Get points of nearest skeleton elements (defined above)
boutons[,c("X","Y","Z")] = points.new # Set bouton positions to their nearest skeleton elements
boutons$treenode = near
points3d(boutons[,c("X","Y","Z")], col = "cyan", size = 5) # Plot the new boutons

# 2. Move the skeleton towards the boutons
nopen3d()
trans = computeTransform(x=points, y=points.new, type = "affine") # compute transform between
xyzmatrix(neuron) <- applyTransform(xyzmatrix(neuron), trans)
plot3d(as.neuronlist(neuron), col = "black", soma = FALSE, lwd = 4) # Plot neuron
plot3d(as.neuronlist(neuron), col = "green", soma = FALSE, lwd = 4)