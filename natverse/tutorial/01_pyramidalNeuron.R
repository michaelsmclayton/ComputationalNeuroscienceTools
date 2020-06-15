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

# 1. Move bouton locations towards skeleton
nopen3d()
points = boutons[,c("X","Y","Slice")] # Get bouton points
near = nabor::knn(data=xyzmatrix(neuron), query=points, k=1)$nn.idx # Find idxs of nearest skeleton elements to each bouton position
points.new = xyzmatrix(neuron)[near,] # Get points of nearest skeleton elements (defined above)
boutons[,c("X","Y","Z")] = points.new # Set bouton positions to their nearest skeleton elements
boutons$treenode = near
plotNeuronWithBoutons(neuron,boutons,dims=c("X","Y","Z"),color="blue")

# 2. Move the skeleton towards the boutons
nopen3d()
trans = computeTransform(x=points, y=points.new, type = "affine") # compute transform between
xyzmatrix(neuron) <- applyTransform(xyzmatrix(neuron), trans)
original_boutons = read.csv("./data/27_06_19_s4_c7_Pom_boutons.csv")
plotNeuronWithBoutons(neuron,original_boutons,dims=c("X","Y","Slice"),color="green")


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


# ----------------------------------------------
# Calculate Strahler across neuron (a numerical measure of its branching complexity)
# ----------------------------------------------

# Assign the Strahler number (and plot over neuron)
"NOTE: for this to work properly, the root of the neuron MUST be the soma"
nopen3d()
neuron = assign_strahler(neuron)
strahler_orders = neuron$d$strahler_order
plot3d(as.neuronlist(neuron), col="black", soma=FALSE, lwd=1)
points3d(xyzmatrix(neuron), col=strahler_orders) # Plot branches coloured by Strahler order

# Assign Strahler numbers to branchpoinrs and boutons
boutons$strahler = strahler_orders[boutons$treenode]
boutonsStrahler = table(boutons$strahler)
branchPointStrahler = table(strahler_orders)
print(boutonsStrahler) # most of the boutons are on Strahler order 1 branches (i.e. the leaf branches of the neuron)

# Set missing column values to 0
boutonsStrahler[setdiff(names(branchPointStrahler),names(boutonsStrahler))] = 0

# Calculate boutons per micron by Strahler order (which shows that they are is different by Strahler order)
print(boutonsStrahler/branchPointStrahler)


# ----------------------------------------------
# Perform Sholl analysis (dendritic intersections vs. distance from soma)
# ----------------------------------------------
"Sholl analysis is a method of quantitative analysis commonly used in neuronal studies to characterize
the morphological characteristics of an imaged neuron, first used to describe the differences in the
visual and motor cortices of cats.
  The analysis includes counting the number of dendritic intersections that occur at fixed distances
from the soma in concentric circles. This analysis reveals the number of branches, branch geometry,
and overall branching patterns of neurons."

# Perform Sholl analysis
sholla = sholl_analysis(neuron)

# Plot the result
if (rstudioRunning()==FALSE){X11()} # new ggplot figure
ggplot(sholla, aes(x=radii, y=intersections, color="darkgrey")) + xlab('Distance from soma') +
  ylab('Dendritic intersections') + geom_line()+theme_minimal()

# We could also plot the distance to boutons on this graph
if (rstudioRunning()==FALSE){X11()}
root = neuron$d[rootpoints(neuron),c("X","Y","Z")]
near2 = nabor::knn(data = root, query = xyzmatrix(boutons),k = 1)$nn.dist
bdist = data.frame(intersections = 0, dist = near2)
ggplot(data = bdist, aes(x=dist, color="red", fill = "red")) + 
  geom_density()+
  theme_minimal()


# ----------------------------------------------
# Work out which layers each bouton is in
# ----------------------------------------------

# Load layer information
surface = read.csv("data/layer_lines_csv/surface_MAX_27_06_19_s4_c7_Pom.csv")[,c("X","Y")]
surface$lboundary = "l1_surface"
l23 = read.csv("data/layer_lines_csv/L23_L1_MAX_27_06_19_s4_c7_Pom.csv")[,c("X","Y")]
l23$lboundary = "l23_l1"
l4 = read.csv("data/layer_lines_csv/L4_L23_MAX_27_06_19_s4_c7_Pom.csv")[,c("X","Y")]
l4$lboundary = "l4_23"
l5 = read.csv("data/layer_lines_csv/L5_L4_MAX_27_06_19_s4_c7_Pom.csv")[,c("X","Y")]
l5$lboundary = "l5_l4"
l6 = read.csv("data/layer_lines_csv/L6_L5_MAX_27_06_19_s4_c7_Pom.csv")[,c("X","Y")]
l6$lboundary = "l6_l5"
wm = read.csv("data/layer_lines_csv/wm_L6_MAX_27_06_19_s4_c7_Pom.csv")[,c("X","Y")]
wm$lboundary = "wm_l6"

# Combine layer information together
layers = rbind(surface,l23,l4,l5,l6,wm)
layers$lower = gsub("_.*","",layers$lboundary) # Get lower layer
layers$upper = gsub(".*_","",layers$lboundary) # Get upper layer

# Convert values to microns (these coordinates are 2D, and in pixels)
layers[,c("X","Y")] = layers[,c("X","Y")]/2.4089

# Add a Z dimension (mean z-axis value)
layers$Z = colMeans(xyzmatrix(neuron))["Z"]

# Define function to assign layers
assignLayers <- function(data){
  "To assign a given layer, we need to find the closest point, and see if it is above or below the layer line"
  near = nabor::knn(query = xyzmatrix(data), data = xyzmatrix(layers), k = 1)$nn.idx
  data$nearest.lboundary = layers[near,"lboundary"]
  data$nearest.lboundary.x = layers[near,"X"]
  data$layer = ifelse(data$nearest.lboundary.x>data$X, 
                      gsub("_.*","",data$nearest.lboundary),
                      gsub(".*_","",data$nearest.lboundary))
  return(data)
}

# Assign the branchpoint and boutons to layers
boutons = assignLayers(boutons)
neuron$d = assignLayers(neuron$d)

# Calculate bouton density per layer
node.layers = table(neuron$d$layer)
boutons.layers = table(boutons$layer)[names(node.layers)]
bouton.density = boutons.layers / node.layers

# Plot neuron with respect to column divisions (and with boutons colored for each layer)
clear3d()
plot3d(as.neuronlist(neuron), col = "black", lwd = 3)
spheres3d(boutons[,c("X","Y","Z")], col = layer.cols[boutons$layer], radius = 10)
points3d(xyzmatrix(layers), col = layer.cols[layers$lower])


# ----------------------------------------------
# Combine data and plot results
# ----------------------------------------------

# Make data frame for plotting
df = rbind(node.layers, boutons.layers,bouton.density)
df = melt(df) # row for each data point
colnames(df) = c("type", "layer", "value")

# Nodes per layer
ggplot(subset(df, grepl("node.layers",type)), aes(x=type, y=value, fill=layer)) + 
  geom_bar(stat="identity") + scale_fill_manual(values = layer.cols) + 
  ggtitle("Nodes per layer") + theme_minimal() + theme(plot.title = element_text(hjust = 0.5))

# Boutons per layer
ggplot(subset(df, grepl("boutons.layers",type)), aes(x=type, y=value, fill=layer)) +
  geom_bar(stat="identity") + scale_fill_manual(values = layer.cols) +
  ggtitle("Boutons per layer") + theme_minimal() + theme(plot.title = element_text(hjust = 0.5))

# Bouton density per layer (bar chart)
ggplot(subset(df, grepl("bouton.density",type)), aes(x=type, y=value, fill=layer)) +
  geom_bar(stat="identity") + scale_fill_manual(values = layer.cols) +
  ggtitle("Bouton density per layer") + theme_minimal() + theme(plot.title = element_text(hjust = 0.5))


# ----------------------------------------------
# Analyse Euclidean distance from neuron midline
# ----------------------------------------------

# Load midline data
axis = read.csv("data/Axis_line/Axis_line_27_06_19_s4_c7.csv")[,c("X","Y")]
axis[,c("X","Y")] = axis[,c("X","Y")]/2.4089 # Convert values to microns
axis$Z = colMeans(xyzmatrix(neuron))["Z"] # Add z-axis

# Get Euclidean bouton distances to the midline
bouton.points = boutons[,c("X","Y","Slice")] # let's use the original bouton locations for this
boutons$dist.to.midline = unlist(nabor::knn(query = bouton.points, data = axis, k = 1)$nn.dists)

# Plot Boutons Euclidean Distance to midline
ggplot(boutons, aes(x=dist.to.midline, color = layer, fill = layer)) + geom_histogram(binwidth=50) +
  scale_fill_manual(values = layer.cols) + scale_color_manual(values = layer.cols) +
  ggtitle("Boutons Euclidean Distance to midline") + theme_minimal() + theme(plot.title = element_text(hjust = 0.5))


# ----------------------------------------------
# Analyse Geodesic distance from neuron midline
# ----------------------------------------------
"To calculate geodesic distance, we are just going to count the number of points that separate boutons.
Let's work this out per layer"

# Create a directed graph
graph = as.ngraph(neuron) # We can now use this representation with functions from the package iGraph.

# Get a single distance value per layer
boutons$interbouton.distance  = NA
interbouton = data.frame()
for(l in unique(boutons$layer)){
  # Just boutons on one layers
  bouton.nodes = unique(subset(boutons, layer == l)$treenode)
  # So we want to take the mean distance for the bouton behind, and in front, of each bouton of interest
  ## the distance to the next bouton, away from the root.
  dists1 = distances(graph, v = bouton.nodes, to = bouton.nodes, mode = c("out"), weights = NULL)
  diag(dists1) = NA
  dist.out = apply(dists1, 1, function(x) min(x, na.rm = TRUE)) # node that cannot be reached have an infinite value
  dist.out = dist.out[!is.infinite(dist.out)]
  ## And towards it
  dists2 = distances(graph, v = bouton.nodes, to = bouton.nodes, mode = c("in"), weights = NULL)
  diag(dists2) = NA
  dist.in = apply(dists2, 1, function(x) min(x, na.rm = TRUE)) # node that cannot be reached have an infinite value
  dist.in = dist.in[!is.infinite(dist.in)]
  interbouton = rbind(interbouton, data.frame(
    layer = l,
    mean = mean( c(dist.out,dist.in) ) / stepsize,
    sd = sd( c(dist.out,dist.in) ) / stepsize,
    number = length(bouton.nodes)
  ))
  ## Calculate on a single neuron basis
  dist.both = intersect(names(dist.in), names(dist.out))
  dists = c(dist.in[dist.both] + dist.out[dist.both])/2
  missing1 = names(dist.out)[!names(dist.out)%in%dist.both]
  missing2 = names(dist.in)[!names(dist.in)%in%dist.both]
  dists[missing1] = dist.out[missing1]
  dists[missing2] = dist.in[missing2]
  dists = dists / stepsize
  match.to.bouton = match(names(dists), boutons$treenode)
  boutons[match.to.bouton,]$interbouton.distance = dists
}

# Plot Boutons Geodesic Distance to midline (jitter plot)
ggplot(boutons, aes(x=layer, y=interbouton.distance, color = layer)) + 
  geom_jitter(position=position_jitter(0.2)) + 
  stat_summary(fun.data="mean_sdl", mult=1, geom="crossbar", width=0.5, col = "grey70") + 
  stat_summary(fun.data=mean_sdl, mult=1, geom="pointrange", color="red")+
  scale_color_manual(values = layer.cols) +
  ggtitle("Boutons Geodesic Distance to midline") + theme_minimal() + theme(plot.title = element_text(hjust = 0.5))
