import numpy as np
import matplotlib.pylab as plt
from graspy.datasets import load_drosophila_right
from graspy.plot import heatmap
from graspy.utils import binarize, symmetrize

''' In this script, we will try to model a larval Drosophila
connectome using random graph models. Note that, in all of these
models, connectivity is sampled using a Bernoulli distribution
with a given probabilitiy. '''

# ---------------------------------------
# Load data to be modelled
# ---------------------------------------

# Load Drosophila melanogaster larva, right MB connectome (Eichler et al. 2017)
'''here we consider a binarized and directed version of the graph'''
adj, labels = load_drosophila_right(return_labels=True)
adj = binarize(adj) # adjacency matrix

# Plot adjacency matrix
def plotHeatmap(data, title, params={}):
	heatmap(X=data, inner_hier_labels=labels, hier_label_fontsize=8.0,
	font_scale=0.5, title=title, sort_nodes=True, **params)
plotHeatmap(adj, "Drosophila right MB")


# ---------------------------------------
# Erdos-Reyni (ER) model
# ---------------------------------------

'''The Erdos-Reyni (ER) model is the simplest random graph model one could
write down. We are interested in modeling the probability of an edge existing
between any two nodes, 𝑖 and 𝑗. We denote this probability 𝑃𝑖𝑗. For the ER model:
𝑃𝑖𝑗 = 𝑝, for any combination of 𝑖 and 𝑗. This means that the one parameter 𝑝 is the
overall probability of connection for any two nodes'''

from graspy.models import EREstimator
er = EREstimator(directed=True,loops=False) # Create Erdos-Reyni estimator
er.fit(adj) # Fit Erdos-Reyni model

# Show results
fig,ax = plt.subplots(1,2)
plotHeatmap(er.p_mat_, "ER probability matrix", params={'vmin':0,'vmax':1,'ax':ax[0]})
plotHeatmap(er.sample()[0], "ER sample", params={'ax':ax[1]})
fig.suptitle('Erdos-Reyni (ER) model', fontsize=16)
print(f"ER \"p\" parameter: {er.p_}")
plt.savefig('figure.png')

# ---------------------------------------
# Degree-corrected Erdos-Reyni (DCER) model
# ---------------------------------------

'''A slightly more complicated variant of the ER model is the degree-corrected
Erdos-Reyni model (DCER). Here, there is still a global parameter 𝑝 to specify
relative connection probability between all edges. However, we add a promiscuity
parameter 𝜃𝑖 for each node 𝑖 which specifies its expected degree relative to other
nodes: 𝑃𝑖𝑗=𝜃𝑖𝜃𝑗𝑝, so the probility of an edge from 𝑖 to 𝑗 is a function of the two nodes'
degree-correction parameters, and the overall probability of an edge in the graph'''

from graspy.models import DCEREstimator
dcer = DCEREstimator(directed=True,loops=False)
dcer.fit(adj) # Fit Degree-corrected Erdos-Reyni model
promiscuities = dcer.degree_corrections_

# Show results
fig,ax = plt.subplots(1,2)
plotHeatmap(dcer.p_mat_, "DCER probability matrix", params={'vmin':0,'vmax':1,'ax':ax[0]})
plotHeatmap(dcer.sample()[0], "DCER sample", params={'ax':ax[1]})
fig.suptitle('Degree-corrected Erdos-Reyni (ER) model', fontsize=16)
print(f"DCER \"p\" parameter: {dcer.p_}")
plt.savefig('figure.png')