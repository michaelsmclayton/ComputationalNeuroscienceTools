import numpy as np
import pandas as pd
import bokeh
import hvplot.pandas
import holoviews as hv
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
renderer = hv.renderer('bokeh')

# Import neuprint
from neuprint import Client, fetch_roi_hierarchy, fetch_neurons, fetch_adjacencies, \
        NeuronCriteria as NC, merge_neuron_properties, connection_table_to_matrix, \
        fetch_synapses, SynapseCriteria as SC, fetch_synapse_connections
from neuprint.client import setup_debug_logging
'''notes from https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html'''

# ------------------------------------------
# Setup client
# ------------------------------------------
''' All communication with the neuPrintHTTP server is done via a Client object.
notes from https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html
'''
# Get personal authentication tokens
with open('../authToken', 'r') as file:
    authToken = file.read()

# Create Client object
c = Client(server='neuprint.janelia.org', dataset='hemibrain:v1.0.1', token=authToken)
# c.fetch_version() # '0.1.0'


# ------------------------------------------
# Cypher Logging
# ------------------------------------------
'''Tip: Inspect all cypher queries by enabling debug logging'''
# setup_debug_logging()


# ------------------------------------------
# Execute a custom query
# ------------------------------------------

'''This query will return all neurons in the ROI ‘AB’ that have greater than 10 pre-synaptic sites.
Results are ordered by total synaptic sites (pre+post)'''
query = """\
    MATCH (n :Neuron {`AB(R)`: true})
    WHERE n.pre > 10
    RETURN n.bodyId AS bodyId, n.name AS name, n.pre AS numpre, n.post AS numpost
    ORDER BY n.pre + n.post DESC
"""
results = c.fetch_custom(query)
print(f"Found {len(results)} results")
print(results.head())


# ------------------------------------------
# ROIs
# ------------------------------------------
'''In neuprint, each neuron is annotated with the list of regions (ROIs) it intersects, along
with the synapse counts in each. The ROIs comprise a hierarchy, with smaller ROIs nested within
larger ROIs. Furthermore, primary ROIs are guaranteed not to overlap, and they roughly tile the
entire brain (with some gaps)
'''

'''For a quick overview of the ROI hierarchy, use fetch_roi_hierarchy()'''
print(fetch_roi_hierarchy(include_subprimary=True, mark_primary=True, format='text')) # Shows ROI hierarchy, with primary ROIs marked with '*'


# ------------------------------------------
# Neuron Search Criteria
# ------------------------------------------
''' Specify neurons of interest by bodyId, type/instance, or via a NeuronCriteria object. With
NeuronCriteria, you can specify multiple search constraints, including the ROIs in which matched
neurons must contain synapses.
'''

# Example: Select several, specific bodies
criteria = [387023620, 387364605, 416642425]
criteria = NC(bodyId=[387023620, 387364605, 416642425])

# Example: Select bodies by exact type
criteria = 'PEN_b(PEN2)'
criteria = NC(type='PENPEN_b(PEN2)')

# Example: Select bodies by exact instance
criteria = 'PEN(PB06)_b_L4'
criteria = NC(type='PEN(PB06)_b_L4')

# Example: Select bodies by type name pattern
criteria = NC(type='PEN.*', regex=True)

# Example: Select bodies by region (input or output)
criteria = NC(rois=['PB', 'EB'])

# Example: Select traced neurons which intersect the PB ROI with at least 100 inputs (PSDs).
criteria = NC(inputRois=['PB'], min_roi_inputs=100, status='Traced', cropped=False)


# ------------------------------------------
# Fetch neuron properties
# ------------------------------------------
''' Neuron properties and per-ROI synapse distributions can be obtained with fetch_neurons(). Two
dataframes are returned: one for neuron properties, and one for the counts of synapses in each ROI.
'''
neuron_df, roi_counts_df = fetch_neurons(criteria) # return properties and per-ROI synapse counts for a set of neurons
print(neuron_df[['bodyId', 'instance', 'type', 'pre', 'post', 'status', 'cropped', 'size']])
print(roi_counts_df.query('bodyId==5813128308')) # synapse counts for ROIs
# print(neuron_df.columns)
# neuron_df, roi_counts_df = fetch_neurons(NC(type='MBON.*', regex=True)) # get mushroom body output neurons

# ------------------------------------------
# Fetch connections
# ------------------------------------------
''' Find synaptic connection strengths between one set of neurons and another using fetch_adjacencies().
The “source” and/or “target” neurons are selected using NeuronCriteria. Additional parameters allow you
to filter by connection strength or ROI. Two DataFrames are returned, for properties of pre/post synaptic
neurons and per-ROI connection strengths (for each pre-post pair)
'''

# Example: Fetch all downstream connections FROM a set of neurons
neuron_df, conn_df = fetch_adjacencies(sources=[387023620, 387364605, 416642425], targets=None)

# Example: Fetch all upstream connections TO a set of neurons
neuron_df, conn_df = fetch_adjacencies(sources=None, targets=[387023620, 387364605, 416642425])

# Example: Fetch all direct connections between a set of upstream neurons and downstream neurons
neuron_df, conn_df = fetch_adjacencies(sources=NC(type='Delta.*', regex=True), targets=NC(type='PEN.*', regex=True))

# Print connections in descending order of strength
print(conn_df.sort_values('weight', ascending=False))


# ------------------------------------------
# Connection matrix
# ------------------------------------------

# Create a connection table with merge_neuron_properties()
'''(adding 'type' and 'instance' properties pre/post neurons from neuron_df to conn_df table)'''
merge_conn_df = merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])

# Convert a connection table into a connectivity matrix via connection_table_to_matrix()
matrix = connection_table_to_matrix(merge_conn_df, 'bodyId', sort_by='type')
print(matrix.iloc[:10, :10])

# Render with hvplot
heatMap = matrix.hvplot.heatmap(height=600, width=700).opts(xrotation=60)
renderer.save(heatMap, 'connectivityGraph.html')

# Render with matplotlib
# plt.imshow(np.array(matrix)); plt.show()


# ------------------------------------------
# Synapses
# ------------------------------------------
''' Fetch synapses for a set of bodies using NeuronCriteria, and optionally apply additional
filtering with SynapseCriteria. Below, this is performed for the ellipsoid body (EB)
'''

# Define neural bodies to fetch synapses for (here 'fan-shaped body; 4Y)
fanshapedNeurons = NC(status='Traced', type='FB4Y', cropped=False, inputRois=['EB'], min_roi_inputs=100, min_pre=400)
''' inputRois: Only neurons which have inputs in EVERY one of the given ROIs will be matched (``regex`` cannot apply)
    min_roi_inputs: How many input (post) synapses a neuron must have in each ROI to satisfy criteria (only if 'inputRois' provided)
    min_pre: Exclude neurons that don't have at least this many t-bars (outputs) overall, regardless of how many t-bars exist in any particular ROI
'''

# Define synapse filter (i.e. which filters by roi, type, confidence)
ellipsoid_synapses = synapse_criteria = SC(rois='EB', primary_only=True)
'''i.e. get only synapses are found in the ellipsoid body (EB)'''

# Get synapses (from neuron)
synapses = fetch_synapses(fanshapedNeurons, ellipsoid_synapses)

# Plot the synapse positions in a 2D projection
fig,ax = plt.subplots(1,1)
fig.set_figwidth(7); fig.set_figheight(8)
ax.scatter(synapses['x'], synapses['z'], s=3)
ax.invert_yaxis(); plt.grid(); plt.box(False)
plt.show()


# ------------------------------------------
# Synapse connections
# ------------------------------------------
''' Fetch all synapse-synapse connections from a set of neurons. Provide a NeuronCriteria for the source
or target neurons (or both) to filter the neurons of interest, and optionally filter the synapses themselves
via SynapseCriteria
'''
ellipsoid_conns = fetch_synapse_connections(source_criteria=fanshapedNeurons, target_criteria=None, synapse_criteria=ellipsoid_synapses)
# ellipsoid_conns.head()

# Retrieve the types of the post-synaptic neurons
post_neurons, _ = fetch_neurons(ellipsoid_conns['bodyId_post'].unique())
eb_conns = merge_neuron_properties(post_neurons, ellipsoid_conns, 'type')
postSynapticNeuronTypes = eb_conns['type_post'].value_counts()


# ------------------------------------------
# Skeletons
# ------------------------------------------

# Get skeletons
skeletons = []; max_number = 50
bodyIds = ellipsoid_conns['bodyId_post'].unique()[0:max_number]
for i, bodyId in enumerate(bodyIds):
    print('Getting skeleton %s/%s...' % (i+1, len(bodyIds)))
    s = c.fetch_skeleton(bodyId, format='pandas')
    skeletons.append(s)

# Plot skeletons
fig = plt.figure(figsize=(8,12),facecolor='black')
ax = fig.gca(projection='3d')
[ax.plot(skel.x, skel.y, skel.z, linewidth=.25) for skel in skeletons]
ax.scatter(ellipsoid_conns.x_pre, ellipsoid_conns.y_pre, ellipsoid_conns.z_pre, s=.01, c='white')
ax.set_facecolor((0,0,0))
plt.axis('off')
plt.show()

# # Combine into one big table for convenient processing
# skeletonsCombined = pd.concat(skeletons, ignore_index=True)
# skeletonsCombined.head()