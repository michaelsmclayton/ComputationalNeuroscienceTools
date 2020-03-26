from brian2 import *
import networkx as nx
import matplotlib.pylab as plt
import numpy as np
# 'An integrative model of the intrinsic hippocampal theta rhythm'
# https://izhikevich.org/publications/hybrid_spiking_models.pdf


# ------------------------------------------------------
# Neural dynamics
# ------------------------------------------------------


# Key parameters
tau = 1 *ms
neuronTypes = ['CA3Pyra', 'GC', 'OLM', 'BC']
'''Due to the striking similarity of OLM and HIPP cells, we used the same model for both cell type'''

# Get parameters
def getParameters(neuronType):
    if neuronType=='CA3Pyra':
        C = 24; k = 1.5; a = .010; b = 2; c = -63; d = 60; vThresh = -75; vRest = -58; vPeak = 29
        simulationLength = 500 * ms
    elif neuronType=='GC':
        C = 24; k = 1.0; a = .015; b = 3; c = -62; d = 3; vThresh = -73; vRest = -53; vPeak = 32
        simulationLength = 250 * ms
    elif neuronType=='OLM':
        C = 80; k = 1.5; a = .003; b = 10; c = -65; d = 30; vThresh = -68; vRest = -53; vPeak = 30 
        simulationLength = 500 * ms     
    elif neuronType=='BC':
        C = 16; k = 1.5; a = .900; b = 2; c = -80; d = 400; vThresh = -65; vRest = -50; vPeak = 28
        simulationLength = 125 * ms
    return k, C, a, b, c, d, vRest, vThresh, vPeak, simulationLength


# Loop over neuron types
plt.figure(figsize=(8,6))
for i, neuronType in enumerate(neuronTypes): 

    # Get parameters
    print('Analysing %s...' % (neuronType))
    k, C, a, b, c, d, vRest, vThresh, vPeak, simulationLength = getParameters(neuronType)

    # Make neuron
    eqs = """
        dv/dt = ((k*(v-vRest)*(v-vThresh) -u + I)/C) / tau: 1
        du/dt = a * (b * (v-vRest) - u) / tau: 1
        I : 1
    """

    pop = NeuronGroup(1, eqs, threshold='v > vPeak', reset='v=c; u+=d', method="euler")
    trace = StateMonitor(pop, ['v','I'], record=True)
    pop.v = vRest

    # Run simulation
    run(.2*simulationLength)
    pop.I = 180
    run(simulationLength)
    pop.I = 0
    run(.2*simulationLength)

    # Run simulation
    plt.subplot(len(neuronTypes),1,i+1)
    plt.title(neuronType)
    plt.plot(trace.v[0])
    plt.axis('off')

plt.show()


# ------------------------------------------------------
# Network structure
# ------------------------------------------------------

# Notes
'''The model DG region had 384 granule cells, 32 BCs, and 32 HIPP interneurons,
while the model CA3 region contained 63 pyramidal cells, 8 BCs, and 8 OLM cells
The model EC region had 30 regular spiking cells.
The entorhinal cortex provides inputs to the hippocampus through the perforant pathway
that projects to the entire hippocampal formation
'''

# Initiate graph
G = nx.MultiDiGraph() # directed graph

# Function to create edges
def addEdges(connectionNames, params):
    def addEdge(fromName,toName, params):
        G.add_edge(fromName, toName, connectivity=params['connectivity'], synapticDelay=params['synapticDelay'], weight=params['weight'])
    for connection in connectionNames:
        addEdge(fromName=connection[0], toName=connection[1], params=params)

# Function to create dictionary of all parameters
def createParamsDict(*kargs):
    return {
        'connectivity': kargs[0], 'synapticDelay': kargs[1],
        'AMPA_GABAa_rise_decay_time constants': kargs[2],
        'NMDA_GABAb_rise_decay_time constants': kargs[3], 'weight': kargs[4],
        'synapticPlasticity': kargs[5], 'bACh': kargs[6]
    }

# Parameters
metaParameters = {
    'ECInput': {
        'connectionNames' : [('EC','BC'), ('EC','Pyr1'), ('EC','Pyr2')],
        'params': createParamsDict('diffuse', 5, (1.7,10.9), (25,300), 2, 'None', -0.5)},
    'Recurrent': {
        'connectionNames' : [('Pyr1','Pyr2'), ('Pyr2','Pyr1')],
        'params': createParamsDict('homogenous', 2, (1.1,5), (25,300), 0.4, 'Depressing', -0.85)},
    'Pyr_OLM': {
        'connectionNames': [('Pyr1','OLM'), ('Pyr2','OLM')],
        'params': createParamsDict('reciprocal', 0.9, (0.27,0.57), (25,150), 3, 'Facilitating', None)},
    'Pyr_BC': {
        'connectionNames': [('Pyr1','BC'), ('Pyr2','BC')],
        'params': createParamsDict('homogenous', 0.9, (0.27,0.57), (25,150), 3, 'Depressing', None)},
    'OLM_Pyr': {
        'connectionNames': [('OLM','Pyr1'), ('OLM','Pyr2')],
        'params': createParamsDict('dense', 0.8, (2.8,20.8), (11.1,125), 3, 'None', None)},
    'BC_pyr': {
        'connectionNames': [('BC','Pyr1'), ('BC','Pyr2')],
        'params': createParamsDict('diffuse', 0.8, (0.21,3.3), (11.1,125), 3, 'Depressing', -0.5)},
}

# Add input node
G.add_edge('INPUT', 'Pyr1')

# Create connections
for connectionType in list(metaParameters):
    connectionNames = metaParameters[connectionType]['connectionNames']
    parameters = metaParameters[connectionType]['params']
    addEdges(connectionNames, parameters)
    
# Get summary node/edge information
nx.get_edge_attributes(G,'connectivity')
# nx.get_node_attributes(G, ...)

# State node positions
# options = {'node_color': 'black', 'node_size': 100, 'width': 3}
positions = {'EC': (.6,1), 'BC': (.6,.6), 'Pyr1': (.2,.3),  'Pyr2': (1,.3),  'OLM': (.6,0), 'INPUT': (0,.6)}
nx.draw_networkx(G, positions, arrows=True)#, **options)
plt.show()

# nx.shortest_path(G, 'EC', 'D', weight='weight')
