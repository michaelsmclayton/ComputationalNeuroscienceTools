from brian2 import *
from matplotlib.pyplot import *
import matplotlib.animation as animation
import numpy as np

# Define simulation parameters
simulationLength = 250*ms
useVariableDelays = True
groupLength = 7
def getCentralLocation(groupLength):
    firstRowLength = (groupLength-1)
    columnsToCenter = groupLength * (int(groupLength/2)-1)
    rowsToCenter = int(groupLength/2)+1
    return firstRowLength + columnsToCenter + rowsToCenter
sourceNeuron = getCentralLocation(groupLength)

# -------------------------------------------
# Helper functions
# -------------------------------------------

# Define Gaussian function
def gaussian(distance, sig):
    return np.exp(-np.power(distance, 2.) / (2 * np.power(sig, 2.)))

# Get Euclidean distance
def euclideanDistance(postSynapticLocation, preSynapticLocation):
    return np.linalg.norm(postSynapticLocation - preSynapticLocation)

# Define function to get weight between two neurons
locations = [[x,y] for x in range(groupLength) for y in range(groupLength)]
@implementation('numpy', discard_units=True)
@check_units(i=1, j=1, result=1)
def getDistance(i,j):
    preSynapticLocation = np.array(locations[int(i)])
    postSynapticLocation = np.array(locations[int(j)])
    distance = euclideanDistance(postSynapticLocation, preSynapticLocation)
    return gaussian(distance, sig=2)


# -------------------------------------------
# Create network
# -------------------------------------------
print('Creating network...')

# Define neurons
N = groupLength**2
tau = 10.0*ms
eqs = '''
    du/dt = (-u + ISyn)/ tau : 1
    dISyn/dt = -ISyn * ms**-1 : 1
'''
G = NeuronGroup(N, eqs, threshold='u>1', reset='u=0', method='exact')
trace = StateMonitor(G, ['u', 'ISyn'], record=True)

# Define synapses
S = Synapses(G, G, on_pre='''ISyn += 3*getDistance(i,j)''', multisynaptic_index='synapse_number', method='euler')
S.connect(condition='i!=j')
# S = Synapses(G, G, on_pre='''ISyn += 2''', method='euler')
# S.connect(condition='i!=j', p='getDistance(i,j)') # Use connection probability instead?

# Define synapse delays
if useVariableDelays==True:
    iterableLocations = [(i_row, i_col, j_row, j_col) for i_row in range(groupLength) \
        for i_col in range(groupLength) for j_row in range(groupLength) \
            for j_col in range(groupLength) if not(i_row==i_col and j_row==j_col)]
    for index, location in enumerate(iterableLocations):
        preSynapticLocation = np.array(location[0:1])
        postSynapticLocation = np.array(location[2:])
        distance = euclideanDistance(postSynapticLocation, preSynapticLocation)
        S.delay[index] = 3*(1/gaussian(distance, sig=2))*ms

# Connect random input
randomInput = PoissonGroup(50, np.arange(50)*Hz + 80*Hz)
Sinput = Synapses(randomInput, G[sourceNeuron], on_pre = '''ISyn += .6''', method='euler')
Sinput.connect()
print('Running simulation...')
run(simulationLength, report='text')


# -------------------------------------------
# Plot results
# -------------------------------------------

# Animation
fig = figure(2, figsize=(4,4))

# Initialise figure
neuronPlots = []
for neuron in range(trace.u.shape[0]):
    # if neuron==sourceNeuron: continue
    currentColor = str(trace.u[neuron][0])
    neuronPlots.append(scatter(locations[neuron][0], locations[neuron][1], color=currentColor, s=500))
axis('off')

# Loop figure
def animate(t):
    for neuron in range(trace.u.shape[0]):
        # if neuron==sourceNeuron: continue
        currentColor = str(trace.u[neuron][t])
        neuronPlots[neuron].set_color(currentColor)
    return neuronPlots
myAnimation = animation.FuncAnimation(fig, animate, frames=trace.u.shape[1], interval=1, blit=True, repeat=False)
show()


