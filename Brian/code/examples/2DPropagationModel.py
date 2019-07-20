from brian2 import *
from matplotlib.pyplot import *
import matplotlib.animation as animation
import numpy as np

# Define simulation parameters
simulationLength = 100*ms
useVariableDelays = True
groupLength = 13
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
@check_units(i=1, j=1, sig=1, result=1)
def getDistance(i, j, sig=3):
    preSynapticLocation = np.array(locations[int(i)])
    postSynapticLocation = np.array(locations[int(j)])
    distance = euclideanDistance(postSynapticLocation, preSynapticLocation)
    return gaussian(distance, sig)


# -------------------------------------------
# Create network
# -------------------------------------------
print('Creating network...')

# Define neurons
N = groupLength**2
tau = 10*ms
eqs = '''
    du/dt = (-u + ISyn)/tau + (.1*xi*tau**-0.5) : 1
    dISyn/dt = -ISyn * ms**-1 : 1
'''
G = NeuronGroup(N, eqs, threshold='u>1', reset='u=0', method='euler')
trace = StateMonitor(G, ['u', 'ISyn'], record=True)

# Define synapses
S = Synapses(G, G, on_pre='''ISyn += 3''', method='euler')
S.connect(condition='i!=j', p='getDistance(i,j)') # Use connection probability instead?
# S = Synapses(G, G, on_pre='''ISyn += 3*getDistance(i,j)''', multisynaptic_index='synapse_number', method='euler')
# S.connect(condition='i!=j')

# Define synapse delays
if useVariableDelays==True:
    delayGaussianSigma = 3
    for syn in range(len(S)):
        currentI, currentJ = S.i[syn], S.j[syn]
        S.delay[syn] = 1/getDistance(currentI, currentJ, delayGaussianSigma) * ms

# Connect random input
randomInput = PoissonGroup(50, np.arange(50)*Hz + 25*Hz)
Sinput = Synapses(randomInput, G[sourceNeuron], on_pre = '''ISyn += .6''', method='euler')
Sinput.connect()
print('Running simulation...')
run(simulationLength, report='text')


# -------------------------------------------
# Plot results
# -------------------------------------------

# Animation
fig = figure(2, figsize=(4,4))
fig.set_facecolor((.8,.8,.8))
gca().set_facecolor((.8,.8,.8))

# Initialise figure
neuronPlots = []
for neuron in range(trace.u.shape[0]):
    # if neuron==sourceNeuron: continue
    currentColor = str(trace.u[neuron][0])
    neuronPlots.append(scatter(locations[neuron][0], locations[neuron][1], color=currentColor, s=400))
axis('off')

# Loop figure
def animate(t):
    for neuron in range(trace.u.shape[0]):
        # if neuron==sourceNeuron: continue
        currentColor = str(np.clip(trace.u[neuron][t], a_min=0, a_max=1))
        neuronPlots[neuron].set_color(currentColor)
    return neuronPlots
myAnimation = animation.FuncAnimation(fig, animate, frames=trace.u.shape[1], interval=1, blit=True, repeat=False)
show()


