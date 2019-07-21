from brian2 import *
from matplotlib.pyplot import *
import matplotlib.animation as animation
import numpy as np

# Define simulation parameters
simulationLength = 500*ms
neuronType = 'Kuramoto' # 'LIF'
useVariableDelays = True
groupLength = 13
N = groupLength**2
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

# Define LIF neurons and synapses
if neuronType == 'LIF': 
    tau = 10*ms
    eqs = '''
        du/dt = (-u + ISyn)/tau : 1
        dISyn/dt = -ISyn * ms**-1 + (xi*tau**-0.5) : 1
    '''
    G = NeuronGroup(N, eqs, threshold='u>1', reset='u=0', method='euler')
    trace = StateMonitor(G, ['u', 'ISyn'], record=True)
    S = Synapses(G, G, on_pre='''ISyn += 4''', method='euler')

# Define Kuramoto neurons and synapses
if neuronType == 'Kuramoto':
    eqs = '''
        dTheta/dt = ((freq + (kN * PIF)) * ms**-1) : 1 # + (sigma*xi*2*ms**-0.5) : 1
        PIF = .5 * (sin(ThetaPreInput - Theta)) : 1
        ThetaPreInput : 1
        freq : 1
        kN : 1
    '''
    G = NeuronGroup(N, eqs, threshold='True', method='euler')
    G.Theta = '1-(randn()*2)'
    G.freq = '2*(i/N)'
    trace = StateMonitor(G, ['Theta'], record=True)
    S = Synapses(G, G, on_pre = '''ThetaPreInput_post = Theta_pre''', method='euler')

# Connect synapses with distance-dependent connectivity
S.connect(condition='i!=j', p='getDistance(i,j)')

# Define synapse delays
if useVariableDelays==True:
    delayGaussianSigma = 3
    for syn in range(len(S)):
        currentI, currentJ = S.i[syn], S.j[syn]
        S.delay[syn] = 1/getDistance(currentI, currentJ, delayGaussianSigma) * ms

# # Connect random input
# randomInput = PoissonGroup(50, np.arange(50)*Hz + 25*Hz)
# Sinput = Synapses(randomInput, G[sourceNeuron], on_pre = '''ISyn += .6''', method='euler')
# Sinput.connect()

# Run simulation
print('Running simulation...')
duration = 10*ms
G.kN = 3
run(duration, report='text')
G.kN = 0
run(duration*6, report='text')
G.kN = 3
run(duration, report='text')
G.kN = 0
run(duration*6, report='text')


# -------------------------------------------
# Plot results
# -------------------------------------------

# Animation
fig = figure(2, figsize=(4,4))
fig.set_facecolor((.8,.8,.8))
gca().set_facecolor((.8,.8,.8))

# Get trace of interest
if neuronType == 'LIF': 
    traceOfInterest = trace.u
elif neuronType == 'Kuramoto': 
    traceOfInterest = (cos(trace.Theta)+1)/2

# Initialise figure
neuronPlots = []
for neuron in range(traceOfInterest.shape[0]):
    # if neuron==sourceNeuron: continue
    currentColor = str(traceOfInterest[neuron][0])
    neuronPlots.append(scatter(locations[neuron][0], locations[neuron][1], color=currentColor, s=500))
axis('off')

# Loop figure
def animate(t):
    for neuron in range(traceOfInterest.shape[0]):
        # if neuron==sourceNeuron: continue
        currentColor = str(np.clip(traceOfInterest[neuron][t], a_min=0, a_max=1))
        neuronPlots[neuron].set_color(currentColor)
    return neuronPlots
myAnimation = animation.FuncAnimation(fig, animate, frames=traceOfInterest.shape[1], interval=1, blit=True, repeat=False)
show()


