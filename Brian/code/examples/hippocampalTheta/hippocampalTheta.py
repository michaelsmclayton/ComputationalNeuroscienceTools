from brian2 import *
from matplotlib.pyplot import *
import numpy as np
# 'An integrative model of the intrinsic hippocampal theta rhythm'
# https://izhikevich.org/publications/hybrid_spiking_models.pdf

# Key parameters
tau = 1 *ms
neuronTypes = ['CA3Pyra', 'GC', 'OLM', 'BC']

# Get parameters
def getParameters(neuronType):
    if neuronType=='CA3Pyra':
        C = 24; k = 1.5; a = .010; b = 2; c = -63; d = 60; vThresh = -75; vRest = -58; vPeak = 29
        simulationLength = 400 * ms
    elif neuronType=='GC':
        C = 24; k = 1.0; a = .015; b = 3; c = -62; d = 3; vThresh = -73; vRest = -53; vPeak = 32
        simulationLength = 200 * ms
    elif neuronType=='OLM':
        C = 80; k = 1.5; a = .003; b = 10; c = -65; d = 30; vThresh = -68; vRest = -53; vPeak = 30 
        simulationLength = 400 * ms     
    elif neuronType=='BC':
        C = 16; k = 1.5; a = .900; b = 2; c = -80; d = 400; vThresh = -65; vRest = -50; vPeak = 28
        simulationLength = 200 * ms
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