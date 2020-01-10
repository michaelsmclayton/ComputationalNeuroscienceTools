from scipy.stats import vonmises
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy.random import normal, uniform

# Parameters
numberOfNeurons = 30
spaceRange = [-2.5,2.5]
pi = np.pi

# -----------------------------------------
# Core functions
# -----------------------------------------

# Make neurons
def makeNeuron():
    return {
        'xPos': uniform(spaceRange[0],spaceRange[1]),
        'yPos': uniform(spaceRange[0],spaceRange[1]),
        'nSynapses': int(uniform(50,100)),
        'angleHomogeneity': uniform(1,60),
        'meanAngle': uniform(0, pi),
        'angleDistScale': 1, # divides the distribution
        'distanceSTD': normal(1.0,.2), 
    }

# Get connections
def getConnections(neuron):

    # Sample from circular distribution
    synapseAngles = vonmises.rvs(
        kappa = neuron['angleHomogeneity'],
        loc = neuron['meanAngle'],
        scale = neuron['angleDistScale'],
        size = neuron['nSynapses'])

    # Get synapse distances (sampled from normal distribution)
    synapseDistances = normal(loc=0.0, scale=neuron['distanceSTD'], size=neuron['nSynapses'])
    synapseDistances = np.abs(synapseDistances) # Take just positive side of distribution

    # Generate synapse lines
    synapses = []
    for s in range(neuron['nSynapses']):
        xEnd = neuron['xPos'] + (np.cos(synapseAngles[s])*synapseDistances[s])
        yEnd = neuron['yPos'] + (np.sin(synapseAngles[s])*synapseDistances[s])
        synapses.append([[neuron['xPos'], xEnd], [neuron['yPos'], yEnd]])

    return synapses


# -----------------------------------------
# Main script
# -----------------------------------------

# Make neurons
neurons = [makeNeuron() for i in range(numberOfNeurons)]

# # Get connections that touch (...work in progress...)
# print('Getting valid connections...')
# threshold = .15
# validConnections = []
# for sourceNeuron in range(numberOfNeurons):
#     synapses = getConnections(neurons[sourceNeuron])
#     for synapse in synapses:
#         synapseXLine = np.linspace(synapse[0][0], synapse[0][1])
#         synapseYLine = np.linspace(synapse[1][0], synapse[1][1])
#         for targetNeuron in range(numberOfNeurons):
#             if sourceNeuron==targetNeuron: continue
#             targetPosition = [neurons[targetNeuron]['xPos'], neurons[targetNeuron]['yPos']]
#             for i in range(len(synapseXLine)):
#                 currentLinePosition = [synapseXLine[0], synapseYLine[0]]
#                 distance = np.linalg.norm(np.array(currentLinePosition) - np.array(targetPosition))
#                 if distance < threshold:
#                     validConnections.append([sourceNeuron, targetNeuron, synapse])

# Plot neurons
for neuron in neurons:
    plt.scatter(neuron['xPos'], neuron['yPos'], c='black')
    synapses = getConnections(neuron)
    for synapse in synapses:
        plt.plot([synapse[0][0], synapse[0][1]], [synapse[1][0], synapse[1][1]], alpha=0.2)
plt.show()
    
# # Plot valid connections
# for connection in validConnections:
#     synapseInfo = connection[2]
#     plt.plot([synapseInfo[0][0], synapseInfo[0][1]], [synapseInfo[1][0], synapseInfo[1][1]])

# Legacy code ----
# # Plot connections    
# iterations = 2; plotIndex = 1
# for i in range(iterations):

#     # Loop over neurons
#     for neuron in neurons:

#         # Get synapse angles (sampled from circular distribution)
#         x = np.linspace(-np.pi,np.pi,50)
#         circularDistribution = vonmises.pdf(x, kappa=neuron['angleHomogeneity'], loc=neuron['meanAngle'], scale=neuron['angleDistScale'])
#         synapseAngles = vonmises.rvs(
#             kappa = neuron['angleHomogeneity'],
#             loc = neuron['meanAngle'],
#             scale = neuron['angleDistScale'],
#             size = neuron['nSynapses'])

#         # Get synapse distances (sampled from normal distribution)
#         synapseDistances = normal(loc=0.0, scale=neuron['distanceSTD'], size=neuron['nSynapses'])
#         synapseDistances = np.abs(synapseDistances) # Take just positive side of distribution

#         # Plot results
#         # fig = plt.figure()
#         plt.subplot(iterations,numberOfNeurons,plotIndex)
#         # ax = plt.axes(projection='3d')
#         # ax.plot3D(np.cos(x), np.sin(x), circularDistribution, 'gray')
#         for s in range(neuron['nSynapses']):
#             plt.plot([0, np.cos(synapseAngles[s])*synapseDistances[s]], [0,np.sin(synapseAngles[s])*synapseDistances[s]])
#         plt.ylim([-3, 3])
#         plt.xlim([-3,3])
#         plotIndex += 1
# plt.show()