from scipy.stats import vonmises
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy.random import normal, uniform
from scipy.spatial import distance

# Parameters
numberOfNeurons = 4
populationSize = 10
generations = 15
spaceRange = [-2.5,2.5]
pi = np.pi

# -----------------------------------------
# Setup functions
# -----------------------------------------

# Make neurons
def makeNeuron(x, y):
    return {
        'xPos': x, # uniform(spaceRange[0],spaceRange[1]),
        'yPos': y, # uniform(spaceRange[0],spaceRange[1]),
        'nSynapses': 50, # int(uniform(50,100)),
        'angleHomogeneity': uniform(1,60),
        'meanAngle': uniform(0, pi),
        'angleDistScale': 1, # divides the distribution
        'distanceSTD': normal(1.0,.3), 
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
# Fitness and reproduction functions
# -----------------------------------------

# Get connections that touch
def getFitness(population):
    # threshold = .25
    # validConnections = []
    minDistances = []
    for sourceNeuron in range(numberOfNeurons):
        synapses = getConnections(population[sourceNeuron])
        for synapse in synapses:
            synapseXLine = np.linspace(synapse[0][0], synapse[0][1])
            synapseYLine = np.linspace(synapse[1][0], synapse[1][1])
            for targetNeuron in range(numberOfNeurons,numberOfNeurons+1): #range(numberOfNeurons):
                if sourceNeuron==targetNeuron: continue
                targetPosition = [population[targetNeuron]['xPos'], population[targetNeuron]['yPos']]
                distances = []
                for i in range(len(synapseXLine)):
                    currentLinePosition = [synapseXLine[i], synapseYLine[i]]
                    currentDistance = distance.euclidean(np.array(currentLinePosition), np.array(targetPosition))
                    distances.append(currentDistance)
                    # if currentDistance < threshold:
                    #     validConnections.append([sourceNeuron, targetNeuron, synapse])
                minDistances.append(np.min(distances))
    # fitness = len(validConnections)
    fitness = 100 - np.mean(minDistances)
    return fitness #, validConnections

# Make new generation
def makeNewGeneration(populations, fitnessesInOrder):

    # Select parents
    def getParent(fitnessesInOrder):
        randomNumber = uniform(0,1)
        if randomNumber<.5:
            parent = fitnessesInOrder[-1]
        elif randomNumber<.75:
            parent = fitnessesInOrder[-2]
        elif randomNumber<.875:
            parent = fitnessesInOrder[-3]
        else:
            parent = fitnessesInOrder[-3]
        return parent

    # Make new populations
    newPopulations = []
    for p in range(populationSize):
        
        # Get parents
        parentA = []; parentB = []
        while parentA == parentB:
            parentA = getParent(fitnessesInOrder)
            parentB = getParent(fitnessesInOrder)
        parents = [populations[parentA], populations[parentB]]

        # Breed parents
        parents = [populations[fitnessesInOrder[-1]], populations[fitnessesInOrder[-2]]]
        newSpecimen = [{} for i in range(numberOfNeurons)]
        values = list(parents[0][0].keys())
        for i in range(numberOfNeurons):
            for value in values:
                if uniform(0,1)>.5:
                    parent = 0
                else:
                    parent = 1
                newSpecimen[i][value] = parents[parent][i][value]
                # Random mutation
                if uniform(0,1)<.025:
                    newNeuron = makeNeuron(parents[parent][i]['xPos'], parents[parent][i]['yPos'])
                    newSpecimen[i][value] = newNeuron[value]

        newSpecimen.append(makeNeuron(x=(numberOfNeurons-1)/2,y=1))
        newPopulations.append(newSpecimen)
    
    return newPopulations

# -----------------------------------------
# Main script
# -----------------------------------------

# Setup initial population
populations = []
for p in range(populationSize):
    neurons = [makeNeuron(x,y=0) for x in range(numberOfNeurons)]
    neurons.append(makeNeuron(x=(numberOfNeurons-1)/2,y=1))
    populations.append(neurons)

# Loop over generations
for g in range(generations):
    print('---Generation %s' % (g))

    # Get fitnesses
    fitnesses = []
    for p in range(populationSize):
        currentFitness = getFitness(populations[p])
        print('Population %s fitness = %s' % (p, currentFitness))
        fitnesses.append(currentFitness)
    fitnessesInOrder = np.argsort(fitnesses)
    fittestNetwork = populations[fitnessesInOrder[-1]]
    print('Mean fitness = %s' % (np.mean(fitnesses)))

    # Get new populations
    populations = makeNewGeneration(populations, fitnessesInOrder)


# Plot final fittest network
def plotFittestNetwork(fittestNetwork):
    # Plot neurons
    for neuron in fittestNetwork:
        plt.scatter(neuron['xPos'], neuron['yPos'], c='black')
        synapses = getConnections(neuron)
        for synapse in synapses:
            plt.plot([synapse[0][0], synapse[0][1]], [synapse[1][0], synapse[1][1]], alpha=0.2)
    plt.show()

# # Plot final fittest network
# def plotFittestNetwork(fittestNetwork, validConnections):
#     # Plot neurons
#     for neuron in fittestNetwork:
#         plt.scatter(neuron['xPos'], neuron['yPos'], c='black')
#         synapses = getConnections(neuron)
#         for synapse in synapses:
#             plt.plot([synapse[0][0], synapse[0][1]], [synapse[1][0], synapse[1][1]], alpha=0.2)
#     # Plot valid connections
#     for connection in validConnections:
#         synapseInfo = connection[2]
#         plt.plot([synapseInfo[0][0], synapseInfo[0][1]], [synapseInfo[1][0], synapseInfo[1][1]], color='black')
#     plt.show()

plotFittestNetwork(fittestNetwork)



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