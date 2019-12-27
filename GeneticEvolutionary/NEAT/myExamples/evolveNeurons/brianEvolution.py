import numpy as np
from brian2 import * # Import Brian
import pprint; op = pprint.PrettyPrinter(depth=6).pprint

# General parameters
intendedSpikeRate = 6
numberOfGenerations = 20
numberOfGenomes = 5
mutationRate = .1

# ---------------------------------------
# Define functions for making genomes and neuron
# ---------------------------------------

# Function to drawn from uniform distribution
def uniformSample(low, high):
    return low + ((high-low)*np.random.sample())

# Function to define a new neuron
def neuron():
    return {
        'a': uniformSample(low=.02, high=.1),
        'b': uniformSample(low=.2, high=.25),
        'c': uniformSample(low=-65, high=-50),
        'd': uniformSample(low=2, high=8),
    }

# Define function for creating neuron
def createIzhikevitchNeurons(N):
    eqs = '''
        tau = .5 * ms : second
        dv/dt = ( .04*v**2 + 5*v + 140 - u + I ) / tau : 1
        du/dt = ( a * (b*v - u) ) / tau : 1
        a : 1
        b : 1
        c : 1
        d : 1
        I : 1
    '''
    return NeuronGroup(N, model=eqs, threshold="v>=30", reset="v=c; u+=d", method='euler')

# Define function to create neural network from genome
def createNetworkFromGenome(genome):
    # Create neurons
    N = len(genome['neurons'])
    pop = createIzhikevitchNeurons(N)
    for n in range(N):
        pop.a[n] = genome['neurons'][n]['a']
        pop.b[n] = genome['neurons'][n]['b']
        pop.c[n] = genome['neurons'][n]['c']
        pop.d[n] = genome['neurons'][n]['d']
        pop.v[n] = -82.16
    return pop

# Define function to make a genome
def createGenome(nOfNeurons):
    return {
        'neurons': [ neuron() for i in range(nOfNeurons)],
        'fitness': None
    }

# Calculate cost
def getCost(spikemon, intendedSpikes=intendedSpikeRate):
    return 100 - (np.abs(spikemon.num_spikes-intendedSpikes))

# Evaluate genome
def evaluateGenome(genome, plotResults=False):

    # Make neural network
    pop = createNetworkFromGenome(genome)
    trace = StateMonitor(pop, ['v','u'], record=True)
    spikemon = SpikeMonitor(pop)

    # Run network
    pop.I = 0.0
    run(100*ms)
    pop.I = 10.0
    run(100*ms)
    pop.I = 0.0
    run(100*ms)

    # Plot results
    if plotResults==True:
        for i in range(len(trace.v)):
            plt.plot(trace.v[i])
        plt.show()

    # Return cost function
    return getCost(spikemon, intendedSpikeRate)


# --------------------------------
# Reproduction!
# --------------------------------

# Evaluate genome fitnesses
def getFitnesses(genomes):
    fitnesses = []
    highestFitness = 0; winnerIndex = None
    print('Evaluating genomes...')
    for g in range(numberOfGenomes):
        fitness = evaluateGenome(genomes[g])
        fitnesses.append(fitness)
        genomes[g]['fitness'] = fitness
        if fitness > highestFitness:
            highestFitness = fitness
            winnerIndex = g
    return fitnesses, winnerIndex

# Produce new set of genomes
def getNewGenomes(fitnesses):

    # Get mating pool
    matingPoolSize = 4
    matingPool = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[-matingPoolSize:]

    # Create new genomes
    newGenomes = []
    for g in range(numberOfGenomes):
        newGenomes.append(createGenome(nOfNeurons=1))

    # Set value
    def getRandomNewValue(parameter, matingPool):
        # parent = np.random.randint(low=0, high=len(matingPool))
        randomNumber = np.random.rand()
        if randomNumber < .54:
            parent = 0
        elif randomNumber < .80:
            parent = 1
        elif randomNumber < .93:
            parent = 2
        else:
            parent = 3
        return genomes[parent]['neurons'][0][parameter]

    for g in range(numberOfGenomes):
        for parameter in ['a','b','c','d']:
            newNeuron = neuron()
            if np.random.rand()>mutationRate:
                newGenomes[g]['neurons'][0][parameter] = getRandomNewValue(parameter, matingPool)
            else:
                newGenomes[g]['neurons'][0][parameter] = newNeuron[parameter]

    return newGenomes

# ---------------------------------------
# Run generations
# ---------------------------------------

# Create initial genomes
genomes = []
for g in range(numberOfGenomes):
    genomes.append(createGenome(nOfNeurons=1))

# Loop over generations
bestGenome = None; bestFitness = 0
for generation in range(numberOfGenerations):
    print('Generation %s' % (generation))

    # Get fitness
    fitnesses, winnerIndex = getFitnesses(genomes)
    winnerFitness = fitnesses[winnerIndex]
    if winnerFitness > bestFitness:
        bestFitness = winnerFitness
        bestGenome = genomes[winnerIndex]
    # print('Average fitness = %s' % (np.mean(fitnesses)))
    print('Best fitness: %s' % (bestFitness))
    print(' ')

    if bestFitness == 100:
        break

    # Get new genomes
    genomes = getNewGenomes(fitnesses)


# Find genome with highest fitness
# winner = genomes[winnerIndex]
evaluateGenome(bestGenome, plotResults=True)