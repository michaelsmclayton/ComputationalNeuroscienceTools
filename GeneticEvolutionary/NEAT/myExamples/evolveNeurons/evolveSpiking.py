
import os
import numpy as np
import neat
import pprint; op = pprint.PrettyPrinter(depth=6).pprint
import sys; sys.path.append('..')
import matplotlib.pyplot as plt

# Set desired firing rate
desiredFiringRate = 4
desiredISI = 1000/desiredFiringRate

# Load configuration
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-evolveSpiking')
config = neat.Config(
    neat.iznn.IZGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path)
config.output_nodes = 1

# Make population
pop = neat.population.Population(config)

# Add a stdout reporter to show progress in the terminal.
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

# ----------------------------------------
# Define functions to assess genome
# ----------------------------------------
# def getAverageTimeBetweenSpikes(spikes): # Attempt to reward consistent ISI times
#     spikes = np.array(spikes)
#     spikeTimes = np.where(spikes==1.0)
#     isis = []
#     for i in range(len(spikeTimes)-1):
#         spikeTimes.append(spikeTimes[i+1]-spikeTimes[i])
#     return np.mean(spikeTimes)

def eval_genome(genome, config):
    spikes, vValues = assessGenomeFitness(genome, config)
    return 100 - (np.abs(np.sum(spikes)-desiredFiringRate)) # - (getAverageTimeBetweenSpikes(spikes)-desiredISI)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def assessGenomeFitness(genome, config):
    # Create and set neural network
    net = neat.iznn.IZNN.create(genome, config)
    net.reset()
    net.set_inputs([1.0])
    net.neurons[0].inputs = [(-1, 0), (0, 0)]
    net.neurons[0].current = 0.0

    # Run simulation
    dt = .25
    vValues = []; firedValues = []
    for i in range(1200):
        if (i > 200 and i < 1000):
            net.neurons[0].current = 15.0
        else:
            net.neurons[0].current = 0.0
        vValues.append(net.neurons[0].v)
        firedValues.append(net.neurons[0].fired)
        net.neurons[0].advance(dt)
    
    # Return firing rate
    return firedValues, vValues

winner = pop.run(eval_genomes, n=10)
spikes, vValues = assessGenomeFitness(winner, config)
print('Desired firing rate: %s, Actual firing rate: %s' % (desiredFiringRate, np.sum(spikes)))
print('a: %s, b: %s, c: %s, d: %s, ' % (winner.nodes[0].a, winner.nodes[0].b, winner.nodes[0].c, winner.nodes[0].d))
plt.plot(vValues); plt.show()


import visualize
# visualize.draw_net(config, genome, filename='hello', view=True)