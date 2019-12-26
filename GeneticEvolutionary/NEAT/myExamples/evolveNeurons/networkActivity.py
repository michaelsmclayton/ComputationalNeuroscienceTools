
import os
import numpy as np
import neat
import pprint; op = pprint.PrettyPrinter(depth=6).pprint
import sys; sys.path.append('..')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.spatial import distance

# Set desired firing rate
desiredFiringRate = 4
generations = 10
desiredISI = 1000/desiredFiringRate

# Load configuration
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-evolveNetwork')
config = neat.Config(
    neat.iznn.IZGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path)
config.output_nodes = 2

# Make population
pop = neat.population.Population(config)

# Add a stdout reporter to show progress in the terminal.
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

genome = pop.population[1]
net = neat.iznn.IZNN.create(genome, config)
net.set_inputs([0.0])

# Set neuron constants
for n in range(len(net.neurons)):
    net.neurons[n].v = -65
    net.neurons[n].u = 0.0
    net.neurons[n].current = 0.0
    op(net.neurons[n].__dict__)

# Set neuron weights
for n in range(len(net.neurons)):
    otherNeuron = 1-n
    net.neurons[n].inputs = [(-1,1), (otherNeuron,-10.00)]

neuronData = [[],[]]
for t in range(1400):
    if t > 200:
        net.set_inputs([10.0])
    if t > 1200:
        net.set_inputs([0.0])
    net.advance(.25)
    for i in range(len( net.neurons)):
        neuronData[i].append(net.neurons[i].v)

for i in range(len( net.neurons)): 
    plt.plot(neuronData[i])
plt.show()