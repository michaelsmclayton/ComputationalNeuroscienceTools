
import os
import numpy as np
import neat
import math
import pprint; op = pprint.PrettyPrinter(depth=6).pprint
import sys; sys.path.append('..')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.spatial import distance

# Load configuration
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-evolveNetwork')
config = neat.Config(
    neat.iznn.IZGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path)

# Make population
pop = neat.population.Population(config)

# Add a stdout reporter to show progress in the terminal.
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

# Run single genome
genome = pop.population[1]
net = neat.iznn.IZNN.create(genome, config)
net.set_inputs([0.0])
numberOfNeurons = len(net.neurons)

# Set neuron constants
for n in range(numberOfNeurons):
    net.neurons[n].v = -75
    # net.neurons[n].u = 0.0
    net.neurons[n].current = 0.0
    op(net.neurons[n].__dict__)

# Set all neuron weights to some value
for n in range(numberOfNeurons):
    otherNeuron = 1-n
    for c in range(len(net.neurons[n].inputs)):
        if net.neurons[n].inputs[c][0]==-1:
            net.neurons[n].inputs[c] = (-1,1.0)
        else:
            net.neurons[n].inputs[c] = (net.neurons[n].inputs[c][0], 0.0)

# Run neural net (step current stimulation)
neuronData = [list() for i in range(numberOfNeurons)]
for t in range(1400):
    if t > 200:
        net.set_inputs([40.0]) # Step current
        # net.set_inputs([20+math.sin(t/100)*20]) # Sinusoidal current
    if t > 1200:
        net.set_inputs([0.0])
    net.advance(.25)
    for i in range(numberOfNeurons):
        neuronData[i].append(net.neurons[i].v)

# Plot results
plt.figure()
for i in range(len( net.neurons)): 
    plt.plot(neuronData[i])

# Plot overall LFP trace
plt.figure()
neuronDataNP = np.array(neuronData)
lfp = np.mean(neuronDataNP, axis=0)
cutLFP = lfp[400:1000]
plt.plot(cutLFP); plt.show()

from scipy.signal import periodogram
powerSpectrum = periodogram(cutLFP)
plt.plot(powerSpectrum[0], powerSpectrum[1]); plt.show()