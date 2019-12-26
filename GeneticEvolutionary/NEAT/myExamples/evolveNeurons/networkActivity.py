
import os
import numpy as np
import neat
import math
import pprint; op = pprint.PrettyPrinter(depth=6).pprint
import sys; sys.path.append('..')
import visualize
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.spatial import distance
from scipy.signal import periodogram, savgol_filter

# Evolution parameters
generations = 10

# Stimulation parameters
stimulationStart = 200
stimulationLength = 2000
postStimulationLength = 200

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

# Define fucntions to assess genome fitness

def eval_genome(genome, config):
    fitness, powerSpectrum, cutLFP = assessGenomeFitness(genome, config)
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def assessGenomeFitness(genome, config):

    # Run single genome
    net = neat.iznn.IZNN.create(genome, config)
    net.set_inputs([0.0])

    # # Set neuron constants
    # for n in range(len(net.neurons)):
    #     net.neurons[n].v = -75
    #     net.neurons[n].current = 0.0
    #     # op(net.neurons[n].__dict__)

    # Get neuron indices
    neuronIndices = [i[0] for i in net.neurons.items()]

    # Set all neuron weights to some value
    for n in neuronIndices:
        for c in range(len(net.neurons[n].inputs)):
            if net.neurons[n].inputs[c][0]==-1:
                net.neurons[n].inputs[c] = (-1,1.0)
            # else:
            #     net.neurons[n].inputs[c] = (net.neurons[n].inputs[c][0], 0.0)

    # Run neural net (step current stimulation)
    neuronData = [list() for i in range(len(net.neurons))]
    stimulationTime = stimulationStart + stimulationLength + postStimulationLength
    for t in range(stimulationTime):
        if t > stimulationStart:
            net.set_inputs([40.0]) # Step current
            # net.set_inputs([20+math.sin(t/100)*20]) # Sinusoidal current
        if t > (stimulationLength+stimulationStart):
            net.set_inputs([0.0])
        net.advance(.25)
        for i in neuronIndices:
            neuronData[i].append(net.neurons[i].v)

    # Get stable LFP
    neuronDataNP = np.array(neuronData)
    lfp = np.mean(neuronDataNP, axis=0)
    cutLFP = lfp[stimulationStart+200:(stimulationStart+stimulationLength)]

    # Get alpha power ratio
    powerSpectrum = periodogram(cutLFP, fs=1000)
    minValueIndex = np.min(np.where(powerSpectrum[0]>7))
    maxValueIndex = np.max(np.where(powerSpectrum[0]<13))
    lowPower = np.mean(powerSpectrum[1][0:minValueIndex-1])
    alphaPower = np.mean(powerSpectrum[1][minValueIndex:maxValueIndex])
    highPower = np.mean(powerSpectrum[1][maxValueIndex+1:-1])
    fitness = (alphaPower/lowPower) * (alphaPower/highPower)
    return fitness, powerSpectrum, cutLFP

# Run evolution
winner = pop.run(eval_genomes, n=generations)
# visualize.draw_net(config, winner, filename='hello', view=True)

# Get results of winner
fitness, powerSpectrum, cutLFP = assessGenomeFitness(winner, config)

# Plot smoothed power spectrum of winner
smoothed = savgol_filter(powerSpectrum[1], window_length=11, polyorder=3)
plt.figure()
plt.plot(powerSpectrum[0], smoothed)
plt.xlim([0,100])
plt.figure()
plt.plot(cutLFP)
plt.show()