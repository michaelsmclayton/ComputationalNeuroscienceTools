#------------------------------------------
# Import dependencies
#------------------------------------------
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy as np
import nest
import os; os.system('clear') # Clear terminal screen

#------------------------------------------
# Generating connected neurons
#------------------------------------------

# In the example below, we create two neurons: one recieving a constant input, and
# the other recieving synaptic input from the first. We make the connection between
# neuron1 and neuron2 using the Connect function. This means that, when neuron1 spikes,
# it will be pass post-synaptic potentials on to neuron2. The strength of this synaptic
# effect is determined by the "weight" parameters in "syn_spec". A delay of 1 ms is also
# set in the example below.

# Create objects 
neuron1 = nest.Create("iaf_psc_alpha", params={"I_e": 376.0})
neuron2 = nest.Create("iaf_psc_alpha")
multimeter = nest.Create("multimeter", params={"withtime": True, "record_from":["V_m"]})

# Make connections
nest.Connect(neuron1, neuron2, syn_spec={"weight": 20, "delay": 1.0})
nest.Connect(multimeter, neuron1)
nest.Connect(multimeter, neuron2)

# # Simulate and plot results
# nest.Simulate(1000.0)
# dmm = nest.GetStatus(multimeter)[0]["events"]
# Vms = dmm["V_m"]
# ts = dmm["times"]
# senders = dmm["senders"]
# pylab.figure(1)
# for i in range(1,3):
#     pylab.subplot(2,1,i)
#     indicesOfInterest = np.where(senders==i)
#     pylab.plot(ts[indicesOfInterest], Vms[indicesOfInterest])
# pylab.savefig("currentFigure.png", dpi=100)


#------------------------------------------
# Generating populations of neurons with deterministic connections
#------------------------------------------

# Create objects
populationSize = 10
pop1 = nest.Create("iaf_psc_alpha", populationSize, params={"I_e": 376.0})
pop2 = nest.Create("iaf_psc_alpha", populationSize)
multimeter = nest.Create("multimeter", params={"withtime":True, "record_from":["V_m"]})

# Randomise starting voltages
def setRandomValues(population, randParams):
    minVal = randParams['low']
    maxVal = randParams['high']
    randValues = minVal + ((maxVal-minVal)*np.random.rand(len(population)))
    nest.SetStatus(population, randParams['label'], randValues)
setRandomValues(pop1, randParams={'label': 'I_e', 'low': 350, 'high': 500})

# If no connectivity pattern is specified, the populations are connected via the default rule, namely
# 'all_to_all'. Therefore, in the current case, each neuron of pop1 is connected to every neuron in
# pop2, resulting in populationSize^2 connections. Alternatively, the neurons can be connected with
# the 'one_to_one'. This means that the first neuron in pop1 is connected to the first neuron in pop2,
# the second to the second, etc., creating 10 connections in total.

# Make connections
nest.Connect(pop1, pop2, "one_to_one", syn_spec={"weight":20.0, "delay":1.0})
# nest.Connect(pop1, pop2, "all_to_all", syn_spec={"weight":20.0, "delay":1.0})
nest.Connect(multimeter, pop2)

# # Simulate and plot results
# nest.Simulate(1000.0)
# dmm = nest.GetStatus(multimeter)[0]["events"]
# Vms = dmm["V_m"]
# ts = dmm["times"]
# senders = dmm["senders"]
# pylab.figure(1)
# neuronsToPlot = np.unique(senders)
# pylab.hold(True)
# for i in range(len(neuronsToPlot)):
#     indicesOfInterest = np.where(senders==neuronsToPlot[i])
#     pylab.plot(ts[indicesOfInterest], Vms[indicesOfInterest])
# pylab.savefig("currentFigure.png", dpi=100)


#------------------------------------------
# Connecting populations with random connections
#------------------------------------------

# Often we will want to look at networks with a sparser connectivity than 'all-to-all'.
# 