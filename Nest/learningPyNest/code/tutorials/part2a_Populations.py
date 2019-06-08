#------------------------------------------
# Import dependencies
#------------------------------------------
import matplotlib
matplotlib.use('Agg')
import pylab # interface to Matplotlib
import numpy as np
import nest
import os; os.system('clear') # Clear terminal screen

#------------------------------------------
# Creating homogeneously parameterised populations of nodes
#------------------------------------------

# The most basic way of creating a batch of identically parameterised neurons
# is to exploit the optional arguments of Create(). Parameterising the neurons
# at creation is more efficient than using SetStatus() after creation, so try
# to do this wherever possible.
numberOfNeurons = 6
ndict = {"I_e": 200.0, "tau_m": 20.0} # Set parameters
neuronpop = nest.Create("iaf_psc_alpha", numberOfNeurons, params=ndict)
# print(neuronpop) # Returns a tuple of all the ids of the created neurons.

# We can also set the parameters of a neuron model before creation, which allows
# us to define a simulation more concisely in many cases. If many individual
# batches of neurons are to be produced, it is more convenient to set the
# defaults of the model, so that all neurons created from that model will
# automatically have the same parameters. The defaults of a model can be
# queried with GetDefaults(model), and set with SetDefaults(model, params),
# where params is a dictionary containing the desired parameter/value pairings.
ndict = {"I_e": 200.0, "tau_m": 20.0}
nest.SetDefaults("iaf_psc_alpha", ndict) # Set default values for chosen model type
neuronpop1 = nest.Create("iaf_psc_alpha", 100) # Create population with new parameters
neuronpop2 = nest.Create("iaf_psc_alpha", 100)
neuronpop3 = nest.Create("iaf_psc_alpha", 100)

# You can also save a created model using the CopyModel function. For example,
# if we want to make a new model ("inh_iaf_psc_alpha") from "iaf_psc_alpha", with
# new parameters (idict), we can use the following code. The newly defined models
# can now be used to generate neuron populations and will also be returned by the
# function Models().
idict = {"I_e": 300.0}
nest.CopyModel("iaf_psc_alpha", "inh_iaf_psc_alpha", params=idict)


#------------------------------------------
# Creating heterogeneously parameterised populations of nodes
#------------------------------------------

# We will often want populations will heterogeneous parameters. One way of doing
# this is to supply, as 'params', a list of dictionaries of the same length as the
# number of neurons (or synapses) created. See below for as example:
parameter_list = [{"I_e": 200.0, "tau_m": 20.0}, {"I_e": 150.0, "tau_m": 30.0}]
epop3 = nest.Create("inh_iaf_psc_alpha", 2, parameter_list)

# The other way is to generate a random list of values, and then set these values
# using the SetStatus function:
Vth, Vrest =-55, -70.
# dVms =  [{"V_m": Vrest+(Vth-Vrest)\*numpy.random.rand()} for x in epop1]
Vms = Vrest+(Vth-Vrest)*np.random.rand(len(neuronpop)) # Create a random array of values (between -70 and -55)
I_e = 200+(100*np.random.rand(len(neuronpop)))
nest.SetStatus(neuronpop, "I_e", I_e)
nest.SetStatus(neuronpop, "V_m", Vms)

# Simulate and plot results
multimeter = nest.Create("multimeter", params={"withtime": True, "record_from": ["V_m"]})
nest.Connect(multimeter, neuronpop)
nest.Simulate(1000.0)
dmm = nest.GetStatus(multimeter)[0]["events"]
Vms = dmm["V_m"]
ts = dmm["times"]
senders = dmm["senders"]
pylab.figure(1)
pylab.hold(True)
for i in range(numberOfNeurons):
    indicesOfInterest = np.where(senders==i)
    pylab.plot(ts[indicesOfInterest], Vms[indicesOfInterest])
pylab.savefig("currentFigure.png", dpi=100)

#------------------------------------------
# Generating populations of neurons with deterministic connections
#------------------------------------------

