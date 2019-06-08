
# -----------------------------------------------
# Import dependencies
# -----------------------------------------------

# Set up matplotlib and pylab
import matplotlib
matplotlib.use('Agg')
import pylab # interface to Matplotlib

# Import nest (and clear screen output)
import nest
import os; os.system('clear') # Clear terminal screen
import numpy as np

# # List all available functions in Nest
# allAvailableFunctions = dir(nest)
# print(allAvailableFunctions)

# # List all available neural models in Nest
# allAvailableNeuralModels = nest.Models()
# print(allAvailableNeuralModels)

# -----------------------------------------------
# Creating nodes
# -----------------------------------------------

# A neural network in NEST consists of two basic element types: nodes and
# connections. Nodes are either neurons, devices or sub-networks. (Devices
# are used to stimulate neurons, or to record from them). Nodes can be
# arranged in sub-networks to build hierarchical networks such as layers,
# columns, and areas). For now we will work in the default sub-network which
# is present when we start NEST, known as the 'root node'.

# To begin with, the root sub-network is empty. New nodes are created with the
# command 'Create', which takes as input arguments the model name of the desired
# node type. Optionally, this function also takes as inputs the number of nodes
# to be created, and the initialising parameters. As outputs, the function returns
# a list of handles to the new nodes (which you can assign to a variable for later
# use. These handles are integer numbers, called 'ids'.

# Creating a neuron -----------------------------

# As an example below, we will create a a neuron of type "iaf_psc_alpha". This
# neuron is an integrate-and-fire neuron with alpha-shaped postsynaptic currents.
neuron = nest.Create("iaf_psc_alpha") # "iaf_psc_exp"

# We can use the GetStatus function to return information about the neuron we
# just created. We can also include, as a second argument, either a single string,
# or string array, to specific which information we would like to be returned
print(nest.GetStatus(neuron))
print(nest.GetStatus(neuron, "I_e"))
print(nest.GetStatus(neuron, ["V_reset", "V_th"]))

# To modify the properties in the dictionary, we use 'SetStatus'. In the following
# example, the background current is set to 376.0pA, a value causing the neuron to
# spike periodically. Note that we can set several properties at the same time by
# giving multiple comma separated key:value pairs in the dictionary. Also, be aware
# that NEST is type sensitive - if a particular property is of type double, then you
# need to explicitly write the decimal point. 'nest.SetStatus(neuron, {"I_e": 376})',
# for example, will result in an error. 
nest.SetStatus(neuron, {"I_e": 376.0})
print(nest.GetStatus(neuron, "I_e"))


# Creating (detector) devices -----------------------------

# Next we create a multimeter, a device we can use to record the membrane voltage of
# a neuron over time. We set its property 'withtime' such that it will also record the
# points in time at which it samples the membrane voltage. The property record_from
# expects a list of the names of the variables we would like to record. The variables
# exposed to the multimeter vary from model to model. For a specific model, you can
# check the names of the exposed variables by looking at the neuron’s property 'recordables'.

# Create a multimeter
multimeter = nest.Create("multimeter")

# Get list of recordable variables from the current neuron model
print(nest.GetStatus(neuron, "recordables"))

# Set up multimeter
nest.SetStatus(multimeter, {
    "withtime": True, # record the points in time at which membrane voltage is sampled
    "record_from": ["V_m"] # set variables to record
})

# We now create a spikedetector , another device that records the spiking events
# produced by a neuron. We use the optional keyword argument params to set its
# properties. This is an alternative to using SetStatus . The property withgid
# indicates whether the spike detector is to record the source id from which it
# received the event (i.e. the id of our neuron).
spikedetector = nest.Create("spike_detector", \
    params = { # an alternative to using SetStatus
        "withgid": True, # record the source id from which events are recieved
        "withtime": True
    }
)

# -----------------------------------------------
# Connecting nodes with default connections
# -----------------------------------------------

# Now we know how to create individual nodes, we can start connecting them to form
# a small network (see below). Note that the order in which the arguments to 'Connect'
# are specified reflects the flow of events: if the neuron spikes, it sends an event
# to the spike detector. Conversely, the multimeter periodically sends requests to the
# neuron to ask for its membrane potential at that point in time.
nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikedetector)

# Now we have connected the network, we can start the simulation. We have to inform the
# simulation kernel how long the simulation is to run. Here we choose 1000ms. Once complete,
# you will have just simulated your first network in NEST!
# nest.Simulate(1000.0) # Comment to focus on simulation perform later in this script

# -----------------------------------------------
# Extracting and plotting data from devices
# -----------------------------------------------

# After the simulation has finished, we can obtain the data recorded by the multimeter.

# In this first line, we obtain the list of status dictionaries for all queried nodes.
# Here, the variable multimeter is the id of only one node, so the returned list just
# contains one dictionary. We extract the first element of this list by indexing it
# (hence the [0] at the end).
dmm = nest.GetStatus(multimeter)[0]

# This dictionary contains an entry named events which holds the recorded data. It is
# itself a dictionary with the entries 'V_m' and 'times' (which we store below in Vms and ts)
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

# Now we are ready to display the data in a figure. To this end, we make use of pylab. The
# first plot the membrane voltage over time
pylab.figure(1)
pylab.subplot(2, 1, 1)
pylab.plot(ts, Vms)

# The second plot shows spikes over time. This is done by taking the event dictionary from
# the spike detector. (Here we extract the events more concisely by using the optional
# keyword argument keys to GetStatus . This extracts the dictionary element with the key
# events rather than the whole status dictionary). We can then get the "senders" (i.e. spike
# events), and "times", which we then plot below
pylab.subplot(2, 1, 2)
dSD = nest.GetStatus(spikedetector, keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]
pylab.plot(ts, evs, ".")
def saveFigure():
    fname = "currentFigure.png"
    pylab.savefig(fname, dpi=100)
saveFigure()

# -----------------------------------------------
# Connecting nodes with specific connections
# -----------------------------------------------

# A commonly used model of neural activity is the Poisson process. We now adapt
# the previous example so that the neuron receives 2 Poisson spike trains, one
# excitatory and the other inhibitory. Hence, we need a new device, the
# 'poisson_generator'. After creating the neurons, we create these two generators
# and set their rates to 80000Hz and 15000Hz, respectively. Additionally, the
# constant input current ("I_e") should be set to 0.
noise_ex = nest.Create("poisson_generator")
noise_in = nest.Create("poisson_generator")
nest.SetStatus(noise_ex, {"rate": 80000.0})
nest.SetStatus(noise_in, {"rate": 15000.0})
nest.SetStatus(neuron, {"I_e": 1.0})

# Each event of the excitatory generator should produce a postsynaptic current of
# 1.2pA amplitude, an inhibitory event of -2.0pA. The synaptic weights can be defined
# in a dictionary, which is passed to the Connect function using the keyword syn_spec
# (synapse specifications). In general all parameters determining the synapse can be
# specified in the synapse dictionary, such as "weight" , "delay" , the synaptic model
# ( "model" ) and parameters specific to the synaptic model.
syn_dict_ex = {"weight": 1.2} # pA
syn_dict_in = {"weight": -2.0}
nest.Connect(noise_ex, neuron, syn_spec=syn_dict_ex)
nest.Connect(noise_in, neuron, syn_spec=syn_dict_in)
nest.Connect(noise_ex, spikedetector) # Connect noise_ex to spikedector, to track spiking

nest.Simulate(1000.0) # Run simulation

# Plot results
pylab.close(); pylab.figure(1)
pylab.subplot(2, 1, 1)
dmm = nest.GetStatus(multimeter, keys="events")[0]
Vms = dmm["V_m"]
ts = dmm["times"]
pylab.plot(ts, Vms)
pylab.subplot(2, 1, 2); pylab.hold(True)
dSD = nest.GetStatus(spikedetector, keys="events")[0]
excitatorySpikes = dSD["senders"]
excitatorySpikeTimes = dSD["times"]
pylab.plot(excitatorySpikeTimes, excitatorySpikes)
saveFigure()




