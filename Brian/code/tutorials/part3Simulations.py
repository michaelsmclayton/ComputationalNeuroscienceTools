
# to install - pip3 install brian2
from brian2 import * # Import Brian
from matplotlib.pyplot import *

# State sections to run
sectionToRun = 'multipleRuns'

# This tutorial is about managing the slightly more complicated tasks that crop up in
# research problems, rather than the toy examples we've been looking at so far. So we
# cover things like inputting sensory data, modelling experimental conditions, etc.

#------------------------------------------
# Multiple runs
#------------------------------------------
if sectionToRun == 'multipleRuns':

    # Let's start by looking at a very common task: doing multiple runs of a
    # simulation with some parameter that changes. Let's start off with something
    # very simple, how does the firing rate of a leaky integrate-and-fire neuron
    # driven by Poisson spiking neurons change depending on its membrane time
    # constant? Below is an implementation of this model.

    # When running this kind of code, it is important to think about where neural
    # models are created. In the commented code below, the script will run slowly
    # because for each loop, you're recreating the objects (e.g. PoissonGroup)
    # from scratch. We can improve this speed by setting up the network just once,
    # before the start of the loop. Importantly, we can also store a copy of the
    # state of the network before the loop, and restore it at the beginning of
    # each iteration.

    # That is a very simple example of using store and restore, but you can use it
    # in much more complicated situations. For example, you might want to run a
    # long training run, and then run multiple test runs afterwards. Simply put a
    # store after the long training run, and a restore before each testing run.

    # Parameters
    num_inputs = 100
    input_rate = 10*Hz
    weight = 0.1

    # Range of time constants
    tau_range = linspace(1, 10, 30)*ms

    # Use this list to store output rates
    output_rates = []

    # Construct the network just once
    P = PoissonGroup(num_inputs, rates=input_rate)
    eqs = '''
    dv/dt = -v/tau : 1
    '''
    G = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
    S = Synapses(P, G, on_pre='v += weight')
    S.connect()
    M = SpikeMonitor(G)

    # Store the current state of the network
    store()

    # Iterate over range of time constants
    for tau in tau_range:

        # # Construct the network each time
        # P = PoissonGroup(num_inputs, rates=input_rate)
        # eqs = '''
        # dv/dt = -v/tau : 1
        # '''
        # G = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
        # S = Synapses(P, G, on_pre='v += weight')
        # S.connect()
        # M = SpikeMonitor(G)

        # Restore the original state of the network
        restore()

        # Run it and store the output firing rate in the list
        run(1*second)
        output_rates.append(M.num_spikes/second)

    # And plot it
    plot(tau_range/ms, output_rates)
    xlabel(r'$\tau$ (ms)')
    ylabel('Firing rate (sp/s)')
    show()