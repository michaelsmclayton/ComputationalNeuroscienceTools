
# to install - pip3 install brian2
from brian2 import * # Import Brian
from matplotlib.pyplot import *

# State sections to run
sectionToRun = 'parameters'

#------------------------------------------
# Units
#------------------------------------------
if sectionToRun == 'Units':
    # Brian has a system for using quantities with physical dimensions.
    # All of the basic SI units can be used (volt, amp, etc.) along with
    # all the standard prefixes (m=milli, p=pico, etc.), as well as a few
    # special abbreviations like mV for millivolt, pF for picofarad, etc.
    print(20*volt, ';', 1000*amp, ';', 1e6*volt, ';', 1000*namp)
    # Note that, if you try to add two differnet quantitis together (which
    # is impossible), you will get an error
    # 20*volt + 1000*amp


#------------------------------------------
# A simple model
#------------------------------------------
if sectionToRun == 'SimpleModel':
    # Let's start by defining a simple neuron model. In Brian, all models
    # are defined by systems of differential equations. Here's a simple
    # example of what that looks like:
    tau = 10*ms
    eqs = '''
    dv/dt = (1-v)/tau : 1
    '''
    # eqs = ''' # Another differential equation
    # dv/dt = (sin(2*pi*100*Hz*t)-v)/tau : 1
    # '''
    # In Python, the notation ''' is used to begin and end a multi-line
    # string. So the equations are just a string with one line per equation.
    # The equations are formatted with standard mathematical notation, with
    # one addition. At the end of a line you write : unit where unit is the
    # SI unit of that variable (in this case, 'v').

    # Now let's use this definition to create a neuron.
    G = NeuronGroup(1, eqs, method='euler') 
    # In Brian, you only create groups of neurons, using the class NeuronGroup.
    # The first two arguments when you create one of these objects are the
    # number of neurons (in this case, 1) and the defining differential equations.

    # 'method' defines the numerical integration method. We could use 'exact',
    # 'euler', etc. Note that the 'exact' method is not applicable to stochastic
    # differential equations. Instead, you should use the 'euler' method.

    # It is important to note that the definition of 'tau' in 'eqs' above is
    # required. This is because, without it, the left hand side dv/dt has units
    # of 1/second but the right hand side 1-v is dimensionless. This is important
    # because, for quantities with physical dimensions, it is incorrect because
    # the results would change depending on the unit you measured it in (e.g. if
    # you measured it in seconds the same equation would behave differently to
    # how it would if you measured time in milliseconds). To avoid this, Brian
    # insists that you always specify dimensionally consistent equations.

    # We can now run this differential equation, and track its state using a
    # 'StateMonitor'. This is used to record the values of a neuron variable
    # while the simulation runs. The first two arguments are the group to record
    # from, and the variable you want to record from. We can specify 'record' as
    # True (all neurons?), or a specific value like 0. This later entry means that
    # we record all values for neuron 0. We have to specify which neurons we want
    # to record because in large simulations with many neurons it usually uses up
    # too much RAM to record the values of all neurons.
    M = StateMonitor(G, 'v', record=True)

    # We run the simulation using the run() function, and a time window
    run(30*ms)

    # We can then plot the results. Given the differential equation we used
    # (dv/dt = (1-v)/tau), this will show a negative exponential curves trending
    # to the value of 1.
    plot(M.t/ms, M.v[0])
    xlabel('Time (ms)')
    ylabel('v')
    show()

#------------------------------------------
# Adding spikes
#------------------------------------------
if sectionToRun == 'AddingSpikes':

    # So far we haven't done anything neuronal, just played around with differential
    # equations. Now let's start adding spiking behaviour. We can do this by adding
    # two new keywords to the NeuronGroup declaration: threshold='v>0.8' and
    # reset='v = 0'. What this means is that when v>0.8 we fire a spike, and
    # immediately reset v = 0 after the spike. We can put any expression and
    # series of statements as these strings.

    #  You can see, at the beginning the behaviour is the same as before until v
    # crosses the threshold v>0.8 at which point you see it reset to 0. Internally,
    # Brian has registered this event as a spike. You can do this using a SpikeMonitor.
    # The SpikeMonitor object takes the group whose spikes you want to record as its
    # argument and stores the spike times in the variable t.
    tau = 10*ms
    eqs = '''dv/dt = (1-v)/tau : 1'''
    G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', method='exact')
    spikemon = SpikeMonitor(G)
    M = StateMonitor(G, 'v', record=0)
    run(50*ms)
    plot(M.t/ms, M.v[0])
    for t in spikemon.t:
        axvline(t/ms, ls='--', c='C1', lw=3)
    xlabel('Time (ms)')
    ylabel('v');
    show()


#------------------------------------------
# Refractoriness
#------------------------------------------
if sectionToRun == 'Refractoriness':

    # A common feature of neuron models is refractoriness. This means that after the
    # neuron fires a spike it becomes refractory for a certain duration and cannot fire
    # another spike until this period is over.

    tau = 10*ms
    eqs = '''dv/dt = (1-v)/tau : 1 (unless refractory)''' # '(unless refractory)' added
    G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', refractory=5*ms, method='exact') # 'refractory=5*ms' added
    statemon = StateMonitor(G, 'v', record=0)
    spikemon = SpikeMonitor(G)
    run(50*ms)

    plot(statemon.t/ms, statemon.v[0])
    for t in spikemon.t:
        axvline(t/ms, ls='--', c='C1', lw=3)
    xlabel('Time (ms)')
    ylabel('v')
    show()

    # As you can see in this figure, after the first spike, v stays at 0 for around 5 ms before
    # it resumes its normal behaviour. To do this, we've done two things. Firstly, we've added
    # the keyword 'refractory=5*ms' to the NeuronGroup declaration. On its own, this only means
    # that the neuron cannot spike in this period, but doesn't change how v behaves. In order
    # to make v stay constant during the refractory period, we have to add '(unless refractory)'
    # to the end of the definition of v in the differential equations. What this means is that
    # the differential equation determines the behaviour of v unless it's refractory in which
    # case it is switched off.


#------------------------------------------
# Multiple neurons
#------------------------------------------
if sectionToRun == 'multipleNeurons':

    # We will now create a neural area with multiple neurons. For this we add two new things:
    # Firstly, we've got a new variable N determining the number of neurons.
    # Secondly, we added the statement G.v = 'rand()' before the run. What this does is initialise
    # each neuron with a different uniform random value between 0 and 1. We've done this just so
    # each neuron will do something a bit different.
    N = 100
    tau = 10*ms
    eqs = '''dv/dt = (2-v)/tau : 1'''
    G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='exact')
    G.v = 'rand()' # Randomise starting voltage of neurons
    spikemon = SpikeMonitor(G)
    run(50*ms)

    # The other big change is how we plot the data in the end. As well as the variable spikemon.t
    # with the times of all the spikes, we've also used the variable spikemon.i which gives the
    # corresponding neuron index for each spike, and plotted a single black dot with time on the
    # x-axis and neuron index on the y-value. This is the standard "raster plot" used in neuroscience.
    plot(spikemon.t/ms, spikemon.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    show()

#------------------------------------------
# Parameters
#------------------------------------------
if sectionToRun == 'parameters':

    # To make these multiple neurons do something more interesting, let's introduce per-neuron
    # parameters that don't have a differential equation attached to them.

    N = 100
    tau = 10*ms
    v0_max = 3.
    duration = 1000*ms
    eqs = '''
    dv/dt = (v0-v)/tau : 1 (unless refractory)
    v0 : 1
    '''
    # v0 : 1 declares a new per-neuron parameter v0 with units 1 (i.e. dimensionless).
    G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', refractory=5*ms, method='exact')
    M = SpikeMonitor(G)
    G.v0 = 'i*v0_max/(N-1)'
    # The line G.v0 = 'i*v0_max/(N-1)' initialises the value of v0 for each neuron varying
    # from 0 up to v0_max. The symbol i when it appears in strings like this refers to the
    # neuron index. So in this example, we're driving the neuron towards the value v0 exponentially,
    # but when v crosses v>1, it fires a spike and resets. The effect is that the rate at which it
    # fires spikes will be related to the value of v0. For v0<1 it will never fire a spike, and as
    # v0 gets larger it will fire spikes at a higher rate.
    run(duration)

    figure(figsize=(12,4))
    subplot(121)
    plot(M.t/ms, M.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    subplot(122)
    plot(G.v0, M.count/duration) # Note that number of spikes / simulation duration = spike rate
    xlabel('v0')
    ylabel('Firing rate (sp/s)')
    show()

#------------------------------------------
# Stochastic neurons
#------------------------------------------
if sectionToRun == 'stochasticNeurons':

    # Often when making models of neurons, we include a random element to model the effect of various
    # forms of neural noise. In Brian, we can do this by using the symbol 'xi' in differential equations.
    # Strictly speaking, this symbol is a "stochastic differential" but you can sort of thinking of it
    # as just a Gaussian random variable with mean 0 and standard deviation 1. We do have to take into
    # account the way stochastic differentials scale with time, which is why we multiply it by tau**-0.5
    # in the equations below (see a textbook on stochastic differential equations for more details). Note
    # that we also changed the method keyword argument to use 'euler' (which stands for the Euler-Maruyama
    # method); the 'exact' method that we used earlier is not applicable to stochastic differential equations.

    N = 100
    tau = 10*ms
    v0_max = 3.
    duration = 1000*ms
    sigma = 0.2
    eqs = '''
    dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1 (unless refractory)
    v0 : 1
    '''
    G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', refractory=5*ms, method='euler')
    M = SpikeMonitor(G)

    G.v0 = 'i*v0_max/(N-1)'

    run(duration)

    figure(figsize=(12,4))
    subplot(121)
    plot(M.t/ms, M.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    subplot(122)
    plot(G.v0, M.count/duration)
    xlabel('v0')
    ylabel('Firing rate (sp/s)')
    show()