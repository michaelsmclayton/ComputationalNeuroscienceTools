from brian2 import *
from matplotlib.pyplot import *

# -----------------------------------------------
# Define Izhikevich neurons
# -----------------------------------------------

# Get model parameters for different neuron types
def getModelParameters(neuronType):
    ''' a = time scale of recovery variable 'u'
        b = sensitivity of 'u' to the subthreshold fluctuations of membrane potential 'v'
        c = post-spike reset value of the membrane potential 'v'
        d = post-spike increase of the recovery variable 'u'
    '''
    if neuronType=='RS': # regular spiking
        a, b, c, d = .02, .2, -65, 8
    elif neuronType=='IB': # intrinsically bursting
        a, b, c, d = .02, .2, -55, 4
    elif neuronType=='CH': # chattering
        a, b, c, d = .02, .2, -50, 2
    elif neuronType=='FS': # fast spiking
        a, b, c, d = .1, .2, -65, 2
    elif neuronType=='TC': # thalamocortical
        a, b, c, d = .02, .25, -65, .05
    elif neuronType=='RZ': # resonator
        a, b, c, d = .1, .26, -65, 2
    elif neuronType=='LTS': # low-threshold spiking
        a, b, c, d = .02, .25, -65, 2
    return a, b, c, d

# Function to define a neural population
def createIzhikevichPopulation(neuronType, N, initialVoltage=-70, variablesToRecord=['v','u']):

    # Define model parameters
    a, b, c, d = getModelParameters(neuronType)
    eqs = '''
        tau = .5 * ms : second
        dv/dt = ( .04*v**2 + 5*v + 140 - u + I ) / tau : 1
        du/dt = ( a * (b*v - u) ) / tau : 1
        a = %s : 1
        b = %s : 1
        c = %s : 1
        d = %s : 1
        I : 1
    ''' % (a, b, c, d)

    # Create population
    population = NeuronGroup(N, model=eqs, threshold="v>=30", reset="v=c; u+=d", method='euler')
    population.v = initialVoltage

    # Create monitors for population
    trace = StateMonitor(population, variablesToRecord, record=True)
    spikemon = SpikeMonitor(population)

    # Return population
    return population, trace, spikemon


# -----------------------------------------------
# Create Izhikevich populations
# -----------------------------------------------
randomInput = PoissonGroup(100, np.arange(100)*Hz + 10*Hz)
n_exc, n_inh = 80, 20
excPop, excTrace, excSpikemon = createIzhikevichPopulation('RS', N=n_exc)
inhPop, inhTrace, inhSpikemon = createIzhikevichPopulation('FS', N=n_inh)


# -----------------------------------------------
# Create synapses
# -----------------------------------------------
connectionProbability = .1
S = Synapses(randomInput, excPop, on_pre='v_post += 10')
Sei = Synapses(excPop, inhPop, 'w : 1', on_pre='v_post += 10')
Sie = Synapses(inhPop, excPop, 'w : 1', on_pre='v_post -= 10')
S.connect(p=connectionProbability)
Sei.connect(p=connectionProbability)
Sie.connect(p=connectionProbability)

# # Create gap junctions between inhibitory neurons
# Sii = Synapses(inhPop, inhPop, '''
#              w_gap : 1 # gap junction conductance
#              Igap = w_gap * (v_pre - v_post) : 1
#              ''', on_pre='v_post += Igap')
# Sii.connect(p=.10)
# Sii.w_gap = .2


# Sie.delay = 10 * ms


# Run simulation
run(500*ms)

# Plot results
# figure(1)
# subplot(2,1,1)
# for trace in excTrace.v:
#     plot(excTrace.t/ms, trace)
# subplot(2,1,2)
# for trace in inhTrace.v:
#     plot(inhTrace.t/ms, trace)

figure(2, figsize=[7,10])
dotSize = 10
scatter(inhSpikemon.t/ms, inhSpikemon.i, s=dotSize, c='r')
scatter(excSpikemon.t/ms, excSpikemon.i+n_inh+1, s=dotSize, c='b')
xlabel('Time (ms)')
ylabel('Neuron index')
show()