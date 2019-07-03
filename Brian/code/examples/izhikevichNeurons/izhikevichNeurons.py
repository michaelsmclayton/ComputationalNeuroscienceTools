from brian2 import *
from matplotlib.pyplot import *

# -----------------------------------------------
# Izhikevich model
# -----------------------------------------------

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
    # elif neuronType=='TC': # thalamocortical
    #     a, b, c, d = .02, .25, -65, .05
    # elif neuronType=='RZ': # resonator
    #     a, b, c, d = .1, .26, -65, 2
    elif neuronType=='LTS': # low-threshold spiking
        a, b, c, d = .02, .25, -65, 2
    return a, b, c, d

tau = .5 * ms
a, b, c, d = getModelParameters('CH')
eqs = '''
    dv/dt = ( .04*v**2 + 5*v + 140 - u + I ) / tau : 1
    du/dt = ( a * (b*v - u) ) / tau : 1
    I : 1
'''
neurons = NeuronGroup(1, model=eqs, threshold="v>=30", reset="v=c; u+=d", method='euler')
trace = StateMonitor(neurons, ['v', 'u'], record=True)
neurons.v = -70

# Run simulation
neurons.I = 0
run(100*ms)
neurons.I = 10
run(200*ms)
neurons.I = 0
run(100*ms)

# Plot results
plot(trace.t/ms, trace.v[0]); show()
# plot(trace.t/ms, trace.u[0]); show()
