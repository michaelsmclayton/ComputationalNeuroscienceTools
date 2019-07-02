from brian2 import *
from matplotlib.pyplot import *
import numpy as np

# Brian implementation of:
# 'Gap junction plasticity as a mechanism to regulate network-wide oscillations'

# -----------------------------------------------
# Fast Spiking (FS) interneurons (with Izhikevich type neuron models)
# -----------------------------------------------

''' Note that the v_threshold and v_reset values from the paper lead to
relentless spiking of this neuron?
'''

# Constants
tau_v = 17 * ms # membrane time constant
v_ra = -75 * mV # membrane resting potential
v_rb = -60 * mV # membrane threshold potential
v_threshold = v_rb # 25 * mV
v_reset = v_ra # -47 * mV # reset potential
Ku = 10 * ohm # coupling parameters to the adaptation variable 'u'
R = 8 * ohm # resistance
tau_u = 10 * ms # membrane time constant
v_rc = -64 * mV # voltage constant
a = 1 * nS # coupling parameters
b = 50 * pA # current constant

# Equations
fastSpikingEqs = '''
    dv/dt = ( ((v - v_ra)*(v - v_rb)/mV) - Ku*u + R * I ) / tau_v : volt
    du/dt = (a * (v - v_rc) - u) / tau_u : amp
    I : amp
'''

# -----------------------------------------------
# Leaky integrated-and-fire model (excitatory)
# -----------------------------------------------


# -----------------------------------------------
# Create network and un
# -----------------------------------------------

# Create network
neurons = NeuronGroup(1, model=fastSpikingEqs, threshold="v>=v_threshold", reset="v=v_reset; u+=b", method='euler')
neurons.v = -75 * mV
trace = StateMonitor(neurons, ['v', 'u'], record=True)
spikes = SpikeMonitor(neurons)

# Run simulation
neurons.I = 0 * mA
run(100*ms)
neurons.I = 7.6 * mA
run(1000*ms)
neurons.I = 0 * mA
run(100*ms)

# Plot results
print(len(spikes.t))
plot(trace.t/ms, trace.v[0]); show()