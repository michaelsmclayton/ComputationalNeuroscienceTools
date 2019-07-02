from brian2 import * # Import Brian
from matplotlib.pyplot import *
import numpy as np

# Adaptive exponential integrate-and-fire model

# Parameters
C = 281 * pF
gL = 30 * nS
taum = C / gL
EL = -70.6 * mV
VT = -50.4 * mV
DeltaT = 2 * mV
Vcut = VT + 5 * DeltaT

# Pick an electrophysiological behaviour
#tauw, a, b, Vr = 144*ms, 4*nS, 0.0805*nA, -70.6*mV # Regular spiking (as in the paper)
tauw,a,b,Vr=20*ms,4*nS,0.5*nA,VT+5*mV; EL = VT # Bursting
#tauw,a,b,Vr=144*ms,2*C/(144*ms),0*nA,-70.6*mV # Fast spiking

# Model equations
eqs = '''
    dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I + Igap - w)/C : volt
    dw/dt = (a*(vm - EL) - w)/tauw : amp
    I : amp
    Igap : amp
'''

# Create neuronss
N = 4
neurons = NeuronGroup(N, model=eqs, threshold='vm>Vcut',
                     reset="vm=Vr; w+=b", method='euler')
neurons.vm = EL
randomInputs = 1*nA + .5*np.random.rand(N)*nA
trace = StateMonitor(neurons, 'vm', record=True)
spikes = SpikeMonitor(neurons)

# Create gap junctions
S = Synapses(neurons, neurons, '''
             w_gap : siemens # gap junction conductance
             Igap_post = w_gap * (vm_pre - vm_post) : amp (summed)
             ''')
S.connect(condition='i!=j')

# Save initial state of network
store()

# Run analysis twice (for gap vs no gap junctions)
fig, axs = plt.subplots(len(trace.vm), 2)
for iteration, conductance in enumerate([0, .05]):
    restore()
    S.w_gap = conductance * uS
    run(20 * ms)
    neurons.I = randomInputs
    run(100 * ms)
    # neurons.I = 0*nA
    # run(20 * ms)
    # We draw nicer spikes
    for index, vm in enumerate(trace.vm):
        #vm = trace[0].vm[:]
        axs[index, iteration].plot(trace.t/ms, vm/ mV)
show()