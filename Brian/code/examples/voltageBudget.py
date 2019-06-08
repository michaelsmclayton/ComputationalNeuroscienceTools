
from brian2 import *
from matplotlib.pyplot import *
import numpy as np

# --------------------------------------------------------------------
# Define code to create random, uniform distributions of variables 
# --------------------------------------------------------------------

# Plant all the seeds!
seed_value=42
seed(seed_value)
prng = np.random.RandomState(seed_value)

# This will create random elements (of length p, between x_min and x_max)
def _parse_membrane_param(x, N, prng):
    try:
        if len(x) == 2:
            x_min, x_max = x
            x = prng.uniform(x_min, x_max, N)
        else:
            raise ValueError("Parameters must be scalars, or 2 V_lement lists")
    except TypeError:
        pass
    return x, prng

# Set simulation parameters
defaultclock.dt = 0.01*ms
numberOfRecievingNeurons = 10

# Set neural parameters
C = [100, 200] * pfarad # total capacitance
gL = [10, 18] * nsiemens # total leak conductance
EL = [-70, -50] * mvolt # effective rest potential
DeltaT = 2 * msecond # threshold slope factors (?)
VT = [-58, -46] * mvolt # effective threshold potential

# Get random, uniform distribution parameters
C, prng = _parse_membrane_param(C, numberOfRecievingNeurons, prng)
gL, prng = _parse_membrane_param(gL, numberOfRecievingNeurons, prng)
EL, prng = _parse_membrane_param(EL, numberOfRecievingNeurons, prng)
VT, prng = _parse_membrane_param(VT, numberOfRecievingNeurons, prng)
taum = C / gL # time scale
Vcut = VT + 5 * DeltaT



# Pick an electrophysiological behaviour
#   tauw = time constant
#   a = conductance
#   b = spike triggered adaptation
#   Vr = rest potential
firingPattern = 'fastSpiking' # regularSpiking, bursting, fastSpiking
if firingPattern == 'regularSpiking':
    tauw, a, b, Vr = 144*ms, 4*nS, 0.0805*nA, -70.6*mV # Regular spiking (as in the paper)
elif firingPattern == 'bursting':
    tauw, a, b, Vr = 20*ms, 4*nS, 0.5*nA, VT+5*mV # Bursting
elif firingPattern == 'fastSpiking':
    tauw, a, b, Vr = 144*ms, 2*C/(144*ms), 0*nA, -70.6*mV # Fast spiking

# Define AdEx differential equations

# exp((vm - VT)/DeltaT):
# Creates an exponential increase in voltage if current voltage (vm) is above
# firing theshold (VT). (i.e. when the difference in voltage is negative, the exponential
# function will give small results, but when the difference is positive and large, the
# exponential function will give large results). The resulting upswing in voltage is
# stopped at a reset threshold which we fix at 0 mV. The downswing of the action potential, is replaced by the reset condition:

# w = adaptation variable:
# During spike trains, w will accumulate by 'b' (i.e. "w+=b")

eqs = """
    dvm/dt = ((- gL * (vm - EL)) + (gL * DeltaT * exp((vm - VT)/DeltaT)) + I - w)/C : volt
    dw/dt = (a*(vm - EL) - w)/tauw : amp
    I : amp
    # I = I_in + I_osc(t) + I_noise + I_ext + bias_in
"""

# Create models
N = 10
neuron = NeuronGroup(numberOfRecievingNeurons, model=eqs, threshold='vm>Vcut',
                     reset="vm=Vr; w+=b", method='euler')
neuron.vm = EL #-40*mV -(30*(rand()))*mV
trace = StateMonitor(neuron, 'vm', record=0)
spikes = SpikeMonitor(neuron)

# From paper...
# P_n = NeuronGroup(
#     N,
#     model=eqs,
#     threshold='v > V_thresh',
#     reset="v = V_rheo; w += b",
#     method='euler')

# # -----------------------------------------------------------------
# # Add synaptic input into the network.
# if ns.size > 0:
#     P_stim = SpikeGeneratorGroup(np.max(ns) + 1, ns, ts * second)
#     C_stim = Synapses(
#         P_stim, P_n, model='w_in : siemens', on_pre='g_in += w_in')
#     C_stim.connect()

#     # (Finally) Potentially random weights
#     w_in, prng = _parse_membrane_param(w_in, len(C_stim), prng)
#     C_stim.w_in = w_in * siemens


# Run stimulation
run(20 * ms)
neuron.I = 1*nA
run(200 * ms)
neuron.I = 0*nA
run(100 * ms)

# We draw nicer spikes
vm = trace[0].vm[:]
for t in spikes.t:
    i = int(t / defaultclock.dt)
    vm[i] = 20*mV

plot(trace.t / ms, vm / mV)
xlabel('time (ms)')
ylabel('membrane potential (mV)')
show()

# plot(spikes.t/ms, spikes.i, '.k')
# xlabel('Time (ms)')
# ylabel('Neuron index')
# show()




#     # -----------------------------------------------------------------
#     # Define an adex neuron, and its connections

#         eqs = \
#     # 
#     """
#     dv/dt = fOfV + I_noise + I_ext + bias_in - w) / C : volt
#     fOfV = (-g_l * (v - V_l) + (g_l * delta_t * exp((v - V_t) / delta_t)) + I_in + I_osc(t))
#     """ + """
#     dw/dt = (a * (v - V_l) - w) / tau_w : amp
#     dg_in/dt = -g_in / tau_in : siemens
#     dg_noise/dt = -(g_noise + (sigma * sqrt(tau_in) * xi)) / tau_in : siemens
#     I_in = g_in * (v - V_l) : amp
#     I_noise = g_noise * (v - V_l) : amp
#     C : farad   # Membrane capacitance
#     g_l : siemens # Leak conductance
#     a : siemens
#     b : amp
#     delta_t : volt
#     tau_w : second
#     V_rheo : volt
#     V_l : volt
#     bias_in : amp
#     tau_in : second
#     """


# # def adex(N,
# #          time,
# #          ns,
# #          ts,
# #          E=0,
# #          n_cycles=1,
# #          w_in=0.8e-9,
# #          tau_in=5e-3,
# #          bias_in=0.0e-9,
# #          V_t=-50.0e-3,
# #          V_thresh=0.0,
# #          f=0,
# #          A=.1e-9,
# #          phi=0,
# #          sigma=0,
# #          C=200e-12,
# #          g_l=10e-9,
# #          V_l=-70e-3,
# #          a=0e-9,
# #          b=10e-12,
# #          tau_w=30e-3,
# #          V_rheo=-48e-3,
# #          delta_t=2e-3,
# #          time_step=1e-5,
# #          budget=True,
# #          report=None,
# #          save_args=None,
# #          pulse_params=None,
# #          seed_value=42):