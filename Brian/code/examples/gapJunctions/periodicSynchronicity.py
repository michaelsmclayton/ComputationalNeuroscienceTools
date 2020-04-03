from brian2 import *
from matplotlib.pyplot import *
import numpy as np

# Brian 2 implementation of:
# 'Gap junction plasticity as a mechanism to regulate network-wide oscillations'

# -----------------------------------------------
# Fast Spiking (FS) interneurons (with Izhikevich type neuron model)
# -----------------------------------------------

# Parameters
tau_v = 17 * ms # membrane time constant
v_ra = -75 * mV # membrane resting potential
v_rb = -60 * mV # membrane threshold potential
v_threshold = -60 * mV # v_rb
v_reset = -47 * mV # v_ra # reset potential
Ku = 10 * ohm # coupling parameters to the adaptation variable 'u'
R = 8 * ohm # resistance
tauburst = 8 *  ms
tau_u = 10 * ms # membrane time constant
v_rc = -64 * mV # voltage constant
a = 1 * nS # coupling parameters
b = 50 * pA # current constant

# Model equations
fastSpikingEqs = '''
    dv/dt = ( ((v - v_ra)*(v - v_rb)/mV) - Ku*u + R * (I+Igap) ) / tau_v : volt
    du/dt = (a * (v - v_rc) - u) / tau_u : amp
    I = Ispike + Igap + Inoise + Iext: amp
    # dsp/dt = : 1
    dburst/dt = -burst / tauburst : 1
    dIspike/dt = (-Ispike / tauI): amp
    Igap : amp
    Inoise : amp
    Iext : amp
'''

# -----------------------------------------------
# Leaky integrated-and-fire model (excitatory)
# -----------------------------------------------

# Parameters
tau_m = 40 * ms # membrane time constant
tau_e = 12 * ms
Rm = 0.6 * ohm # resistance
v_threshold = 0 * mV
v_reset = -70 * mV

# Model equations
LIFeqs = '''
    dv/dt = ( -v + Rm * I) / tau_m : volt
    I = Iinput + Ispike : amp
    dIspike/dt = (-Ispike / tau_e) : amp
    Iinput : amp
'''

# -----------------------------------------------
# Define custom functions
# -----------------------------------------------
@implementation('numpy', discard_units=True)
@check_units(x=1, result=1)
def heaviside(x):
    return 1 if x > 0 else 0

# -----------------------------------------------
# Create populations
# -----------------------------------------------

# Define inhibitory population
n_inh = 30
inhPop = NeuronGroup(n_inh, model=fastSpikingEqs, threshold="v>=v_threshold", reset="v=v_reset; u+=b", method='euler')
trace = StateMonitor(inhPop, ['v', 'burst'], record=True)

# -----------------------------------------------
# Create synapses
# -----------------------------------------------

# # Define inhibitory gap junction synapses
Wii = 0 * mA
tauI = 10 * ms
thetaBurst = 1.3
alphaLTD = 100 * uS * ms**-1
gammab = 10 * siemens
S = Synapses(inhPop, inhPop, on_pre='Ispike_post+=Wii', on_post='burst_post+=1', model='''
             Igap_post = gamma * (v_pre - v_post) : amp (summed)
             dgamma/dt = (gammaIncrease + gammaDecrease)/ms : siemens (clock-driven) # gap junction conductance
             gammaIncrease = alphaLTD * ((gammab - gamma)/gammab) * ms : siemens
             gammaDecrease = -alphaLTD * (heaviside(burst_pre-thetaBurst)+heaviside(burst_post-thetaBurst)) * ms : siemens
             ''', method='euler')
S.connect()
traceS = StateMonitor(S, ['gamma'], record=True)


# -----------------------------------------------
# Run simulation
# -----------------------------------------------
print('Running simulation...')

# Run simulation
inhPop.Iext = [(8+np.random.randn()*4)*mA for i in range(n_inh)]
run(20000*ms, report='stdout')

def removeBox(ax):
    for line in ['top', 'right','bottom','left']:
        ax.spines[line].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

# Plot results
fig = figure()
ax1 = subplot(2,1,1)
meanSignal = np.mean(trace.v, axis=0)
plot(meanSignal[1000:195000], color='k', linewidth=.05)
ylabel('Mean inhibitory population activity')
ax1.get_xaxis().set_ticks([])
ax2 = subplot(2,1,2)
meanGamma = np.mean(traceS.gamma, axis=0)
plot(trace.t[1000:195000], meanGamma[1000:195000], color='k', linewidth=1)
ylabel('Gap junction coupling')
xlabel('Time (seconds)')
removeBox(ax1); removeBox(ax2); 
show()