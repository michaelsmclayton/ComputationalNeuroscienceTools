from brian2 import*
import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
simulationLength = 3000*ms
externalInput = 1.0*nA
baseInput = 0.25*nA

# Common functions
getOscPeriod = lambda freq : 1/(freq)/ms
gaussianInput = lambda amp,mu,sig : '''%s*exp(-((tC-%s)**2)/%s)*nA''' % (amp,mu,sig)
sawtooth = lambda A, tS : '''%s*((((t/ms)-%s)/T)-floor(((t/ms)-%s)/T))''' % (A,tS,tS)
def sawtoothCycles(A,tS):
    return '''
    tC = '''+sawtooth(A,tS)+''': 1 # cycle time
    T : 1
    # dT/dt = (baseT-T)/second + (.01*xi_3*ms**-.5) : 1'''

# Adaptive exponential integrate-and-fire model
C = 281*pF # membrane capacitance
gL = 30*nS # leak conductance
EL = -70.6*mV # leak reversal potential
deltaT = 4*mV # slope factor
def adexNeuron():
    return '''
    du/dt = ( -gL*(u-EL) + gL*deltaT*exp((u - VT)/deltaT) - w + I ) / C + ( xi_1*mV*ms**-.5 ) : volt
    dw/dt = ( a*(u-EL) - w ) / tau_w + ( 10*xi_2*pA*ms**-.5 ): amp
    tau_w : second \n a : siemens \n b : amp \n Vr : volt \n VT : volt'''
def initialiseAdexNeurons(neurons, tau_w, a, b, Vr, VT, T=None):
    neurons.u = EL; neurons.w = .7*nA
    neurons.tau_w = tau_w # adaptation time constant
    neurons.a = a # subthreshold adaptation
    neurons.b = b # spike-triggered adaptation
    neurons.Vr = VT + Vr # post-spike reset potential
    neurons.VT = VT # spike threshold
    if T!=None:
        neurons.T = T # cycle period (ms)
    return neurons


# ---------------------------------------------------------
# High-threshold bursting cells
# ---------------------------------------------------------
''' With no external input (i.e. spontaneously), these neurons will fire at a base rate of ~8Hz.
During this mode of spontaneous firing, spikes will prodominantly be single spikes. When external
input is greater (e.g. 1*nA), firing rate increase to >8Hz, and spikes come is periodic bursts.
(see Fig 4 of "Cellular Dynamics of Cholinergically Induced (8 – 13 Hz) Rhythms in Sensory Thalamic
Nuclei In Vitro)'''
htBurstBasePeriod = getOscPeriod(8*Hz) # Note that the base frequency is 8Hz. The actual firing frequency increases when driven by excited, faster firing HT neurons
htburst = adexNeuron() + '''
    I = I_ext + gamma*I_spont : amp
    I_spont = '''+gaussianInput(6,htBurstBasePeriod-5,25)+''' : amp # spontaneous firing
    baseT = (1 / baseFreq) / ms : 1
    gamma = (1 / (1+exp(-3*(I_ext/nA)))): 1 # neuron excitability
    I_ext : amp
    baseFreq : hertz
'''+sawtoothCycles(A=htBurstBasePeriod,tS='0')
htBurstingCells = NeuronGroup(N=1, model=htburst, threshold='u>20*mV', reset="u=Vr; w+=b", method='euler')
htBurstingCells = initialiseAdexNeurons(htBurstingCells, tau_w=144*ms, a=4*nS, b=0.0805*nA, Vr=-5*mV, VT=-40.4*mV, T=htBurstBasePeriod)
htBurstingCells.baseFreq = 8*Hz

# Define recordings
htBursting_trace = StateMonitor(htBurstingCells, ['u','w','I','I_ext'], record=True)
htBursting_spikes = SpikeMonitor(htBurstingCells)


# ---------------------------------------------------------
# Interneuron cells
# ---------------------------------------------------------
'''With no external input (i.e. spontaneously), these neurons will fire single spikes sporadically,
but always in-line with ongoing alpha oscillations. When the cell is inhibited, it will display
only membrane voltage fluctuations, and no spiking. However, if the cell is activated (e.g. ~1*nA),
the cell will fire in the following pattern: a single spike around the start of a 100*ms cycle,
followed by a burst of spikes ~50% into the cycle'''
interneuronBasePeriod = getOscPeriod(8*Hz)
burstTime = .5*interneuronBasePeriod; burstWidth = burstTime/2 # moment and width of secondary, bursting activity
tau_syn = 5*ms; tau_gamma = 250*ms
interneuron = adexNeuron() + '''
    I = I_ext + (gamma*I_exp + gamma**4*I_gaus) : amp # Note that I_gaus is multiplied by gamma^4, such that it smaller than I_exp other than when gamma ~= §
    I_exp = '''+gaussianInput(10,10,1)+''' : amp # (initial spike)
    I_gaus = '''+gaussianInput(8,burstTime,burstWidth)+''' : amp # (post-spike burst)
    dI_syn/dt = -I_syn/tau_syn: amp
    '''+sawtoothCycles(A=interneuronBasePeriod,tS='tS/ms')+'''
    tS : second
    dgamma/dt = (.5-gamma) / tau_gamma: 1 # neuron excitability
    baseT = '''+str(interneuronBasePeriod)+''' : 1
    I_ext : amp
'''
interneuronCells = NeuronGroup(N=1, model=interneuron, threshold='u>20*mV', reset="u=Vr; w+=b", events={'phase_reset': 'I_syn>2*pA'}, method='euler')
interneuronCells = initialiseAdexNeurons(interneuronCells, tau_w=144*ms, a=4*nS, b=0.0805*nA, Vr=-35*mV, VT=-50.4*mV, T=interneuronBasePeriod)
interneuronCells.run_on_event('phase_reset', "tS=t; I_syn=0*pA; gamma+=.08")
interneuronCells.gamma = .5
# Define recordings
interneuron_trace = StateMonitor(interneuronCells, ['u','w','I','I_syn','tC','T','tS','gamma'], record=True)
interneuron_spikes = SpikeMonitor(interneuronCells)

# -------------------------------
# Relay mode neurons (thalamocortical neuron)
# -------------------------------
tauI = 15*ms 
relayMode = '''du/dt = ((0.04/mV)*u**2 + 5*u + 140.25*mV - w + (I_syn/gRelay)) / ms : volt
         dw/dt = a*(b*u-w) / ms : volt
         dI_syn/dt = -I_syn / tauI: amp
         a = 0.02 : 1 \n b = 0.25 : 1 \n c=-65*mV : volt \n d=.05*mV : volt \n gRelay = 40*nS : siemens'''
relayCells = NeuronGroup(N=1, model=relayMode, threshold="u>=30*mV", reset="u=c; w+=d", method='euler')
trace_relayCells = StateMonitor(relayCells, ['u','w','I_syn'], record=True)

# -------------------------------
# Readout oscillator
# -------------------------------
tau = 50*ms
def getSineFreq(freq,samplingRate=1000):
    return ((freq/Hz)*(2*pi))/samplingRate
readoutOscillatorEqs = '''
    u = ((scale+scale_syn) + cos((%s*t/ms))) : 1
    dscale/dt = (1-scale) / tau : 1
    scale_syn = %s*exp(-(((t-tSpike)/ms)**2)/%s) : 1
    tSpike : second
    ''' % (getSineFreq(freq=8*Hz),1,1000)
readoutOscillator = NeuronGroup(N=1, model=readoutOscillatorEqs, method='euler')
readoutOscillator_trace = StateMonitor(readoutOscillator, ['u','scale','scale_syn'], record=True)


# ---------------------------------------------------------
# Synapses
# ---------------------------------------------------------
S1 = Synapses(htBurstingCells, interneuronCells, on_pre='I_syn_post += 1*pA', method='euler')
S1.connect()
S2 = Synapses(interneuronCells, relayCells, on_pre='I_syn_post-=1*nA', method='euler')
S2.connect()
S3 = Synapses(relayCells, readoutOscillator, on_pre='tSpike = t+(62.5*ms)', method='euler')
S3.connect()

# ---------------------------------------------------------
# Run network and plot results
# ---------------------------------------------------------

# Run simulation
for i in range(2):
    htBurstingCells.I_ext = baseInput
    run(simulationLength,report='stdout')
    htBurstingCells.I_ext = externalInput
    run(simulationLength,report='stdout')
htBurstingCells.I_ext = baseInput
run(simulationLength,report='stdout')

# Plot results
fig,ax = plt.subplots(5,1,sharex=True)
ax[0].plot(readoutOscillator_trace.t/ms, readoutOscillator_trace.u[0], color='k', linewidth=.5)
ax[1].plot(trace_relayCells.t/ms, np.mean(trace_relayCells.u,axis=0), color='k', linewidth=.5)
ax[2].plot(interneuron_trace.t/ms, np.mean(interneuron_trace.u,axis=0), color='k', linewidth=.5)
# interneuron_spikeTrain = interneuron_spikes.spike_trains()[0]/ms
# ax[1].scatter(interneuron_spikeTrain,np.zeros(shape=interneuron_spikeTrain.shape))
ax[3].plot(htBursting_trace.t/ms, htBursting_trace.u[0], color='k', linewidth=.5)
ax[4].plot(htBursting_trace.t/ms, htBursting_trace.I_ext[0], color='k', linewidth=.5)
ax[4].set_ylim([-.5*nA,1.5*nA]/amp)
# plt.xlim([1200,2800]) # For looking at low input, high alpha sections
# plt.xlim([3600,5200]) # For looking at high input, low alpha sections
fig.tight_layout()
[ax[i].axis('off') for i in range(len(ax))] 
fig.set_figheight(8)
fig.set_figwidth(6)
plt.show()

# Get spike time interavls
def getSpikeIntervals(spikes):
    spikeTrain = spikes.spike_trains()[0]
    spikeIntervals = np.zeros(shape=len(spikeTrain)-1)
    for i in range(len(spikeTrain)-1):
        spikeIntervals[i] = (spikeTrain[i+1]-spikeTrain[i])/ms
    return spikeIntervals
# plt.hist(getSpikeIntervals(spikes)); plt.show()
# plt.plot(interneuron_trace.T[0]); plt.show()

# See whether spikes are more likely to be inline with oscillation peaks
# plt.hist(spikeTrain % A); plt.show()

# plt.subplot(2,1,1)
# plt.plot(readoutOscillator_trace.t/ms, readoutOscillator_trace.u[0], color='k', linewidth=.5)
# plt.subplot(2,1,2)
# plt.plot(readoutOscillator_trace.t/ms, readoutOscillator_trace.scale_syn[0], color='k', linewidth=.5)
# plt.show()

