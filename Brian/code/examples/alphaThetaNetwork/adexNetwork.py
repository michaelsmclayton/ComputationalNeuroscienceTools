from brian2 import*
import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
simulationLength = 1000*ms
externalInput = 1.0*nA
''' < -0.5 = no spiking (i.e. membrane voltage fluctuations only)
    ~ 0.0*nA = sporadic, single spiking
    ~ 0.2*nA = regular, single spiking
    > 1.0*nA : periodic bursting'''

# Common functions
getOscPeriod = lambda freq : 1/(freq)/ms
gaussianInput = lambda amp,mu,sig : '''%s*exp(-((tC-%s)**2)/%s)*nA''' % (amp,mu,sig)
sawtooth = lambda A : '''%s*((((t/ms))/T)-floor(((t/ms))/T))''' % (A)
def sawtoothCycles(A):
    return '''
    tC = '''+sawtooth(A)+''': 1 # cycle time
    dT/dt = (baseT-T)/second + (.01*xi_3*ms**-.5) : 1'''

# Adaptive exponential integrate-and-fire model
C = 281*pF # membrane capacitance
gL = 30*nS # leak conductance
EL = -70.6*mV # leak reversal potential
VT = -50.4*mV # spike threshold
deltaT = 4*mV # slope factor
Vcut = VT + 5*deltaT
def adexNeuron():
    return '''
    du/dt = ( -gL*(u-EL) + gL*deltaT*exp((u - VT)/deltaT) - w + I ) / C + ( xi_1*mV*ms**-.5 ) : volt
    dw/dt = ( a*(u-EL) - w ) / tau_w + ( 10*xi_2*pA*ms**-.5 ): amp
    tau_w : second \n a : siemens \n b : amp \n Vr : volt'''
def initialiseAdexNeurons(neurons, tau_w, a, b, Vr,T):
    neurons.u = EL; neurons.w = .7*nA
    neurons.tau_w = tau_w # adaptation time constant
    neurons.a = a # subthreshold adaptation
    neurons.b = b # spike-triggered adaptation
    neurons.Vr = Vr # post-spike reset potential
    neurons.T = T # cycle period (ms)
    return neurons


# ---------------------------------------------------------
# High-threshold bursting cells
# ---------------------------------------------------------

htBurstBasePeriod = getOscPeriod(8*Hz)
htburst = adexNeuron() + '''
    I = I_ext + I_spont : amp
    I_spont = '''+gaussianInput(1.5,.1*htBurstBasePeriod,htBurstBasePeriod/2)+''' : amp # spontaneous firing
    I_ext : amp
    baseT = 1 / (baseFreq + maxFreqChange*(2*(1/(1+exp(-3*I_ext/nA)))-1)) / ms : 1
    baseFreq : hertz
    maxFreqChange : hertz
'''+sawtoothCycles(A=htBurstBasePeriod)
htBurstingCells = NeuronGroup(N=1, model=htburst, threshold='u>20*mV', reset="u=Vr; w+=b", method='euler')
htBurstingCells = initialiseAdexNeurons(htBurstingCells, tau_w=144*ms, a=4*nS, b=0.0805*nA, Vr=VT+5*mV, T=htBurstBasePeriod)
htBurstingCells.I_ext = externalInput
htBurstingCells.baseFreq = 8*Hz
htBurstingCells.maxFreqChange = 8*Hz

# Define recordings
htBursting_trace = StateMonitor(htBurstingCells, ['u','w','I'], record=True)
htBursting_spikes = SpikeMonitor(htBurstingCells)


# ---------------------------------------------------------
# Interneuron cells
# ---------------------------------------------------------

interneuronBasePeriod = getOscPeriod(10*Hz)
burstTime = .5*interneuronBasePeriod; burstWidth = burstTime/2 # moment and width of secondary, bursting activity
interneuron = adexNeuron() + '''
    I = (I_ext + gamma*I_exp + gamma**4*I_gaus) : amp # Note that I_gaus is multiplied by gamma^4, such that it smaller than I_exp other than when gamma ~= §
    I_exp = '''+gaussianInput(10,5,1)+''' : amp # (initial spike)
    I_gaus = '''+gaussianInput(4,burstTime,burstWidth)+''' : amp # (post-spike burst)
    tC = '''+sawtooth(A=interneuronBasePeriod)+''': 1 # cycle time
    dT/dt = (100-T)/second + (.01*xi_3*ms**-.5) : 1
    gamma = 1 / (1+exp(-3*(I_ext/nA))): 1 # neuron excitability
    I_ext : amp
'''
interneuronCells = NeuronGroup(N=1, model=interneuron, threshold='u>20*mV', reset="u=Vr; w+=b", method='euler')
interneuronCells = initialiseAdexNeurons(interneuronCells, tau_w=144*ms, a=4*nS, b=0.0805*nA, Vr=EL+5*mV, T=interneuronBasePeriod)
interneuronCells.I_ext = externalInput

# Define recordings
interneuron_trace = StateMonitor(interneuronCells, ['u','w','I','tC','T'], record=True)
interneuron_spikes = SpikeMonitor(interneuronCells)


# ---------------------------------------------------------
# Run network and plot results
# ---------------------------------------------------------

# Run simulation
run(simulationLength,report='stdout')

# Plot bursting results results
plt.figure()
plt.plot(htBursting_trace.u[0])
# plt.show()

# Plot interneuron results
spikeTrain = interneuron_spikes.spike_trains()[0]/ms
fig,ax = plt.subplots(1,1)
ax.plot(interneuron_trace.t/ms, np.mean(interneuron_trace.u,axis=0), color='k', linewidth=.5)
ax.scatter(spikeTrain,np.zeros(shape=spikeTrain.shape))
ax.set_ylim([-.14,.02])
# plt.plot(interneuron_trace.t/ms, interneuron_trace.I[0], color='k', linewidth=.5)

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
print(interneuron_spikes.spike_trains()[0]/ms)

# See whether spikes are more likely to be inline with oscillation peaks
# plt.hist(spikeTrain % A); plt.show()