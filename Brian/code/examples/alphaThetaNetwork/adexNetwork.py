from brian2 import*
import matplotlib.pyplot as plt
import numpy as np

# Stimulation amplitude
externalInput = 0.0*nA
''' < -0.5 = no spiking (i.e. membrane voltage fluctuations only)
    ~ 0.0*nA = sporadic, single spiking
    ~ 0.2*nA = regular, single spiking
    > 0.5*nA : periodic bursting'''

# Adaptive exponential integrate-and-fire model
C = 281*pF # membrane capacitance
gL = 30*nS # leak conductance
EL = -70.6*mV # leak reversal potential
VT = -50.4*mV # spike threshold
deltaT = 2*mV # slope factor
Vcut = VT + 5*deltaT
tau_w = 144*ms # adaptation time constant
a = 4*nS # subthreshold adaptation
b = 0.0805*nA # spike-triggered adaptation
Vr = EL-5*mV # post-spike reset potential
adex = '''
    du/dt = ( -gL*(u-EL) + gL*deltaT*exp((u - VT)/deltaT) - w + I ) / C + ( xi_1*mV*ms**-.5 ) : volt
    dw/dt = ( a*(u-EL) - w ) / tau_w + ( 10*xi_2*pA*ms**-.5 ): amp
'''

# Bursting parameters
A = 125 # period of the sawtooth oscillations
burstTime = .5*A; burstWidth = burstTime/2 # moment and width of secondary, bursting activity
gaussianInput = lambda amp,mu,sig : '''%s*exp(-((tC-%s)**2)/%s)*nA''' % (amp,mu,sig)
sawtooth = lambda A : '''%s*((((t/ms))/T)-floor(((t/ms))/T))''' % (A)
periodic = '''
    I = I_ext + gamma*(I_exp + I_gaus) : amp
    I_exp = '''+gaussianInput(10,5,1)+''' : amp # (initial spike)
    I_gaus = '''+gaussianInput(4,burstTime,burstWidth)+''' : amp # (post-spike burst)
    tC = '''+sawtooth(A)+''': 1 # cycle time
    dT/dt = (100-T)/second + (.01*xi_3*ms**-.5) : 1
    gamma = 1 / (1+exp(-3*(I_ext/nA))): 1 # neuron excitability
    I_ext : amp
'''

# Create neurons
numberOfNeurons = 1
neurons = NeuronGroup(N=numberOfNeurons, model=adex+periodic, threshold='u>20*mV', reset="u=Vr; w+=b", method='euler')
neurons.u = EL
neurons.w = .7*nA
neurons.T = 100
neurons.I_ext = externalInput

# Define recordings
trace = StateMonitor(neurons, ['u','w','I','tC','T'], record=True)
spikes = SpikeMonitor(neurons)

# Run simulation
run(1000*ms,report='stdout')

# Plot results
spikeTrain = spikes.spike_trains()[0]/ms
fig,ax = plt.subplots(1,1)
ax.plot(trace.t/ms, np.mean(trace.u,axis=0), color='k', linewidth=.5)
ax.scatter(spikeTrain,np.zeros(shape=spikeTrain.shape))
ax.set_ylim([-.1,.02])
plt.plot(trace.t/ms, trace.I[0], color='k', linewidth=.5)
plt.show()

# Get spike time interavls
def getSpikeIntervals(spikes):
    spikeTrain = spikes.spike_trains()[0]
    spikeIntervals = np.zeros(shape=len(spikeTrain)-1)
    for i in range(len(spikeTrain)-1):
        spikeIntervals[i] = (spikeTrain[i+1]-spikeTrain[i])/ms
    return spikeIntervals
# plt.hist(getSpikeIntervals(spikes)); plt.show()
# plt.plot(trace.T[0]); plt.show()
print(spikes.spike_trains()[0]/ms)


