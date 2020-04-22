from brian2 import*
import matplotlib.pyplot as plt
import numpy as np

# Adaptive exponential integrate-and-fire model
C = 281*pF
gL = 30*nS
EL = -70.6*mV
VT = -50.4*mV
deltaT = 2*mV
Vcut = VT + 5*deltaT
tau_w, a, b, Vr = 144*ms, 4*nS, 0.0805*nA, EL-5*mV # regular spiking parameters
adex = '''
    du/dt = ( -gL*(u-EL) + gL*deltaT*exp((u - VT)/deltaT) - w + I ) / C + ( xi_1*mV*ms**-.5 ) : volt
    dw/dt = ( a*(u-EL) - w ) / tau_w + ( 10*xi_2*pA*ms**-.5 ): amp
'''

# Bursting parameters
A = 100 # period of the sawtooth oscillations
mu = .5*A; sig = mu/2 # moment and width of secondary, bursting activity
excitability = 0; burstTime = 40; burstWidth = 100
gaussianInput = lambda amp,mu,sig : '''%s*exp(-((tC-%s)**2)/%s)*nA''' % (amp,mu,sig)
sawtooth = lambda A : '''%s*((((t/ms))/T)-floor(((t/ms))/T))''' % (A)
periodic = '''
    I = I_syn*(I_exp + I_gaus) : amp
    I_exp = '''+ gaussianInput(10,5,1)+''' : amp # (initial spike)
    I_gaus = '''+gaussianInput(excitability,burstTime,burstWidth)+''' : amp # (post-spike burst)
    I_syn : 1
    tC = '''+sawtooth(A)+''': 1 # cycle time
    dT/dt = (100-T)/second + (.01*xi_3*ms**-.5) : 1
'''

# Create neurons
numberOfNeurons = 1
neurons = NeuronGroup(N=numberOfNeurons, model=adex+periodic, threshold='u>20*mV', reset="u=Vr; w+=b", method='euler')
neurons.u = EL
neurons.w = .7*nA
neurons.T = 100
neurons.I_syn = 1 # .5

# Define recordings
trace = StateMonitor(neurons, ['u','w','I','tC','T'], record=True)
spikes = SpikeMonitor(neurons)

# Run simulation
run(2000*ms,report='stdout')

# Plot results
fig,ax = plt.subplots(1,1)
ax.plot(trace.t/ms, np.mean(trace.u,axis=0), color='k', linewidth=.5)
ax.set_ylim([-.1,.02])
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