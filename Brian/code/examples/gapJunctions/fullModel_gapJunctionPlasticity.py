from brian2 import *
from matplotlib.pyplot import *
import numpy as np
import scipy.signal as signal

# Brian 2 implementation of:
# 'Gap junction plasticity as a mechanism to regulate network-wide oscillations'

tau = .1 * ms

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
    dsp/dt = -sp / ms: 1
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
sigma = 1*mV
tauSig = 10*ms
LIFeqs = '''
    dv/dt = (( -v + Rm * I) / tau_m) + (sigma*sqrt(2/tau)*xi): volt
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

# Define excitatory population
n_exc = 40 #800
excPop = NeuronGroup(n_exc, model=LIFeqs, threshold="v>=v_threshold", reset="v=v_reset", method='euler')
trace_excPop = StateMonitor(excPop, ['v'], record=True)

# Define inhibitory population
n_inh = 10
inhPop = NeuronGroup(n_inh, model=fastSpikingEqs, threshold="v>=v_threshold", reset="v=v_reset; u+=b", method='euler')
trace_inhPop = StateMonitor(inhPop, ['v', 'burst'], record=True)

# -----------------------------------------------
# Create synapses
# -----------------------------------------------

# Define inhibitory-inhibitory gap junction synapses
Wii = -80 * mA
tauI = 10 * ms
thetaBurst = 1.3
alphaLTD = 15.69 * nS * ms**-1
alphaLTP = alphaLTD # 2.9 * alphaLTD
gammab = 10 * siemens
inhibInhibGapJuncs = Synapses(inhPop, inhPop, on_pre='Ispike_post+=Wii; sp_post+=1; burst_post+=1', model='''
    Igap_post = gamma * (v_pre - v_post) : amp (summed)
    dgamma/dt = (gammaIncrease + gammaDecrease)/ms : siemens (clock-driven) # gap junction conductance
    gammaIncrease = alphaLTP * ((gammab - gamma)/gammab)*(sp_pre+sp_post) * ms : siemens
    gammaDecrease = -alphaLTD * (heaviside(burst_pre-thetaBurst)+heaviside(burst_post-thetaBurst)) * ms : siemens
    ''', method='euler')
inhibInhibGapJuncs.connect('i!=j')
trace_inhibInhibGapJuncs = StateMonitor(inhibInhibGapJuncs, ['gamma'], record=True)

# Define inhibitory-excitatory synapses
Wie = -5000 * mA
inhibExc = Synapses(inhPop, excPop, on_pre='Ispike_post+=Wie')
inhibExc.connect()

# Define excitatory-excitatory synapses
Wee = 500 * mA
inhibExc = Synapses(excPop, excPop, on_pre='Ispike_post+=Wee')
inhibExc.connect('i!=j')

# Define excitatory-inhibitory synapses
Wei = 300 * mA
inhibExc = Synapses(excPop, inhPop, on_pre='Ispike_post+=Wei')
inhibExc.connect()

# -----------------------------------------------
# Run simulation
# -----------------------------------------------
print('Running simulation...')

# Run simulation
# randomNumber = lambda : (np.random.randn()+2)/2
excPop.Iinput = [np.random.uniform()*300*pA for i in range(n_exc)]
run(20000*ms, report='stdout')

# Plot results
figure()
meanSignal = np.mean(trace_excPop.v, axis=0)
meanSignal = meanSignal[10000:]
subplot(3,1,1)
plot(meanSignal, linewidth=1)
subplot(3,1,2)
meanGamma = np.mean(trace_inhPop.v, axis=0)
plot(meanGamma, linewidth=1)
subplot(3,1,3)
meanGamma = np.mean(trace_inhibInhibGapJuncs.gamma, axis=0)
plot(meanGamma, linewidth=1)
show()


figure()
for i in range(n_inh):
    subplot(n_inh,1,i+1)
    plot(trace_inhPop.t/ms, trace_inhPop.burst[i])



# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = signal.butter(order, [low, high], btype='band')
#     return b, a

# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = signal.lfilter(b, a, data)
#     return y


# a = butter_bandpass_filter(meanSignal, .1, 200, fs)






# widths = np.arange(1, int(len(meanSignal)/2))
# wavelet = signal.ricker
# cwtmatr = signal.cwt(meanSignal, wavelet, widths)


# specgram(meanSignal,Fs=fs); show()



# signal.spectrogram(meanSignal, fs, )

# fs = (1/(trace_excPop.t[1] - trace_excPop.t[0]))/Hz # sampling rate

# widths = np.arange(1, 100)
# cwtmatr = signal.cwt(meanSignal, signal.ricker, widths)
# imshow(cwtmatr); show()

# sos = signal.butter(10, (1,20), 'hp', fs=fs, output='sos')
# filtered = signal.sosfilt(sos, meanSignal)


# from pyrates.utility import time_frequency

# signal.freqs







# from scipy.signal import stft


# specgram(meanSignal, NFFT=64, Fs=fs, noverlap=32)
# imshow(Sxx, aspect='auto', cmap='hot_r', origin='lower'); print(f); show(); 

#  = spectrogram(meanSignal, fs)
# pcolormesh(t, f, np.abs(Zxx)); ylim(0,20); show()
# # f,t,Sxx = spectrogram(meanSignal, fs=samplingRate, nperseg=512*20, window='hann')
# # imshow(Sxx, aspect='auto', cmap='hot_r', origin='lower'); print(f); show(); 


# # inhPop.Iext = 0 * mA
# # run(200*ms)

# # # Plot results
# # figure()
# # for i in range(n_inh):
# #     subplot(n_inh,1,i+1)
# #     plot(trace.t/ms, trace.v[i])

# # figure()
# # for i in range(n_inh):
# #     subplot(n_inh,1,i+1)
# #     plot(trace.t/ms, trace.burst[i])

# # figure()
# # meanSignal = np.mean(trace_inhPop.v, axis=0)
# # plot(meanSignal, linewidth=2)



# # from scipy.signal import savgol_filter
# # figure()
# # # for i in range(n_inh): plot(trace.v[i], linewidth=.5);
# # meanSignal = np.mean(trace.v, axis=0)
# # # plot(savgol_filter(meanSignal, 1001, 1), linewidth=2); show()

# # N = 1000
# # plot(np.convolve(meanSignal, np.ones((N,))/N, mode='valid'), linewidth=2); show()


# # figure()
# # for i in range(n_inh):
# #     subplot(n_inh,1,i+1);
# #     plot(trace.t/ms, traceS.gamma[i])
# # show()




# # results = []
# # tau = 1
# # spikeTime = 10
# # for i in range(spikeTime,spikeTime+10):
# #     result = math.exp(-((i-spikeTime)/tau))
# #     results.append(result)

# # Run simulation
# # excPop.Iinput = '20 * mA'
# # excPop.v = [-60*mV, -45*mV]
# # run(1000*ms)
# # # excPop.Iinput = 10 * mA
# # # run(400*ms)
# # # excPop.Iinput = 0 * mA
# # # run(200*ms)

# # # figure(1)
# # # subplot(2,1,1)
# # # for i in range(2):
# # #     plot(trace.v[i])
# # # subplot(2,1,2)
# # # for i in range(2):
# # #     plot(trace.I[i])
# # # show()


# # # pop = NeuronGroup(1, model=eqs, method='euler')
# # # pop.I = 1
# # # trace = StateMonitor(pop, ['I'], record=True)
# # # run(100*ms)
# # # plot(trace.I[0]); show()


# trace_excPop.t[1] - trace_excPop.t[0]