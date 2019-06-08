from brian2 import *
from matplotlib.pyplot import *

defaultclock.dt = 0.01*ms

# ------------------------------------------------------------
# Define model equations
# ------------------------------------------------------------

# Voltage differential equation (dv/dt)
Cm = 1*uF # /cm**2 # membrane time constant
Iapp = 2*uA # injected current
dvdt = '''
    dv/dt = (-I_Na - I_K - I_L + Iapp) / Cm : volt
'''

# Ion currents
gNa = 35*msiemens # Na conductance
ENa = 55*mV # Na reversal potential
gL = 0.1*msiemens # leak current conductance
EL = -65*mV # leak reversal potential
gK = 9*msiemens # K conductance
EK = -90*mV # K reversal potential
ionCurrents = '''
    I_Na = gNa*m**3*h*(v-ENa)  : amp #
    I_K = gK*n**4*(v-EK)    : amp #
    I_L = gL*(v-EL) : amp # Leak current
'''

# # Synapse equations
# gSyn = .1*msiemens
# ESyn = -75*mV
# alpha = 12*(10^-1)*msecond
# beta = .1*(10^-1)*msecond
# deltaSyn = 0*mV
# synapseCurrents = '''
#     ds/dt = alpha * f_vpre * (1 - s) - beta: 1
#     f_vpre = 1/(1 + exp(-(v_pre - deltaSyn)/2)) : volt
#     I_syn = gSyn * s * (v - ESyn) : volt
# '''

# Na activation = m
# Na inactivation = h
# K activation = n

# Na activation equation (m)
m = '''
    m = alpha_m/(alpha_m+beta_m) : 1
'''
naActivationParameters = '''
    alpha_m = -0.1/mV*(v+35*mV)/(exp(-0.1/mV*(v+35*mV))-1)/ms : Hz
    beta_m = 4*exp(-(v+60*mV)/(18*mV))/ms : Hz
'''

# Na inactivation variable differential equation (dh/dt)
dhdt = '''
    dh/dt = 5*(alpha_h*(1-h)-beta_h*h) : 1
'''
naInactivationParameters = '''
    alpha_h = 0.07*exp(-(v+58*mV)/(20*mV))/ms : Hz
    beta_h = 1./(exp(-0.1/mV*(v+28*mV))+1)/ms : Hz
'''

# K activation differential equation
dndt = '''
    dn/dt = 5*(alpha_n*(1-n)-beta_n*n) : 1
'''
kActivationParameters = '''
    alpha_n = -0.01/mV*(v+34*mV)/(exp(-0.1/mV*(v+34*mV))-1)/ms : Hz
    beta_n = 0.125*exp(-(v+44*mV)/(80*mV))/ms : Hz
'''

# Define full equation
eqs = dvdt + ionCurrents + m + naActivationParameters + dhdt \
+ naInactivationParameters + dndt + kActivationParameters


# ------------------------------------------------------------
# Define neurons (and state monitor)
# ------------------------------------------------------------
neuron = NeuronGroup(1, eqs, method='exponential_euler')
neuron.v = -70*mV
neuron.h = 1
M = StateMonitor(neuron, 'v', record=0) # record=True)


# ------------------------------------------------------------
# Run simulation
# ------------------------------------------------------------
run(100*ms, report='text')

# ------------------------------------------------------------
# Plot results
# ------------------------------------------------------------
plot(M.t/ms, M[0].v/mV)
show()