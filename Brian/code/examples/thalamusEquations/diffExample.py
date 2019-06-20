from brian2 import * # Import Brian
from matplotlib.pyplot import *
import math

# Taken from 'Causal Role of Thalamic Interneurons in
# Brain State Transitions- A Study Using a Neural Mass
# Model Implementing Synaptic Kinetics'

# General parameters
tau = 10 * ms

# -----------------------------
# Equations for AMPA and GABAa channels
# -----------------------------

# Equation 1
Tmax = 1
Vthr = -32 # Point where output reaches .5 mM
omega = 3.8 # steepness of the sigmoid function
# Vpre = Vthr
equation1 = '''T = Tmax / (1 + exp(-((Vpre - Vthr)/omega))) : 1'''

# Equation 2
alpha = 1000**-1 # * mM**-1 * second**-1
beta = 50**-1 # * second**-1 # reverse rates of chemical reactions
Vrest = -70
equation2 = '''
dVpre/dt = .01*(Vrest-Vpre) / tau : 1
dr/dt = ( alpha * T * (1 - r) - beta * r ) / tau : 1
diff = ( alpha * T * (1 - r) - beta * r ) : 1
'''

# Combined equation for AMPA and GABAa synapses
eqs = equation1 + equation2
G = NeuronGroup(1, eqs, method='euler')
G.r = .001
G.Vpre = 0
M = StateMonitor(G, [ 'T', 'Vpre', 'r', 'diff'], record=True)
run(2000*ms)

subplot(2,2,1)
plot(M.t/ms, M.Vpre[0])
ylabel('Vpre')
subplot(2,2,2)
plot(M.t/ms, M.T[0])
ylabel('T')
subplot(2,2,3)
plot(M.t/ms, M.r[0])
ylabel('r')
subplot(2,2,4)
plot(M.t/ms, M.diff[0])
ylabel('dr/dt')
show()


# # # Equation 1 - neurotransmitter concentrations

# tau = 10 * ms
# omega = 3.8 * mV # steepness of the sigmoid function
# Tmax = 1 * mM
# Vthr = -32 * mV # Point where output reaches .5 mM
# alpha = 1000 * mM**-1 * second**-1
# beta = 50 * second**-1 # reverse rates of chemical reactions
# eqs = '''
# dVpre/dt = -100 * mV / tau : volt
# T = Tmax / (1 + exp(-((Vpre - Vthr)/omega))) : mM
# dr/dt = alpha * T * (1 - r) - beta * r / tau : mM
# '''
# G = NeuronGroup(1, eqs, method='euler')
# G.Vpre = -10 * mV
# M = StateMonitor(G, [ 'T', 'Vpre', 'r'], record=True)
# run(5*ms)

# r = .01
# alpha * T * (1 - r) - beta * r