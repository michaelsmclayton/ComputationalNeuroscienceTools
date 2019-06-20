from brian2 import * # Import Brian
from matplotlib.pyplot import *
import math

# Taken from 'Causal Role of Thalamic Interneurons in
# Brain State Transitions- A Study Using a Neural Mass
# Model Implementing Synaptic Kinetics'

# General parameters
tau = 10 * ms
equations = []

# Membrane voltage dynamics
membraneVoltageEquation = '''
    dVpre/dt = .01*(Vrest-Vpre) / tau : 1
'''

# -----------------------------
# Equations for AMPA and GABAa channels
# -----------------------------

# Equation 1
Tmax = 1
Vthr = -32 # Point where output reaches .5 mM
omega = 3.8 # steepness of the sigmoid function
equation1 = '''
    T = Tmax / (1 + exp(-((Vpre - Vthr)/omega))) : 1
'''
equations.append(equation1)

# Equation 2
alpha = 1000**-1 # * mM**-1 * second**-1
beta = 50**-1 # * second**-1 # reverse rates of chemical reactions
Vrest = -70
equation2 = '''
    dr/dt = ( alpha * T * (1 - r) - beta * r ) / tau : 1
    diff = ( alpha * T * (1 - r) - beta * r ) : 1
'''
equations.append(equation2)

# # Combined equation for AMPA and GABAa synapses
# eqs = equation1 + equation2
# G = NeuronGroup(1, eqs, method='euler')
# G.r = .001
# G.Vpre = 0
# M = StateMonitor(G, [ 'T', 'Vpre', 'r', 'diff'], record=True)
# run(2000*ms)

# subplot(2,2,1)
# plot(M.t/ms, M.Vpre[0])
# ylabel('Vpre')
# subplot(2,2,2)
# plot(M.t/ms, M.T[0])
# ylabel('T')
# subplot(2,2,3)
# plot(M.t/ms, M.r[0])
# ylabel('r')
# subplot(2,2,4)
# plot(M.t/ms, M.diff[0])
# ylabel('dr/dt')
# savefig('equation1and2_results.png')

# -----------------------------
# Equations for GABAb channel
# -----------------------------

# Equation 3 - metabotropic synapses (AMPA, GABAa)
''' Get the fraction of activated GABAB receptors'''
alpha1 = 10**-1 # forward rates of chemical reactions
beta1 = 25**-1 # reverse rates of chemical reactions
equation3 = '''
    dR/dt = ( alpha1 * T * (1 - R) - beta1 * R ) / tau : 1
'''
equations.append(equation3)

# Equation 4 - secondary messenger concentration
'''Get the concentration of the activated G-protein'''
'''dX(t)/dt'''
alpha2 = 15**-1
beta2 = 5**-1
equation4 = '''
    dX/dt = ( alpha2 * R - beta2 * X ) / tau : 1
'''
equations.append(equation4)

# Equation 5 - get fraction of open ion channels
'''Get the fraction of open ion channels caused by binding of [X]'''
n = 4
Kd = 100
equation5 = '''
    r = X**n / (X**n + Kd) : 1
'''
equations.append(equation5)

# Combined equation for GABAb synapses
eqs = membraneVoltageEquation
eqsOfInterest = [0,2,3,4]
for eq in eqsOfInterest:
    eqs += equations[eq]
G = NeuronGroup(1, eqs, method='euler')
M = StateMonitor(G, [ 'Vpre', 'R', 'X', 'r'], record=True)
run(2000*ms)

subplot(2,2,1)
plot(M.t/ms, M.Vpre[0])
ylabel('Vpre')
subplot(2,2,2)
plot(M.t/ms, M.R[0])
ylabel('R')
subplot(2,2,3)
plot(M.t/ms, M.X[0])
ylabel('X')
subplot(2,2,4)
plot(M.t/ms, M.r[0])
ylabel('r')
# show()
savefig('equation3to5_results.png')



# # # # Equation 1 - neurotransmitter concentrations

# # tau = 10 * ms
# # omega = 3.8 * mV # steepness of the sigmoid function
# # Tmax = 1 * mM
# # Vthr = -32 * mV # Point where output reaches .5 mM
# # alpha = 1000 * mM**-1 * second**-1
# # beta = 50 * second**-1 # reverse rates of chemical reactions
# # eqs = '''
# # dVpre/dt = -100 * mV / tau : volt
# # T = Tmax / (1 + exp(-((Vpre - Vthr)/omega))) : mM
# # dr/dt = alpha * T * (1 - r) - beta * r / tau : mM
# # '''
# # G = NeuronGroup(1, eqs, method='euler')
# # G.Vpre = -10 * mV
# # M = StateMonitor(G, [ 'T', 'Vpre', 'r'], record=True)
# # run(5*ms)

# # r = .01
# # alpha * T * (1 - r) - beta * r