from brian2 import * # Import Brian
from matplotlib.pyplot import *
from equations import equations, getEquations
from synapses import *

tau = 10 * ms

#-------------------------------------------
# Population
#-------------------------------------------

# Population parameters
km = 10 * ms #uF/cm2
gLeak = 10 / 1000 #* uS/cm2
Eleak = -55 #* mV

# Create population
eqsOfInterest = [7, 8]
eqs = getEquations(equations, eqsOfInterest)
poisson = PoissonGroup(1, np.arange(1)*Hz + 1000*Hz)
population1 = NeuronGroup(1, threshold='t>0*10*ms', model=eqs, method='euler')
population1.V = 10
population2 = NeuronGroup(1, threshold='t>0*10*ms', model=eqs, method='euler')
population2.V = -10
M1 = StateMonitor(population1, ['Ipsp', 'V'], record=True)
M2 = StateMonitor(population2, ['Ipsp', 'V'], record=True)

#-------------------------------------------
# Synapses
#-------------------------------------------

# Create synapses
ionotropicEquations = getEquations(equations, [1, 2, 6])

# Connectivity
Cuvw = 23.6

# # AMPA
#curentEqs = ionotropicEquations + AMPA_parameters(gSyncMax=100) # AMPA_parameters GABAa_parameters
curentEqs = ionotropicEquations + GABAa_parameters(ESyncRev=-75) # AMPA_parameters GABAa_parameters
S1 = Synapses(population1, population2, model=curentEqs, on_post='''Ipsp += Ipsp_syn''', method='euler')
S1.connect()
S1.r = .001

# GABAa
synapseName = 'S2'
curentEqs = ionotropicEquations + GABAa_parameters(ESyncRev=-85) # AMPA_parameters GABAa_parameters
globals()[synapseName] = Synapses(population2, population1, model=curentEqs, on_post='''Ipsp += Ipsp_syn''', method='euler')
globals()[synapseName].connect()
globals()[synapseName].r = .001
# M = StateMonitor(S1, ['V_pre', 'V_post'], record=True)
# S3 = Synapses(poisson, population1, on_pre='V+=0.1')
# S3.connect()

run(1000*ms)

figure()
nRows = 2; nCols = 2
subplot(nRows, nCols, 1)
plot(M1.t/ms, M1.V[0])
subplot(nRows, nCols, 2)
plot(M1.t/ms, M2.V[0])
subplot(nRows, nCols, 3)
plot(M1.t/ms, M1.Ipsp[0])
subplot(nRows, nCols, 4)
plot(M1.t/ms, M2.Ipsp[0])
show()
# savefig('current.png')