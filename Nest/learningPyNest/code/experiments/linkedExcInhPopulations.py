#------------------------------------------
# Import dependencies
#------------------------------------------
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy as np
import nest
import pprint
pp = pprint.PrettyPrinter(depth=6)
import os; os.system('clear') # Clear terminal screen

# Create neural populations
populationSize = 3
randomInput_Exc = nest.Create("poisson_generator", params={"rate": 2000.0})
randomInput_Inh = nest.Create("poisson_generator", params={"rate": 1500.0})
popExc = nest.Create("iaf_psc_alpha", populationSize)
popInh = nest.Create("iaf_psc_alpha", populationSize)
multimeter = nest.Create("multimeter", 2, params={"withtime":True, "record_from":["V_m"]})
pp.pprint(nest.GetStatus(popExc))

# Randomise starting voltages
def setRandomValues(population, randParams):
    minVal = randParams['low']
    maxVal = randParams['high']
    randValues = minVal + ((maxVal-minVal)*np.random.rand(len(population)))
    nest.SetStatus(population, randParams['label'], randValues)
setRandomValues(popExc, randParams={'label': 'V_m', 'low': -70, 'high': -55})
setRandomValues(popInh, randParams={'label': 'V_m', 'low': -70, 'high': -55})
setRandomValues(popInh, randParams={'label': 'I_e', 'low': 100, 'high': 200})

# Make connections
nest.Connect(randomInput_Exc, popExc, "all_to_all", syn_spec={"weight":30.0, "delay":1.0})
nest.Connect(randomInput_Inh, popInh, "all_to_all", syn_spec={"weight":10.0, "delay":1.0})
nest.Connect(popExc, popInh, "all_to_all", syn_spec={"weight":1000.0, "delay":1.0})
nest.Connect(popInh, popExc, "all_to_all", syn_spec={"weight":-100.0, "delay":1.0})
nest.Connect(multimeter, popExc)
nest.Connect(multimeter, popInh)

# Simulate and plot results
nest.Simulate(1000.0)
pylab.figure(1)
multimeterStatus = nest.GetStatus(multimeter)
dmm = multimeterStatus[0]["events"]
populations = [popExc, popInh]
for pop in range(len(populations)):
    pylab.subplot(len(multimeterStatus),1,pop+1); pylab.hold(True)
    neuronsToPlot = np.unique(dmm["senders"][np.isin(dmm["senders"], populations[pop])])
    for i in range(len(neuronsToPlot)):
        indicesOfInterest = np.where(dmm["senders"]==neuronsToPlot[i])
        pylab.plot(dmm["times"][indicesOfInterest], dmm["V_m"][indicesOfInterest])
pylab.savefig("currentFigure.png", dpi=100)