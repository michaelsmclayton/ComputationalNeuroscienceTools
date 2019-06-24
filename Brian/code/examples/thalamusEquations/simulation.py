from brian2 import * # Import Brian
from matplotlib.pyplot import *
from equations import equations, getEquations
from populations import *
from synapses import *
import pickle

# General parameters
tau = 1 * ms
integrationMethod = 'rk2' # integrationMethod rk2, rk4
km = 10 * ms # uF/cm2
simulationLength = 25000 * ms

#-------------------------------------------
# # Create populations (and recorders)
#-------------------------------------------

# Set variables to record
variablesToRecord = ['V']

# Loop over populations
for pop in populations:

    # Make populations
    params = populations[pop]['customParameters'] # Get parameters
    if pop == 'RET': # retina
        eqs = '''V : 1'''
        meanInput = str(params['outputMean'])
        stdInput = str(params['outputSTD'])
        RET = NeuronGroup(1, threshold='t>0*10*ms',
            reset='V = '+meanInput+' + (('+stdInput+'*rand())-1)',
            model=eqs, method=integrationMethod)
        RET.V = params['outputMean']

    else:
        eqs = getEquations(equations, [7, 8]) + '''
            gLeak = ''' + str(params['gLeak']) + ''' : 1
            Eleak = ''' + str(params['Eleak']) + ''' : 1'''
        globals()[pop] = NeuronGroup(1, threshold='t>0*10*ms', model=eqs, method=integrationMethod)

        # Set initial conditions
        globals()[pop].V = params['Vrest']

    # Create recording devices
    globals()[pop+'data'] = StateMonitor(globals()[pop], variablesToRecord, record=True)


#-------------------------------------------
# Create synapses
#-------------------------------------------

# Define function to extract synapse equations
def getSynapseEquations(currentSynapse):

    # Get base equations
    synapseType = currentSynapse['synType']
    if synapseType == 'AMPA' or synapseType == 'GABAa':
        synapseEqs = getEquations(equations, [1, 2, 6]) # ionotropic
    elif synapseType == 'GABAb':
        synapseEqs = getEquations(equations, [1, 3, 4, 5, 6]) # second-order
    
    # Get synapse parameters
    if synapseType == 'AMPA':
        synapseParams = AMPA_parameters(gSyncMax=currentSynapse['gSyncMax'])
    elif synapseType == 'GABAa':
        synapseParams = GABAa_parameters(ESynRev=currentSynapse['ESynRev'])
    elif synapseType == 'GABAb':
        synapseParams = GABAb_parameters()

    # Get connection weights:
    connectionWeight = '''
        Cuvw = ''' + str(currentSynapse['connectionStrength']/100) + ''' : 1
    '''

    # Return equations
    return synapseEqs + synapseParams + connectionWeight


# Make all connections
for efferent in connections:
    for afferent in connections[efferent]:
        
        # Create synapse
        currentSynapse = connections[efferent][afferent]
        eqs = getSynapseEquations(currentSynapse) # Get synapse equations
        efferentPop = globals()[efferent]
        afferentPop = globals()[afferent]
        globals()[efferent + '_' + afferent] = \
            Synapses(efferentPop, afferentPop, model=eqs,
                on_post = '''Ipsp += Ipsp_syn''', method=integrationMethod)

        # Connect populations with synapses
        globals()[efferent + '_' + afferent].connect()

        # Set initial conditions
        synapseType = currentSynapse['synType']
        if synapseType == 'AMPA' or synapseType == 'GABAa':
            globals()[efferent + '_' + afferent].r = .001
        elif synapseType == 'GABAb':
            globals()[efferent + '_' + afferent].R = .001
            globals()[efferent + '_' + afferent].X = .001


#-------------------------------------------
# Run simulation!
#-------------------------------------------
BrianLogger.log_level_debug()
seed(123)
run(simulationLength)


#-------------------------------------------
# Analyse / plot results
#-------------------------------------------
populationData = {}
figure()
areasToPlot = ['RET', 'TCR', 'IN', 'TRN']
plotWindow = [150000, 250000]
for index, pop in enumerate(areasToPlot):
    populationData[pop] = globals()[pop+'data'].V[0]
    times = globals()[pop+'data'].t/ms
    subplot(len(areasToPlot), 1, index+1)
    plot(times[plotWindow[0]:plotWindow[1]], populationData[pop][plotWindow[0]:plotWindow[1]], linewidth=.5)
    ylabel(pop)
    # xlim(15000, 25000)

show()

# #-------------------------------------------
# # Store results
# #-------------------------------------------
# with open('simulationResults.pickle', 'wb') as handle:
#     pickle.dump(populationData, handle, protocol=pickle.HIGHEST_PROTOCOL)

# figure()
# nRows = 2; nCols = 2
# subplot(nRows, nCols, 1)
# plot(M1.t/ms, M1.V[0])
# subplot(nRows, nCols, 2)
# plot(M1.t/ms, M2.V[0])
# subplot(nRows, nCols, 3)
# plot(M1.t/ms, M1.Ipsp[0])
# subplot(nRows, nCols, 4)
# plot(M1.t/ms, M2.Ipsp[0])
# show()
# # savefig('current.png')