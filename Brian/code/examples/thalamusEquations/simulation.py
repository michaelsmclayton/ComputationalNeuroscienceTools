from brian2 import * # Import Brian
from matplotlib.pyplot import *
from equations import equations, getEquations
from populations import *
from synapses import *
from analyse import plotResult
import pickle

# General parameters
dt = 1 * ms
tau = 1 * ms
integrationMethod = 'rk2' # integrationMethod rk2, rk4
km = 10 * ms # uF/cm2
simulationLength = 40000 * ms

# Remove connections fron IN
# del connections['IN']

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
        meanInput = params['outputMean']
        stdInput = params['outputSTD']
        RET = NeuronGroup(1, threshold='t>0*10*ms',
            reset = 'V = %s + (randn()*%s)' % (meanInput, stdInput),
            model = eqs, method=integrationMethod, dt=dt)
        RET.V = params['outputMean']

    else:
        eqs = getEquations(equations, [7, 8]) + '''
            gLeak = ''' + str(params['gLeak']) + ''' : 1
            Eleak = ''' + str(params['Eleak']) + ''' : 1'''
        globals()[pop] = NeuronGroup(1, threshold='t>0*10*ms', dt=dt, model=eqs, method=integrationMethod)

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
        synapseParams = AMPA_parameters(gSynMax=currentSynapse['gSynMax'])
    elif synapseType == 'GABAa':
        synapseParams = GABAa_parameters(ESynRev=currentSynapse['ESynRev'])
    elif synapseType == 'GABAb':
        synapseParams = GABAb_parameters()

    # Get connection weights:
    connectionWeight = '''
        Cuvw = ''' + str(currentSynapse['connectionStrength']) + ''' : 1
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
                on_post = '''Ipsp += Ipsp_syn''', dt=dt, method=integrationMethod)

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
seed(int(rand()*10000))
run(simulationLength)
# print("Time is %s" % (t))


#-------------------------------------------
# Combine results into dictionary
#-------------------------------------------
populationData = {}
areasToPlot = ['RET', 'TCR', 'IN', 'TRN']
for index, pop in enumerate(areasToPlot):
    populationData[pop] = globals()[pop+'data'].V[0]
populationData['times'] = globals()[pop+'data'].t/ms

#-------------------------------------------
# Store results
#-------------------------------------------
f = open("simulationResults_new.pkl","wb")
pickle.dump(populationData, f)
f.close()

# Plot result
plotResult(populationData)