import os
import numpy as np
import neuron
import LFPy
import matplotlib.pyplot as plt
import pprint; pp = pprint.PrettyPrinter(depth=10).pprint
from matplotlib.animation import FuncAnimation
from neuron import h, gui
from helperFunctions import getData, makeInhibitoryCellTemplate
h("forall delete_section()")

# Simulation parameters
global dt; dt = .1
simulationLength = 105
timepoints = int(simulationLength/dt)+1
numberOfCells = 10
xGap = 100 # x gap between neurons

# Download data (if not present)
if not(os.path.isdir('./SS-cortex/')):
    getData()

# Define template directory, and load compiled .mod files (from NEURON file)
templateDirectory = './SS-cortex/'
h.nrn_load_dll(templateDirectory + 'x86_64/.libs/libnrnmech.so') # Load compiled .mod files


# ------------------------------------------------------------------------------------------------
#                       Define neural network (creating cells from .hoc templates)
# ------------------------------------------------------------------------------------------------

# --------------------------------------
# Get cells (LFPy.NetworkCell)
# --------------------------------------

# --------------------------
# Define excitatory cells
# --------------------------

# Function to get general parameters
def getCellParams(name):
    return dict(
        morphology='empty.hoc', # Note: it seems quite strange and unsatisfying that this works
        templatefile='%s/sj3-cortex.hoc' % (templateDirectory),
        templatename = name,
        templateargs=None, v_init=-75,
        delete_sections = False, # important so that all sections are kept when creating different populations
        dt = dt,
        # nsegs_method='fixed_length', # To determine segment length
        # max_nsegs_length= np.inf
    )

# Get excitatory cell parameters
Layer2_pyr_params = getCellParams(name='Layer2_pyr')
Layer5_pyr_params = getCellParams(name='Layer5_pyr')


# --------------------------
# Define inhibitory cells
# --------------------------

# Create inhibitory template file if none exists
inhibTempName = 'Inhib.hoc'
if not(os.path.isfile(inhibTempName)):
    makeInhibitoryCellTemplate(inhibTempName, templateDirectory)

# Define inhibitory cell parameters
inhibitory_params = getCellParams(name='inhib')
inhibitory_params['templatefile'] = inhibTempName
inhibitory_params['Ra'] = 200
inhibitory_params['cm'] = .85


# --------------------------------------
# Build network (LFPy.Network(); composed of LFPy.NetworkPopulation)
# --------------------------------------

# Create network
network = LFPy.Network(tstop=simulationLength,OUTPUTPATH='./')

# Define populations
def addPopulation(network, cellParams, N, name):
    populationParameters = dict(
        Cell=LFPy.NetworkCell,
        cell_args=cellParams,
        pop_args=dict(radius=1.,loc=0.,scale=0),
        rotation_args=dict(x=0, y=0))
    network.create_population(name=name, POP_SIZE=N, **populationParameters)

# Add dummy cell (to load sections from sj3-cortex.hoc, and then delete all sections)
dummy = LFPy.TemplateCell(**Layer2_pyr_params)
h('forall delete_section()')

# Add populations to network
addPopulation(network, Layer2_pyr_params, numberOfCells, 'Layer2_pyr_pop')
addPopulation(network, Layer5_pyr_params, numberOfCells, 'Layer5_pyr_pop')
addPopulation(network, inhibitory_params, 3, 'Layer2_inh_pop')
addPopulation(network, inhibitory_params, 3, 'Layer5_inh_pop')
# h("for i=0,2 { IPL5[i] = new Inhib() }") 

# Rotate all cells
for pop in network.populations.keys():
    rotations = network.populations[pop].rotations
    for i, cell in enumerate(network.populations[pop].cells):
        currentZRot = rotations[i] # correct for automatic, random z-axis rotation of cells
        cell.set_rotation(x=2*np.pi, z=-currentZRot)

# Position cells
for i, cell in enumerate(network.populations['Layer2_pyr_pop'].cells):
    cell.set_pos(x=(i*xGap)-100,y=500)
for i, cell in enumerate(network.populations['Layer5_pyr_pop'].cells):
    cell.set_pos(x=(i*xGap),y=0)
for i, cell in enumerate(network.populations['Layer2_inh_pop'].cells):
    cell.set_pos(x=(i*xGap)+50,y=700)
for i, cell in enumerate(network.populations['Layer5_inh_pop'].cells):
    cell.set_pos(x=(i*xGap)+50,y=200)


# -------------------------------------
# Add connectivity
# -------------------------------------

# Define local network synaptic connection parameters (from Table 2 of original paper)
localNetworkParameters = {
    'L2/3e to L5e': {
        'pre': 'Layer2_pyr_pop',
        'post': 'Layer5_pyr_pop',
        'params': {
            'max_conductance': .00025,
            'weight_space': 3, 'min_delay': 3, 'delay_space': 3}},
    # 'L2/3e to L2/3e': {
    #     'pre': 'Layer2_pyr_pop',
    #     'post': 'Layer2_pyr_pop',
    #     'params': {
    #         'max_conductance': {'AMPA': .001,'NMDA': .0005},
    #         'weight_space': 3, 'min_delay': 1, 'delay_space': 3}},
}

# Define function to return weights and delays (Gaussian based on synaptic distance)
def getWeightsAndDelays(npre, npost, max_conductance, weight_space, min_delay, delay_space):
    weights, delays = np.zeros(shape=(npre,npost)), np.zeros(shape=(npre,npost))
    for i in range(npre):
        for j in range(npost):
            if i==j: weights[i,j], delays[i,j] = max_conductance, min_delay
            else:
                weights[i,j] = max_conductance * np.exp(-(np.abs(i-j))**2 / (weight_space**2))
                delays[i,j] = min_delay * 1/(np.exp(-(np.abs(i-j))**2 / (delay_space**2)))
    return weights, delays

# Define delay and weight functions
def makeIteratorFunction(iteratorType,values,scalar):
    def function(**kwargs):
        global iterators
        value = values[iterators[iteratorType]]
        iterators[iteratorType] += 1
        return np.ndarray(1, dtype=float) + (value*scalar)
    return function

# Loop over all synapse types
for synapseType in localNetworkParameters.keys():
    currentType = localNetworkParameters[synapseType]
    # Get pre and post population information
    prePop, postPop = currentType['pre'], currentType['post']
    npre = len(network.populations[prePop].cells)
    npost = len(network.populations[postPop].cells)
    # Get connections, weights, and delays
    connectivity = np.ones(shape=(npre,npost),dtype=np.bool) # Boolean matrix of True values
    weights, delays = getWeightsAndDelays(npre, npost, **currentType['params'])
    connections = np.array([[[i,j] for j in range(npost)] for i in range(npre)]) # Define connections
    # Reshape to 1-D arrays
    connections = np.reshape(connections, (connections.shape[0]*connections.shape[1],connections.shape[2]))
    weights = np.reshape(weights, (weights.shape[0]*weights.shape[1]))
    delays = np.reshape(delays, (delays.shape[0]*delays.shape[1]))
    # Create delay and weight producing functions
    iterators = {'delay': 0, 'weight': 0}
    delayFun = makeIteratorFunction('delay', delays, scalar=.5)
    weightFun = makeIteratorFunction('weight', weights, scalar=150)
    # Connect!
    network.connect(pre=prePop, post=postPop, connectivity=connectivity, syn_pos_args=dict(section=['soma']), \
        delayfun=delayFun, weightfun=weightFun)


# -----------------------------------------
# Define electrodes
# -----------------------------------------

# Define stimulus device
def makeStimulus(cell):
    return LFPy.StimIntElectrode(
    cell=cell, idx=0, pptype='IClamp',
    amp=1+(.05*np.random.randn()),dur=100., delay=5.,
    record_current=True)

# # Attach simulus electrodes (to all cells)
# for pop in ['Layer5_inh_pop']: # network.populations.keys():
#     for cell in network.populations[pop].cells:
#         makeStimulus(cell)
makeStimulus(network.populations['Layer2_pyr_pop'].cells[4])

# Define grid recording electrode
gridLims = {'x': [-500,(numberOfCells*xGap)+300], 'y': [-600,2200]}
X, Y = np.mgrid[gridLims['x'][0]:gridLims['x'][1]:25, gridLims['y'][0]:gridLims['y'][1]:25]
Z = np.zeros(X.shape)
grid_electrode = LFPy.RecExtElectrode(**{
    'sigma' : 0.3,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten()})

# ------------------------------------------------------------------------------------------------
#                                   Simulate, and plot results
# ------------------------------------------------------------------------------------------------

# Simulate
network.simulate(rec_vmem=True, rec_imem=True)

# Make dummy cell
dummyCellParams = {}; dataTypes = ['xstart', 'ystart', 'zstart', 'xmid', 'ymid', 'zmid', 'xend', 'yend', 'zend', 'diam', 'area']
dummyCellParams['totnsegs'] = 0; dummyCellParams['imem'] = np.empty((0,timepoints))
for dataType in dataTypes: dummyCellParams[dataType] = np.empty(0)
for pop in network.populations.keys():
    for cell in network.populations[pop].cells:
        dummyCellParams['totnsegs'] += cell.totnsegs
        dummyCellParams['imem'] = np.vstack((dummyCellParams['imem'], cell.imem))
        for dataType in dataTypes:
            dummyCellParams[dataType] = np.hstack((dummyCellParams[dataType], getattr(cell,dataType)))
fullNetwork = LFPy.network.DummyCell(**dummyCellParams)
fullNetwork.verbose = False
# _, fullNetwork = network._create_network_dummycell() # Note that this does not seem to work

# Get LFP (reshaping into matrix: x,y,time)
grid_electrode.calc_lfp(cell=fullNetwork)
LFP = grid_electrode.LFP
x, y = X.shape; time = LFP.shape[1]
LFP = np.reshape(LFP,(x,y,time))

# Define plotting functions
def showNeuron(cell,ax):
    for xStart,xEnd,yStart,yEnd,diam in zip(cell.xstart,cell.xend,cell.ystart,cell.yend,cell.diam):
        ax.plot([xStart,xEnd], [yStart,yEnd], linewidth=diam/16, color='k', alpha=.8)

# Figure
fig, axs = plt.subplots(ncols=1, nrows=5)
fig.set_figheight(8); fig.set_figwidth(8)
gs = axs[0].get_gridspec()
ax0 = fig.add_subplot(gs[0:])
lfpPlot = ax0.imshow(np.rot90(LFP[:,:,0]), extent=np.r_[gridLims['x'],gridLims['y']], vmin=np.min(LFP), vmax=np.max(LFP), cmap='gist_gray')
for pop in network.populations.keys():
    for cell in network.populations[pop].cells:
        showNeuron(cell,ax0)
showNeuron(network.populations['Layer5_pyr_pop'].cells[0],ax0)
#ax1 = fig.add_subplot(gs[0])
#ax1.plot(network.populations['Layer5_pyr_pop'].cells[0].vmem.T, color='r', alpha=.1)
#ax1.plot(network.populations['Layer2_pyr_pop'].cells[0].vmem.T, color='b', alpha=.1)
# line, = ax1.plot([0,0],[-80,50], color='k')
for ax in axs: ax.axis('off')
# ax1.axis('off')
ax0.axis('off'); 

# Define animation function
def updatefig(t):
    lfpPlot.set_data(np.rot90(LFP[:,:,int(t)]))
    # line.set_xdata([t,t])
    return lfpPlot,#line

# Animate
ani = FuncAnimation(fig, updatefig, frames=range(LFP.shape[2]), interval=2)
plt.show()








# a = network.populations['Layer5_inh_pop'].cells[0].vmem
# plt.plot(a.T); plt.show()


# # Load original model
# h("load_file(\"./SS-cortex/sj3-cortex.hoc\")")
# h("load_file(\"./SS-cortex/wiring_proc.hoc\")")
# h("load_file(\"./SS-cortex/wiring-config_suprathresh.hoc\")")

# AMPAconnects = h.AMPAconnects
# NMDAconnects = h.NMDAconnects
# GABAAconnects = h.GABAAconnects
# GABABconnects = h.GABABconnects

# for connect in [AMPAconnects,NMDAconnects,GABAAconnects,GABABconnects]:
#     for i, syn in enumerate(connect):
#         # print(np.round(i/len(connect),2), syn.preseg(), syn.postseg(), syn.delay, syn.weight[0])
#         print(np.round(i/len(connect),2), syn.preseg(), syn.postseg(), syn.delay, syn.weight[0])


# $1 = max_weight
# $2 = space_constant
# $3 = compartment
# $4 = receptor_type
# $5 = minimum_delay
# $6 = delay_space_constant
# {weight = max_weight * exp( -(abs(i - j))^2 / ( space_constant^2) )}
# if (i==j) {delay = minimum_delay}else {delay = minimum_delay * 1 / (exp( -(abs(i - j))^2 / ( delay_space_constant^2) ) )}