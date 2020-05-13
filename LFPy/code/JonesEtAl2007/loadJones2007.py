import os
import numpy as np
import neuron
import LFPy
import matplotlib.pyplot as plt
import pprint; pp = pprint.PrettyPrinter(depth=6).pprint
from matplotlib.animation import FuncAnimation
from neuron import h #, gui

# Simulation parameters
global dt; dt = .1
simulationLength = 105
timepoints = int(simulationLength/dt)+1
numberOfCells = 9
xGap = 100 # x gap between neurons

# Download data (if not present)
if not(os.path.isdir('./SS-cortex/')):
    os.system("python3 getData.py")


# Define template directory, and load compiled .mod files (from NEURON file)
templateDirectory = './SS-cortex/'
h.nrn_load_dll(templateDirectory + 'x86_64/.libs/libnrnmech.so') # Load compiled .mod files


# ------------------------------------------------------------------------------------------------
#                       Define neural network (creating cells from .hoc templates)
# ------------------------------------------------------------------------------------------------

# --------------------------------------
# Get cells (LFPy.NetworkCell)
# --------------------------------------

# Define cells
def getCellParams(name):
    return dict(
        morphology='empty.hoc', # Note: it seems quite strange and unsatisfying that this works
        templatefile='%s/sj3-cortex.hoc' % (templateDirectory),
        templatename = name,
        templateargs=None, v_init=-75,
        delete_sections = False, # important so that all sections are kept when creating different populations
        dt = dt,
        # pt3d=True,
        # nsegs_method='fixed_length', # To determine segment length
        # max_nsegs_length= np.inf
    )
L2params = getCellParams(name='Layer2_pyr')
L5params = getCellParams(name='Layer5_pyr')


# --------------------------------------
# Build network (LFPy.Network(); composed of LFPy.NetworkPopulation)
# --------------------------------------

# Create network
network = LFPy.Network(tstop=simulationLength)

# Define populations
def addPopulation(network, cellParams, N, name):
    populationParameters = dict(
        Cell=LFPy.NetworkCell,
        cell_args=cellParams,
        pop_args=dict(radius=1.,loc=0.,scale=0),
        rotation_args=dict(x=0, y=0))
    network.create_population(name=name, POP_SIZE=N, **populationParameters)

# Add populations to network
addPopulation(network, L2params, numberOfCells, 'L2pop')
addPopulation(network, L5params, numberOfCells, 'L5pop')

# Rotate all cells
for pop in network.populations.keys():
    rotations = network.populations[pop].rotations
    for i, cell in enumerate(network.populations[pop].cells):
        currentZRot = rotations[i] # correct for automatic, random z-axis rotation of cells
        cell.set_rotation(x=2*np.pi, z=-currentZRot)

# Position cells
for i, cell in enumerate(network.populations['L2pop'].cells):
    cell.set_pos(x=(i*xGap)-100,y=500)
for i, cell in enumerate(network.populations['L5pop'].cells):
    cell.set_pos(x=(i*xGap),y=0)

# -------------------------------------
# Add connectivity
# -------------------------------------

# Define function which returns delays of increasing duration for each synapse
iterator = 0
def progressiveDelay(**kwargs):
    global iterator; iterator += 1
    return np.ndarray(1, dtype=float) + (iterator*.25)

# Define connectivity matrix (Boolean matrix)
connectivity = np.zeros(shape=(numberOfCells,numberOfCells),dtype=np.bool) # Boolean matrix of False values
connections = [[4,i] for i in range(numberOfCells)]
for pre,post in connections:
    connectivity[pre,post] = True

# Connect!
network.connect(pre='L5pop', post='L2pop', connectivity=connectivity, syn_pos_args=dict(section=['dend[3]']), \
    delayfun=progressiveDelay)


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
# for pop in ['L5pop']: # network.populations.keys():
#     for cell in network.populations[pop].cells:
#         makeStimulus(cell)
makeStimulus(network.populations['L5pop'].cells[4])

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

# Get LFP (reshaping into matrix: x,y,time)
grid_electrode.calc_lfp(cell=fullNetwork)
LFP = grid_electrode.LFP
x, y = X.shape; time = LFP.shape[1]
LFP = np.reshape(LFP,(x,y,time))

# Define plotting functions
def showNeuron(cell,ax):
    for xStart,xEnd,yStart,yEnd,diam in zip(cell.xstart,cell.xend,cell.ystart,cell.yend,cell.diam):
        ax.plot([xStart,xEnd], [yStart,yEnd], linewidth=diam/4, color='k', alpha=.8)

# Figure
fig, axs = plt.subplots(ncols=1, nrows=5)
fig.set_figheight(8); fig.set_figwidth(8)
gs = axs[0].get_gridspec()
ax0 = fig.add_subplot(gs[0:])
lfpPlot = ax0.imshow(np.rot90(LFP[:,:,0]), extent=np.r_[gridLims['x'],gridLims['y']], vmin=np.min(LFP), vmax=np.max(LFP), cmap='gist_gray')
for pop in network.populations.keys():
    for cell in network.populations[pop].cells:
        showNeuron(cell,ax0)
showNeuron(network.populations['L5pop'].cells[0],ax0)
#ax1 = fig.add_subplot(gs[0])
#ax1.plot(network.populations['L5pop'].cells[0].vmem.T, color='r', alpha=.1)
#ax1.plot(network.populations['L2pop'].cells[0].vmem.T, color='b', alpha=.1)
# line, = ax1.plot([0,0],[-80,50], color='k')
for ax in axs: ax.axis('off')
# ax1.axis('off')
ax0.axis('off'); 

# Define animation function
def updatefig(t):
    # print(L5Cell.tvec[t])
    lfpPlot.set_data(np.rot90(LFP[:,:,int(t)]))
    # line.set_xdata([t,t])
    return lfpPlot,#line

# Animate
ani = FuncAnimation(fig, updatefig, frames=range(LFP.shape[2]), interval=2)
plt.show()