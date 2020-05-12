

import numpy as np
import neuron
import LFPy
import matplotlib.pyplot as plt
import pprint; pp = pprint.PrettyPrinter(depth=6).pprint
from matplotlib.animation import FuncAnimation
from neuron import h

# Simulation parameters
global dt; dt = .1
simulationLength = 1000
timepoints = int(simulationLength/dt)+1

# Define template directory, and load compiled .mod files (from NEURON file)
templateDirectory = '../../../NEURON/code/hoc&mod/bushSejnowski/'
h.nrn_load_dll(templateDirectory + '/x86_64/.libs/libnrnmech.so') # Load compiled .mod files

# -----------------------------------------
# Define neural network (creating cells from .hoc templates)
# -----------------------------------------

# --------------------------------------
# Get cells (LFPy.NetworkCell)
# --------------------------------------

# Define cells
def getCellParams(file,name):
    return dict(
        morphology='empty.hoc', # Note: it seems quite strange and unsatisfying that this works
        templatefile='%scells/%s.hoc' % (templateDirectory, file),
        templatename = name,
        templateargs=None, v_init=-75,
        delete_sections = False, # important so that all sections are kept when creating different populations
        dt = dt,
        # pt3d=True,
        # nsegs_method='fixed_length', # To determine segment length
        # max_nsegs_length= np.inf
    )
L2params = getCellParams(file='L2Pyramidal',name='Layer2_pyr')
L5params = getCellParams(file='L5Pyramidal',name='Layer5_pyr')


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
        rotation_args=dict(x=np.pi/2, y=0),
    )
    network.create_population(name=name, POP_SIZE=N, **populationParameters)

# Add populations to netowkr
addPopulation(network, L2params, 1, 'L2pop')
addPopulation(network, L5params, 1, 'L5pop')


# Rotate all cells
network.populations['L2pop'].cells[0].set_rotation(**{'x': -.2*np.pi, 'y': 0})
network.populations['L5pop'].cells[0].set_rotation(x=180,y=0,z=0)
network.populations['L2pop'].cells[0].set_pos(x=-100,y=500)
network.populations['L5pop'].cells[0].set_pos(x=00,y=0)

# -----------------------------------------
# Define electrodes
# -----------------------------------------

# Define stimulus device
def makeStimulus(cell):
    return LFPy.StimIntElectrode(
    cell=cell, idx=0, pptype='IClamp',
    amp=5,dur=100., delay=5.,
    record_current=True)
makeStimulus(network.populations['L2pop'].cells[0])
makeStimulus(network.populations['L5pop'].cells[0])

# Define grid recording electrode
gridLims = {'x': [-450,300], 'y': [-350,2000]}
X, Y = np.mgrid[gridLims['x'][0]:gridLims['x'][1]:25, gridLims['y'][0]:gridLims['y'][1]:25]
Z = np.zeros(X.shape)
grid_electrode = LFPy.RecExtElectrode(**{
    'sigma' : 0.3,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten()})

# -----------------------------------------
# Simulate, and plot results
# -----------------------------------------
network.simulate(rec_vmem=True, rec_imem=True)

# Make dummy cell
totnsegs = 0; imem = np.empty((0,timepoints))
xstart, ystart, zstart = [np.empty(0) for i in range(3)]
xmid, ymid, zmid = [np.empty(0) for i in range(3)]
xend, yend, zend = [np.empty(0) for i in range(3)]
diam, area = [np.empty(0) for i in range(2)]
for pop in network.populations.keys():
    for cell in network.populations[pop].cells:
        totnsegs += cell.totnsegs
        imem = np.vstack((imem, cell.imem))
        # for dataType in ['xstart', 'ystart', 'zstart', 'xmid', 'ymid', 'zmid', 'xend', 'yend', 'zend', 'diam', 'area']:
        #     exec('np.hstack((%s,cell.%s)))' % (dataType, dataType))
        xstart = np.hstack((xstart,cell.xstart))
        ystart = np.hstack((ystart,cell.ystart))
        zstart = np.hstack((zstart,cell.zstart))
        xmid = np.hstack((xmid,cell.xmid))
        ymid = np.hstack((ymid,cell.ymid))
        zmid = np.hstack((zmid,cell.zmid))
        xend = np.hstack((xend,cell.xend))
        yend = np.hstack((yend,cell.yend))
        zend = np.hstack((zend,cell.zend))
        diam = np.hstack((diam,cell.diam))
        area = np.hstack((area,cell.area))
fullNetwork = LFPy.network.DummyCell(totnsegs=totnsegs, imem=imem, xstart=xstart, ystart=ystart, \
    zstart=zstart, xmid=xmid, ymid=ymid, zmid=zmid, xend=xend, yend=yend, zend=zend, diam=diam, area= area)
fullNetwork.verbose = False

# Get LFP (reshaping into matrix: x,y,time)
grid_electrode.calc_lfp(cell=fullNetwork)
LFP = grid_electrode.LFP
x, y = X.shape; time = LFP.shape[1]
LFP = np.reshape(LFP,(x,y,time))

# Define plotting functions
def showNeuron(cell,ax):
    for xStart,xEnd,yStart,yEnd,diam in zip(cell.xstart,cell.xend,cell.ystart,cell.yend,cell.diam):
        ax.plot([xStart,xEnd], [yStart,yEnd], linewidth=diam/8, color='k', alpha=.8)

# Figure
fig, axs = plt.subplots(ncols=1, nrows=5)
fig.set_figheight(8); fig.set_figwidth(3)
gs = axs[0].get_gridspec()
ax0 = fig.add_subplot(gs[1:])
lfpPlot = ax0.imshow(np.rot90(LFP[:,:,0]), extent=np.r_[gridLims['x'],gridLims['y']], vmin=np.min(LFP), vmax=np.max(LFP), cmap='gist_gray')
showNeuron(network.populations['L2pop'].cells[0],ax0)
showNeuron(network.populations['L5pop'].cells[0],ax0)
# ax1 = fig.add_subplot(gs[0])
# ax1.plot(L5Cell.vmem.T, color='k', alpha=.1)
# line, = ax1.plot([0,0],[np.min(L5Cell.vmem), np.max(L5Cell.vmem)], color='k')
for ax in axs: ax.axis('off')
# ax1.axis('off')
# ax0.axis('off'); 

# Define animation function
def updatefig(t):
    # print(L5Cell.tvec[t])
    lfpPlot.set_data(np.rot90(LFP[:,:,int(t)]))
    # line.set_xdata([t,t])
    return lfpPlot,#line

# Animate
ani = FuncAnimation(fig, updatefig, frames=range(LFP.shape[2]), interval=2)
plt.show()