

import numpy as np
import neuron
import LFPy
import matplotlib.pyplot as plt 
import pprint; pp = pprint.PrettyPrinter(depth=6).pprint
from matplotlib.animation import FuncAnimation
from neuron import h

# Define template directory, and load compiled .mod files (from NEURON file)
templateDirectory = '../../../NEURON/code/hoc&mod/bushSejnowski/'
h.nrn_load_dll(templateDirectory + '/x86_64/.libs/libnrnmech.so') # Load compiled .mod files

# -----------------------------------------
# Create cell from .hoc  template
# -----------------------------------------
cellParameters = dict(
    morphology='empty.hoc', # Note: it seems quite strange and unsatisfying that this works
    templatefile=templateDirectory+'cells/L5Pyramidal.hoc',
    templatename='Layer5_pyr',
    templateargs=None, v_init=-75,
    # nsegs_method='fixed_length', # To determine segment length
    # max_nsegs_length= np.inf
)
cell = LFPy.NetworkCell(tstart=0., tstop=60.,**cellParameters)
# cell.set_pos(0,0,0)

# -----------------------------------------
# Define electrodes
# -----------------------------------------

# Define stimulus device
iclamp = LFPy.StimIntElectrode(
    cell=cell, idx=0, pptype='IClamp',
    amp=1,dur=100., delay=5.,
    record_current=True)

# Define grid recording electrode
gridLims = {'x': [-450,300], 'y': [-350,2000]}
X, Y = np.mgrid[gridLims['x'][0]:gridLims['x'][1]:25, gridLims['y'][0]:gridLims['y'][1]:25]
Z = np.zeros(X.shape)
grid_electrode = LFPy.RecExtElectrode(cell, **{
    'sigma' : 0.3,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten()})


# -----------------------------------------
# Simulate, and plot results
# -----------------------------------------
cell.simulate(rec_vmem=True, rec_imem=True)

# Get LFP (reshaping into matrix: x,y,time)
grid_electrode.calc_lfp()
LFP = grid_electrode.LFP
x, y = X.shape; time = LFP.shape[1]
LFP = np.reshape(LFP,(x,y,time))

# Figure
fig, axs = plt.subplots(ncols=1, nrows=5)
fig.set_figheight(8); fig.set_figwidth(3)
gs = axs[0].get_gridspec()
ax0 = fig.add_subplot(gs[1:])
lfpPlot = ax0.imshow(np.rot90(LFP[:,:,0]), extent=np.r_[gridLims['x'],gridLims['y']], vmin=np.min(LFP), vmax=np.max(LFP), cmap='gist_gray')
for xStart,xEnd,yStart,yEnd,diam in zip(cell.xstart,cell.xend,cell.ystart,cell.yend,cell.diam):
    ax0.plot([xStart,xEnd], [yStart,yEnd], linewidth=diam/8, color='k', alpha=.8)
ax1 = fig.add_subplot(gs[0])
ax1.plot(cell.vmem.T, color='k', alpha=.1)
line, = ax1.plot([0,0],[np.min(cell.vmem), np.max(cell.vmem)], color='k')
for ax in axs: ax.axis('off')
ax0.axis('off'); ax1.axis('off')

# Define animation function
def updatefig(t):
    # print(cell.tvec[t])
    lfpPlot.set_data(np.rot90(LFP[:,:,int(t)]))
    line.set_xdata([t,t])
    return lfpPlot,line

# Animate
ani = FuncAnimation(fig, updatefig, frames=range(LFP.shape[2]), interval=2)
plt.show()


# cell.get_idx('Layer5_pyr[0].soma')