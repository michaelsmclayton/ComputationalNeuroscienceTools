import os
import numpy as np
import neuron
import LFPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pprint; pp = pprint.PrettyPrinter(depth=10).pprint
from neuron import h, gui
from helperFunctions import getData
h("forall delete_section()")

# ----------------------------------------------------
# Load model
# ----------------------------------------------------

# Download data (if not present)
if not(os.path.isdir('./SS-cortex/')):
    getData()

# Define template directory, and load compiled .mod files (from NEURON file)
templateDirectory = './SS-cortex/'
h.nrn_load_dll(templateDirectory + 'x86_64/.libs/libnrnmech.so') # Load compiled .mod files

# Turn on calculation of i_membrane (i.e. transmembrane currents)
cv = h.CVode()
cv.use_fast_imem(True)

# Load model
h.chdir("./SS-cortex") # fix path
h.load_file("mosinit.hoc")
# h("load_file(\"./SS-cortex/sj3-cortex.hoc\")")
# h("load_file(\"./SS-cortex/wiring_proc.hoc\")")

# Load specific configuration
h("suprathresh()")

# Change layer 2 pyramidal neuron locations
h("for i=0,9 { PL2[i].position(i*100,0,0) }")

# ----------------------------------------------------
# Define cells of interest and recorders
# ----------------------------------------------------

# Collect cells of interest and setup recorders
cellTypes = ['PL2', 'PL5']
cells = {}; imem_recorders = []
areas, diams = [], []
for cellType in cellTypes:
    currentCells = getattr(h,cellType)
    cells[cellType] = []
    for cellInx in range(len(currentCells)):
        currentCell = currentCells[cellInx]
        cells[cellType].append(currentCell)
        for seg in np.r_[currentCell.soma, currentCell.dend]:
            areas.append(seg(.5).area())
            diams.append(seg(.5).diam)
            imem_recorders.append(h.Vector().record(seg(.5)._ref_i_membrane_))
times = h.Vector().record(h._ref_t) # Time stamp vector

# Record dipoles
dipole_recorders = {}
dipole_names = ['dipoleL2', 'dipoleL5']
for popName in dipole_names:
    dipole_recorders[popName] = []
    for cell in getattr(h,popName):
        for dipole in cell.bd:
            dipole_recorders[popName].append(h.Vector().record(dipole._ref_Qsum))

# Define grid recording electrode
gridLims = {'x': [-400,1100], 'y': [-300,1400]}
X, Y = np.mgrid[gridLims['x'][0]:gridLims['x'][1]:25, gridLims['y'][0]:gridLims['y'][1]:25]
Z = np.zeros(X.shape)
grid_electrode = LFPy.RecExtElectrode(**{
    'sigma' : 0.3,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten()})

# ----------------------------------------------------
# Run model!
# ----------------------------------------------------
h('run()')


# ----------------------------------------------------
# Create dummy cell and LFP/dipole information
# ----------------------------------------------------

# Get segment positions in xyz space
xyz_starts, xyz_ends = np.zeros(shape=(0,3)), np.zeros(shape=(0,3))
getXYZ = lambda seg,loc : [seg.x3d(loc), seg.y3d(loc), seg.z3d(loc)]
for cellType in cellTypes:
    for cell in cells[cellType]:
        cellPos = np.array([cell.x, cell.y, cell.z])
        # Loop over segments
        for seg in np.r_[cell.soma, cell.dend]:
            start,end = [getXYZ(seg,0), getXYZ(seg,1)]
            if not('soma' in seg.name()): # for some reason, the some xyz coordinates already contain cell position
                start += cellPos; end += cellPos; 
            xyz_starts = np.vstack((xyz_starts, start))
            xyz_ends = np.vstack((xyz_ends, end))

# Get dummy cell parameters
dummyCellParams = {}
for i,dim in enumerate(['x','y','z']):
    start, end = xyz_starts[:,i], xyz_ends[:,i]
    dummyCellParams['%sstart' % (dim)] = start
    dummyCellParams['%smid' % (dim)] = start+((end-start)/2)
    dummyCellParams['%send' % (dim)] = end
dummyCellParams['totnsegs'] = len(imem_recorders)
dummyCellParams['imem'] = np.array(imem_recorders)
dummyCellParams['area'] = np.array(areas)
dummyCellParams['diam'] = np.array(diams)
fullNetwork = LFPy.network.DummyCell(**dummyCellParams)
fullNetwork.verbose = False

# Get LFP (reshaping into matrix: x,y,time)
grid_electrode.calc_lfp(cell=fullNetwork)
LFP = grid_electrode.LFP
x, y = X.shape; time = LFP.shape[1]
LFP = np.reshape(LFP,(x,y,time))

# Get dipole information
dipoleL2 = np.array(dipole_recorders['dipoleL2'])
dipoleL5 = np.array(dipole_recorders['dipoleL5'])

# Downsample
downsampleFactor = 10
LFP = LFP[:,:,0::downsampleFactor]
times = np.array(times)[0::downsampleFactor]
dipoleL2 = dipoleL2[:,0::downsampleFactor]
dipoleL5 = dipoleL5[:,0::downsampleFactor]


# ----------------------------------------------------
# Plot results
# ----------------------------------------------------

fig, axs = plt.subplots(ncols=1, nrows=5)
fig.set_figheight(8); fig.set_figwidth(8)
gs = axs[0].get_gridspec()

# Plot dipoles
ax0 = fig.add_subplot(gs[0])
l2dipole_mean, l5dipole_mean = np.mean(dipoleL2,axis=0), np.mean(dipoleL5,axis=0)
ax0.plot(times, l2dipole_mean, label="L2 dipole")
ax0.plot(times, l5dipole_mean, label="L5 dipole")
ax0.plot(times, l2dipole_mean+l5dipole_mean, '--', color='k', alpha=.5, label="Combined dipole")
ax0.legend(frameon=False, loc='lower left')
plt.box(False)
line, = ax0.plot([0,0],[-200,200], color='k')

# Show LFP animation
ax1 = fig.add_subplot(gs[1:])
def showNeurons(ax):
    for xStart,xEnd,yStart,yEnd,diam in zip(dummyCellParams['xstart'],dummyCellParams['xend'],dummyCellParams['ystart'],dummyCellParams['yend'],dummyCellParams['diam']):
        ax.plot([xStart,xEnd], [yStart,yEnd], linewidth=diam/8, color='k', alpha=.5)
lfpPlot = ax1.imshow(np.rot90(LFP[:,:,0]), extent=np.r_[gridLims['x'],gridLims['y']], vmin=np.min(LFP), vmax=np.max(LFP))#, cmap='gist_gray')
showNeurons(ax1)
for ax in axs: ax.axis('off')
# ax0.axis('off')
ax1.axis('off')

# Define animation function
def updatefig(t):
    # print(np.round(times[t],0),h.tstop)
    lfpPlot.set_data(np.rot90(LFP[:,:,int(t)]))
    line.set_xdata([times[t],times[t]])
    return lfpPlot, line

# Animate
ani = FuncAnimation(fig, updatefig, frames=range(LFP.shape[2]), interval=1)
plt.show()
