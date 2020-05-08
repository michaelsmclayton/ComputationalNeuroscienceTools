import LFPy
import numpy as np
import os
from urllib.request import urlopen
import ssl
import zipfile
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from os.path import join
from matplotlib.animation import FuncAnimation
# code adapted from https://github.com/LFPy/LFPy/blob/master/examples/LFPy-example-3.ipynb

# ------------------------------------------------------------------
# Get data
# ------------------------------------------------------------------

# Get the model files (https://senselab.med.yale.edu/ModelDB/ShowModel?model=2488#tabs-2)
if not os.path.isfile(join('cells', 'cells', 'j4a.hoc')):
    modelLink = 'http://senselab.med.yale.edu/ModelDB/eavBinDown.asp?o=2488&a=23&mime=application/zip'
    u = urlopen(modelLink, context=ssl._create_unverified_context())
    # Save data to .zip
    localFile = open('patdemo.zip', 'wb')
    localFile.write(u.read())
    localFile.close()
    # Unzip file
    myzip = zipfile.ZipFile('patdemo.zip', 'r')
    myzip.extractall('.')
    myzip.close()


# ------------------------------------------------------------------
# Define cell
# ------------------------------------------------------------------

# Define cell parameters
cell_parameters = {
    'morphology' : join('cells', 'cells', 'j4a.hoc'), # from Mainen & Sejnowski, J Comput Neurosci, 1996
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150.,        # axial resistance
    'v_init' : -65.,    # initial crossmembrane potential
    'passive' : True,   # turn on NEURONs passive mechanism for all sections
    'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -65},
    'nsegs_method' : 'lambda_f', # spatial discretization method
    'lambda_f' : 100.,           # frequency where length constants are computed
    'dt' : 2.**-3,      # simulation time step size
    'tstart' : 0.,      # start time of simulation, recorders start at t=0
    'tstop' : 40.,     # stop simulation at 100 ms.
}

# Create cell
cell = LFPy.Cell(**cell_parameters)

# Align cell
cell.set_rotation(x=4.99, y=-4.33, z=3.14)


# ------------------------------------------------------------------
# Define synapse (i.e. input stimulus)
# ------------------------------------------------------------------

# Define synapse parameters
def getSynapseParams(inputLoc, weight):
    return {
    'idx' : cell.get_closest_idx(inputLoc['x'], inputLoc['y'], inputLoc['z']),
    'e' : 0.,                   # reversal potential
    'syntype' : 'ExpSyn',       # synapse type
    'tau' : 5.,                 # synaptic time constant
    'weight' : weight,            # synaptic weight
    'record_current' : True,    # record synapse current
    }

# Create synapse and set time of synaptic input
def makeSynapse(inputLoc,weight,inputTimes):
    synapseParams = getSynapseParams(inputLoc,weight)
    synapse = LFPy.Synapse(cell, **synapseParams)
    synapse.set_spike_times(np.array(inputTimes))

makeSynapse({'x': -200., 'y': 0., 'z': 800.}, .004, [10.,20.,30.])
makeSynapse({'x': 300., 'y': 0., 'z': 1000.}, .002, [15.,25.,35.])
makeSynapse({'x': 100., 'y': 0., 'z': -100.}, .001, [12.5,22.5,32.5])


# ------------------------------------------------------------------
# Create measurement grid (to measure field) and recording electrodes
# ------------------------------------------------------------------

# Create a grid of measurement locations, in (mum)
X, Z = np.mgrid[-700:701:25, -400:1201:25]
Y = np.zeros(X.shape)

# Define electrode parameters
grid_electrode_parameters = {
    'sigma' : 0.3,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten()
}

# Define electrode parameters
point_electrode_parameters = {
    'sigma' : 0.3,  # extracellular conductivity
    'x' : np.array([-130., -220.]),
    'y' : np.array([   0.,    0.]),
    'z' : np.array([   0.,  700.]),
}

# Create electrode objects
grid_electrode = LFPy.RecExtElectrode(cell,**grid_electrode_parameters)
point_electrode = LFPy.RecExtElectrode(cell,**point_electrode_parameters)


# ------------------------------------------------------------------
# Run simulation and pre-process results
# ------------------------------------------------------------------

# Run simulation, electrode object argument in cell.simulate
cell.simulate(rec_imem=True)

# Calculate LFPs
grid_electrode.calc_lfp()
point_electrode.calc_lfp()

# Get LFP (reshaping into matrix: x,y,time)
LFP = grid_electrode.LFP
x, y = X.shape
time = LFP.shape[1]
LFP = np.reshape(LFP,(x,y,time))


# ------------------------------------------------------------------
# Plot simulation output
# ------------------------------------------------------------------

fig = plt.figure(dpi=160)
ax = plt.subplot('111')

# Plot LFP
lfpPlot = ax.imshow(np.rot90(LFP[:,:,0]),extent=[-700,701,-400,1201], cmap="bwr", vmin=np.min(LFP), vmax=np.max(LFP))
# contour = np.abs(LFP[:,:,200])
# im = ax.contour(X, Z, np.log10(contour), 50, cmap='inferno', zorder=-2)

# plot morphology
zips = []
for x, z in cell.get_idx_polygons():
    zips.append(list(zip(x, z)))
polycol = PolyCollection(zips, edgecolors='none', facecolors='k')
ax.add_collection(polycol)
ax.plot([350, 450], [-200, -200], 'k', lw=1, clip_on=False)
ax.text(400, -150, r'100$\mu$m', va='center', ha='center')
ax.axis('off')

# Define animation function
def updatefig(t):
    lfpPlot.set_data(np.rot90(LFP[:,:,int(t)]))
    return lfpPlot,

# Animate
ani = FuncAnimation(fig, updatefig, frames=range(LFP.shape[2]), interval=1)
plt.show()






