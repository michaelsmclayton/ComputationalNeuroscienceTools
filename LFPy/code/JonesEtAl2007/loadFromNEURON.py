import os
import numpy as np
import neuron
import LFPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pprint; pp = pprint.PrettyPrinter(depth=10).pprint
from neuron import h, gui
from helperFunctions import getData
h("forall delete_section()")

# ----------------------------------------------------
# Load model and run simulation
# ----------------------------------------------------

# Define template directory, and load compiled .mod files (from NEURON file)
templateDirectory = './SS-cortex/'
h.nrn_load_dll(templateDirectory + 'x86_64/.libs/libnrnmech.so') # Load compiled .mod files

# Load original model
h("load_file(\"./SS-cortex/sj3-cortex.hoc\")")
h("load_file(\"./SS-cortex/wiring_proc.hoc\")")
# h('chdir("./SS-cortex")') # fix path
# h('load_file("mosinit.hoc")')
# # Load specific instantiation
# h('suprathresh()')

# Turn on calculation of i_membrane (i.e. transmembrane currents)
cv = h.CVode()
cv.use_fast_imem(True)

# Set segments to record
segToRecord = h.PL2[0].soma(0)
imem = h.Vector().record(segToRecord._ref_i_membrane_) # also "_ref_v"
t = h.Vector().record(h._ref_t) # Time stamp vector

# Run model!
h('run()')


# ----------------------------------------------------
# Interrogate results
# ----------------------------------------------------

# Define cell types of interest
cellTypes = ['PL2', 'PL5']

# Collect all cells
cells = {}
for cellType in cellTypes:
    currentCells = getattr(h,cellType)
    cells[cellType] = []
    for cellInx in range(len(currentCells)):
        cells[cellType].append(currentCells[cellInx])

# Get segment positions in xyz space
xyz_starts, xyz_ends = np.zeros(shape=(0,3)), np.zeros(shape=(0,3))
getXYZ = lambda seg,loc : [seg.x3d(loc), seg.y3d(loc), seg.z3d(loc)]
for cellType in cellTypes:
    for cell in cells[cellType]:
        cellPos = np.array([cell.x, cell.y, cell.z])
        # Loop over segments
        for seg in np.r_[cell.soma, cell.dend]:
            xyz_starts = np.vstack((xyz_starts, cellPos+getXYZ(seg,0)))
            xyz_ends = np.vstack((xyz_ends, cellPos+getXYZ(seg,1)))

# Plot cells
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(xyz_starts)):
    ax.plot([xyz_starts[i,2],xyz_ends[i,2]], [xyz_starts[i,0],xyz_ends[i,0]], [xyz_starts[i,1],xyz_ends[i,1]], color='k')
ax.view_init(1, 0)
plt.show()

# Plot imem results
plt.plot(t,imem)
plt.show()