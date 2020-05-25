import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from tqdm import tqdm
from neuprint import Client, NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapse_connections

# Setup client
print('Creating client...')
with open('../authToken', 'r') as file:
    authToken = file.read()
c = Client(server='neuprint.janelia.org', dataset='hemibrain:v1.0.1', token=authToken)

# Pick save/load functions
def savePickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
def loadPickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Get neurons and synapses
connectionsFile = 'connections.pkl'
print('Getting neurons and synapses...')
if not(os.path.isfile(connectionsFile)):
    fanshapedNeurons = NC(status='Traced', type='FB4Y', cropped=False, inputRois=['EB'], min_roi_inputs=100, min_pre=400)
    ellipsoid_synapses = synapse_criteria = SC(rois='EB', primary_only=True)
    ellipsoid_conns = fetch_synapse_connections(source_criteria=fanshapedNeurons, target_criteria=None, synapse_criteria=ellipsoid_synapses)
    savePickle(connectionsFile, ellipsoid_conns)
else:
    ellipsoid_conns = loadPickle(connectionsFile)

# Get skeletons
skeletonFile = 'skeletons.pkl'
print('Getting skeletons...')
if not(os.path.isfile(skeletonFile)):
    skeletons = []; # max_number = 100
    bodyIds = ellipsoid_conns['bodyId_post'].unique()#[0:max_number]
    for i, bodyId in tqdm(enumerate(bodyIds)):
        try:
            s = c.fetch_skeleton(bodyId, format='pandas')
            skeletons.append(s)
        except:
            print('; Skeleton %s cannot be downloaded' % (i))
    savePickle(skeletonFile, skeletons)
else:
    skeletons = loadPickle(skeletonFile)

# Plot skeletons
fig = plt.figure(figsize=(8,12),facecolor='black')
ax = fig.gca(projection='3d')
[ax.plot(skel.x, skel.y, skel.z, linewidth=.25) for skel in skeletons]
ax.scatter(ellipsoid_conns.x_pre, ellipsoid_conns.y_pre, ellipsoid_conns.z_pre, s=.01, c='white')
ax.set_facecolor((0,0,0))
plt.axis('off')

# Define animation and run
def animate_func(angle):
    ax.view_init(10, angle*5)
    return ax
anim = animation.FuncAnimation(fig, animate_func, interval=1) # 1000 / fps)
plt.show()