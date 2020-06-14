import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pylab as plt
from tqdm import tqdm

# Import neuprint
from neuprint import Client, NeuronCriteria as NC, SynapseCriteria as SC, \
    fetch_synapse_connections, fetch_roi_hierarchy, fetch_neurons

# Define inputs/outputs for each region
analysedRegion = 'PN'
regionsParams = {
    'EB': { # projections to ellipsoid body from fan-shaped body neurons (FB4Y)
        'sourceParams': {'status': 'Traced', 'type': 'FB4Y', 'regex':True,
                    'cropped':False, 'inputRois':['EB'], 'min_roi_inputs':100, 'min_pre':400},
        'synapseParams': {'rois': 'EB', 'primary_only':True},
        'max_neurons': None},
    'MB': { # Kenyon cells (of the mushroom body)
        'sourceParams': {'status': 'Traced', 'type': 'KC.*', 'regex':True, 'cropped':False},
        'synapseParams': None,
        'max_neurons': 300},
    'PN': { # projection neurons from the right antennal lobe
        'sourceParams': {'inputRois': ['AL(R)'], 'outputRois': ['CA(R)'], 'status': 'Traced', 'cropped': False},
        'synapseParams': None,
        'max_neurons': 50}
}

# Setup client
print('Creating client...')
with open('../authToken', 'r') as file:
    authToken = file.read()
c = Client(server='neuprint.janelia.org', dataset='hemibrain:v1.0.1', token=authToken)
# Print regions (primary ROIs marked with '*')
# print(fetch_roi_hierarchy(include_subprimary=True, mark_primary=True, format='text'))

# Function to get body IDs
def getBodyIDs(sourceParams,synapseParams):
    print('Getting neurons/synapses...')
    sourceNeurons = NC(**sourceParams)
    if not(synapseParams==None):
        synapseCriteria = SC(**synapseParams)
        synapses = fetch_synapse_bodyIDs(source_criteria=sourceNeurons, target_criteria=None, synapse_criteria=synapseCriteria)
        bodyIds = synapses['bodyId_post'].unique()#[0:max_number]
    else:
        neuron_df, roi_counts_df = fetch_neurons(sourceNeurons)
        bodyIds = neuron_df.bodyId; synapses = None
    return bodyIds, synapses

# Pick save/load functions
def savePickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
def loadPickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Get neurons and synapses
params = regionsParams[analysedRegion]
bodyIDsFile = '%s_bodyIDs.pkl' % (analysedRegion)
if not(os.path.isfile(bodyIDsFile)):
    bodyIds, synapses = getBodyIDs(params['sourceParams'], params['synapseParams'])
    savePickle(bodyIDsFile, [bodyIds,synapses])
else:
    bodyIds, synapses = loadPickle(bodyIDsFile)

# Taken random subsample of neurons (if required)
if not(params['max_neurons']==None):
    neuronIDs = np.arange(len(bodyIds))
    np.random.shuffle(neuronIDs)
    selectedIDs = neuronIDs[0:params['max_neurons']]
    bodyIds = bodyIds.iloc[selectedIDs]

# Get skeletons
skeletonFile = '%s_skeletons.pkl' % (analysedRegion)
randomColor = lambda : '#'+''.join(['0123456789ABCDEF'[np.random.randint(16)] for i in range(6)])
print('Getting skeletons...')
if not(os.path.isfile(skeletonFile)):
    skeletons = []; # max_number = 100
    for i, bodyId in tqdm(enumerate(bodyIds)):
        try:
            s = c.fetch_skeleton(bodyId, format='pandas')
            s['bodyId'] = bodyId
            s['color'] = randomColor()
            skeletons.append(s)
        except:
            print('; Skeleton %s cannot be downloaded' % (i))
    savePickle(skeletonFile, skeletons)
else:
    skeletons = loadPickle(skeletonFile)


# Load MB neurons for comparison with PN neurons
if analysedRegion == 'PN':
    getRandGrey = lambda : '#' + 'ABCDEF'[np.random.randint(low=0, high=6)]*6
    mb_skeletons = loadPickle('MB_skeletons.pkl')
    neuronIDs = np.arange(len(mb_skeletons))
    np.random.shuffle(neuronIDs)
    # Change colours to white and extract random 50 neurons
    for i in neuronIDs[0:75]:
        color = getRandGrey()
        mb_skeletons[i].color = [color for i in range(len(mb_skeletons[i].color))]
        skeletons.append(mb_skeletons[i])


################################################
# Plotting
################################################

# Import plotting libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Get segments (joining parent/child nodes for plotting as line segments)
segments = [skeletons[i].merge(skeletons[i], 'inner', left_on=['bodyId', 'rowId'], \
    right_on=['bodyId', 'link'], suffixes=['_child', '_parent']) for i in range(len(skeletons))]

# Divide segements into groups (depending on identification of large jumps)
def getSubsegments(seg, threshold=5):
    # Define function to join segments
    def getLine(segment,axis):
        positions = np.zeros(shape=(len(segment)*2))
        positions[0::2] = getattr(segment,'%s_child' % (axis))
        positions[1::2] = getattr(segment,'%s_parent' % (axis))
        return positions
    xyz = np.array([getLine(seg,'x'), getLine(seg,'y'), getLine(seg,'z')]).T
    diffMags = np.linalg.norm(xyz[1:]-xyz[0:-1],axis=1)
    jumpIndices = np.where(diffMags > (np.mean(diffMags)+threshold*np.std(diffMags)))[0]
    segStarts, segEnds = np.r_[0,jumpIndices+1], np.r_[jumpIndices,len(xyz)]
    return [xyz[segStarts[i]:segEnds[i],:] for i in range(len(segStarts))]

# Plot skeletons (matplotlib)
fig = plt.figure(figsize=(8,12),facecolor='black')
ax = fig.gca(projection='3d')
for segment in segments:
    subsegments = getSubsegments(segment)
    for seg in subsegments:
        ax.plot(seg[:,0], seg[:,1], -seg[:,2], linewidth=.25, color=segment.color_parent[0])
if not(synapses is None): ax.scatter(synapses.x_pre, synapses.y_pre, synapses.z_pre, s=.01, c='white')
ax.set_facecolor((0,0,0))
plt.axis('off')

# Define animation and run
def animate_func(angle):
    ax.view_init(10, angle*5)
    return ax
anim = animation.FuncAnimation(fig, animate_func, interval=1) # 1000 / fps)
plt.show()