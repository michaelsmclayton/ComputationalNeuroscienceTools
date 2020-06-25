import numpy as np
import scipy.io as sio
from scipy.stats import gaussian_kde, zscore
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import neuron data data
baseDirectory = './data/'
mt = sio.loadmat('%snatimg2800_M170714_MP032_2017-09-14.mat' % baseDirectory)

# Get cell information
med = mt['med']  # cell centers (X Y Z)
numberOfNeurons = med.shape[0]

# Normalise data (for color information)
def normalise(med):
    norm = med + np.min(med)
    norm = norm / np.max(norm)
    return norm
normMed = normalise(med)
normMed[:,0] = (normMed[:,0]*-1)+1 # Reverse red channel

# Add alpha channel
alpha = 1
normMed = np.hstack((normMed, alpha*np.ones(shape=[len(normMed),1])))

# Show neurons
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
dots = ax.scatter(med[:,0], med[:,1], med[:,2], s=2, color=normMed)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_facecolor((0,0,0))
plt.axis('off')

# Rotate the axes and update
for angle in range(0,360,1):
    print(angle)
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.0001)