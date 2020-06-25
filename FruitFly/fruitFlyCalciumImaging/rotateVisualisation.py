import numpy as np
import matplotlib.pylab as plt
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D

# Load data
evokedData = np.load('GOL1visualevokedAverage.npy')

# Get voxel locations (downsampled)
downsamplingFactors = [2,2,1]
getAxisPixels = lambda axis, step : np.arange(evokedData.shape[axis], step=step)
pixelCoords = [ \
    getAxisPixels(axis=0, step=downsamplingFactors[0]),
    getAxisPixels(axis=1, step=downsamplingFactors[1]),
    getAxisPixels(axis=3, step=downsamplingFactors[2])]

# Get data for each voxel
timepoints = evokedData.shape[2]
numberOfPixels = np.prod([len(pixelCoords[i]) for i in range(3)])
pixelIntensities = np.zeros(shape=[numberOfPixels, timepoints])
pixelLocations = np.zeros(shape=[numberOfPixels,3])
pixelIndex = 0
for x in pixelCoords[0]:
    for y in pixelCoords[1]:
        for z in pixelCoords[2]:
            pixelIntensities[pixelIndex,:] = evokedData[x,y,:,z]
            pixelLocations[pixelIndex,:] = np.array([x,y,z])
            pixelIndex += 1

# Get voxel transparency (i.e. alpha) and intensity data
def normalise(data):
    data -= np.min(data)
    data /= np.max(data)
    return data
alphaData = np.var(pixelIntensities, axis=1)
alphaData = normalise(alphaData)
intensityData = zscore(pixelIntensities[:,300])
intensityData = normalise(intensityData)

# Get voxel colour data
colourData = np.zeros(shape=[numberOfPixels,4])
# getAlpha = lambda intensity : (.5 + np.cos(intensity*(2*np.pi))/2)
for i, intensity in enumerate(intensityData):
    colourData[i,:] = [np.sin(intensity), intensity, np.cos(intensity), alphaData[i]]

# Plot result
fig = plt.figure()
data = evokedData[:,:,0,:]
ax = fig.add_subplot(111, projection='3d')
intensityData = intensityData / np.max(intensityData)
dots = ax.scatter(pixelLocations[:,1], pixelLocations[:,2], pixelLocations[:,0], s=20, color=colourData)
ax.set_facecolor((0,0,0))
plt.axis('off')
ax.set_ylim(-20,20)
ax.view_init(-180,90)

# Animate
# for angle in range(90,360,10):
#     print(angle)
#     ax.view_init(-180,angle)
#     plt.draw()
#     plt.pause(1)

plt.show()

