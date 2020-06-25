import numpy as np
import matplotlib.pylab as plt
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D

evokedData = np.load('GOL1visualevokedAverage.npy')

getAxisPixels = lambda axis, step : np.arange(evokedData.shape[axis], step=step)

pixelData = {'data', 'location'}
timepoints = evokedData.shape[2]

downsamplingFactors = [3,3,1]
pixelCoords = [ \
    getAxisPixels(axis=0, step=downsamplingFactors[0]),
    getAxisPixels(axis=1, step=downsamplingFactors[1]),
    getAxisPixels(axis=3, step=downsamplingFactors[2])]

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



def normalise(data):
    data -= np.min(data)
    data /= np.max(data)
    return data

alphaData = np.std(pixelIntensities, axis=1)
alphaData = normalise(alphaData)
intensityData = zscore(pixelIntensities[:,100])
intensityData = normalise(intensityData)

colourData = np.zeros(shape=[numberOfPixels,4])
# getAlpha = lambda intensity : (.5 + np.cos(intensity*(2*np.pi))/2)
for i, intensity in enumerate(intensityData):
    colourData[i,:] = [intensity, intensity, intensity, alphaData[i]]

fig = plt.figure()
data = evokedData[:,:,0,:]
ax = fig.add_subplot(111, projection='3d')
intensityData = intensityData / np.max(intensityData)
dots = ax.scatter(pixelLocations[:,1], pixelLocations[:,2], pixelLocations[:,0], color=colourData)
ax.set_ylim(-25,25)
ax.view_init(-180,90)

# for angle in range(90,360,10):
#     print(angle)
#     ax.view_init(-180,angle)
#     plt.draw()
#     plt.pause(1)

plt.show()

