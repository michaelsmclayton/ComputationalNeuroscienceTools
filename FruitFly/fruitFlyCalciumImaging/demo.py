import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import zscore
from scipy.io import loadmat
import gzip

# Data and code from:
# https://portal.nersc.gov/project/crcns/download/fly-1
# https://github.com/sophie63/FlyLFM

# "Fast near-whole brain imaging in adult Drosophila during responses to stimuli and behavior.
# Sophie Aimon, Takeo Katsuki, Tongqiu Jia, Logan Grosenick, Michael Broxton, Karl Deisseroth,
# Terrence J. Sejnowski, Ralph J Greenspan bioRxiv 033803; doi: https://doi.org/10.1101/033803"

# --------------------------------------------
# Load data
# --------------------------------------------
print('Loading data...')
def loadNII(image_path):
    img = nib.load(image_path)
    img_data = img.get_fdata()
    img_data = np.transpose(img_data, axes=(0,1,3,2))
    return img_data
def takeTimeExtract(startIndex, endIndex):
    return img_data[:,:,startIndex:endIndex,:] # At 200 there is a visual stimulus

# # Load raw fluorescence image ----------------
# rawData = loadNII(image_path='./AOL1b.nii')
# sliceImage = np.mean(rawData[:,:,:,-500],axis=2)
# np.save('sliceImage', sliceImage)
# sliceImage = np.load('sliceImage.npy')

# # Load image data ----------------
# img_data = loadNII(image_path='./AOL1_psfproj_dff_kf.nii')
# cutImgData = takeTimeExtract(startIndex=700, endIndex=6700)
# cutImgData = np.transpose(cutImgData, axes=(1,0,2,3))
# np.save('cutImgData', cutImgData)
cutImgData = np.load('cutImgData.npy')


# --------------------------------------------
# Import stimulation information
# --------------------------------------------
print('Getting stimulation information...')
matImport = loadmat("AOL1FlashOdor.mat")
def getStimulationTimes(odorType='visual'):
    odorIndex = 0 if odorType=='visual' else 1
    stimTimes = np.where(matImport['X'][odorIndex,:]==1)[0]
    def getStarts(stimTimes):
        diffs = [stimTimes[i+1]-stimTimes[i] for i in range(len(stimTimes)-1)]
        diffIndices = np.where(np.array(diffs)>1)
        return np.hstack((stimTimes[0], stimTimes[diffIndices]))
    return getStarts(stimTimes)
flashTimes = getStimulationTimes(odorType='visual')
odorTimes = getStimulationTimes(odorType='odor')

# --------------------------------------------
# Get evoked brain activity
# --------------------------------------------
frequency = 200 # Hz
baselineLength = 1 * frequency # seconds
evokedLength = 2 * frequency
def getMeanEvokedActivity(stimulationTimes):
    evokedData = []
    for time in stimulationTimes:
        startTime = time - baselineLength
        endTime = time + evokedLength
        evokedData.append(img_data[:,:,startTime:endTime,:])
    evokedData = np.array(evokedData)

    # # Baseline subtraction
    # for t in range(len(stimulationTimes)):
    #     print(t)
    #     currentEvoked = evokedData[t,:,:,:,]
    #     for i in range(np.max(currentEvoked.shape)):
    #         currentEvoked[:,:,i,:] -= np.mean(currentEvoked[:,:,0:evokedLength,:],axis=2)
    #     evokedData[t,:,:,:,] = currentEvoked

    # Return average across evoked responses
    evokedAverage = np.mean(evokedData, axis=0)
    evokedAverage = np.transpose(evokedAverage, axes=(1,0,2,3))
    return evokedAverage

# visualEvokedAverage = getMeanEvokedActivity(stimulationTimes=flashTimes)
# odorEvokedAverage = getMeanEvokedActivity(stimulationTimes=odorTimes)
# np.save('visualEvokedAverage', visualEvokedAverage)
visualEvokedAverage = np.load('baselineCorrectedVisualEvokedAverage.npy')
times = np.linspace(-baselineLength, evokedLength, num=visualEvokedAverage.shape[2])/frequency

# --------------------------------------------
# Visualise brain activity
# --------------------------------------------
# # Show raw fluorescence image
# plt.figure()
# plt.imshow(np.transpose(sliceImage))

# Function to create animation
def makeAnimation(data, sliceIndex=None, showTimes=False):
    def retrievalFunction(data, i, sliceIndex):
        if sliceIndex!=None:
            return data[:,:,i,sliceIndex]
        else:
            return data[:,:,i]
    def animate_func(i):
        im.set_array(retrievalFunction(data, i, sliceIndex))
        if showTimes == True:
            text.set_text("%.2f" % times[i])
            return [im, text]
        else: return im
    fig = plt.figure()# figsize=(10,10))
    im = plt.imshow(retrievalFunction(data, 0, sliceIndex)) #, cmap="gray")
    if showTimes == True: text = plt.text(5,5, s="%.2f" % times[0], color='white')
    frameNumber = np.max(data.shape)
    anim = animation.FuncAnimation(fig, animate_func, frames=frameNumber, interval = 1.0) #1000 / fps)
    plt.axis('off')
    plt.show()

# Dispay evoked average
# makeAnimation(np.mean(visualEvokedAverage, axis=3), sliceIndex=None, showTimes=True)
makeAnimation(np.mean(cutImgData, axis=3), sliceIndex=None)
# makeAnimation(visualEvokedAverage, sliceIndex=1)