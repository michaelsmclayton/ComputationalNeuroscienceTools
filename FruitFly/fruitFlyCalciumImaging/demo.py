import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import zscore
import gzip

# Data from:
# https://portal.nersc.gov/project/crcns/download/fly-1

# "Fast near-whole brain imaging in adult Drosophila during responses to stimuli and behavior.
# Sophie Aimon, Takeo Katsuki, Tongqiu Jia, Logan Grosenick, Michael Broxton, Karl Deisseroth,
# Terrence J. Sejnowski, Ralph J Greenspan bioRxiv 033803; doi: https://doi.org/10.1101/033803"

# --------------------------------------------
# Get data
# --------------------------------------------
# image_path = './AOL1_psfproj_dff_kf.nii' # './AOL1b.nii'
# print('Loading data...')
# img = nib.load(image_path)
# img_data = img.get_fdata()
# img_data = np.transpose(img_data, axes=(0,1,3,2))
# cutDFFData = img_data[:,:,700:6700,:] # At 200 there is a visual stimulus
img_data = np.load('cutDFFData.npy')

# # Save data
# np.save('cutDFFData', cutDFFData)
# # f = gzip.GzipFile("cutData.npy.gz", "w")
# # np.save(file=f, arr=cutDFFData)
# # f.close()

# --------------------------------------------
# Cut data in time
# --------------------------------------------
sliceIndex = 3
cutImgData = img_data[:,:,700:,:]
cutImgData = np.transpose(cutImgData, axes=(1,0,2,3))

# --------------------------------------------
# Define function to get fluorescence
# --------------------------------------------
def getFluorescence(imgData, sliceIndex, i):
    F = imgData[:,:,i,sliceIndex]
    # meanFluorescence = np.mean(imgData[:,:,:,sliceIndex], axis=2)
    return F # - meanFluorescence
    # F0 = np.mean(imgData[:,:,i-11:i-1, sliceIndex], axis=2) # i.e. priorTenFramesAverage
    # return ((F-F0)/F0) # * imgData[:,:,0]

# --------------------------------------------
# Generate slice animation
# --------------------------------------------
def animate_func(i):
    im.set_array(getFluorescence(cutImgData, sliceIndex, i=12+i))
    text.set_text(i)
    return [im, text]
fig = plt.figure( figsize=(5,5))
im = plt.imshow(getFluorescence(cutImgData, sliceIndex, i=12), vmin=-.05, vmax=.05)#, cmap="gray")
text = plt.text(5,5, s=0, color='white')
fps = 50
anim = animation.FuncAnimation(fig, animate_func, frames=None, interval = 1.0)#1000 / fps)
plt.show()

# # --------------------------------------------
# # Animate all slices together
# # --------------------------------------------
# def animate_func(i):
#     for sliceIndex in range(cutImgData.shape[3]):
#         imgs[sliceIndex].set_array(getFluorescence(cutImgData, sliceIndex, i=12+i))
#     return [imgs]
# fig = plt.figure( figsize=(5,15))
# imgs = []
# for sliceIndex in range(cutImgData.shape[3]):
#     plt.subplot(cutImgData.shape[3],1,sliceIndex+1)
#     imgs.append(plt.imshow(getFluorescence(cutImgData, sliceIndex, i=12), vmin=-.05, vmax=.05))#, cmap="gray")
# anim = animation.FuncAnimation(fig, animate_func, frames=None, interval = 1.0)#1000 / fps)
# plt.show()

# a = cutImgData[:,30,:,5]
# plt.plot(np.transpose(a)); plt.show()