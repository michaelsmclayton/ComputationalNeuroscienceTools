import skimage.io as io
import matplotlib.pyplot as plt

# Show probe data
def showProbe(probe):
    img = io.imread('../atlasVolume/atlasVolume.mhd', plugin='simpleitk')
    # Plot single probe
    fig, axes = plt.subplots(2,1)
    fig.set_size_inches(6,10)
    title = fig.suptitle('', y=.9, fontsize=12)
    # Create coronal plot
    halfwayIndex = int(len(probe.coords.ant_post)/2)
    anteriorPosteriorSlice = int(probe.coords.ant_post[halfwayIndex]/25)
    dataToPlot = img[:,:,anteriorPosteriorSlice].T
    '''x25 to rescale image to 25um per pixel'''
    axes[0].imshow(dataToPlot, cmap='gray', extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
    scat1 = axes[0].scatter(probe.coords.left_right, probe.coords.dors_vent, s=4, c=probe.lfp[0,:].values)
    axes[0].set_xlim([0, dataToPlot.shape[1]*25])
    axes[0].set_xlabel('Left-Right')
    axes[0].set_ylabel('Dorsal-Ventral')
    # Create saggital plot
    leftRightSlice = int(probe.coords.left_right[halfwayIndex]/25)
    dataToPlot = img[leftRightSlice,:,:]
    axes[1].imshow(dataToPlot, cmap='gray', extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
    scat2 = axes[1].scatter(probe.coords.ant_post, probe.coords.dors_vent, s=4, c=probe.lfp[0,:].values)
    axes[1].set_xlabel('Anterior-Posterior')
    axes[1].set_ylabel('Dorsal-Ventral')
    plt.show()