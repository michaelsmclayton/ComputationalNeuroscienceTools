

# Import anatomical image and figure setup
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import pprint; op = pprint.PrettyPrinter(depth=11).pprint
import matplotlib.pyplot as plt

# Import base anatomical image
import skimage.io as io
img = io.imread('./atlasVolume/atlasVolume.mhd', plugin='simpleitk')

# Download sessions meta-data
downloadDirectoryName = "example_ecephys_project_cache"
manifest_path = os.path.join(downloadDirectoryName, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# Get summary information
sessions = cache.get_session_table()
channels = cache.get_channels()
probes = cache.get_probes()

# Auxillary function
def getMiddleSlice(axis):
    halfwayIndex = int(len(axis)/2)
    return int(axis[halfwayIndex]/25)
def centreAxes(axes,dataToPlot,probeIndex,row):
    axes[probeIndex,row].set_xlim([0, dataToPlot.shape[1]*25])
    axes[probeIndex,row].set_ylim([dataToPlot.shape[0]*25,0])
    # axes[probeIndex,row].axis('off')
    return axes

# Select probes for current session
for sessionIndex in range(len(sessions.index)):

    # Get session and probe data
    currentSession = sessions.index.values[sessionIndex]
    currentProbes = probes.loc[probes.ecephys_session_id==currentSession]
    sessions.loc[sessions.index==currentSession]

    # Create figure
    fig, axes = plt.subplots(len(currentProbes),3)
    fig.set_size_inches(6,10)
    # Loop over probes
    for probeIndex in range(len(currentProbes)):
        # Get current channels
        currentChannels = channels.loc[channels.ecephys_probe_id==currentProbes.index.values[probeIndex]]
        # Get spatial cooridinats
        ant_post = currentChannels.anterior_posterior_ccf_coordinate.values
        dors_vent = currentChannels.dorsal_ventral_ccf_coordinate.values
        left_right = currentChannels.left_right_ccf_coordinate.values
        # Create coronal plot
        dataToPlot = img[:,:,getMiddleSlice(ant_post)].T
        axes[probeIndex,0].imshow(dataToPlot, cmap='gray', extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
        axes[probeIndex,0].plot(left_right, dors_vent, color='white', linewidth=2)
        axes = centreAxes(axes,dataToPlot,probeIndex,0)
        # Create saggital plot
        dataToPlot = img[getMiddleSlice(left_right),:,:]
        axes[probeIndex,1].imshow(dataToPlot, cmap='gray', extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
        axes[probeIndex,1].plot(ant_post, dors_vent, color='white', linewidth=2)
        axes[probeIndex,1].set_xlim([0, dataToPlot.shape[1]*25])
        axes[probeIndex,1].set_title(" ".join([str(region) for region in currentProbes.ecephys_structure_acronyms.values[probeIndex]]), fontsize=8)
        axes = centreAxes(axes,dataToPlot,probeIndex,1)
        # Create transverse
        dataToPlot = img[:,getMiddleSlice(dors_vent),:].T
        axes[probeIndex,2].imshow(dataToPlot, cmap='gray', extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
        axes[probeIndex,2].plot(left_right, ant_post, color='white', linewidth=2)
        axes[probeIndex,2].set_xlim([0, dataToPlot.shape[1]*25])
        axes = centreAxes(axes,dataToPlot,probeIndex,2)
    title = fig.suptitle('Session: ' + str(currentSession), y=.95, fontsize=12)
    plt.subplots_adjust(top = 0.875, bottom=0.05, hspace=.65, wspace=0.4)
    plt.savefig(str(currentSession))
    plt.close()
    #plt.show()


