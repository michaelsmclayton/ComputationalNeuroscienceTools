# ---------------------------------
# Import dependencies
# ---------------------------------
'''may require pip3 install --upgrade allensdk'''
import os
import numpy as np
import pandas as pd
import pprint; pp = pprint.PrettyPrinter(depth=10).pprint
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pprint; op = pprint.PrettyPrinter(depth=11).pprint
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
'''The EcephysProjectCache is the main entry point to the Visual Coding Neuropixels dataset.
It allows you to download data for individual recording sessions and view cross-session
summary information.
'''

# Analyse flags
analyseLFP = True

# ---------------------------------
# Define data paths and download data
# ---------------------------------

# Define download paths
downloadDirectoryName = "example_ecephys_project_cache"
manifest_path = os.path.join(downloadDirectoryName, "manifest.json")

# Download sessions meta-data
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
# print(cache.get_all_session_types())
'''['brain_observatory_1.1', 'functional_connectivity']'''

# See session information
sessions = cache.get_session_table()
# brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
# brain_observatory_type_sessions.tail()

# Download data from an arbitrary session
print('Getting session data...')
session_id = 791319847
session = cache.get_session_data(session_id)

# ---------------------------------
# Get LFP data
# ---------------------------------

if analyseLFP==True:

    # List the probes recorded from in this session
    # session.probes.head()

    # load up the lfp from one of the probes. This returns an xarray dataarray
    print('Getting LFP data...')
    probe_id = session.probes.index.values[0]
    lfp = session.get_lfp(probe_id)
    # print(lfp)

    aksjd

    # We can figure out where each LFP channel is located in the Brain
    # now use a utility to associate intervals of /rows with structures
    structure_acronyms, intervals = session.channel_structure_intervals(lfp["channel"])
    interval_midpoints = [aa + (bb - aa) / 2 for aa, bb in zip(intervals[:-1], intervals[1:])]
    # print(structure_acronyms)
    # print(intervals)

    # # Get stimulus types and timings
    # stimulusEvents = session.get_stimulus_epochs()
    # flashEvents = stimulusEvents[stimulusEvents['stimulus_name']=='flashes']
    # def getStartAndStopTimes(events):
    #     startTime, stopTime = [np.double(events[time].values) for time in ['start_time', 'stop_time']]
    #     return startTime, stopTime
    # startTime, stopTime = getStartAndStopTimes(flashEvents)

    # Get all flash start and end times
    flashTable = session.get_stimulus_table("flashes")
    flashTable = flashTable[flashTable['color']==1]
    starts, ends = [], []
    for index, row in flashTable.iterrows():
        starts.append(row['start_time'])
        ends.append(row['stop_time'])

    # Combine evoked LFPs together
    evokedLFP = np.zeros(shape=[len(starts), 77, 312])
    for flash in range(len(starts)):
        # print(flash/len(starts))
        window = np.where(np.logical_and(lfp["time"] < ends[flash], lfp["time"] >= starts[flash]))[0]
        currentLFP = lfp[{"time": window}].T
        evokedLFP[flash,:,:] = currentLFP[0:77,0:312]

    # Plot average evoked LFP
    fig, ax = plt.subplots()
    ax.pcolormesh(np.mean(evokedLFP,axis=0))
    ax.set_yticks(intervals)
    ax.set_yticks(interval_midpoints, minor=True)
    ax.set_yticklabels(structure_acronyms, minor=True)
    plt.tick_params("y", which="major", labelleft=False, length=40)
    plt.show()

    # # Get flash responses
    # endFlash = 5
    # startTime = starts[0]
    # endTime = ends[endFlash]
    # window = np.where(np.logical_and(lfp["time"] < endTime, lfp["time"] >= startTime))[0]
    # currentLFP = lfp[{"time": window}].T
    # # currentLFP -= currentLFP[0,:]
    # fig, ax = plt.subplots()
    # ax.pcolormesh(currentLFP)
    # ax.set_yticks(intervals)
    # ax.set_yticks(interval_midpoints, minor=True)
    # ax.set_yticklabels(structure_acronyms, minor=True)
    # #ax.plot(currentLFP[30,:])
    # for i in range(endFlash):
    #     currentStart = np.min(np.where(currentLFP["time"]>starts[i]))
    #     currentEnd = np.min(np.where(currentLFP["time"]>ends[i]))
    #     ax.plot([currentStart,currentStart], ax.get_ylim(), linewidth=1, color='black')
    #     ax.plot([currentEnd,currentEnd], ax.get_ylim(), linewidth=1, color='black')
    # plt.show()
    #
    # # Plot LFP window
    # def showLFP(startTime, endTime):
    #     baseline = .1*(endTime-startTime)
    #     window = np.where(np.logical_and(lfp["time"] < endTime, lfp["time"] >= startTime-baseline))[0]
    #     fig, ax = plt.subplots()
    #     ax.pcolormesh(lfp[{"time": window}].T)
    #     ax.set_yticks(intervals)
    #     ax.set_yticks(interval_midpoints, minor=True)
    #     ax.set_yticklabels(structure_acronyms, minor=True)
    #     plt.tick_params("y", which="major", labelleft=False, length=40)
    #     num_time_labels = 8
    #     time_label_indices = np.around(np.linspace(1, len(window), num_time_labels)).astype(int) - 1
    #     time_labels = [ f"{val:1.3}" for val in lfp["time"].values[window][time_label_indices]]
    #     ax.set_xticks(time_label_indices + 0.5)
    #     # ax.set_xticklabels(time_labels)
    #     ax.set_xlabel("time (s)", fontsize=20)

    # for i in range(5):
    #     startTime = starts[i]
    #     endTime = ends[i]
    #     print(endTime-startTime)
    #     showLFP(startTime, endTime)
    # plt.show()

# probes = cache.get_probes()
# probes.loc[probes.index==probe_id]

# --------------------------------------
# View probe location and activity
# --------------------------------------

# Import anatomical image and figure setup
import skimage.io as io
img = io.imread('./atlasVolume/atlasVolume.mhd', plugin='simpleitk')
# img = io.imread('./P56_Mouse_annotation/annotation.mhd', plugin='simpleitk')

# Get channels
channels = cache.get_channels()

# Get coordinates of LFP data
ant_post_lfp, dors_vent_lfp, left_right_lfp = list(), list(), list()
for channelValue in lfp.channel.values:
    channelInfo = channels.loc[channels.index==channelValue]
    ant_post_lfp.append(channelInfo.anterior_posterior_ccf_coordinate.values[0])
    dors_vent_lfp.append(channelInfo.dorsal_ventral_ccf_coordinate.values[0])
    left_right_lfp.append(channelInfo.left_right_ccf_coordinate.values[0])

# Define animation function
def updatefig(t):
    global scat1, scat2
    scat1.set_array(lfp[t,:].values)
    scat2.set_array(lfp[t,:].values)
    currentTime = str(lfp.time[t].values)
    title.set_text(currentTime[0:4] + " seconds")
    return scat1, scat2, title

# Plot single probe
fig, axes = plt.subplots(2,1)
fig.set_size_inches(6,10)
title = fig.suptitle('', y=.9, fontsize=12)
# Create coronal plot
halfwayIndex = int(len(ant_post_lfp)/2)
anteriorPosteriorSlice = int(ant_post_lfp[halfwayIndex]/25)
dataToPlot = img[:,:,anteriorPosteriorSlice].T
'''x25 to rescale image to 25um per pixel'''
axes[0].imshow(dataToPlot, cmap='gray', extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
scat1 = axes[0].scatter(left_right_lfp, dors_vent_lfp, s=4, c=lfp[0,:].values)
axes[0].set_xlim([0, dataToPlot.shape[1]*25])
axes[0].set_xlabel('Left-Right')
axes[0].set_ylabel('Dorsal-Ventral')
# Create saggital plot
leftRightSlice = int(left_right_lfp[halfwayIndex]/25)
dataToPlot = img[leftRightSlice,:,:]
axes[1].imshow(dataToPlot, cmap='gray', extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
scat2 = axes[1].scatter(ant_post_lfp, dors_vent_lfp, s=4, c=lfp[0,:].values)
axes[1].set_xlabel('Anterior-Posterior')
axes[1].set_ylabel('Dorsal-Ventral')
ani = FuncAnimation(fig, updatefig, frames=range(1, lfp.shape[0], 20), interval=1)
plt.show()


# # Loop over all probes
# fig, axes = plt.subplots(2,len(session.probes), sharex=True)
# fig.set_size_inches(18, 3.5)
# fig.subplots_adjust(wspace=0.01, hspace=0.5)  # hspace controls the hight of space between subplots
# numberOfProbes = len(session.probes)
# for probe in range(numberOfProbes):
#     # Get channel location information from chosen probe
#     probe_id = session.probes.index.values[probe]
#     channelsOfInterest = channels.loc[channels.ecephys_probe_id==probe_id]
#     ant_post = channelsOfInterest.anterior_posterior_ccf_coordinate.values
#     dors_vent = channelsOfInterest.dorsal_ventral_ccf_coordinate.values
#     left_right = channelsOfInterest.left_right_ccf_coordinate.values
#     halfwayIndex = int(len(ant_post)/2)
#     # Create coronal plot
#     anteriorPosteriorSlice = int(ant_post[halfwayIndex]/25)
#     dataToPlot = img[:,:,anteriorPosteriorSlice].T
#     '''x25 to rescale image to 25um per pixel'''
#     axes[0,probe].imshow(dataToPlot, extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
#     axes[0,probe].plot(left_right, dors_vent, color='white', linewidth=2)
#     # Create saggital plot
#     leftRightSlice = int(left_right[halfwayIndex]/25)
#     dataToPlot = img[leftRightSlice,:,:]
#     axes[1,probe].imshow(dataToPlot, extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
#     axes[1,probe].plot(ant_post, dors_vent, color='white', linewidth=2)
#     if probe>0:
#         axes[0,probe].axis('off')
#         axes[1,probe].axis('off')
# plt.show()

# Create transverse
# dorsalVentralSlice = int(dors_vent[halfwayIndex]/25)
# dataToPlot = img[:,dorsalVentralSlice,:]
# axes[2].imshow(dataToPlot, extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
# axes[2].plot(ant_post, left_right, color='white', linewidth=2)


# np.save('neuropixelslocation.npy', [ant_post, dors_vent, left_right])
# locations = np.load('neuropixelslocation.npy')
# ant_post = locations[0,:]
# dors_vent = locations[1,:]
# left_right = locations[2,:]

# # Plot 3D probe location
# fig = plt.figure()
# x = plt.axes(projection='3d')
# ax.plot3D(ant_post, left_right, dors_vent, 'gray')
# ax.set_xlabel('ant_post')
# ax.set_ylabel('left_right')
# ax.set_zlabel('dors_vent')
# plt.show()



unitLocations = np.array([session.units.probe_horizontal_position, session.units.probe_vertical_position])

plt.plot(unitLocations[0,:],unitLocations[1,:])
plt.show()