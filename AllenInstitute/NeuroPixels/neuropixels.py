# ---------------------------------
# Import dependencies
# ---------------------------------
'''may require pip3 install --upgrade allensdk'''
import os
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pprint; objectPrint = pprint.PrettyPrinter(depth=11).pprint
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
'''The EcephysProjectCache is the main entry point to the Visual Coding Neuropixels dataset.
It allows you to download data for individual recording sessions and view cross-session
summary information.
'''

# ---------------------------------
# Define data paths and download data
# ---------------------------------

# Define download paths
downloadDirectoryName = "example_ecephys_project_cache"
manifest_path = os.path.join(downloadDirectoryName, "manifest.json")

# Download sessions meta-data
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
print(cache.get_all_session_types())
'''['brain_observatory_1.1', 'functional_connectivity']'''

# See session information
sessions = cache.get_session_table()
brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
brain_observatory_type_sessions.tail()
objectPrint(sessions)

# Download data from an arbitrary session
print('Getting session data')
session_id = 791319847
session = cache.get_session_data(session_id)

# ---------------------------------
# Get LFP data
# ---------------------------------

# List the probes recorded from in this session
# session.probes.head()

# load up the lfp from one of the probes. This returns an xarray dataarray
print('Getting LFP data')
probe_id = session.probes.index.values[0]
lfp = session.get_lfp(probe_id)
# print(lfp)

# We can figure out where each LFP channel is located in the Brain
# now use a utility to associate intervals of /rows with structures
structure_acronyms, intervals = session.channel_structure_intervals(lfp["channel"])
interval_midpoints = [aa + (bb - aa) / 2 for aa, bb in zip(intervals[:-1], intervals[1:])]
print(structure_acronyms)
print(intervals)

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
    evokedLFP[flash,:,:] = currentLFP[:,0:312]

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

# asldj
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

probes = cache.get_probes()
probes.loc[probes.index==probe_id]

# --------------------------------------
# View probe location
# --------------------------------------
channels = cache.get_channels()
channelsOfInterest = channels.loc[channels.ecephys_probe_id==probe_id]
ant_post = channelsOfInterest.anterior_posterior_ccf_coordinate
dors_vent = channelsOfInterest.dorsal_ventral_ccf_coordinate
left_right = channelsOfInterest.left_right_ccf_coordinate
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(ant_post, left_right, dors_vent, 'gray')
ax.set_xlabel('ant_post')
ax.set_ylabel('left_right')
ax.set_zlabel('dors_vent')
plt.show()

## Look at anatomical images
import skimage.io as io
img = io.imread('./atlasVolume/atlasVolume.mhd', plugin='simpleitk')
#img = io.imread('./P56_Mouse_annotation/annotation.mhd', plugin='simpleitk')
fig, ax = plt.subplots()
dataToPlot = img[:,:,300].T
# Rescale image to 25um per pixel
plt.imshow(dataToPlot, extent=[0, dataToPlot.shape[1]*25, dataToPlot.shape[0]*25, 0])
plt.show()