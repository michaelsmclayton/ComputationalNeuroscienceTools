import os
import numpy as np
import pandas as pd
import pprint; pp = pprint.PrettyPrinter(depth=10).pprint
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pprint; op = pprint.PrettyPrinter(depth=11).pprint
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Cut nan regions
cutNaNRegions = True

# Get session and cache
print('Getting session data...')
cache = EcephysProjectCache.from_warehouse(manifest="./example_ecephys_project_cache/manifest.json")
sessions = cache.get_session_table()
session = cache.get_session_data(session_id=791319847)

# Print probe properties
print('Showing possible probes...')
channelInfo = cache.get_channels()
probeIDs = session.probes.index.values
for probe_id in probeIDs:
    probeChannels = channelInfo.loc[channelInfo.ecephys_probe_id==probe_id].index
    structure_acronyms, intervals = session.channel_structure_intervals(probeChannels)
    print('Probe: %s : %s' % (probe_id, structure_acronyms))

# Select probe and get LFP data
probe_id = 805008602
print('\nGetting LFP data for probe %s...' % (probe_id))
lfp = session.get_lfp(probe_id)

# Get probe information
structure_acronyms, intervals = session.channel_structure_intervals(lfp["channel"])
interval_midpoints = [aa + (bb - aa) / 2 for aa, bb in zip(intervals[:-1], intervals[1:])]


# ---------------------------------
# Get evoked LFP data
# ---------------------------------

# Get all flash start and end times
flashTable = session.get_stimulus_table("flashes")
flashTable = flashTable[flashTable['color']==1]
pretime = .2; posttime = .4
starts, ends = [], []
for index, row in flashTable.iterrows():
    starts.append(row['start_time']-pretime)
    ends.append(row['stop_time']+posttime)

# Get valid channels (i.e. no nan values)
if cutNaNRegions==True:
    validChannels = np.zeros(shape=lfp.shape[1])
    validIntervals, validStructureAcronyms = [], []
    gapLength = 0
    for i, struct in enumerate(structure_acronyms):
        if isinstance(struct, str): # i.e. is not nan value
            validChannels[np.arange(intervals[i],intervals[i+1])] = 1
            validStructureAcronyms.append(struct)
            validIntervals.append(intervals[i]-0)
        else:
            gapLength += intervals[i+1]-intervals[i]
    validIntervals.append(int(np.sum(validChannels)))
    midpoint = lambda data, i : data[i] + (data[i+1]-data[i])/2
    validIntervalMidpoints = [midpoint(validIntervals,i) for i in range(len(validIntervals)-1)]
    intervals = validIntervals
    interval_midpoints = validIntervalMidpoints
    structure_acronyms = validStructureAcronyms
else:
    validChannels = np.ones(shape=lfp.shape[1])

# Combine evoked LFPs together
getWindow = lambda i : np.where(np.logical_and(lfp["time"] < ends[i], lfp["time"] >= starts[i]))[0]
lengthInTime = len(getWindow(0))-1
numberOfChannels = int(np.sum(validChannels))
evokedLFP = np.zeros(shape=[len(starts), numberOfChannels, lengthInTime])
for flash in range(len(starts)):
    # print(flash/len(starts))
    window = getWindow(flash)
    currentLFP = lfp[{"time": window}].T
    # Baseline normalise
    stimIndex = np.argmin(np.abs(lfp["time"]-(starts[flash]+pretime))).values
    stimIndex = np.where(window==stimIndex)[0].tolist()[0]
    currentLFP = currentLFP - np.mean(currentLFP[:,:stimIndex],axis=1)
    # Remove nan channels
    if cutNaNRegions:
        currentLFP = currentLFP[np.where(validChannels==1)[0].tolist(),:]
    # Save to store
    evokedLFP[flash,:,:] = currentLFP[:,:lengthInTime]

# Plot average evoked LFP
fig, ax = plt.subplots()
ax.pcolormesh(np.mean(evokedLFP,axis=0))
ax.set_yticks(intervals)
ax.set_yticks(interval_midpoints, minor=True)
ax.set_yticklabels(structure_acronyms, minor=True)
# Plot times
postStimTime = ends[0]-(starts[0]+pretime)
times = np.linspace(start=-pretime,stop=postStimTime, num=evokedLFP.shape[2])
stimIndex = np.argmin(np.abs(times))
xticks = [0, stimIndex, stimIndex*2, stimIndex*3, stimIndex*4]
ax.set_xticks(xticks)
ax.set_xticklabels(['-200','0','200', '400', '600'])
plt.tick_params("y", which="major", labelleft=False, length=40)
# Plot stimulus time
plt.plot([stimIndex,stimIndex], ax.get_ylim(), color='black')
title = plt.title(f'LFP data (probe {probe_id}): mean activity following visual flashes', fontsize=12)
plt.xlabel('Time (relative to flash presentation)')
plt.show()

# Save results
results = {
    'evokedLFP': evokedLFP,
    'intervals': intervals,
    'interval_midpoints': interval_midpoints,
    'structure_acronyms': structure_acronyms}
np.save(f'{probe_id}_evokedLFP', results)

# # Get stimulus types and timings (legacy)
# stimulusEvents = session.get_stimulus_epochs()
# flashEvents = stimulusEvents[stimulusEvents['stimulus_name']=='flashes']
# def getStartAndStopTimes(events):
#     startTime, stopTime = [np.double(events[time].values) for time in ['start_time', 'stop_time']]
#     return startTime, stopTime
# startTime, stopTime = getStartAndStopTimes(flashEvents)