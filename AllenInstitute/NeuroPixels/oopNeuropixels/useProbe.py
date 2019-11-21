import numpy as np
from setupFunctions import getSummaryData, getSessionData
from supplementaryFunctions import showProbe
from Probe import Probe

# Get summary data
cache, sessions = getSummaryData(dataDirectory="../example_ecephys_project_cache")

# Get session data
session_id = 791319847
session, probes, channels = getSessionData(cache, session_id)

# Get all flash start and end times
flashTable = session.get_stimulus_table("flashes")
flashTable = flashTable[flashTable['color']==1]
flashStarts, flashEnds = [], []
for index, row in flashTable.iterrows():
    flashStarts.append(row['start_time'])
    flashEnds.append(row['stop_time'])

# Define period for analysis
periodStart = flashStarts[0]-2
periodEnd = flashEnds[2]+2

# Get probes
sessionProbes = list()
for probe_id in [805008600, 805008604]:
    probe = Probe(probe_id, session_id)
    probe.getLfpData(session)
    probe.getProbeCoordinates(channels, session)
    probe.cutLFP(startTime=periodStart, endTime=periodEnd)
    sessionProbes.append(probe)
    del probe
np.save('sessionProbes', sessionProbes)

# sessionProbes = np.load('sessionProbes.npy')
# showProbe(sessionProbes[0])

