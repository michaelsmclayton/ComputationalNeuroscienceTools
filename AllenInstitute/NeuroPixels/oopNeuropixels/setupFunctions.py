import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

def getSummaryData(dataDirectory):
    print('Getting cache...')
    manifest_path = os.path.join(dataDirectory, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    channels = cache.get_channels()
    probes = cache.get_probes()
    return cache, sessions, channels, probes

def getSessionData(cache, session_id):
    print('Getting session...')
    session = cache.get_session_data(session_id)
    return session