from types import SimpleNamespace # allows for dot naming of dictionary values
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# --------------------------------
# Probe class
# --------------------------------
class Probe:

    # Constructor function
    def __init__(self, probe_id, session_id):
        self.session_id = session_id
        self.probe_id = probe_id
        self.lfp = None
        self.structures = list()
        self.coords = SimpleNamespace(**{
            'ant_post': list(), 'dors_vent': list(), 'left_right': list()
        })
        
    # Get LFP data
    def getLfpData(self, session):
        print('Getting LFP for probe %s...' % str(self.probe_id))
        self.lfp = session.get_lfp(self.probe_id)

    # Get probe spatial coordinates and structures recorded
    def getProbeCoordinates(self, channels, session):
        print('Getting LFP coordinates...')
        for channelValue in self.lfp.channel.values:
            channelInfo = channels.loc[channels.index==channelValue]
            if len(channelInfo)==0: # if no channel can be found
                self.coords.ant_post.append(float('nan'))
                self.coords.dors_vent.append(float('nan'))
                self.coords.left_right.append(float('nan'))
                self.structures.append(float('nan'))
            else:
                self.coords.ant_post.append(channelInfo.anterior_posterior_ccf_coordinate.values[0])
                self.coords.dors_vent.append(channelInfo.dorsal_ventral_ccf_coordinate.values[0])
                self.coords.left_right.append(channelInfo.left_right_ccf_coordinate.values[0])
                self.structures.append(channelInfo.ecephys_structure_acronym.values[0])

    # Cut LFP
    def cutLFP(self, startTime, endTime):
        print('Cutting LFP within temporal region...')
        self.lfp = self.lfp[(self.lfp.time.values > startTime) & (self.lfp.time.values < endTime)]