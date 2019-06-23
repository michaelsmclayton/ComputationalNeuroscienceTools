
# Define synapse parameters

def AMPA_parameters(gSyncMax):
    return '''
    alpha = 1000**-1 : 1 # * mM**-1 * second**-1
    beta = 50**-1 : 1 # * second**-1 # reverse rates of chemical reactions
    gSynMax = ''' + str(gSyncMax) + ''' / 1000 : 1 #* uS/cm2 # maximum conductance
    ESyncRev = 0 : 1 # * mV # reverse potential
    '''

def GABAa_parameters(ESyncRev):
    return '''
    alpha = 1000**-1 : 1 # * mM**-1 * second**-1
    beta = 40**-1 : 1 # * second**-1 # reverse rates of chemical reactions
    gSynMax = 100 / 1000 : 1 #* uS/cm2 # maximum conductance
    ESyncRev = ''' + str(ESyncRev) + ''' : 1 # * mV # reverse potential
    '''

# Define synapse connections
connections = {
    'RET_TCR': {
        'from': 'RET',
        'to': 'TCR',
        'synType': 'AMPA',
        'synFun': AMPA_parameters,
        'customParameters': {
            'gSyncMax': 300
        }
    }
}

