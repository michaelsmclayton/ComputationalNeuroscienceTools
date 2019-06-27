
# ----------------------------------------
# Define synapse parameters
# ----------------------------------------

def AMPA_parameters(gSynMax):
    return '''
    alpha = 1000**-1 : 1 # * mM**-1 * second**-1
    beta = 50**-1 : 1 # * second**-1 # reverse rates of chemical reactions
    gSynMax = ''' + str(gSynMax) + ''' / 1 : 1 # * uS/cm2 # maximum conductance
    ESynRev = 0 : 1 # * mV # reverse potential
    '''

def GABAa_parameters(ESynRev):
    return '''
    alpha = 1000**-1 : 1 # * mM**-1 * second**-1
    beta = 40**-1 : 1 # * second**-1 # reverse rates of chemical reactions
    gSynMax = 100.0 / 1 : 1 # * uS/cm2 # maximum conductance
    ESynRev = ''' + str(ESynRev) + ''' : 1 # * mV # reverse potential
    '''

def GABAb_parameters():
    return '''
    alpha1 = 10**-1 : 1 # forward rates of chemical reactions
    beta1 = 25**-1 : 1 # reverse rates of chemical reactions
    alpha2 = 15**-1 : 1
    beta2 = 5**-1 : 1
    gSynMax = 60.0 / 1 : 1 # * uS/cm2 # maximum conductance
    ESynRev = -100 : 1 # * mV # reverse potential
    '''

# ----------------------------------------
# Define synapse connections
# ----------------------------------------
connections = {

    # Thalamocortical relay to the...
    'TCR': {
        # thalamic reticular nucleus
        'TRN': {
            'synType': 'AMPA',
            'connectionStrength': 35,
            'gSynMax': 100
        }
    },

    # Interneuron population to the...
    'IN': {
        # thalamocortical relay
        'TCR': {
            'synType': 'GABAa',
            'connectionStrength': (1/2) * 30.9,
            'ESynRev': -85 
        },
        # interneuron population
        'IN': {
            'synType': 'GABAa',
            'connectionStrength': 23.6,
            'ESynRev': -75 
        }
    },

    # Thalamic reticular nucleus to the...
    'TRN': {
        # thalamocortical relay (GABAa)
        'TCR': {
            'synType': 'GABAa',
            'connectionStrength': (3/8) * 30.9,
            'ESynRev': -85 
        },
        # thalamic reticular nucleus
        'TRN': {
            'synType': 'GABAa',
            'connectionStrength': 20,
            'ESynRev': -75
        },
        # thalamocortical relay (GABAb)
        'TCR': {
            'synType': 'GABAb',
            'connectionStrength': (1/8) * 30.9,
            'ESynRev': -100
        }
    },

    # Retina to the...
    'RET': {
        # thalamocortical relay
        'TCR': {
            'synType': 'AMPA',
            'connectionStrength': 7.1,
            'gSynMax': 300
        },
        # interneuron population
        'IN': {
            'synType': 'AMPA',
            'connectionStrength': 47.4,
            'gSynMax': 100
        },
    }
}

