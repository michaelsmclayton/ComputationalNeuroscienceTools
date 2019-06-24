populations = {
    # Retina
    'RET': {
        'customParameters': {
            'Vrest': -65,
            'outputMean': -65,
            'outputSTD': 2
        }
    },
    # Thalamocortical relay
    'TCR': {
        'customParameters': {
            'gLeak': 10,
            'Eleak': -55,
            'Vrest': -65
        }
    },
    # Inhibitory population
    'IN': {
        'customParameters': {
            'gLeak': 10,
            'Eleak': -72.5,
            'Vrest': -75
        }
    },
    # Thalamic-reticular nucleus
    'TRN': {
        'customParameters': {
            'gLeak': 10,
            'Eleak': -72.5,
            'Vrest': -85
        }
    },
}