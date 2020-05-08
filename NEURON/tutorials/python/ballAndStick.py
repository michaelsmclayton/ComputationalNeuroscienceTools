import os
from neuron import h
from neuron.units import ms, mV, um
import matplotlib.pylab as plt
import pprint
pp = pprint.PrettyPrinter(depth=6).pprint
'''https://neuron.yale.edu/neuron/docs/ball-and-stick-model-part-1'''

# Load the standard run library to give us high-level simulation control functions
# (e.g. running a simulation for a given period)
h.load_file('stdrun.hoc')

# Compile .mod files
if not(os.path.isdir("./x86_64")):
    os.system("nrnivmodl ./mod_files/")
    h.nrn_load_dll('./x86_64/.libs/libnrnmech.so') # Load compiled .mod files

class BallAndStick:
    '''A ball-and-stick cell by definition consists of two parts: the soma (ball) and a dendrite (stick).
    We could define two Sections at the top level (as in the previous tutorial ('scriptingBasics.py), but that
    wouldn't give us an easy way to create multiple cells. Instead, let's define a BallAndStick neuron class'''
    def __init__(self):
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.soma.insert('ca')
    '''Any variables that describe properties of the cell must get stored as attributes of self. This is why we
    write self.soma instead of soma. Temporary variables, on the other hand, need not be prefixed with self and
    will simply stop existing when the initialization function ends'''

    '''Adding this __repr__ function means that, when h.topology() is called, the cells will have the label of
    'BallAndStick' (rather than the less nice <__main__.BallAndStick object at 0x10c303940>.soma(0-1)'''
    def __repr__(self):
        return 'BallAndStick'

# Instatiate cells
my_cell = BallAndStick()

# Print current cell topology
h.topology()

pp(my_cell.soma.psection())