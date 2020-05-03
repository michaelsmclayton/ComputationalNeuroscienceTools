from neuron import h
from neuron.units import ms, mV, um
import matplotlib.pylab as plt
import pprint
pp = pprint.PrettyPrinter(depth=6).pprint
'''https://neuron.yale.edu/neuron/docs/scripting-neuron-basics'''


# --------------------------------
# Create a cell
# --------------------------------
'''Here the cell is made just of a single soma'''
soma = h.Section(name='soma')

# View current cell (/network)
print('\nShowing current cell state...')
h.topology()

# View section properties
print('\nShowing cell properties...')
pp(soma.psection())
'''return dictionary the properties of the section'''
# soma.psection()['morphology']['L'] = 100.0 # Get a speific property by name
# soma.L = 100.0
# soma.diam = 500.0


# --------------------------------
# Set the cell's morphological properties
# --------------------------------
'''Since we're simulating a soma, the default length of 100 μm and diameter of 500 μm are
inappropriate. Let's set the length (L) and diameter (diam) to 20 μm instead'''
soma.L = 20*um
soma.diam = 20*um

# --------------------------------
# Insert ion channels
# --------------------------------
'''NEURON comes with a few built in biophysical mechanisms that can be added to a model.
e.g:
    pas	: Passive (“leak”) channel.
    extracellular : For simulating effects of nonzero extracellular potential, as may happen
                    with leaky patch clamps, or detailed propertes of the myelin sheath.
    hh : Hodgkin-Huxley sodium, potassium, and leakage channels'''
soma.insert('hh')
# Note that Hodgkin-Huxley channel kinetics are based on the squid giant axon. If that's not
# your model organism, then for your actual modeling projects, you'll want to use other kinetics

# Sections and segments
'''A NEURON Section is considered a piece of cable. Depending on the resolution desired, it may be
necessary to divide the cable into a number of segments. segments. The number of segments within a
section is given by the variable, nseg. To access a part of the section, specify a value between 0
and 1, where 0 is typically the end closest to the soma and 1 is the distal end'''
soma.nseg # 1
type(soma) # <class 'nrn.Section'>
type(soma(0.5)) # <class 'nrn.Segment'>
soma(0.5).hh.gkbar # .036 (accesssing segment properties)


# --------------------------------
# Insert a stimulus
# --------------------------------

# Let's insert a current clamp (an IClamp object) into the center of the soma to induce some membrane dynamics.
iclamp = h.IClamp(soma(0.5))
'''An IClamp is a Point Process. Point processes are point sources of current. When making a new PointProcess,
you pass the segment to which it will bind'''

# Lets look at the properties of this current clamp object
print('\nShowing properties of current clamp object...')
print([item for item in dir(iclamp) if not item.startswith('__')])
# ['amp', 'baseattr', 'delay', 'dur', 'get_loc', 'get_segment', 'has_loc', 'hname', 'hocobjptr', 'i', 'loc', 'same']
'''In particular, we notice three key properties of a current clamp: amp -- the amplitude (in nA), delay -- the time the
current clamp switches on (in ms), and dur -- how long (in ms) the current clamp stays on. Let's set these values:'''
iclamp.delay = 2*ms
iclamp.dur = 0.1*ms
iclamp.amp = 0.9 #*nA

print('\nShowing representation of model with stimulus input...')
pp(soma.psection())


# --------------------------------
# Set up recording variables
# --------------------------------
'''The cell should be configured to run a simulation. However, we need to indicate which variables we wish to record
from; these will be stored in a NEURON Vector (h.Vector object). For now, we will record the membrane potential, which is
soma(0.5).v and the corresponding time points (h.t). References to variables are available by preceding the last part of
the variable name with a _ref_'''

v = h.Vector().record(soma(0.5)._ref_v)             # Membrane potential vector
t = h.Vector().record(h._ref_t)                     # Time stamp vector


# --------------------------------
# Run the simulation
# --------------------------------
'''By default, the NEURON h module provides the low level fadvance function for advancing one time step. For higher-level
simulation control specification, we load NEURON's stdrun library'''
h.load_file('stdrun.hoc')

'''We can then initialize our simulation such that our cell has a resting membrane potential of -65 mV'''
h.finitialize(-65 * mV)

'''And then continue the simulation from the current time (0) until 40 ms'''
h.continuerun(40 * ms)


# --------------------------------
# Plot the results
# --------------------------------
print('\nPlotting model results...')
plt.figure()
plt.plot(t, v)
plt.xlabel('t (ms)')
plt.ylabel('v (mV)')
plt.show()