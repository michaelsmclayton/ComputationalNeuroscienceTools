

# Import dependencies
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dipde.internals.internalpopulation import InternalPopulation
from dipde.internals.externalpopulation import ExternalPopulation
from dipde.internals.simulation import Simulation
from dipde.internals.connection import Connection as Connection

# Set simulation settings
t0 = 0.
dt = .0001 # time step of the simulation (in seconds)
dv = .0001 # granularity of the voltage domain of the Internal population (in volts)
tf = .1 # Total length of simulation
verbose = True # Print times during simulation
update_method = 'approx' # Update time-stepping method
approx_order = 1 # Fine tuning of the time-stepping method
tol = 1e-14 #   "     "       "


# -------------------------------------------------------
# Single population (singlepop)
# -------------------------------------------------------
"""This singlepop simulation provides a simple feedforward topology that uses
every major class in the core library. A single 100 Hz External population population
provides excitatory input. (Note that although here this frequencys specified as a
string, a floating point or integer specification will also work). This external
population is connected to an Internal population (modeled as a population density pde)
via a delta-distributed synaptic weight distribution, with 5 mV strength. The in-degree
(nsyn) of this Connection is set to 1 for this example. In general, this serves as a
multiplier of the input firing rate of the source population. The internal population
has a linearly binned voltage domain from v_min to v_max. No negative bins (i.e. v_min < 0)
are required here, because no negative synaptic inputs ('weights' in the Connection object)
are defined.
"""

# Create simulation
externalPopFreq = '50'
b1 = ExternalPopulation(externalPopFreq, record=True)
i1 = InternalPopulation(v_min=0, v_max=.02, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)
b1_i1 = Connection(b1, i1, 1, weights=[.005], probs=[1.], delay=0.0)
simulation = Simulation([b1, i1], [b1_i1], verbose=verbose)

# Run simulation
simulation.run(dt=dt, tf=tf, t0=t0)
    
# Visualize results
i1 = simulation.population_list[1]
plt.figure(figsize=(3,3))
plt.plot(i1.t_record, i1.firing_rate_record)
plt.xlim([0,tf])
plt.ylim(ymin=0)
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (Hz)')
plt.tight_layout()
plt.savefig('./singlepop2.png')

