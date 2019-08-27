import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dipde.internals.internalpopulation import InternalPopulation
from dipde.internals.externalpopulation import ExternalPopulation
from dipde.internals.simulation import Simulation
from dipde.internals.connection import Connection as Connection

# Settings:
t0 = 0.
dt = .0001
dv = .0001
tf = .1
verbose = True
update_method = 'approx'
approx_order = 1
tol = 1e-14

# Create simulation:
b1 = ExternalPopulation('100', record=True)
i1 = InternalPopulation(v_min=0, v_max=.02, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)
b1_i1 = Connection(b1, i1, 1, weights=[.005], probs=[1.], delay=0.0)
simulation = Simulation([b1, i1], [b1_i1], verbose=verbose)

# Change update method
def update(self):
    self.update_total_input_dict()
    self.update_propability_mass()
    self.update_firing_rate()
    if self.record == True: self.update_firing_rate_recorder()
#i1.update = update


# Run simulation:
simulation.run(dt=dt, tf=tf, t0=t0)

# Visualize:
i1 = simulation.population_list[1]
# plt.figure(figsize=(3,3))
# plt.plot(i1.t_record, i1.firing_rate_record)
# plt.xlim([0,tf])
# plt.ylim(ymin=0)
# plt.xlabel('Time (s)')
# plt.ylabel('Firing Rate (Hz)')
# plt.tight_layout()
# #plt.savefig('./singlepop2.png')
