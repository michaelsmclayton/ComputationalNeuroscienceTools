
# to install - pip3 install brian2
from brian2 import * # Import Brian
from matplotlib.pyplot import *

# Once you have some neurons, the next step is to connect them up via
# synapses. We'll start out with doing the simplest possible type of
# synapse that causes an instantaneous change in a variable after a spike.
Cm = 250*pF # /cm**2 # membrane time constant
tau_m = 10*ms
eqs = '''
dv/dt = (-v/tau_m) + (I/Cm) : volt
I : volt
'''
G = NeuronGroup(1, eqs, threshold='v>1', reset='v = 0', method='exact')
G.I = 2*mV # Input for neuron one = 2; two = 0

# # Given that neuron two has no driving input (i.e. as I = 0), this neuron
# # will not spike on its own. However, we can influence its activity by linking
# # it to neuron one via a synapse. In Brian, we can define synapses using the
# # following code:
# #   Synapses(source, target, ...) [i.e. going from source neurons to target neurons]
# # In the case below, the source and target are both the same (i.e. the group G).
# # The syntax "on_pre='v_post += 0.2'" means that when a spike occurs in the
# # presynaptic neuron (hence on_pre), it causes an instantaneous change to happen
# # v_post += 0.2. The _post means that the value of v referred to is the post-
# # synaptic value, and it is increased by 0.2. Therefore, in total, what this
# # model says is that whenever two neurons in G are connected by a synapse,
# # when the source neuron fires a spike, the target neuron will have its
# # value of v increased by 0.2.
# S = Synapses(G, G, on_pre='v_post += 0.2')

# # However, at this point we have only defined the synapse model, we haven't
# # actually created any synapses. The next line S.connect(i=0, j=1) creates
# # a synapse from neuron 0 to neuron 1.
# S.connect(i=0, j=1)

# Run simulation and plot
M = StateMonitor(G, 'v', record=True)
run(100*ms)
plot(M.t/ms, M.v[0], label='Neuron 0')
#plot(M.t/ms, M.v[1], label='Neuron 1')
xlabel('Time (ms)')
ylabel('v')
legend()
show()