from pyrates.frontend import OperatorTemplate, NodeTemplate, CircuitTemplate
from brian2 import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# https://github.com/pyrates-neuroscience/PyRates/blob/master/documentation/Tutorial_PyRates_Basics.ipynb


# ---------------------------------------------------
# Define mathematical operators
# ---------------------------------------------------

'''Via these two operators, each population of the Jansen-Rit circuit can be defined'''

# # Parameters
# def returnProcessedParameters(m_max, r, V_thr, h, tau):
#     m_max = m_max * ms
#     r =  (1/r) / mV
#     V_thr = V_thr / mV
#     h = h / mV
#     tau = tau * ms
#     return m_max, r, V_thr, h, tau

# # Define parameters
# m_max = 5 * Hz
# r = .56 * mV ** -1
# V_thr = 6 * mV
# h = 14 * mV
# tau = 100 * second **-1
# m_max, r, V_thr, h, tau = returnProcessedParameters(m_max, r, V_thr, h, tau)

# Parameters
m_max, r, V_thr, h, tau = 5, 560.0, .006, .00325, .01


# Define potential-to-rate operator (PRO) = (VOLTAGE TO RATE; SIGMOIDAL TRANSFER FUNCTION)
PRO = OperatorTemplate(name='PRO', path=None, 
    equations = ['m_out = m_max / (1. + exp(r*(V_thr - PSP)))'],
    variables = {
        'm_out': {'default': 'output'}, # average firing rate
        'PSP': {'default': 'input'}, # membrane potential
        'm_max': {'default': m_max}, # maximum firing rate
        'r': {'default': r}, # firing threshold variance
        'V_thr': {'default': V_thr}}, # average firing threshold
    description="")

# Define rate-to-potential operator (RPO)
RPO_e = OperatorTemplate(name='RPO_e', path=None,
    equations = [ 'd/dt * PSP = PSP_t',
                  'd/dt * PSP_t =  (h/tau*m_in) - 2.*(1./tau)*PSP_t - ((1./tau)^2.)*PSP'],
    variables = {
        'h': {'default': h}, # efficacy of the synapse
        'tau': {'default': tau}, # time-scale of the synapse
        'm_in': {'default': 'input'}, # input firing rate
        'PSP': {'default': 'output'}, # average post-synaptic potential
        'PSP_t': {'default': 'variable'}},
    description="")

# PRO = OperatorTemplate.from_yaml("model_templates.jansen_rit.simple_jansenrit.PRO")
# RPO_e = OperatorTemplate.from_yaml("model_templates.jansen_rit.simple_jansenrit.RPO_e")

# ---------------------------------------------------
# Build the Jansen-Rit circuit (JRC)
# ---------------------------------------------------
"""
Defines a population with 2 operators, one RPO and one PRO, transforming its incoming firing
rates into post-synaptic membrane potential changes and its average membrane potential back
into an average firing rate. (Note that the order in which the operators of a node are defined
does not matter. This is because the operators are re-arranged internally, according to the
hierarchical dependencies defined by their input and output variables.
"""

# Population templates
iin = NodeTemplate(name='IIN', path=None, operators=[PRO,RPO_e]) # Inhibitory interneuron population (IIN)
ein = NodeTemplate(name='EIN', path=None, operators=[PRO,RPO_e]) # Excitatory interneurons population (EIN)
"""While the excitatory interneurons are defined via the same operator structure as the excitatory
interneurons, the pyramidal cells have two different RPOs, representing different synaptic dynamics
for incoming inhibitory and excitatory signals"""
RPO_e_pc = OperatorTemplate.from_yaml("model_templates.jansen_rit.simple_jansenrit.RPO_e_pc")
RPO_i = OperatorTemplate.from_yaml("model_templates.jansen_rit.simple_jansenrit.RPO_i")
pc = NodeTemplate(name='PC', path=None, operators=[RPO_e_pc, RPO_i, PRO]) # Pyramidal cells (PC)

# Circuits (lists of nodes, coupled via edges)
""" Edges are defined by a list with four entries (1/2/3/4):
1) The source variable (PC/PRO/m_out refers to variable m_out in operator PRO of node PC) /
2) The target variable /
3) An edge template with additional operators (null means no particular edge template is used)
4) A dictionary of variables and values that are specific to this edge"""
circuit = CircuitTemplate(
    name="JRC", nodes={'IIN': iin, 'EIN': ein, 'PC': pc},
    edges = [ \
        ["PC/PRO/m_out", "IIN/RPO_e/m_in", None, {'weight': 33.75}],
        ["PC/PRO/m_out", "EIN/RPO_e/m_in", None, {'weight': 135.}],
        ["EIN/PRO/m_out", "PC/RPO_e_pc/m_in", None, {'weight': 108.}],
        ["IIN/PRO/m_out", "PC/RPO_i/m_in", None, {'weight': 33.75}]],
    path=None)

# Instantiate (apply) circuit
circuit = circuit.apply()


# # ---------------------------------------------------
# # Visualise the network
# # ---------------------------------------------------
# pos = nx.spring_layout(circuit.graph)
# nx.draw_shell(circuit.graph, with_labels=True, node_size=2000, arrowsize=30)
# plt.show()


# ---------------------------------------------------
# Run simulation
# ---------------------------------------------------

# Setup backend
T, dt = 10.0, 0.001
compute_graph = circuit.compile(vectorization="nodes", dt=dt, backend="numpy")
"""Here, the keyword argument vectorization is used to group identical nodes such that their
equations can be calculated more efficiently. A such reorganised network consists of 2 instead
of 3 nodes, because the EIN and IIN nodes actually contain the same equations"""
# compute_graph.nodes = NodeView(('vector_node0', 'vector_node1'))

# Define external input
simulationLength = int(T/dt)
random_input = np.random.uniform(120., 320., (simulationLength, 1)) # Define external input
ext_input = random_input # np.zeros(shape=[1,simulationLength])
for i in range(simulationLength):
    ext_input[i] = i/20 # ext_input[i] * (i/simulationLength) + (i*.05)

# Run simulation
results = compute_graph.run(
    simulation_time = T,
    outputs={'V_PC': 'PC/RPO_e_pc/PSP',
            'V_EIN': 'EIN/RPO_e/PSP',
            'V_IIN': 'IIN/RPO_e/PSP'}, 
    inputs={'PC/RPO_e_pc/u': ext_input})

# Plot results
from pyrates.utility import plot_timeseries, plot_psd, time_frequency
times = np.linspace(0,10,10000)
plt.figure(figsize=(20,10))
plt.subplot(3,1,1)
plt.plot(times, ext_input)
plt.xlim(0,10)
ax1 = plt.subplot(3,1,2)
plot_timeseries(results, ax=ax1, ylabel='PSP in V')
plt.xlim(0,10)
ax2 = plt.subplot(3,1,3)
# plot_psd(results, ax=ax2)
timeFreq = time_frequency(results, freqs=np.arange(3,30,1))
plt.pcolormesh(timeFreq[0], alpha=1, cmap="binary")
plt.ylim(0,15)
plt.ylabel('Frequency (Hz)')
plt.show()