import os
import h5py
import numpy as np
from neuron import h
from bmtk.simulator import bionet
from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar, xiter_random
from bmtk.builder.auxi.edge_connectors import connect_random, distance_connector
from bmtk.utils.sim_setup import build_env_bionet
from bmtk.analyzer.cell_vars import plot_report
import matplotlib.pylab as plt
'''NOTE: Run in Docker container to ensure that this scripts runs properly'''

# General parameters
savedir = 'sim_ch02'

# ----------------------------------------------------
# 1. Build the network
# ----------------------------------------------------

##########################################
# Cortex network
##########################################

# Create the cortex network
'''The first step is to use the bmtk Network Builder to create and save the network. We first instantiate a network with a
name or our choosing. As we will use cell models from mouse cortex in this tutorial, we'll call our network 'mcortex'''
cortex = NetworkBuilder('mcortex')

# Create network of cells
'''Once we have a network, we can add nodes (i.e. cells) by calling the add_nodes() method. Here, we have 100 cell. all of
which are of the same type, but with different locations and y-axis rotations. The positions of each cell is defined by the
columinar built-in method, which will random place our cells in a column. The rotation_angle_yaxis is similarly defined by
a built-in function that randomly assigns each cell a given y angle.'''
N = 10
cortex.add_nodes(
    N=N, # number of cells
    pop_name='Scnn1a',
    positions=positions_columinar(N=N, center=[0, 50.0, 0], max_radius=30.0, height=100.0), # position cells within column
    rotation_angle_yaxis=xiter_random(N=N, min_x=0.0, max_x=2*np.pi),
    rotation_angle_zaxis=3.646878266, # note here that z rotations are all the same, but they could be different by using the function above
    potental = 'exc', # indicate that it is an excitatory type cell (optional)
    model_type = 'biophysical', # used by the simulator to indicate that we are using a biophysical cell.
    dynamics_params = '472363762_fit.json', # model parameters (file will be downloaded from the Allen Cell Types Database)
    morphology = 'Scnn1a_473845048_m.swc', # - Model morphology (file will be downloaded from the Allen Cell Types Database)
    model_processing = 'aibs_perisomatic', # - a custom function used by the simulator to load the model into NEURON using Allen Cell-Types files for perisomatic models
    model_template = 'ctdb:Biophys1.hoc'
    # cell_name = 'Scnn1a_473845048', # name/type of cell we will be modeling (optional; when N=1)
)

# Create recurrent connections within population
'''Next we want to add recurrent edges. To create the connections we will use the built-in distance_connector function, which
will assign the number of connections between two cells randomly (between range nsyn_min and nsysn_max) but weighted by distance.
The other parameters, including the synaptic model (AMPA_ExcToExc) will be shared by all connections.
    To use this, or customized connection functions, we must pass the name of our connection function using the "connection_rule"
parameter, and the function parameters through "connection_params" as a dictionary. The connection_rule method isn't explicitly
called by the script. Rather when the build() method is called, the connection_rule will iterate through every source/target node
pair, and use the rule and build a connection matrix.'''
cortex.add_edges(
    source = {'pop_name': 'Scnn1a'}, target = {'pop_name': 'Scnn1a'},
    connection_rule = distance_connector,
    connection_params = {'d_weight_min': 0.0, 'd_weight_max': 0.34, 'd_max': 50.0, 'nsyn_min': 0, 'nsyn_max': 10},
    syn_weight = 2.0e-04,
    distance_range = [30.0, 150.0],
    target_sections = ['basal', 'apical', 'soma'],
    delay = 2.0,
    dynamics_params = 'AMPA_ExcToExc.json',
    model_template = 'exp2syn')


# Build and save the network
cortex.build()
cortex.save_nodes(output_dir = '%s/network' % (savedir))

##########################################
# External network
##########################################

# Add spike generating (thalamus) cells
'''We will also want a collection of external spike-generating cells that will synapse onto our cell. To do this we create
a second network which can represent thalamic input. We will call our network "mthalamus", and it will consist of 10 cells.
These cells are not biophysical but instead "virtual" cells. Virtual cells don't have a morphology or the normal properties
of a neuron, but rather act as spike generators'''
thalamus = NetworkBuilder('mthalamus')
thalamus.add_nodes(
    N = 100,
    pop_name = 'tON',
    potential = 'exc',
    model_type = 'virtual'
)

# Create synapses
'''Now that we built our nodes, we want to create a connection between our 10 thalamic cells onto our cortex cell. To do so
we use the add_edges function like so:'''
thalamus.add_edges(
    # Define source  and target cells
    source = {'pop_name': 'tON'}, # note: we could also use source=thalamus.nodes(), or source={'level_of_detail': 'filter'})
    target = cortex.nodes(),
    # Connection parameters
    connection_rule = connect_random,
    connection_params={'nsyn_min': 0, 'nsyn_max': 12},
    # connection_rule = 5, # how many synapses exists between every source/target pair
    syn_weight = 0.001,
    delay = 2.0, # ms
    weight_function = None, # used to adjust the weights before runtime
    # Determine where on the post-synaptic cell to place the synapse
    target_sections = ['basal', 'apical'],
    distance_range = [0.0, 150.0],
    # Set the parameters of the synpases
    dynamics_params = 'AMPA_ExcToExc.json', # AMPA type synaptic model with an Excitatory connection
    model_template = 'exp2syn')

# Build thalamus (saving nodes and edges)
thalamus.build()
thalamus.save_nodes(output_dir='%s/network' % (savedir))
thalamus.save_edges(output_dir='%s/network' % (savedir))

# Set spike trains
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
spikesFilename = '%s/inputs/mthalamus_spikes.h5' % (savedir)
psg = PoissonSpikeGenerator(population='mthalamus')
psg.add(node_ids=range(10),  # Have 10 nodes to match mthalamus
        firing_rate=15.0,    # 10 Hz, we can also pass in a nonhomoegenous function/array
        times=(0.0, 3.0))    # Firing starts at 0 s up to 3 s
if not(os.path.isfile(spikesFilename)):
    psg.to_sonata(spikesFilename)

# ----------------------------------------------------
# 2. Set up the BioNet environment
# ----------------------------------------------------
'''Before running a simulation, we will need to create the runtime environment, including parameter files, run-script
and configuration files. This will also compile mechanisms'''

# Mechanisms need to be compiled?
compile_mechanisms = True
if os.path.isdir('%s/components/mechanisms/x86_64/' % (savedir)):
    compile_mechanisms = False

# Build network
build_env_bionet(
    base_dir = savedir,      
    network_dir = '%s/network'  % (savedir),
    tstop = 3000.0, dt = 0.1,
    report_vars = ['v', 'cai'],     # Record membrane potential and calcium (default soma)
    spikes_inputs = [('mthalamus', '%s/inputs/mthalamus_spikes.h5'  % (savedir))],  # Name of population which spikes will be generated for   
    include_examples = True,    # Copies components files
    compile_mechanisms = compile_mechanisms   # If true, will try to compile NEURON mechanisms
)

# Update the configuration file to read "thalamus_spikes.csv"
inputsFilename = '%s/simulation_config.json' % (savedir)
with open(inputsFilename, 'r') as json_file:
    jsonText = json_file.readlines()
    for i,line in enumerate(jsonText):
        if '"input_file"' in line:
            jsonText[i] = '      "input_file": "${BASE_DIR}/inputs/mthalamus_spikes.h5",\n'
with open(inputsFilename, 'w') as json_file:
    json_file.write(''.join(jsonText))


# ----------------------------------------------------
# 3. Run the simulation
# ----------------------------------------------------
'''Once our config file is setup we can run a simulation either through the command line:
    $ python run_bionet.py simulation_config.json
or through the script:'''
conf = bionet.Config.from_json('%s/simulation_config.json' % (savedir))
conf.build_env()
net = bionet.BioNetwork.from_config(conf)
sim = bionet.BioSimulator.from_config(conf, network=net)
sim.run()

# ----------------------------------------------------
#  Analyse the run
# ----------------------------------------------------

# # Plot data reports (only reporting cai?)
# plot_report(config_file='sim_ch02/simulation_config.json', report_name='v', report_file='%s/output/v_report.h5' % (savedir))

# # Plot raster (also not working)
# plot_raster(config_file='sim_ch03/simulation_config.json')
# plt.savefig('%s/rasterResults' % (savedir))

# Get data
def getData(var):
    f = h5py.File('%s/output/%s_report.h5' % (savedir, var), 'r')
    data = f['report']['mcortex']
    return np.array(data['data'][()]), data['mapping']
v_rec, v_map = getData('v')
cai_rec, cai_map = getData('cai')

# Plot data
fig,ax = plt.subplots(2,1)
ax[0].plot(v_rec, label='x')
ax[1].plot(cai_rec, label='cai')
[ax[i].legend() for i in range(len(ax))]
plt.savefig('%s/simulationResults' % (savedir))

# # Get spikes
# f = h5py.File('%s/output/spikes.h5' % (savedir), 'r')

# # Get segment locations
# getXYZ = lambda sec,loc : [sec.x3d(loc), sec.y3d(loc), sec.z3d(loc)]
# xyz_starts, xyz_ends = np.zeros(shape=(0,3)), np.zeros(shape=(0,3))
# for sec in h.allsec():
#     start,end = [getXYZ(sec,0), getXYZ(sec,1)]
#     xyz_starts = np.vstack((xyz_starts, start))
#     xyz_ends = np.vstack((xyz_ends, end))

# # Plot segments
# plt.figure()
# # nsegs = 1000
# # xyz_starts = xyz_starts[0:nsegs,:]; xyz_ends = xyz_ends[0:nsegs,:]
# for starts, ends in zip(xyz_starts, xyz_ends):
#     plt.plot([starts[0],ends[0]], [starts[1],ends[1]], linewidth=1, color='k', alpha=.5)
# plt.savefig('network')