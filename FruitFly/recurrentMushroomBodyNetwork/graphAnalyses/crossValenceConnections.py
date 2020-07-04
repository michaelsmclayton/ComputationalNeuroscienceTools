import os
import networkx as nx
import matplotlib.pylab as plt
import numpy as np
import pprint; pp = pprint.PrettyPrinter(depth=11).pprint
import copy

# ------------------------------------------------------
# Analye full graph
# ------------------------------------------------------

# Initiate graph
G = nx.MultiDiGraph() # directed graph

# Load data
cond = 'control'
os.chdir('../') # change working directory to parent directory
exec(open('loadmat.py').read())

# Create graph
for pre in range(J.shape[0]):
    for post in range(J.shape[1]):
        if J[pre,post]>0:
            G.add_edge(names[pre], names[post], weight=J[pre,post])

# State nodes (no links from h1)
nodes = {'appetitive': ['i1','k1'], 'aversive': ['f1','g1','d1']}

# Get links
nodeTypes = list(nodes.keys())
links = {'appetitive': {}, 'aversive': {}}
for i in range(2): # reverse order between appetitive and aversive
    names = nodeTypes if i==0 else [name for name in reversed(nodeTypes)]
    for pre in nodes[names[0]]:
        for post in nodes[names[1]]:
            links[names[0]][f'{pre}-{post}'] = nx.shortest_path(G, source=f'DAN-{pre}', target=f'DAN-{post}')

# Get link weights
weights = copy.deepcopy(links)
for nodeType in nodeTypes:
    currentLinks = weights[nodeType]
    for key in currentLinks.keys():
        currentWeights = []
        currentData = currentLinks[key]
        for i in range(len(currentData)-1):
            currentWeights.append(G.get_edge_data(currentData[i], currentData[i+1], default=0)[0]['weight'])
        currentLinks[key] = currentWeights


# Show links
print('\nLinks...')
pp(links)
print('\nWeights...')
pp(weights)

# Node that appetitive-aversive links exclusively go through FBN (i.e. within compartment feedback),
# while aversive-appetitive links generally go through FAN (feed-across neurons; 4/6)

# ------------------------------------------------------
# Create subgraph
# ------------------------------------------------------

# Define methods to plot subgraph
def plotSubgraph(nodeType, ax):

    # Get opposite type
    oppositeType = 'aversive' if nodeType=='appetitive' else 'appetitive'

    # Initialise subgraph
    subG = nx.MultiDiGraph() # directed graph

    # Get positions
    positions = {}
    currentLinks = links[nodeType]
    for col, key in enumerate(currentLinks.keys()):
        for row, node in enumerate(currentLinks[key]):
            if not(node in positions):
                positions[node] = [row/len(currentLinks[key]),col/len(currentLinks.keys())]

    # Center DANs
    def centerDANs(nodeTypeName,xPos):
        for i, node in enumerate(nodes[nodeTypeName]):
            # positions[f'DAN-{node}'][0] = xPos
            positions[f'DAN-{node}'][1] = np.linspace(.25,.75,len(nodes[nodeTypeName]))[i]
    centerDANs(nodeType, 0)
    centerDANs(oppositeType, 1)

    # Add nodes and edges
    for key in currentLinks:
        for i in range(len(currentLinks[key])-1):
            subG.add_edge(currentLinks[key][i], currentLinks[key][i+1], weight=weights[nodeType][key][i])

    # Plot each neuron type
    def getNodeList(type):
        nodelist = []
        for node in positions:
            if type in node:
                nodelist.append(node)
        return nodelist

    # Draw network
    neuronTypes = ['FAN', 'FBN', 'FB2', 'MBON']
    neuronLists = [getNodeList(neuron) for neuron in neuronTypes]
    neuronLists.append([f'DAN-{neuron}' for neuron in nodes['appetitive']])
    neuronLists.append([f'DAN-{neuron}' for neuron in nodes['aversive']])
    neuronColors = ['#ecff5a', '#ffb544', '#48d627', '#989898', '#7d79f9', '#f96666']
    colors = [data['weight'] for node1, node2, data in subG.edges(data=True)]
    for neuronList,color in zip(neuronLists, neuronColors):
        nx.draw_networkx(subG, arrows=True, node_size=1000, nodelist=neuronList, width=1, font_size=12, font_weight='normal', \
            font_family='Arial', node_color=color, pos=positions, ax=ax) # edge_color=colors, edge_cmap=plt.cm.Greys, vmax=10
    ax.set_title(f'A{nodeType[1:]} DAN to {oppositeType} DAN connections', fontdict={'fontsize': 14,'fontweight' : 'normal','fontfamily': 'Arial'})
    ax.axis('off')


# Display subnetworks
fig, axes = plt.subplots(2,1, figsize=(8, 10))
for i,nodeType in enumerate(nodeTypes):
    plotSubgraph(nodeType, axes[i])
plt.show()
