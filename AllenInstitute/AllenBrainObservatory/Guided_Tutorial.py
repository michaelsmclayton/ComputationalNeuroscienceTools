
# ----------------------------------
# # Import dependencies
# ----------------------------------
import numpy as np
import pandas as pd
from matplotlib.pyplot import *
'exec(%matplotlib inline)'
import pprint
objectPrint = pprint.PrettyPrinter(depth=6).pprint
import os
import copy
import matplotlib.animation as animation

# Import Allen Institute Observatory library
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache()

# ----------------------------------
# Get meta-data
# ----------------------------------

# list of all targeted areas, using Allen Brain Atlas nomenclature
boc.get_all_targeted_structures()
# ['VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl']

# list of all imaging depths
boc.get_all_imaging_depths()
# [175, 185, 195, 200, 205, 225, 250, 265, 275, 276, 285, 300, 320, 325, 335, 350, 365, 375, 390, 400, 550, 570, 625]

# list of all cre driver lines 
boc.get_all_cre_lines()
# ['Cux2-CreERT2', 'Emx1-IRES-Cre', 'Fezf2-CreER', 'Nr5a1-Cre', 'Ntsr1-Cre_GN220', 'Pvalb-IRES-Cre', 'Rbp4-Cre_KL100', 'Rorb-IRES2-Cre', 'Scnn1a-Tg3-Cre', 'Slc17a7-IRES2-Cre', 'Sst-IRES-Cre', 'Tlx3-Cre_PL56', 'Vip-IRES-Cre']

# list of all GCaMP reporter lines
boc.get_all_reporter_lines()
# ['Ai148(TIT2L-GC6f-ICL-tTA2)', 'Ai162(TIT2L-GC6s-ICL-tTA2)', 'Ai93(TITL-GCaMP6f)', 'Ai93(TITL-GCaMP6f)-hyg', 'Ai94(TITL-GCaMP6s)']

# list of all stimuli
boc.get_all_stimuli()
# ['drifting_gratings', 'locally_sparse_noise', 'locally_sparse_noise_4deg', 'locally_sparse_noise_8deg', 'natural_movie_one', 'natural_movie_three', 'natural_movie_two', 'natural_scenes', 'spontaneous', 'static_gratings']


# ----------------------------------
# Experiment containers & sessions
# ----------------------------------
'''The experiment container describes a set of 3 experiment sessions performed
for the same field of view (ie. same targeted area and imaging depth in the same
mouse that targets the same set of cells). Each experiment container has a
unique ID number.'''

# Choose a visual area and Cre line from the lists above
visual_area = 'VISp'
cre_line = 'Cux2-CreERT2'

# Get the list of all the experiment containers for that area and Cre line combination.
exps = boc.get_experiment_containers(
    targeted_structures=[visual_area], 
    cre_lines=[cre_line],
)
# pd.DataFrame(exps)

# Let's look at one experiment container, imaged from Cux2, in VISp, from imaging depth 175 um.
experiment_container_id = 511510736
exp_cont = boc.get_ophys_experiments(
    experiment_container_ids=[experiment_container_id], # Choose from experiment containers
    stimuli=['natural_scenes'] # with given stimuli
)
# pd.DataFrame(exp_cont)
# objectPrint(exp_cont)


# Get data for an experiment
'''The Ophys Experiment data object gives us access to everything in the NWB file for a
single imaging session. Note: if you are following along on your own machine, the following
command will download data if it has not been previously retrieved. You should see a warning
to this effect.'''
session_id = exp_cont[0]['id']
data_set = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)

# Get Maximum projection
'''This is the projection of the full motion corrected movie. It shows all of the cells imaged during the session'''
max_projection = data_set.get_max_projection()

# ROI Masks
'''These are all of the segmented masks for cell bodies in this experiment'''
rois = data_set.get_roi_mask_array()

# Get df/f traces
ts, dff = data_set.get_dff_traces()

# Animate df/f traces
fig =figure(figsize=(8,8))
# imshow(max_projection, cmap='gray', alpha=.5)
blankImage = np.zeros(shape=(rois.shape[1], rois.shape[2]))
my_cmap = copy.copy(cm.get_cmap('gray')) # get a copy of the gray color map
my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values
image = imshow(blankImage, vmin=0, vmax=1.5, cmap=my_cmap)
def animate(i):
    combinedImage = np.zeros(shape=(rois.shape[1], rois.shape[2]))
    for neuron in range(len(rois)):
        currentImage = rois[neuron,:,:] * dff[neuron,i]
        combinedImage = combinedImage + currentImage
    combinedImage[combinedImage<.05] = np.nan
    image.set_array(combinedImage)
    return image
image = animate(0)
anim = animation.FuncAnimation(fig, animate, interval=1)
gca().set_facecolor((0,0,0))
show()