# Import dependencies
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import matplotlib.pyplot as plt
import pandas as pd # pandas for nice tables
import skimage.io as io

# --------------------------
# Cache (between experiment) information
# --------------------------

# Download connectivity data cache
'''The manifest file is a simple JSON file that keeps track of all of
the data that has already been downloaded onto the hard drives.
If you supply a relative path, it is assumed to be relative to your
current working directory.'''
mcc = MouseConnectivityCache()

# Get a list of all of the experiments
all_experiments = mcc.get_experiments(dataframe=True)
print("%d total experiments" % len(all_experiments))

# Get the StructureTree instance
structure_tree = mcc.get_structure_tree()

# Get info on some structures
structures = structure_tree.get_structures_by_name(['Primary visual area', 'Hypothalamus'])
pd.DataFrame(structures)

# --------------------------
# Experiment-specific information
# --------------------------

# Take a look at what we know about an experiment with a retina (default: primary motor) injection
regionOfInterest = 'LGN'
if regionOfInterest == 'retina':
    experientID = 306098703 # (default: 122642490)
elif regionOfInterest == 'LGN':
    experientID = 479891303
all_experiments.loc[experientID]

# Get projection density: number of projecting pixels / voxel volume
projd, projd_info = mcc.get_projection_density(experientID)

# Get projection density (averaged over anterior-posterior axis)
projd_mip = projd.max(axis=0)

# Get annotated anatomical image
template, template_info = mcc.get_template_volume()
annot, annot_info = mcc.get_annotation_volume()

# Plot projection density
colourMap = 'gist_gray'
img = io.imread('./atlasVolume/atlasVolume.mhd', plugin='simpleitk')
plt.imshow(img[:,:,296].T, cmap=colourMap, aspect='equal')
plt.imshow(projd_mip, cmap='gist_heat', alpha=.5, aspect='equal')
plt.show()