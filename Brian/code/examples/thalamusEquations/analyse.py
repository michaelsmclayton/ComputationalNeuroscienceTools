from matplotlib.pyplot import *
import pickle
from populations import *

# Import results
with open('simulationResults.pickle', 'rb') as handle:
    results = pickle.load(handle)

ajksd

#-------------------------------------------
# Analyse / plot results
#-------------------------------------------
populationData = {}
figure()
for index, pop in enumerate(populations):
    populationData[pop] = globals()[pop+'data'].V[0]
    subplot(2, 2, index+1)
    plot(populationData[pop])
    ylabel(pop)
show()