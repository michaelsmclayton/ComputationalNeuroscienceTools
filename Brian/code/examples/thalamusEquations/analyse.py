from matplotlib.pyplot import *
import pickle
from populations import *

# Define plotting function
def plotResult(populationData, plotWindow=[150000, 250000]):

    #-------------------------------------------
    # Analyse / plot results
    #-------------------------------------------
    figure()
    areasToPlot = ['RET', 'TCR', 'IN', 'TRN']
    times = populationData['times']
    plotWindow = [100000, 250000]
    for index, pop in enumerate(areasToPlot):
        subplot(len(areasToPlot), 1, index+1)
        plot(times[plotWindow[0]:plotWindow[1]], populationData[pop][plotWindow[0]:plotWindow[1]], linewidth=.5)
        ylabel(pop)
    show()
    # savefig('current2.png')

if __name__ == "__main__":

    # Import results
    with open('simulationResults_new.pkl', 'rb') as handle:
        populationData = pickle.load(handle)

    # Plot data

# plotWindow = [150000, 250000]
# samplingRate = 1000
# timeSeries = populationData['TCR'][plotWindow[0]:plotWindow[1]]

# results = welch_psd(timeSeries, fs=samplingRate)

# dataLength = len(results[0])
# def getFrequencyIndex(percent):
#     return int(percent*dataLength)
# startIndex = getFrequencyIndex(.05)
# endIndex = getFrequencyIndex(.15)
# rangeOfInterest = range(startIndex, endIndex)
# plot(results[0][rangeOfInterest], results[1][rangeOfInterest])
# show()