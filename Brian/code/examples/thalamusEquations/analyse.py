from matplotlib.pyplot import *
import numpy as np
import pickle
from populations import *
from scipy.signal import butter, sosfilt, sosfreqz, welch

sampleRate = 1000
plotWindow = range(10*sampleRate, 39*sampleRate)

# -----------------------------------------------
#           (PRE)PROCESSING FUNCTIONS
# -----------------------------------------------

# Import results
def importData(filepath):
    with open(filepath, 'rb') as handle:
        populationData = pickle.load(handle)
    return populationData

# Band-pass filtering
def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], analog=False, btype='band', output='sos')
def butter_bandpass_filter(data, lowcut, highcut, fs=sampleRate, order=10):
        sos = butter_bandpass(lowcut, highcut, fs, order)
        y = sosfilt(sos, data)
        return y

# Power analysis
def powerSpectralDensity(data):
    win = 1 * sampleRate # Define window length (2 seconds)
    freqs, psd = welch(data, sampleRate, nperseg = win)
    return freqs, psd


# -----------------------------------------------
#               ANALYSIS FUNCTIONS
# -----------------------------------------------

# Define plotting function
def analyseSingleResult(populationData, plotWindow=plotWindow):

    #-------------------------------------------
    # Analyse / plot results
    #-------------------------------------------
    figure()
    areasToPlot = ['RET', 'TCR', 'IN', 'TRN']
    times = populationData['times']
    for index, pop in enumerate(areasToPlot):
        
        # Process data
        dataToPlot = populationData[pop]
        if pop != 'RET':
            dataToPlot = butter_bandpass_filter(dataToPlot, lowcut=1, highcut=100)
        
        # Plot raw data
        figure(1, figsize=(12,5))
        subplot(len(areasToPlot), 1, index+1)
        plot(times[plotWindow], dataToPlot[plotWindow], linewidth=.5)
        ylabel(pop)
        
        # Plot power spectral density
        figure(2, figsize=(5,12))
        subplot(len(areasToPlot), 1, index+1)
        freqs, psd = powerSpectralDensity(dataToPlot[plotWindow])
        plot(freqs, psd, color='k', lw=2)
        xlabel('Frequency (Hz)')
        ylabel('Power spectral density (V^2 / Hz)')
        title(pop)
        xlim([1, 30])
    show()


def analyseManyResults(numberOfIterations, dataPath, areasToPlot=['RET','TCR','IN','TRN'], plotWindow=plotWindow):

    # Initialise data store
    combinedData = {}
    for pop in areasToPlot:
        combinedData[pop] = np.empty([501, numberOfIterations])

    # Loop over iterations
    for iteration in range(numberOfIterations):

        # Load data
        populationData = importData(dataPath + '%s.pkl' % (iteration))

        # Loop over areas
        for index, pop in enumerate(areasToPlot):

            # Get current data
            currentData = populationData[pop]

            # Filter data
            currentData = butter_bandpass_filter(currentData, lowcut=1, highcut=100)

            # Get power spectral density
            freqs, psd = powerSpectralDensity(currentData[plotWindow])

            # Store current power spectral density
            combinedData[pop][:,iteration] = psd

    # Average over iterations
    for index, pop in enumerate(areasToPlot):
        subplot(1,len(areasToPlot),index+1)
        meanSpectrum = np.nanmean(combinedData[pop], axis=1)
        # meanSpectrum -= max(meanSpectrum)/freqs # ** -4
        plot(freqs, meanSpectrum)
        title(pop)
        # ylabel('Power spectral density (V^2 / Hz)')
        xlim([.1, 40])
    show()

if __name__ == "__main__":
    analyseManyResults(numberOfIterations=2, \
    dataPath='savedData/simulationResults_', \
    areasToPlot=['TCR','IN','TRN'], plotWindow=plotWindow)


    
    
