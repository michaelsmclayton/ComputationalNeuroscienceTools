from matplotlib.pyplot import *
import numpy as np
import pickle
from populations import *
from scipy.signal import butter, sosfilt, sosfreqz, welch
from scipy.signal import savgol_filter
# from fooof import FOOOF

# Analysis settings
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
    win = 4 * sampleRate # Define window length (2 seconds)
    freqs, psd = welch(data, sampleRate, nperseg = win)
    return freqs, psd

# def removeBackroundNoise(spectrum, freqs, freq_range = [3, 40]):
#     # Initialize FOOOF object
#     fm = FOOOF()
#     fm.fit(freqs, spectrum, freq_range)
#     fres = fm.get_results()
#     backgroundParams = fres.background_params
#     backgroundNoise = backgroundParams[0] - np.log(freqs**backgroundParams[1])
#     return spectrum - backgroundNoise


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
        combinedData[pop] = np.empty([2001, numberOfIterations])

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

            # Plot raw data
            if iteration < 7: # Limit plotting to first few
                figure(iteration+1)
                subplot(len(areasToPlot),1,index+1)
                plot(populationData['times'][plotWindow], currentData[plotWindow], linewidth=.5)
                ylabel(pop)

            # Get power spectral density
            freqs, psd = powerSpectralDensity(currentData[plotWindow])

            # Plot spectral data
            figure(202)
            subplot(len(areasToPlot),1,index+1)
            plot(freqs, psd, linewidth=.5)
            ylabel(pop)
            xlim([3, 40])

            # Store current power spectral density
            combinedData[pop][:,iteration] = psd

    # Average over iterations
    for index, pop in enumerate(areasToPlot):
        figure(100)
        subplot(1,len(areasToPlot),index+1)
        meanSpectrum = np.nanmean(combinedData[pop], axis=1)
        meanSpectrum = savgol_filter(meanSpectrum, 19, 7) # Smooth
        # meanSpectrum = removeBackroundNoise(psd, freqs)
        plot(freqs, meanSpectrum)
        title(pop)
        # ylabel('Power spectral density (V^2 / Hz)')
        xlim([3, 40])
    show()
    
    return combinedData, freqs

# Main program
if __name__ == "__main__":
    
    # Perform analysis
    dataPath = 'savedData/fullConductance/withIN_bigConnectivity/simulationResults_'
    areasToPlot = ['TCR','IN','TRN']
    areaNames = ['Thalamocortical relay', 'Thalamic inhibitory population', 'Thalamic reticular nucleus']
    # combinedData, freqs = analyseManyResults(numberOfIterations=20, \
    #     dataPath = dataPath, areasToPlot = areasToPlot, plotWindow=plotWindow)

    # Plot example data
    figure(123, [10,8])
    plotWindow = range(15*sampleRate, 20*sampleRate)
    populationData = importData(dataPath + '0.pkl')
    for index, pop in enumerate(areasToPlot):
        subplot(len(areasToPlot), 1, index+1)
        dataToPlot = populationData[pop][plotWindow]
        plot(populationData['times'][plotWindow], dataToPlot, linewidth=.5)
        bottomTick = True
        if index == 1:
            ylabel('Membrane Potential (mV)')
        if index<len(areasToPlot)-1:
            bottomTick = False
        tick_params(top=False, bottom=bottomTick, left=True, right=False, labelleft=True, labelbottom=bottomTick)
        gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        for splineIndex, spine in enumerate(gca().spines.values()):
            if splineIndex%2 == 1:
                spine.set_visible(False)
        title(areaNames[index])
        # Change labels to seconds
        axisLabels = gca().axes.get_xticks().tolist()
        for i in range(len(axisLabels)):
            newAxisLabel = str(int(axisLabels[i]/1000))
            axisLabels[i]=newAxisLabel
        gca().set_xticklabels(axisLabels)
        if index == len(areasToPlot)-1:
            xlabel('Time (seconds)')

    savefig('rawLFPOutput.png')
    show()

#     # Model the power stectrum with FOOF
# # Initialize FOOOF object
# fm = FOOOF()
# spectrum = np.nanmean(combinedData['IN'], axis=1)
# fm.fit(freqs, spectrum, freq_range=[3,40])
# fres = fm.get_results()

# smoothed = savgol_filter(spectrum, 9, 3)

# backgroundParams = fres.background_params
# backgroundNoise = backgroundParams[0] - np.log(freqs**backgroundParams[1])