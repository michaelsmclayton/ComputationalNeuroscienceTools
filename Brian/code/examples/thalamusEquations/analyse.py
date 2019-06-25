from matplotlib.pyplot import *
import pickle
from populations import *
from scipy.signal import butter, sosfilt, sosfreqz, welch

sampleRate = 1000
plotWindow = [10*sampleRate, 39*sampleRate]

# Define plotting function
def plotResult(populationData, plotWindow=plotWindow):

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
        plot(times[plotWindow[0]:plotWindow[1]], dataToPlot[plotWindow[0]:plotWindow[1]], linewidth=.5)
        ylabel(pop)
        
        # Plot power spectral density
        figure(2, figsize=(5,12))
        subplot(len(areasToPlot), 1, index+1)
        freqs, psd = powerSpectralDensity(dataToPlot[plotWindow[0]:plotWindow[1]])
        plot(freqs, psd, color='k', lw=2)
        xlabel('Frequency (Hz)')
        ylabel('Power spectral density (V^2 / Hz)')
        title(pop)
        xlim([2, 30])
    
    show()

# ---------------------------------
# Filtering
# ---------------------------------
def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], analog=False, btype='band', output='sos')
def butter_bandpass_filter(data, lowcut, highcut, fs=sampleRate, order=10):
        sos = butter_bandpass(lowcut, highcut, fs, order)
        y = sosfilt(sos, data)
        return y


# ---------------------------------
# Power analysis
# ---------------------------------
def powerSpectralDensity(data):

    # Define window length (4 seconds)
    win = 2 * sampleRate
    freqs, psd = welch(data, sampleRate, nperseg=win)
    return freqs, psd


if __name__ == "__main__":

    # Import results
    with open('simulationResults_new.pkl', 'rb') as handle:
        populationData = pickle.load(handle)

    # Plot data
    plotResult(populationData)
