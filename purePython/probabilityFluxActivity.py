import scipy.stats as stats
import matplotlib.pylab as plt
import numpy as np
from scipy.stats.kde import gaussian_kde

# Simulation parameters
numberOfNeurons = 500
steps = 500
stimulusTime = 200

# Fokker-Planck approximation
fokkerPlanckUpdate = lambda loc : stats.norm.rvs(loc=loc,scale=2)
'''Note that this function is equivalent to a proposal distribution in Metropolis-Hastings,
Markov chain Monte Carlo (MCMC) method. At every step, the membrane voltage of a neuron is
updated using this function. If the loc (i.e. Gaussian mean) is greater than 0, the membrane
voltage will tend to increase over time. This increase can be thought of as the drag component
in the Fokker-Planck equation. However, as increases are sampled from a Gaussian distribution
with given scale (i.e. standard deviation), there is a degree of random noise in the process.
Consequently, the membrane voltage will not all change in the same way over time. This second
component can be thought of as analagous to random forces (e.g. Brownian motion) is the Fokker-
Planck equation. As such, starting from a narrow distrubution of voltage distributions centered
on -70mV, this distribition will move upwards and become broader with time. When a neuron's
voltage passes -40mV from below, the voltage is reset to -70. The firing rate of the neural
population at a given moment in time, t, is equal to the probability of a neuron having a
membrane voltage of -40*mV at time, t. Further notes on these kinds of ideas can be found at
https://neuronaldynamics.epfl.ch/online/Ch13.html
'''

# Run simulation
allNeuronVoltages = np.zeros(shape=(numberOfNeurons,steps)) # initialise store of all neuron voltages
print('Running simulation...')
for i in range(numberOfNeurons):
    voltages = np.zeros(shape=steps) # initial store of neuron voltages
    currentVoltage = -70
    # currentVoltage = np.random.uniform(low=-70,high=-40) # Initialise voltage randomly between -70 and -40mV
    for step in range(steps):
        stimAmplitude = 1 + 40 * stats.norm.pdf(step,loc=stimulusTime, scale=1) # constant, positive drive (with stimulus at 'stimulusTime')
        currentVoltage += fokkerPlanckUpdate(loc=stimAmplitude) # Update voltage
        if currentVoltage > -40: # Spike reset
            currentVoltage = -70
        voltages[step] = currentVoltage
    allNeuronVoltages[i,:] = voltages

# Get probability distributions for each step
x = np.arange(-70,-40,.1)
pdfs = np.zeros(shape=(len(x),steps))
for i in range(steps):
    pdf = gaussian_kde(allNeuronVoltages[:,i])
    pdfs[:,i] = pdf(x)

# Plot results
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots(2,1)
ax[0].plot(pdfs[-1,:])
distPlot = ax[1].plot(x,pdf(x))
plt.ylim([0,0.5])
def update(frame):
    pdf = gaussian_kde(allNeuronVoltages[:,frame])
    distPlot[0].set_data(x,pdf(x))
    return distPlot
ani = FuncAnimation(fig, update, frames=range(steps), interval=100, blit=True)
plt.show()