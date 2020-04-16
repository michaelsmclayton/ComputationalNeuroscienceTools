from brian2 import *

# -----------------------------------
# Create shape of activation
# -----------------------------------
import scipy.stats as stats

# Create full cycle (0 to 2pi)
cycleX = np.linspace(0.1,2*pi,num=50) # Full cycle 
cycle = np.array([[np.cos(x), np.sin(x)] for x in cycleX])

# Get negative exponential of cycle
negExp = cycleX**-1 / 10

# Get Gaussian around central angle
angle = pi; std = .15
gaussian = 3 * (stats.norm.pdf(x=cycleX,loc=angle,scale=std) / stats.norm.pdf(x=angle,loc=angle,scale=std))

# Combine negative exponential and Gaussian
inputDrive = .1 * ((negExp+gaussian)-.2)

# # Add activation to unit circle
# unitCycle = np.array([unit * combined[i] for i, unit in enumerate(cycle)])

# # Plot in 3D
# from mpl_toolkits.mplot3d import Axes3D
# fig = figure()
# ax = fig.gca(projection='3d')
# ax.plot(cycle[:,0], cycle[:,1], inputDrive)
# show()

# -----------------------------------
# Perform simulation
# -----------------------------------

# Get input drive
repetitions = 3
fullInputDrive = 10*np.hstack([inputDrive for i in range(repetitions)])
stimulus = TimedArray(fullInputDrive, dt=2*ms)

# Define model
eqs = '''
    dTheta/dt = ( (1 - cos(Theta)) + (I * (1 + cos(Theta)))) * ms**-1 + (.05*xi)*ms**-.5: 1
    I = stimulus(t): 1
'''
G = NeuronGroup(1, eqs, threshold="sin(Theta)>.99", method='euler')
trace = StateMonitor(G, ['Theta'], record=True)

# Run model and plot
run(repetitions*100*ms)
subplot(2,1,1)
plot(np.sin(trace.Theta[0]))
subplot(2,1,2)
plot(fullInputDrive)
show()
