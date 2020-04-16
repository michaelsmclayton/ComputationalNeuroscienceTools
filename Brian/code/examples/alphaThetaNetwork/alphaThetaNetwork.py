from brian2 import *

# Define functions
gaussian = lambda amp,mu,sig : '''%s*exp(-((tC-%s)**2)/%s)''' % (amp,mu,sig)
sawtooth = lambda A, T : '''%s*((t/ms/%s)-floor(t/ms/%s))''' % (A,T,T)

# Define function parameters
A, T = 100, 100 # period of the sawtooth oscillations
mu = 50; sig = 10 # moment and width of secondary, bursting activity

# Define model
eqs = '''
    dTheta/dt = ( (1-cos(Theta)) + (I*(1 + cos(Theta)))) * ms**-1 + (.05*xi)*ms**-.5: 1
    I = (1*(I_exp+I_gaus))-.1 : 1
    I_exp = exp(-tC/2) : 1 # exponential
    I_gaus = '''+gaussian(5,mu,sig)+''' : 1 # Gaussian
    tC = '''+sawtooth(A,T)+''': 1 # cycle time
'''
G = NeuronGroup(1, eqs, method='euler')
trace = StateMonitor(G, ['Theta','I', 'tC'], record=True)

# Run and plot results
run(200*ms)
subplot(3,1,1)
plot(np.sin(trace.Theta[0]))
subplot(3,1,2)
plot(trace.I[0])
subplot(3,1,3)
plot(trace.tC[0])
show()





# # -----------------------------------
# # Create shape of activation
# # -----------------------------------
# import scipy.stats as stats

# # Create full cycle (0 to 2pi)
# cycleX = np.linspace(0.1,2*pi,num=50) # Full cycle 
# cycle = np.array([[np.cos(x), np.sin(x)] for x in cycleX])

# # Get negative exponential of cycle
# negExp = cycleX**-1 / 10

# # Get Gaussian around central angle
# angle = pi; std = .15
# gaussian = 3 * (stats.norm.pdf(x=cycleX,loc=angle,scale=std) / stats.norm.pdf(x=angle,loc=angle,scale=std))

# # Combine negative exponential and Gaussian
# inputDrive = .1 * ((negExp+gaussian)-.2)

# # # Add activation to unit circle
# # unitCycle = np.array([unit * combined[i] for i, unit in enumerate(cycle)])

# # # Plot in 3D
# # from mpl_toolkits.mplot3d import Axes3D
# # fig = figure()
# # ax = fig.gca(projection='3d')
# # ax.plot(cycle[:,0], cycle[:,1], inputDrive)
# # show()

# # -----------------------------------
# # Perform simulation
# # -----------------------------------

# # Get input drive
# repetitions = 3
# fullInputDrive = 10*np.hstack([inputDrive for i in range(repetitions)])
# stimulus = TimedArray(fullInputDrive, dt=2*ms)

# # Define model
# eqs = '''
#     dTheta/dt = ( (1 - cos(Theta)) + (I * (1 + cos(Theta)))) * ms**-1 + (.05*xi)*ms**-.5: 1
#     I = stimulus(t): 1
# '''
# G = NeuronGroup(1, eqs, threshold="sin(Theta)>.99", method='euler')
# trace = StateMonitor(G, ['Theta'], record=True)

# # Run model and plot
# run(repetitions*100*ms)
# subplot(2,1,1)
# plot(np.sin(trace.Theta[0]))
# subplot(2,1,2)
# plot(fullInputDrive)
# show()

# DEFAULT_FUNCTIONS.keys()