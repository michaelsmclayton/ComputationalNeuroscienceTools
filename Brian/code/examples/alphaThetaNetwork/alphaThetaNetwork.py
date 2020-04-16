from brian2 import *

# -------------------------------
# Interneurons
# -------------------------------

# Define functions
gaussian = lambda amp,mu,sig : '''%s*exp(-((tC-%s)**2)/%s)''' % (amp,mu,sig)
sawtooth = lambda A : '''%s*((((t/ms)-tS)/T)-floor(((t/ms)-tS)/T))''' % (A)
thetaNeuron = lambda sig : '''dTheta/dt = ( (1-cos(Theta)) + (I*(1 + cos(Theta)))) * ms**-1 : 1'''# + (%s*xi)*ms**-.5: 1''' % (sig)

# Define function parameters
A = 100 # period of the sawtooth oscillations
mu = 50; sig = 10 # moment and width of secondary, bursting activity
gaussianHeight = 5

# Define interneuron model
numberOfNeurons = 100
eqs = thetaNeuron(sig=.1) + '''
    I = (I_exp+I_gaus)-.1 : 1
    I_exp = exp(-tC/2) : 1 # Exponential (i.e. initial spike)
    I_gaus = '''+gaussian(gaussianHeight,mu,sig)+''' : 1 # Gaussian (i.e. both spike burst)
    tC = '''+sawtooth(A)+''': 1 # cycle time
    tS : 1 # input spike time
    T : 1
'''
interneurons = NeuronGroup(N=numberOfNeurons, model=eqs, threshold="sin(Theta)>.99", method='euler')
interneurons.T = 100+(np.random.randn(numberOfNeurons)*5)
interneurons.tS = -np.abs(np.random.randn(numberOfNeurons))*50
trace_interNeuron = StateMonitor(interneurons, ['Theta','I', 'tC', 'tS'], record=True)

# -------------------------------
# Bursting neurons
# -------------------------------
eqs = thetaNeuron(sig=0) + '''
    I : 1
'''
bursting = NeuronGroup(N=1, model=eqs, threshold="sin(Theta)>.99", method='euler')
trace_bursting = StateMonitor(bursting, ['Theta'], record=True)

# -------------------------------
# Synapses
# -------------------------------
S = Synapses(bursting, interneurons, on_pre='tS_post = t/ms') # reset sawtooth on spike
S.connect()

# Run and plot results
bursting.I = -.1
run(1000*ms)
bursting.I = .001
run(1000*ms)
bursting.I = -.1
run(2000*ms)
fig, ax = plt.subplots(2,1,sharex=True)
#ax[0].plot(trace_interNeuron.t/ms, np.sin(trace_interNeuron.Theta[0]))
ax[0].plot(trace_interNeuron.t/ms, np.mean(np.sin(trace_interNeuron.Theta),axis=0))
ax[1].plot(trace_bursting.t/ms, np.sin(trace_bursting.Theta[0]))
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