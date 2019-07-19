from brian2 import *
from matplotlib.pyplot import *
import matplotlib.animation as animation
from matplotlib import gridspec

# We can visualise the above-defined connectivity using the function below
def visualise_connectivity(S):
    Ns = len(S.source) # Get number of source neurons
    Nt = len(S.target) # Get number of target neurons
    # Get figure
    figure(figsize=(10, 4))
    subplot(121) # Left subplot...
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    show()

# ''' Equations taken from 'Generative models of cortical oscillations- neurobiological
# implications of the Kuramoto model'''
''' Equations taken from 'Theta and Alpha Oscillations Are Traveling Waves in the Human Neocortex'''

# Define Kuramoto neurons
def createKuramotoNeurons(N):
    eqs = '''
        dTheta/dt = (freq + (kN * PIF)) * ms**-1 : 1
        PIF = .2 * (sin(ThetaPreInput - Theta) + sin(ThetaPostInput - Theta)): 1
        ThetaPreInput : 1
        ThetaPostInput : 1
        freq : 1
        kN : 1
    '''
    neurons = NeuronGroup(N, eqs, threshold='True', method='rk4')
    neurons.Theta = '1-(randn()*2)'
    neurons.freq = '.3+(.1*i)'
    trace = StateMonitor(neurons, ['Theta'], record=True)
    return neurons, trace
neurons, trace = createKuramotoNeurons(N=20)

# Define synapses
'''such that Theta_m in the post-synaptic neuron updates to Theta in the pre-synaptic neuron'''
s = Synapses(neurons, neurons, on_pre = ''' ThetaPreInput_post = Theta_pre''', \
    on_post='''ThetaPostInput_pre = Theta_post''', method='euler')
s.connect(condition='i-j==1')
#visualise_connectivity(s)
#s.delay = '(i-j)*10*ms'

# Run and plot
neurons.kN = 0
run(10*ms, report='text')
neurons.kN = 12
run(20*ms, report='text')
neurons.kN = 0
run(10*ms, report='text')
# figure(1, figsize=(12,4))



# -----------------------------------------------
# Create animation
# -----------------------------------------------

# Initialise figure
# fig = figure(figsize=(16, 3)) 
# gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 

# ax0 = plt.subplot(gs[0])
# for currentTrace in trace.Theta:
#     ax0.plot(trace.t/ms, cos(currentTrace))
#     yLimits = [-1, 1]

# Get initial values
neurons = []
fig = figure(1, figsize=(4,6))
fig.set_facecolor((.8,.8,.8))
gca().set_facecolor((.8,.8,.8))
for neuron in range(len(trace.Theta)):
    currentValue = cos(trace.Theta[neuron][0])
    currentColor = str((currentValue+1)/2)
    currentNeuron = scatter(1, neuron+1, s=500, color=currentColor)
    neurons.append(currentNeuron)
axis('off')

# Define animation change
def animate(t):
    newNeurons = []
    for index, neuron in enumerate(neurons):
        currentValue = cos(trace.Theta[index][t])
        currentColor = str((currentValue+1)/2)
        if (trace.t[t]>10*ms and trace.t[t]<30*ms):
            edgeColor = 'red'
        else:
            edgeColor = None
        currentNeuron = scatter(1, index+1, s=500, edgecolors=edgeColor, color=currentColor)
        newNeurons.append(currentNeuron)
    return newNeurons

# create animation using the animate() function
myAnimation = animation.FuncAnimation(fig, animate, frames=len(trace.Theta[0]), interval=1, blit=True, repeat=False)
# myAnimation.save('animation.gif', writer='imagemagick', fps=60)
show()


timeLine[0].set_xdata