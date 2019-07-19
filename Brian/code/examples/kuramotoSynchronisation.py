from brian2 import *
from matplotlib.pyplot import *
import matplotlib.animation as animation

''' Equations taken from 'Generative models of cortical oscillations- neurobiological
implications of the Kuramoto model'''

# Define Kuramoto neurons
def createKuramotoNeurons(N):
    eqs = '''
        dTheta_n/dt = (freq + (kN * PIF)) * ms**-1 : 1
        PIF = -sin(angleDiff + beta) + (R*sin(2*angleDiff)) : 1
        angleDiff = Theta_m - Theta_n : 1
        R = .9 : 1
        beta = .25 : 1
        Theta_m : 1
        freq : 1
        kN : 1
    '''
    neurons = NeuronGroup(N, eqs, threshold='True', method='rk4')
    neurons.Theta_n = '1-(randn()*2)'
    neurons.freq = '1-(randn()*2)'
    trace = StateMonitor(neurons, ['Theta_n'], record=True)
    return neurons, trace
neurons, trace = createKuramotoNeurons(N=15)

# Define synapses
'''such that Theta_m in the post-synaptic neuron updates to Theta_n in the pre-synaptic neuron'''
s = Synapses(neurons, neurons, on_pre='Theta_m_post = Theta_n_pre', method='euler')
s.connect(condition='i!=j')
s.delay = '(i-j)*10*ms'

# Run and plot
neurons.kN = 0
run(10*ms, report='text')
neurons.kN = 5
run(20*ms, report='text')
neurons.kN = 0
run(10*ms, report='text')
figure(1, figsize=(12,4))
for currentTrace in trace.Theta_n:
    plot(trace.t/ms, cos(currentTrace))
# show()


# -----------------------------------------------
# Create animation
# -----------------------------------------------

# Initialise figure
fig, ax = subplots()

# Get initial values
neurons = []
for neuron in range(len(trace.Theta_n)):
    currentValue = cos(trace.Theta_n[neuron][0])
    currentColor = str((currentValue+1)/2)
    currentNeuron = scatter(neuron+1, 1, s=500, color=currentColor)
    neurons.append(currentNeuron)

# Define animation change
def animate(t):
    newNeurons = []
    for index, neuron in enumerate(neurons):
        currentValue = cos(trace.Theta_n[index][t])
        currentColor = str((currentValue+1)/2)
        currentNeuron = scatter(index+1, 1, s=500, color=currentColor)
        newNeurons.append(currentNeuron)
    return newNeurons

# create animation using the animate() function
myAnimation = animation.FuncAnimation(fig, animate, frames=len(trace.Theta_n[0]), interval=1, blit=True, repeat=False)
show()

