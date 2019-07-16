from brian2 import *

''' Equations taken from 'Generative models of cortical oscillations- neurobiological
implications of the Kuramoto model'''

# Define Kuramoto neurons
N = 4
kN_ratio = .05 # k / N
freq = 1.5
eqs = '''
    dTheta_n/dt = (freq + (kN_ratio * sin(Theta_m - Theta_n))) * ms**-1 : 1 #( freq + (kN_ratio * sin(Theta - Theta_m)) ) * ms**-1 : 1
    Theta_m : 1
'''
neurons = NeuronGroup(4, eqs, threshold='True', method='rk4')
neurons.Theta_n = '(2*(i*pi/%s))' % (N)
trace = StateMonitor(neurons, ['Theta_n'], record=True)

# Define synapses
'''such that Theta_m in the post-synaptic neuron updates to Theta_n in the pre-synaptic neuron'''
s = Synapses(neurons, neurons, on_pre='Theta_m_post = Theta_n_pre', method='euler')
s.connect()

# Run and plot
run(75*ms, report='text')
figure(1, figsize=(12,4))
for currentTrace in trace.Theta_n:
    plot(cos(currentTrace))
show()