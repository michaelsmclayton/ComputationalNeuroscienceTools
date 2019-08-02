from brian2 import *

eqs = '''
    dTheta/dt = ( (1 - cos(Theta)) + (I * (1 + cos(Theta)))) * ms**-1 + (.01*xi)*ms**-.5: 1
    I : 1
'''
N = 3
G = NeuronGroup(N, eqs, threshold="sin(Theta)>.99", method='euler')
G.I = .001
trace = StateMonitor(G, ['Theta'], record=True)

run(1000*ms, report='text')
dataToPlot = np.transpose(sin(trace.Theta))
figure(1)
for i in range(N):
    subplot(N,1,i+1)
    plot(dataToPlot[:,i])
show()

