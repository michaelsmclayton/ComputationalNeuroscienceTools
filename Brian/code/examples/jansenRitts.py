from brian2 import *
import numpy as np
import matplotlib.pylab as plt

# from "The role of node dynamics in shaping emergent functional connectivity patterns in the brain"

class jansenRitts:

    def __init__(self, nodeType):
        # Parameters
        self.type = nodeType
        self.vmax = 5 * Hz
        self.r = .56 * mV **-1
        self.v0 = 6 * mV
        # Variables
        self.v = 0. * mV
        self.v_t = 0. * mV
        self.v_input = 100. * mV

    # Rate from potential
    def pro(self, v):
        return self.vmax/(1 + np.exp(self.r*(self.v0-v)))

    # Potential from rate
    def rpo(self, h=6*mV, tau=100*second**-1):
        if self.type=='pyramid':
            return h*tau*self.pro(self.v_input) - (2*tau*self.v_t)/ms - (tau**2)*self.v
        # elif self.type=='excitatory':
            # return h*tau
            # P = 120 * Hz
            # wij = .5
            # e = 0.1
            # P + e * (wij * self.pro())
        else:
            raise NameError('No return value')
    # Update
    def update(self):
        self.v = self.rpo()*ms**2
        return self.v

# v = []
# a = jansenRitts('pyramid')
# for i in range(200):
#     a.v_input = 6 * np.sin(i/5)*mV
#     v.append(a.update()/volt)
# plt.plot(v); plt.show()



h=6*mV
tau=100*second**-1
P = 120 * Hz
wij = .5
e = 0.1
C = [None, 135, 108, 33.75, 33.75]
exMiInh = 5*Hz
pyrad = C[1] * 6*Hz 
result = h/tau * (P + e * wij * exMiInh + C[2]* pyrad)


aslkd


