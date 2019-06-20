import matplotlib as plt
from brian2 import * # Import Brian
import numpy as np
import math

# Define function to plot outputs over voltages
def plotOverVoltages(functionOfInterest):
    voltageRange = range(-70, -10)
    outputs = np.zeros([len(voltageRange), 1])
    for index, voltage in enumerate(voltageRange):
        outputs[index] = functionOfInterest(voltage)
    plt.figure()
    plt.plot(voltageRange, outputs)
    plt.show()

# Taken from 'Causal Role of Thalamic Interneurons in
# Brain State Transitions- A Study Using a Neural Mass
# Model Implementing Synaptic Kinetics'

# Equation 1 - neurotransmitter concentrations
def getTransmitterConcentration(Vpre):
    Tmax = 1 #* mM # Maximum output concentration
    Vthr = -32 #* mV # Point where output reaches .5 mM
    omega = 3.8 #* mV # steepness of the sigmoid function
    return Tmax / (1 + math.exp(-((Vpre - Vthr)/omega)))
# plotOverVoltages(getTransmitterConcentration)

# -----------------------------
# Equations for AMPA and GABAa channels
# -----------------------------

# Equation 2 - ionotropic synapses (AMPA, GABAa)
'''dr(t)/dt'''
def getProportionPostSynOpenChannels(Vpre, r):
    alpha = 10 # forward rates of chemical reactions
    beta = 25 # reverse rates of chemical reactions
    # r = .01 # Current proportion
    return alpha * getTransmitterConcentration(Vpre) * (1 - r) - (beta * r)
plotOverVoltages(getProportionPostSynOpenChannels)

masd

# -----------------------------
# Equations for GABAb channel
# -----------------------------

# Equation 3 - metabotropic synapses (AMPA, GABAa)
''' Get the fraction of activated GABAB receptors'''
'''dR(t)/dt'''
def getDiffOfFractionOfGABAbReceptors(Vpre):
    alpha1 = 10 # forward rates of chemical reactions
    beta1 = 25 # reverse rates of chemical reactions
    R = .01 # Current fraction
    return alpha1 * getTransmitterConcentration(Vpre) * (1 - R) - (beta1 * R)
# plotOverVoltages(getDiffOfFractionOfGABAbReceptors)



# Equation 4 - secondary messenger concentration
'''Get the concentration of the activated G-protein'''
'''dX(t)/dt'''
def getDiffOf2ndaryMessenger():
    alpha2 = 15
    beta2 = 5
    R = .01
    X = .01
    return alpha2 * R - beta2 * X
# print(getDiffOf2ndaryMessenger())

# Equation 5 - get fraction of open ion channels
'''Get the fraction of open ion channels caused by binding of [X]'''
def getFractionOfIonChannels():
    n = 4
    X = 1
    Kd = 100 # dissociation constant of binding of [X] with the ion channels
    return math.pow(X, n) / (math.pow(X, n) + Kd)
# print(getFractionOfIonChannels())

# -----------------------------
# Equation for post-synaptic current
# -----------------------------
# Equation 6 - post-synaptic current
'''Returns current density A/m2'''
def getPostSynapticCurrent(Cuvw, gSynMax, r, Vpsp, ESyncRev):
    return Cuvw * gSynMax * r * (Vpsp - ESyncRev)
# print(getPostSynapticCurrent())

# -----------------------------
# Equation for post-synaptic membrane potential
# -----------------------------
'''Returns current density A/m2'''
def getLeakCurrent(Vpsp, gLeak, Eleak):
    return gLeak * (Vpsp - Eleak)

def getPostSynapticMembranePotential(Vpsp):

    Vpsp = Vpsp * mV

    # Get Ipsp
    Cuvw = 23.6
    gSynMax = 100 * uS/cm2 # maximum conductance
    r = 0.4 # Current proportion of post-synaptic open channel
    ESyncRev = -75 * mV # reverse potential
    Ipsp = getPostSynapticCurrent(Cuvw, gSynMax, r, Vpsp, ESyncRev)
    # print('Ipsp', Ipsp)

    # Get Ileak
    gLeak = 10 * uS/cm2
    Eleak = -55 * mV
    Ileak = getLeakCurrent(Vpsp, gLeak, Eleak)
    # print('Ileak', Ileak)

    km = 1 * uF/cm2
    return (-(Ipsp - Ileak))/km

# plotOverVoltages(getPostSynapticMembranePotential)
print(getPostSynapticMembranePotential(-85)*(8*ohm))

# Dynamic varibles = X, R, r