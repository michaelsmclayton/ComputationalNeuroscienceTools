from brian2 import * # Import Brian
from matplotlib.pyplot import *
import math

# Taken from 'Causal Role of Thalamic Interneurons in
# Brain State Transitions- A Study Using a Neural Mass
# Model Implementing Synaptic Kinetics'

# General parameters
tau = 10 * ms
equations = []

# Set Membrane voltage dynamics
Vrest = -65
membraneVoltageEquation = '''
    dVpre/dt = .001*(Vrest-Vpre) / tau : 1
'''
sigma = .2
randomInput = '''
    dVpre/dt = sigma*sqrt(2/tau)*xi : 1
'''
sinusoidalInput = '''
    dVmag/dt = .002*(0 - Vmag) / tau : 1
    Vpre = -65 + (Vmag*sin(2*pi*(.4*Hz)*t)) : 1
'''

# -----------------------------
# Equations for AMPA and GABAa channels
# -----------------------------

# Equation 1
Tmax = 1
Vthr = -32 # Point where output reaches .5 mM
omega = 3.8 # steepness of the sigmoid function
equation1 = '''
    T = Tmax / (1 + exp(-((Vpre - Vthr)/omega))) : 1
'''
equations.append(equation1)

# Equation 2
alpha = 1000**-1 # * mM**-1 * second**-1
beta = 50**-1 # * second**-1 # reverse rates of chemical reactions
equation2 = '''
    dr/dt = ( alpha * T * (1 - r) - beta * r ) / tau : 1
    diff = ( alpha * T * (1 - r) - beta * r ) : 1
'''
equations.append(equation2)

# # Combined equation for AMPA and GABAa synapses
# eqs = equation1 + equation2
# G = NeuronGroup(1, eqs, method='euler')
# G.r = .001
# G.Vpre = 0
# M = StateMonitor(G, [ 'T', 'Vpre', 'r', 'diff'], record=True)
# run(2000*ms)

# subplot(2,2,1)
# plot(M.t/ms, M.Vpre[0])
# ylabel('Vpre')
# subplot(2,2,2)
# plot(M.t/ms, M.T[0])
# ylabel('T')
# subplot(2,2,3)
# plot(M.t/ms, M.r[0])
# ylabel('r')
# subplot(2,2,4)
# plot(M.t/ms, M.diff[0])
# ylabel('dr/dt')
# savefig('equation1and2_results.png')

# -----------------------------
# Equations for GABAb channel
# -----------------------------

# Equation 3 - metabotropic synapses (AMPA, GABAa)
''' Get the fraction of activated GABAB receptors'''
alpha1 = 10**-1 # forward rates of chemical reactions
beta1 = 25**-1 # reverse rates of chemical reactions
equation3 = '''
    dR/dt = ( alpha1 * T * (1 - R) - beta1 * R ) / tau : 1
'''
equations.append(equation3)

# Equation 4 - secondary messenger concentration
'''Get the concentration of the activated G-protein'''
'''dX(t)/dt'''
alpha2 = 15**-1
beta2 = 5**-1
equation4 = '''
    dX/dt = ( alpha2 * R - beta2 * X ) / tau : 1
'''
equations.append(equation4)

# Equation 5 - get fraction of open ion channels
'''Get the fraction of open ion channels caused by binding of [X]'''
n = 4
Kd = 100
equation5 = '''
    r = X**n / (X**n + Kd) : 1
'''
equations.append(equation5)

# # Combined equation for GABAb synapses
# eqs = membraneVoltageEquation + ''' : 1'''
# eqsOfInterest = [0,2,3,4]
# for eq in eqsOfInterest:
#     eqs += equations[eq]
# G = NeuronGroup(1, eqs, method='euler')
# M = StateMonitor(G, [ 'Vpre', 'R', 'X', 'r'], record=True)
# run(2000*ms)

# subplot(2,2,1)
# plot(M.t/ms, M.Vpre[0])
# ylabel('Vpre')
# subplot(2,2,2)
# plot(M.t/ms, M.R[0])
# ylabel('R')
# subplot(2,2,3)
# plot(M.t/ms, M.X[0])
# ylabel('X')
# subplot(2,2,4)
# plot(M.t/ms, M.r[0])
# ylabel('r')
# # show()
# savefig('equation3to5_results.png')


# -----------------------------
# Equation for post-synaptic current
# -----------------------------

# Equation 6 - post-synaptic current
'''Returns current density A/m2'''
Cuvw = 7.1 # 23.6
gSynMax = 300 / 1000 #* uS/cm2 # maximum conductance
ESyncRev = 0 #* mV # reverse potential
equation6 = '''
    Ipsp = Cuvw * gSynMax * r * (Vpsp - ESyncRev) : 1
'''
equations.append(equation6)
#Cuvw * gSynMax * M.r[0][0] * (M.Vpsp[0][0] - ESyncRev)

# Equation 7 - post-synaptic membrane potential
km = 10 * ms #uF/cm2
equation7 = '''
    dVpsp/dt = - (Ipsp + Ileak) / km : 1
'''
equations.append(equation7)
'''Note that this equation is different to the one listed in this paper.
Here, Ipsp and Ileak are summed rather than subtracted. See the following
webpage https://www.st-andrews.ac.uk/~wjh/hh_model_intro/ to see an explanation
of why I think that summing them is correct. Suggesting to subtract them may have
been a mistake in the original paper?'''

# Equation 8 - leak current
gLeak = 10 / 1000 #* uS/cm2
Eleak = -55 #* mV
equation8 = '''
    Ileak = gLeak * (Vpsp - Eleak) : 1
    # Ileak = gLeak * (Eleak - Vpsp) : 1
'''
equations.append(equation8)
# gLeak/1000 * (M.Vpsp[0][0] - Eleak)

# # Combined equation for GABAb synapses
# # eqs = membraneVoltageEquation + ''' : 1'''
# eqs = sinusoidalInput
# eqsOfInterest = [0,1,5,6,7]
# for eq in eqsOfInterest:
#     eqs += equations[eq]
# G = NeuronGroup(1, eqs, method='euler')
# # G.Vpre = -20
# G.Vpsp = -65
# G.Vmag = 49
# G.r = .001
# M = StateMonitor(G, [ 'Vpre', 'Vpsp', 'r', 'Ipsp', 'Ileak'], record=True)
# run(10000*ms)

# print(M.Ipsp[0])
# rows = 2; columns = 3
# subplot(rows,columns,1)
# plot(M.t/ms, M.Vpre[0])
# ylabel('Vpre')
# subplot(rows,columns,2)
# plot(M.t/ms, M.Vpsp[0])
# ylabel('Vpsp')
# subplot(rows,columns,3)
# plot(M.t/ms, M.Ipsp[0])
# ylabel('Ipsp')
# subplot(rows,columns,4)
# plot(M.t/ms, M.Ileak[0])
# ylabel('Ileak')
# subplot(rows,columns,5)
# plot(M.t/ms, M.r[0])
# ylabel('r')
# show()
# # savefig('equation6to8_results.png')



'''

        ##########################
        SYNAPSES
        ##########################

        # PARAMETERS

        {
            # Synaptic dynamics
            alpha / alpha 1 & alpha 2
            beta / beta 1 & beta 2
                / Kd n
            gSynMax
            ESyncRev # reverse potential

            # Connection strength
            Cuvw
        }
        
        Presynaptic activity
        (For GABAb)
        R = fraction of activated GABAB receptors, which acts as a catalyst for G-protein (X)
        X = concentration of the activated G-protein
        (For AMPA and GABAa)
        r = proportion of open ion-channels on the post-synaptic population
        (caused by the binding of the neurotransmitters)

        Equations 1, 2, 3, 4, 5, 6

        v

        Postsynaptic current

        ##########################
        POPULATIONS
        ##########################

        # PARAMETERS

        gLeak
        Eleak
        vRest


        Equation 7



Area = {
    constants: {
        gLeak
        Eleak
        vRest
    }

    inputVariables = {
        Ipsp : from synapse

    }

    equations = eqs (Equations 7 and 8)

    stateVariables = {
        'Vm', # Equation 7
        'Ileak', # Equation 8
    }
    
}

Connection = {
    connectivity = Cuvm
    synapticDynamicsConstants = {

    }
}
        

'''


