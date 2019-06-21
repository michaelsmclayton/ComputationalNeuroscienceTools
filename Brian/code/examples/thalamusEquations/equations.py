# Taken from 'Causal Role of Thalamic Interneurons in
# Brain State Transitions- A Study Using a Neural Mass
# Model Implementing Synaptic Kinetics'

# tau = 10 * ms

# ---------------------------------------------------
# Pre-synaptic dynamics
# ---------------------------------------------------

# -----------------------------
# Equations for AMPA and GABAa channels
# -----------------------------

# Equation 1
equation1 = '''
    Tmax = 1 : 1
    Vthr = -32 : 1 # Point where output reaches .5 mM
    omega = 3.8 : 1 # steepness of the sigmoid function
    T = Tmax / (1 + exp(-((Vpre - Vthr)/omega))) : 1
'''

# Equation 2
equation2 = '''
    dr/dt = ( alpha * T * (1 - r) - beta * r ) / tau : 1
    diff = ( alpha * T * (1 - r) - beta * r ) : 1
'''

# -----------------------------
# Equations for GABAb channel
# -----------------------------

# Equation 3 - metabotropic synapses (AMPA, GABAa)
''' Get the fraction of activated GABAB receptors'''
equation3 = '''
    dR/dt = ( alpha1 * T * (1 - R) - beta1 * R ) / tau : 1
'''

# Equation 4 - secondary messenger concentration
'''Get the concentration of the activated G-protein'''
'''dX(t)/dt'''
equation4 = '''
    dX/dt = ( alpha2 * R - beta2 * X ) / tau : 1
'''

# Equation 5 - get fraction of open ion channels
'''Get the fraction of open ion channels caused by binding of [X]'''
equation5 = '''
    n = 4 : 1
    Kd = 100 : 1
    r = X**n / (X**n + Kd) : 1
'''

# ---------------------------------------------------
# Post-synaptic dynamics
# ---------------------------------------------------

# -----------------------------
# Equation for post-synaptic current
# -----------------------------

# Equation 6 - post-synaptic current
'''Returns current density A/m2'''
equation6 = '''
    Ipsp = Cuvw * gSynMax * r * (Vpsp - ESyncRev) : 1
'''

# Equation 7 - post-synaptic membrane potential
equation7 = '''
    km = 10 * ms : second
    dVpsp/dt = - (Ipsp + Ileak) / km : 1
'''
# Note that this equation is different to the one listed in this paper.
# Here, Ipsp and Ileak are summed rather than subtracted. See the following
# webpage https://www.st-andrews.ac.uk/~wjh/hh_model_intro/ to see an
# explanation of why I think that summing them is correct. Suggesting to
# subtract them may have been a mistake in the original paper?

# Equation 8 - leak current
equation8 = '''
    Ileak = gLeak * (Vpsp - Eleak) : 1
    # Ileak = gLeak * (Eleak - Vpsp) : 1
'''


# ---------------------------------------------------
# Additional, membrane input dynamics
# ---------------------------------------------------

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


