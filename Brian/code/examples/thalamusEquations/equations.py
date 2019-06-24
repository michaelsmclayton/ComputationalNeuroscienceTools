# Taken from 'Causal Role of Thalamic Interneurons in
# Brain State Transitions- A Study Using a Neural Mass
# Model Implementing Synaptic Kinetics'

# Function to return equations of interest
def getEquations(equations, eqsOfInterest):
    eqs = ''''''
    for eq in eqsOfInterest:
        eqs += equations[eq]
    return eqs

# Initialise store (with empty 1st entry, so Equation 1 = index 1)
equations = ['''empty''']

# ---------------------------------------------------
# Pre-synaptic dynamics
# ---------------------------------------------------

# -----------------------------
# Equations for AMPA and GABAa channels
# -----------------------------

# Equation 1
equations.append('''
    Tmax = 1 : 1
    Vthr = -32 : 1 # Point where output reaches .5 mM
    omega = 3.8 : 1 # steepness of the sigmoid function
    T = Tmax / (1 + exp(-((V_pre - Vthr)/omega))) : 1
''')

# Equation 2
equations.append('''
    dr/dt = ( alpha * T * (1 - r) - beta * r ) / tau : 1
''')

# -----------------------------
# Equations for GABAb channel
# -----------------------------

# Equation 3 - metabotropic synapses (AMPA, GABAa)
''' Get the fraction of activated GABAB receptors'''
equations.append( '''
    dR/dt = ( alpha1 * T * (1 - R) - beta1 * R ) / tau : 1
''')

# Equation 4 - secondary messenger concentration
'''Get the concentration of the activated G-protein'''
'''dX(t)/dt'''
equations.append('''
    dX/dt = ( alpha2 * R - beta2 * X ) / tau : 1
''')

# Equation 5 - get fraction of open ion channels
'''Get the fraction of open ion channels caused by binding of [X]'''
equations.append('''
    n = 4 : 1
    Kd = 100 : 1
    r = X**n / (X**n + Kd) : 1
''')

# ---------------------------------------------------
# Post-synaptic dynamics
# ---------------------------------------------------

# -----------------------------
# Equation for post-synaptic current
# -----------------------------

# Equation 6 - post-synaptic current
'''Returns current density A/m2'''
equations.append('''
    Ipsp_syn = Cuvw * gSynMax * r * (V - ESynRev) : 1
''')

# Equation 7 - post-synaptic membrane potential
equations.append('''
    dV/dt = - (Ipsp + Ileak) / km : 1
    Ipsp : 1
''')
# Note that this equation is different to the one listed in this paper.
# Here, Ipsp and Ileak are summed rather than subtracted. See the following
# webpage https://www.st-andrews.ac.uk/~wjh/hh_model_intro/ to see an
# explanation of why I think that summing them is correct. Suggesting to
# subtract them may have been a mistake in the original paper?

# Equation 8 - leak current
equations.append('''
    Ileak = gLeak * (V - Eleak) : 1
''')


# ---------------------------------------------------
# Additional, membrane input dynamics
# ---------------------------------------------------

# Set Membrane voltage dynamics
Vrest = -65
membraneVoltageEquation = '''
    dV/dt = .001*(Vrest-V) / tau : 1
'''
sigma = .2
randomInput = '''
    dV/dt = sigma*sqrt(2/tau)*xi : 1
'''
sinusoidalInput = '''
    dVmag/dt = .002*(0 - Vmag) / tau : 1
    V = -65 + (Vmag*sin(2*pi*(.4*Hz)*t)) : 1
'''


