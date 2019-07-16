from brian2 import * # Import Brian

# Define capacitance
area = 2*um2
Cm = ((1*uF)/cm2) * area

# Define Hodgkin-Huxley model
ENa = 55 * mV;  gNa = 40 * (mS/cm2) * area
EK = -77 * mV;  gK = 35 * (mS/cm2) * area
Eleak = -65 * mV;   gleak = 0.3 * (mS/cm2) * area
I = 0 * mA
n = .5; h = .5
currentEquations = '''
    du/dt = (-Ik + I) / Cm : volt
    Ik = INa + IK + Ileak : amp
    INa = (gNa * m**3 * h) * (u-ENa)  : amp
    IK = (gK * n**4) * (u-EK) : amp
    Ileak = gleak * (u-Eleak) : amp
'''
u = -75*mV
gatingEquations = '''
    dm/dt = alpha_m * (1-m) - beta_m * m : 1
    alpha_m = (.182*((u/mV)+35)) / (1 - exp(-((u/mV)-35)/9)) * ms**-1 : Hz
    beta_m = (-.124*((u/mV)+35)) / (1 - exp(-((u/mV)-35)/9)) * ms**-1 : Hz
'''
eqs = currentEquations + gatingEquations
neurons = NeuronGroup(1, eqs, method='rk4')
trace = StateMonitor(neurons, ['u'], record=True)
run(1000*ms, report='text')

plot(trace.u[0])
show()