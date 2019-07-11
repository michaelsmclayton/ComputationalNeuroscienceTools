from brian2 import *

# ------------------------------------------
# Time constant of RC circuit
# ------------------------------------------
'''
The time constant (τ) for an RC circuit is the product of resistance (ohms)
and capacitance (farads).
ohm (resistance) = kilogram * meter**2 * second**-3 * amp**-2
farad (capacitance)= kilogram**-1 * meter**-2 * second**-4 * amp**2
therefore ohm * farad = second**1
'''
resistance = 10 * kohm
capacitance = 100 * ufarad
timeConstant = resistance * capacitance
print(timeConstant)


# ------------------------------------------
# Voltage (from current divided by capacitance)
# ------------------------------------------
'''
amps (current) = amp**1
farad (capacitance) = kilogram**-1 * meter**-2 * second**-4 * amp**2
voltage (potential) = kilogram**1 * meter**2 * second**-3 * amp**-1
(remember that division is opposite of multiplication)
so amps / farad = voltage
'''
current = 1 * amp
capacitance = 1 * farad
print((current / capacitance)*second)

# Example using neuron
Cm = 1*ufarad
g = 2 * usiemens
uRev = -70 * mV
eqs = '''
    du/dt = -(g*(u-uRev)) / Cm : volt
'''
neurons = NeuronGroup(1, eqs, method='exact')
trace = StateMonitor(neurons, ['u'], record=True)
run(1000*ms, report='text')
plot(trace.u[0])
show()