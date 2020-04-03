import numpy as np

################################################################################
#                 A SIMPLE MODEL OF PATTERN ASSOCIATION MEMORY
###############################################################################
# from Chapter 2 of 'Neural Networks and Brain Function' (Rolls & Treves, 1998)

# ---------------------------------------
# Set parameters and initialise synapses (to zeros)
# ---------------------------------------
k = 1
n_uncondInputs = 4
n_condInputs = 6
repeatWithNoUCStimuli = False
gracefulDegredation = True
w = np.zeros(shape=(n_condInputs, n_uncondInputs)) # weights between conditioned inputs, and output neurons
h = np.zeros(shape=(n_uncondInputs)) # dendritic activation
randint = np.random.randint

# ---------------------------------------
# Functions
# ---------------------------------------

# Activation function
def activationFunction(e):
    return e # linear

# Hebbian learning
def hebbRule(r,rPrime, k=1):
    return k * r * rPrime.T

# Synaptic activation
'''Note that, as this computation can be achieved with the dot product pf wi and rPrime, this computation
essentially returns the difference in the vectors of wi and rPrime (given that dot products provide
a direct measure of how similar two vectors are (although mainly when using unit vectors. Consequently,
a fundamental operation many neurons perform is effectively to compute how similar an input pattern vector
rPrime is to a stored weight vector'''
def synapticActivation(rPrime,w):
    return [np.nansum([rPrime[0,j] * w[j,i] for j in range(n_condInputs)]) for i in range(n_uncondInputs)]
    # return np.array([np.dot(rPrime, w[:,i]) for i in range(n_uncondInputs)]).T[0]

# Threshold output firing (i.e. fire only if activity is greater than threshold)
def outputFiring(h, threshold=2):
    return [1 if hVal>=threshold else 0 for hVal in h]

# ---------------------------------------
# Run the model
# ---------------------------------------

# Set inputs
eInputs = np.array([\
    [[1.,1.,0.,0.]], \
    [[0.,1.,0.,1.]], \
    [[0.,0.,0.,0.]]]) # unconditioned input stimulus
rPrimeInputs = np.array([\
    [[1.,0.,1.,0.,1.,0.]], \
    [[1.,1.,0.,0.,0.,1.]], \
    [[1.,1.,0.,1.,0.,0.]]]) # conditioned input stimulus

# Generalisation
'''Note here that, for the third set of inputs here, the conditioned input
is different, but quite similar to the conditioned input for the second set
(i.e. 110001 vs 110100). However, when the third stimulus set is run through
the network (note without any unconditioned stimulus), as the third unconditioned
stimulus is sufficiently similar to the second, this third input set leads to
the same output as the second. In this sense, the pattern associator network is
able to generalise across different stimuli: if two stimuli are similar, they
are likely be treated as the same.
'''

# Graceful degradation (or fault tolerance)
'''An important propert of associative memory is their graceful degradation in
response to missing neurons or synapses. Due to the dot product and generalisation
principles described above, inputs with missing values can often still lead to
perfect recall. This is quite different to computer memory, which often produces
incorrect data if even only 1 storage location of their memory cannot be accessed.
This property of must have had great adaptive value for biological systems.
'''
numberOfDeadSynapses = 2
deadSynapses = [[randint(n_condInputs), randint(n_uncondInputs)] for i in range(numberOfDeadSynapses)]
if gracefulDegredation==True:
    for synInx in deadSynapses:
        w[synInx[0],synInx[1]] = np.float('NaN')

# A note on distributed representations (full vs sparse representation)
'''In this network, many different stimuli can be represented using different combinations
of 1s and 0s across just a handful of neurons. We can represent many different events or
stimuli with such overlapping sets of elements, because in general any one element cannot
be used to identify the stimulus, but instead the information about which stimulus is present
is distributed over the population of elements or neurons (this is called a distributed
representation). If, for binary neurons, half the neurons are in one state (e.g. 0), and the
other are in the other state (e.g. 1), then the representation is described as 'fully
distributed'. However, if only a smaller proportion of the neurons is active to represent a
stimulus, then this is a 'sparse representation'.
'''

# Run model
reps = 2 if repeatWithNoUCStimuli==True else 1
for e, rPrime,i in zip(eInputs, rPrimeInputs, range(len(eInputs))):
    print('\nStimulus pair %s' % (i+1))
    for i in range(reps):
        if i!=0: e = np.zeros(shape=e.shape) # Run network with no unconditioned stimulus (after learning)
        print('Uncond input = %s' % (e))
        print('Cond input = %s' % (rPrime))

        # Perform iteration of algorithm
        r = activationFunction(e)
        w += hebbRule(r,rPrime)
        h = synapticActivation(rPrime,w)
        outputRate = outputFiring(h)
        print(np.vstack((w,h,outputRate)))