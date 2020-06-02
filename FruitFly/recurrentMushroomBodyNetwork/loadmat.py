#this file is part of eschbach_et_al_recurrent_2020
#Copyright (C) 2019 Ashok Litwin-Kumar
#see README for more information

import csv
import numpy as np
import torch
from torch.autograd import Variable

# # For debugging
# import matplotlib.pylab as plt
# cond = 'control'
# B = 30

# Parameters
f = 0.1
S = 70 # size of input vector (i.e. KC input)
U = 2 # size of US vector
R = 1 # size of output vector
X = 1 # position (not used for this one)
probex = 0.5 # prob. of extinction trial 
tau = 5 # stdp time constant
tauw = 5
wmax = 1./(S*f)
minit = 0.
oinit = .1
dinit = .1

# Define helper functions
def initrand(X,Y,scalefac=.5):
    return (scalefac*np.random.standard_normal([X,Y])/np.sqrt(Y)).astype(np.float32)

def initrandJ(J,scalefac=0.5):
    Jnew = J.astype(np.float32)
    N0 = Jnew.shape[0]
    for ii in range(N0):
        if np.sum(Jnew[ii,:] > 0):
            Jnew[ii,:] = Jnew[ii,:] / np.sqrt(np.sum(Jnew[ii,:]**2))
    return scalefac*Jnew

def geninitialcond(B):
    '''At the beginning of a trial, MBON rates are initialized to 0, whereas DAN and feedback neuron rates are initialized to 0.1'''
    m0 = minit*np.ones([M,B],dtype=np.float32) # MBON rates
    d0 = dinit*np.ones([D,B],dtype=np.float32) # DAN rates
    o0 = oinit*np.ones([O,B],dtype=np.float32) # feedback rates
    w0 = np.ones([M,S,B],dtype=np.float32)/(S*f) # set weights to 1/(S*f)
    sbar0 = np.zeros([S,B],dtype=np.float32)
    dabar0 = np.zeros([M,B],dtype=np.float32)
    return m0,d0,o0,w0,sbar0,dabar0

# Get connectivity data
ff = open('data/Supp_Connectivity_Matrix_MBIN-MBON-FBN-FAN-FB2IN-FFN.csv')

# Get neuron names
rows = ff.read().split('\n')
names = rows[0].split(',')[1:]
names = [x.strip("\" ") for x in names]
Nnames = len(names)
J = np.zeros([Nnames,Nnames])

# Get connectivity matrix
namesCheck = []
for ii in range(Nnames):
    row = rows[ii+1].split(',')
    row = [x.strip("\" ") for x in row]
    namesCheck.append(row[0])
    J[ii,:] = [int(x) for x in row[1:]]
J = J.T #change pre->post to post->pre
assert(names==namesCheck) # check that both names variables are the same

# Get indices of left vs. right neurons
indsl = [("LEFT" in x) for x in names]
indsr = [("RIGHT" in x) for x in names]

# Get names of left vs. right neurons
namesl = np.array(names)[indsl]
namesr = np.array(names)[indsr]
namesl = [x.replace(" LEFT","") for x in namesl]
namesr = [x.replace(" RIGHT","") for x in namesl]
assert(namesl == namesr) # check that neurons on left and right sides are identical
names = np.array(namesl)
Nnames = len(names) # update Nnames

# Get connectivity values (between and within hemispheres)
Jll = J[indsl,:][:,indsl] # find connectivity values when pre- or post-synaptic neuron is left
Jrr = J[indsr,:][:,indsr] # find connectivity values when pre- or post-synaptic neuron is right
Jlr = J[indsl,:][:,indsr] # find connectivity values when pre-synaptic is left, and post-synaptic neuron is right
Jrl = J[indsr,:][:,indsl] # find connectivity values when pre-synaptic is right, and post-synaptic neuron is left
Jl = Jll + Jlr # Get connectivity when pre-synaptic neuron is left
Jr = Jrl + Jrr # Get connectivity when pre-synaptic neuron is right

# Reduce connectivity matrix to include only reliable connections
'''"We consider ‘reliable’ connections those for which the connections between the left and right
homologous neurons have at least three synapses each and their sum is at least 10"'''
Jipsi = (Jll > 2) * (Jrr > 2) # At least three synapses
Jcontra = (Jlr > 2) * (Jrl > 2)
J1 = Jipsi | Jcontra
J10 = (Jl + Jr) > 9 # Their sum is at least 10
J = (J1*J10) * (Jl+Jr)/2

# Get indices for each neural type (i.e. mushroom body output, modulatory, and modulatory input (feedback + feedforward))
minds = ["MBON" in x for x in names]
dinds = np.array(["DAN" in x for x in names]) | np.array(["OAN" in x for x in names]) | np.array(["MBIN" in x for x in names])
fbinds = np.array(["FBN" in x for x in names]) | np.array(["FB2IN" in x for x in names]) | np.array(["FAN" in x for x in names])
ffinds = np.array(["FFN" in x for x in names])
oinds = fbinds | ffinds # combine feedback and feedforward
M = np.sum(minds)
D = np.sum(dinds)
O = Nnames - M - D

# Get names for each neural type
mnames = names[minds]
dnames = names[dinds]
onames = names[oinds]
fb2inds = ["FB2IN" in x for x in onames]

# Get initial connectivity values (from EM)
connectivityDictionary = {}
indices, neuralnames = [minds,dinds,oinds], ['m','d','o']
for preinds,prename in zip(indices,neuralnames):
    for postinds,postname in zip(indices,neuralnames):
        connectivityDictionary[f'J{prename}{postname}0'] = J[preinds,:][:,postinds]
locals().update(connectivityDictionary)

# Get names of inhibitory and excitatory neurons
ff = open('data/inh.txt')
namesinh = ff.read().split('\n')
ff = open('data/exc.txt')
namesexc = ff.read().split('\n')

# Get indices of inhibitory vs. excitatory neurons for each neuron type
minh = [x in namesinh for x in mnames]
oinh = [x in namesinh for x in onames]
mexc = [x in namesexc for x in mnames]
oexc = [x in namesexc for x in onames]

# Get names of MBONS coding for negative vs positive valence
ff = open('data/pipos.txt')
namespos = ff.read().split('\n')
ff = open('data/pineg.txt')
namesneg = ff.read().split('\n')
mpos = [x in namespos for x in mnames]
mneg = [x in namesneg for x in mnames]

# Get compartment names (and a matrix of compartment membership for MBONs and DANs)
comids = np.unique([x[-2] for x in np.append(mnames,dnames)]) # compartments names (a-q)
Ncom = len(comids) # number of compartments
comm = np.zeros([Ncom,M]) # matrix of compartment membership for MBONs
comd = np.zeros([Ncom,D]) # matrix of compartment membership for DANs 
for mi in range(M):
    comm[:,mi] = mnames[mi][-2] == comids
for di in range(D):
    comd[:,di] = dnames[di][-2] == comids

# Get indices of DAN connections that go from DAN to FBN/FFN
como = np.zeros([Ncom,O])
comoinds = np.argmax(comd @ (Jdo0), axis=0) # comd @ (Jdo0) = all DAN connections that go from DAN to FBN/FFN 
como[comoinds,np.arange(O)] = 1

# Set connection values to zero (depending on current condition)
if cond == "control": # no change
    pass
elif cond == "control2":
    Jmm0[:,:] = 0
    Jdm0[:,:] = 0
elif cond == "nofb": # no feedback
    Joo0[:] = 0
    Jmo0[:] = 0
    Jdo0[:] = 0
    Jdm0[:] = 0
    Jdd0[:] = 0
elif cond == "nofbns": # no feedback neurons
    Joo0[:] = 0
    Jmo0[:] = 0
    Jdo0[:] = 0
elif cond == "nofb2ins": # no two-step feedback
    Joo0[:,fb2inds] = 0
    Jmo0[:,fb2inds] = 0
    Jdo0[:,fb2inds] = 0
elif cond == "no2step": # !!!!!!!!!!!!!!!!!!!!!!!!!!!! SOMETHING VERY ODD HAS HAPPENED HERE! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''nofb2ins above doesn't seem to be used on the training script, but no2step is. However, this doesn't seem right, with
    fb2inds only used once below?'''
    Joo0[:,:] = 0
    Jmo0[:,:] = 0
    Jdo0[:,fb2inds] = 0
elif cond == "no2step_dirind":
    Joo0[:,:] = 0
    Jmo0[:,:] = 0
    Jdo0[:,fb2inds] = 0
    Jmm0[:,:] = 0
    Jdm0[:,:] = 0
elif cond == "nocrosscom": # no cross-compartment communication
    Joo0 = Joo0 * (como.T @ como)
    Jmo0 = Jmo0 * (comm.T @ como)
    Jdo0 = Jdo0 * (comd.T @ como)
    Jom0 = Jom0 * (como.T @ comm)
    Jod0 = Jod0 * (como.T @ comd)
elif cond == "nowithincom": # no within-compartment communication
    Joo0 = Joo0 * (1-(como.T @ como))
    Jmo0 = Jmo0 * (1-(comm.T @ como))
    Jdo0 = Jdo0 * (1-(comd.T @ como))
    Jom0 = Jom0 * (1-(como.T @ comm))
    Jod0 = Jod0 * (1-(como.T @ comd))
elif cond == "nocrosscom_dirind":
    Joo0 = Joo0 * (como.T @ como)
    Jmo0 = Jmo0 * (comm.T @ como)
    Jdo0 = Jdo0 * (comd.T @ como)
    Jom0 = Jom0 * (como.T @ comm)
    Jod0 = Jod0 * (como.T @ comd)
    Jmm0 = Jmm0 * (comm.T @ comm)
    Jmd0 = Jmd0 * (comm.T @ comd)
    Jdd0 = Jdd0 * (comd.T @ comd)
    Jdm0 = Jdm0 * (comd.T @ comm)
else:
    print("WARNING: INVALID CONDITION")

# Get indices of where given connectivity values do not equal zero
Jmmbin = torch.from_numpy((Jmm0 != 0).astype(np.float32))
Jmdbin = torch.from_numpy((Jmd0 != 0).astype(np.float32))
Jmobin = torch.from_numpy((Jmo0 != 0).astype(np.float32))
Jdmbin = torch.from_numpy((Jdm0 != 0).astype(np.float32))
Jddbin = torch.from_numpy((Jdd0 != 0).astype(np.float32))
Jdobin = torch.from_numpy((Jdo0 != 0).astype(np.float32))
Jombin = torch.from_numpy((Jom0 != 0).astype(np.float32))
Jodbin = torch.from_numpy((Jod0 != 0).astype(np.float32))
Joobin = torch.from_numpy((Joo0 != 0).astype(np.float32))

# Initialise random bias (valence) values for MBONs
mbias = np.random.randn(1,M)/np.sqrt(M)
mbias[:,mpos] = np.abs(mbias[:,mpos])
mbias[:,mneg] = -np.abs(mbias[:,mneg])

# Declare Torch trained variables
Jmm = Variable(torch.from_numpy(initrandJ(Jmm0)),requires_grad=True)
Jmd = Variable(torch.from_numpy(initrandJ(Jmd0)),requires_grad=True)
Jmo = Variable(torch.from_numpy(initrandJ(Jmo0)),requires_grad=True)
Jdm = Variable(torch.from_numpy(initrandJ(Jdm0)),requires_grad=True)
Jdd = Variable(torch.from_numpy(initrandJ(Jdd0)),requires_grad=True)
Jdo = Variable(torch.from_numpy(initrandJ(Jdo0)),requires_grad=True)
Jom = Variable(torch.from_numpy(initrandJ(Jom0)),requires_grad=True)
Jod = Variable(torch.from_numpy(initrandJ(Jod0)),requires_grad=True)
Joo = Variable(torch.from_numpy(initrandJ(Joo0)),requires_grad=True)
bm  = Variable(torch.from_numpy(0.*np.ones([M,1]).astype(np.float32)),requires_grad=True)
bd  = Variable(torch.from_numpy(0.*np.ones([D,1]).astype(np.float32)),requires_grad=True)
bo  = Variable(torch.from_numpy(0.*np.ones([O,1]).astype(np.float32)),requires_grad=True)
wdu = Variable(torch.from_numpy(np.random.standard_normal([D,U]).astype(np.float32)),requires_grad=True)
wou = Variable(torch.from_numpy(np.random.standard_normal([O,U]).astype(np.float32)),requires_grad=True)
wrm = Variable(torch.from_numpy(mbias.astype(np.float32)),requires_grad=True)
train_vars = [Jmm, Jmd, Jmo, Jdm, Jdd, Jdo, Jom, Jod, Joo, bm, bd, bo, wdu, wou, wrm]

