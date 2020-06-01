#this file is part of eschbach_et_al_recurrent_2020
#Copyright (C) 2019 Ashok Litwin-Kumar
#see README for more information

resettimes = [int(30/dt),int(60/dt)] # 
dus = int(2/dt) # delay to from CS to US
cslen = int(3/dt) # length of CS presentation
uslen = int(3/dt) # length of US presentation
pvalid = 0.5 
'''On each trial, there is a 50% probability that one of the signals (for example, the US) will
be omitted, or a CS−odor will replace a CS+ odor, and the network will report a valence of 0 in
these cases, ensuring that only valid CS–US contingencies are learned.'''

# Generate random trials (either exctinction of second-order trials)
def genrandtrials(B):
    # T = max time; S = size of input vector (i.e. KC input); B = batch size; U = size of US vector; R = size of output vector
    s = np.zeros([T,S,B],dtype=np.float32) # KC input
    u = np.zeros([T,U,B],dtype=np.float32) # US input
    rtarg = np.zeros([T,R,B],dtype=np.float32) # target (i.e. correct) output
    # Define helper function
    randBool = lambda : np.random.randint(2)
    # Loop over batches
    for bi in range(B):
        # Choose either negative of positive valence
        val = np.random.choice([-1,1])
        if np.random.rand() < probex: # prob. of extinction trial 
            if np.random.rand() < pvalid:
                s[:,:,bi], u[:,:,bi], rtarg[:,:,bi] = extinctiontrial(val,doA=True,doUS=True,doA2=True,doUS2=False)
            else:
                s[:,:,bi], u[:,:,bi], rtarg[:,:,bi] = extinctiontrial(val,doA=randBool(),doUS=randBool(),doA2=randBool(),doUS2=randBool())
        else:
            if np.random.rand() < pvalid:
                s[:,:,bi], u[:,:,bi], rtarg[:,:,bi] = secondordertrial(val,doA=True,doUS=True,doA2=True,doC=False)
            else:
                s[:,:,bi], u[:,:,bi], rtarg[:,:,bi] = secondordertrial(val,doA=randBool(),doUS=randBool(),doA2=randBool(),doC=randBool())
    return s, u, rtarg # KC input, US input, target (i.e. correct) output


# # General function to generate array of trials
# def gentrials(ttype,B):
#     # T = max time; S = size of input vector (i.e. KC input); B = batch size; U = size of US vector; R = size of output vector
#     s = np.zeros([T,S,B],dtype=np.float32) # KC input
#     u = np.zeros([T,U,B],dtype=np.float32) # US input
#     rtarg = np.zeros([T,R,B],dtype=np.float32) # output
#     # Loop over batches
#     for bi in range(B):
#         # Choose either negative of positive valence
#         val = np.random.choice([-1,1])
#         # Add trials
#         if ttype == "extinction":
#             s[:,:,bi],u[:,:,bi],rtarg[:,:,bi] = extinctiontrial(valence=val, doUS=True, doUS2=False, doA=True, doA2=True)
#         elif ttype == "secondorder":
#             s[:,:,bi],u[:,:,bi],rtarg[:,:,bi] = secondordertrial(val,True,True,True,False)
#         elif ttype == "firstorder":
#             s[:,:,bi],u[:,:,bi],rtarg[:,:,bi] = firstordertrial(np.random.choice([-1,1]),True,False)
#     return s,u,rtarg

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def initialiseStore():
    s = np.zeros([T,S]) # KC inputs
    u = np.zeros([T,U]) # US inputs
    rtarg = np.zeros([T,R]) # target (i.e. correct) output
    return s,u,rtarg

def getKCInputIndices():
    inds = np.random.choice(S,int(f*S),replace=False)
    stim = np.zeros(S); stim[inds] = 1
    return stim

def getStimulusTimes():
    tA = int(np.random.randint(5,15)/dt) # Set time of initial CS (occurs randomly 5-15 ms)
    tA2 = int(np.random.randint(35,45)/dt) # Set time of an additional CS+ presentation (occurs randomly 35-45 ms)
    tUS = tA + dus # Set time of US following first CS
    tUS2 = tA2 + dus # Set time of US following second CS
    ttest = int(np.random.randint(65,75)/dt) # Set time of test CS+ presentation (65-75 ms)
    return tA,tA2,tUS,tUS2,ttest

# -------------------------------------------------------------------
# Generate extinction trial
# -------------------------------------------------------------------
def extinctiontrial(valence, doA, doUS, doA2, doUS2, returntimes=False):

    # -------------------
    # Setup
    # -------------------
    s,u,rtarg = initialiseStore() # Initialise stores
    stimA, stimB = [getKCInputIndices() for i in range(2)] # Get random indices for first CS (a) and second CS (b) KC cells
    tA,tA2,tUS,tUS2,ttest = getStimulusTimes() # Set times

    # -------------------
    # Add stimulus (CS and US) presentations
    # -------------------
    # Add first CS presentation to KC inputs store (if doA == True)
    if doA:
        s[tA:(tA+cslen),:] = stimA # note: tA = time of CS presentation; cslen = length of CS presentation
    # Set valence of first US presentation
    if doUS:
        if valence > 0:
            u[tUS:(tUS+uslen),0] = 1. # note: tUS = time of US presentation; uslen = length of US presentation
        else:
            u[tUS:(tUS+uslen),1] = 1.

    # Add second CS presentation to KC inputs store (if doA == True)
    if doA2:
        s[tA2:(tA2+cslen),:] = stimA
    else:
        s[tA2:(tA2+cslen),:] = stimB
    # Set valence of second US presentation
    if doUS2:
        if valence > 0:
            u[tUS2:(tUS2+uslen),0] = 1.
        else:
            u[tUS2:(tUS2+uslen),1] = 1.

    # Add test CS presentation
    s[ttest:(ttest+cslen),:] = stimA

    # -------------------
    # Add target outputs
    # -------------------
    # Set target (i.e. correct) outputs
    if doA: # if there is a first CS...
        if doUS: # ...and a first US...
            if doA2: # ...and a second CS...
                rtarg[tA2:(tA2+cslen),0] = valence
                if doUS2: # two pairings
                    rtarg[ttest:(ttest+cslen),0] = valence
                else: # one pairing, extinction
                    rtarg[ttest:(ttest+cslen),0] = valence/2 # the magnitude of the valence is halved for the final test CS+ presentation
            else: #one pairing, second odor presentation omitted (no extinction)
                rtarg[ttest:(ttest+cslen),0] = valence
        else: # no initial pairing
            if doA2 and doUS2:
                rtarg[ttest:(ttest+cslen),0] = valence
    else: # no initial odor
        if doA2 and doUS2:
            rtarg[ttest:(ttest+cslen),0] = valence

    # return results...
    if returntimes:
        return s,u,rtarg,tA,tA2,ttest
    else:
        return s,u,rtarg

# -------------------------------------------------------------------
# Generate second-order trial
# -------------------------------------------------------------------
def secondordertrial(valence,doA,doUS,doA2,doC,returntimes = False):
    
    # -------------------
    # Setup
    # -------------------
    s,u,rtarg = initialiseStore() # Initialise stores
    stimA, stimB, stimC = [getKCInputIndices() for i in range(3)] # Get random indices for first CS (a) and second CS (b) KC cells
    tA,tB,tUS,tA2,ttest = getStimulusTimes() # Set times

    # -------------------
    # Add stimulus (CS and US) presentations
    # -------------------
    if doA:
        s[tA:(tA+cslen),:] = stimA
    if doUS:
        if valence > 0:
            u[tUS:(tUS+uslen),0] = 1.
        else:
            u[tUS:(tUS+uslen),1] = 1.
    s[tB:(tB+cslen),:] = stimB

    if doA2:
        s[tA2:(tA2+cslen),:] = stimA
    elif doC:
        s[tA2:(tA2+cslen),:] = stimC
    s[ttest:(ttest+cslen),:] = stimB

    # -------------------
    # Add target outputs
    # -------------------
    if doUS and doA and doA2:
        rtarg[ttest:(ttest+cslen),0] = valence
        rtarg[tA2:(tA2+cslen),0] = valence

    # return results...
    if returntimes:
        return s,u,rtarg,tA,tB,ttest
    else:
        return s,u,rtarg


# -------------------------------------------------------------------
# Generate first-order trial
# -------------------------------------------------------------------
def firstordertrial(valence,doUS,doC,returntimes=False):

    # -------------------
    # Setup
    # -------------------
    s,u,rtarg = initialiseStore() # Initialise stores
    stimA, stimC = [getKCInputIndices() for i in range(2)] # Get random indices for first CS (a) and second CS (b) KC cells
    tA,_,tUS,_,ttest = getStimulusTimes() # Set times

    # -------------------
    # Add stimulus (CS and US) presentations
    # -------------------
    if doC:
        s[ttest:(ttest+cslen),:] = stimC
    else:
        s[ttest:(ttest+cslen),:] = stimA

    if doUS:
        if valence > 0:
            u[tUS:(tUS+uslen),0] = 1.
        else:
            u[tUS:(tUS+uslen),1] = 1.

    # -------------------
    # Add target outputs
    # -------------------
    if doUS and not doC:
        rtarg[ttest:(ttest+cslen),0] = valence

    if returntimes:
        return s,u,rtarg,tA,tUS,ttest
    else:
        return s,u,rtarg