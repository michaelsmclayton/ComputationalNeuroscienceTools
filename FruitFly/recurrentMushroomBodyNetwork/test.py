import torch
import time
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Setup
# --------------------------------------------------------

# Training parameters
Nepochs = 15 # original = 1,500
Ntrain = 20 # number of training iterations
lr = 2*np.logspace(-3,-3,Nepochs) # learning rate
alphada = .1 # ???????????????????????????????????????????????
B = 30 # batch size
dt = 0.5
T = int(80/dt) # max time
Nepochspc = max(int(Nepochs/100),1)

# Analysis parameters
ttypes = ["classical"] # types: (epochIndexther "classical" or "context") 
conds = ["control", "nofbns", "no2step", "nocrosscom", "nowithincom"] # conditions
trialtypes = ["extinction", "secondorder", "firstorder"]
Nttypes, Nconds = len(ttypes), len(conds)
doplot = False
ttype = 'classical'

# --------------------------------------------------------
# Test
# --------------------------------------------------------

# Initialise results store
resultsStore = {}
for trialtype in trialtypes:
    resultsStore[trialtype] = {}
    for cond in conds:
        resultsStore[trialtype][cond] = []

# Iterate over trial types
for trialType in trialtypes:
    print(f'Running {trialType}...')
    # Iterate over network types
    for cond in conds:

        # Loop over networks
        for net in range(1,21):
            print(f'... {cond} - net {net}')

            # Load data
            exec(open('loadmat.py').read())

            # Load trial running functions
            if ttype == "classical":
                exec(open("classical.py").read())
            elif ttype == "context":
                exec(open("context.py").read())
            else:
                print("ERROR: invalid trial type")

            # Generate random trials and initial conditions
            '''For networks trained on first-order conditioning, second-order conditioning and extinction, training consists
            of random second-order conditioning and extinction trials (for which first-order conditioning is a subcomponent)'''
            s0, u0, rtarg0 = gentrials(trialType,B)
            m0, d0, o0, w0, sbar0, dastdpbar0 = geninitialcond(B)

            # Load network values
            train_vars,track_loss = torch.load(f'./original/Fig_7_code/classical-{cond}-{net}.pt')
            # train_vars,track_loss = torch.load(f'./results/classical-control-1.pt')
            Jmm, Jmd, Jmo, Jdm, Jdd, Jdo, Jom, Jod, Joo, bm, bd, bo, wdu, wou, wrm = train_vars

            # Run model!
            exec(open("runmodel.py").read())

            # Get loss
            loss_err = torch.sum(torch.pow(r-rtarg,2))/B # squared distance between the actual and target valence (summer over timesteps)
            loss_da = alphada*dacost/B # regularisation term for DAN activity
            loss = loss_err + loss_da # total loss (i.e. error + regularisation)

            # Store loss
            resultsStore[trialType][cond].append(np.array(loss_err.detach()).tolist())

# Plot results
fig,axs = plt.subplots(1,3)
for i,trialType in enumerate(trialtypes):
    for b,cond in enumerate(conds):
        data = resultsStore[trialType][cond]
        axs[i].scatter(b*np.ones(len(data)), data)
    # axs[i].set_xticklabels(conds)
plt.show()