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
Nttypes, Nconds = len(ttypes), len(conds)
doplot = False

# Initialise loss
lossa = np.zeros([Nttypes,Nconds,Ntrain,Nepochs])


# --------------------------------------------------------
# Training
# --------------------------------------------------------

# Iterate over training iterations
for itrain in range(Ntrain):
    # Iterate over training types
    for ittype in range(Nttypes):
        ttype = ttypes[ittype] # get current type
        # Iterate over conditions
        for icond in range(Nconds):

            # Skip if there is already a (non-zero) loss value saved
            if lossa[ittype,icond,itrain,-1] != 0:
                continue

            # Get current condition
            cond = conds[icond]

            # Print properties of current iteration
            print("\n",ttype,", trial type ",ittype+1,"/",Nttypes,"; ",cond,", condition ",icond+1,"/",Nconds,"; training iteration ",itrain+1,"/",Ntrain)

            # Define model file name
            modelfname = './results/%s-%s-%s.pt' % (ttype, cond, str(itrain+1))
            print("model filename: ",modelfname)
            
            # Load data
            # train_vars, S, U, R, probex, f, minit = getData(cond, B) # not working
            exec(open('loadmat.py').read())

            # Run a trial (depending on trial type)
            if ttype == "classical":
                exec(open("classical.py").read())
            elif ttype == "context":
                exec(open("context.py").read())
            else:
                print("ERROR: invalid trial type")

            # Define optimiser
            opt = torch.optim.RMSprop(train_vars,lr=lr[0])

            # Intialise loss trackers
            track_loss = np.zeros(Nepochs)
            track_loss_da = np.zeros(Nepochs)

            # Iterate over epochs
            lastt = time.time()
            for epochIndex in range(Nepochs):

                # Set learning rates for each epoch
                for g in opt.param_groups:
                    g['lr'] = lr[epochIndex]

                # Generate random trials and initial conditions
                '''For networks trained on first-order conditioning, second-order conditioning and extinction, training consists
                of random second-order conditioning and extinction trials (for which first-order conditioning is a subcomponent)'''
                s0, u0, rtarg0 = genrandtrials(B)
                m0, d0, o0, w0, sbar0, dastdpbar0 = geninitialcond(B)

                # Run model!
                exec(open("runmodel.py").read())

                # Get loss
                loss_err = torch.sum(torch.pow(r-rtarg,2))/B # squared distance between the actual and target valence (summer over timesteps)
                loss_da = alphada*dacost/B # regularisation term for DAN activity
                loss = loss_err + loss_da # total loss (i.e. error + regularisation)

                # Update loss values in stores
                track_loss[epochIndex] = loss_err
                track_loss_da[epochIndex] = loss_da
                
                # Perform optimisation
                loss.backward() # compute gradients
                opt.step() # perform single optimisation step
                opt.zero_grad() # clear gradients

                # Enforce sparse connectivity (J**bin = 1 where initial connectivity is not zero)
                '''I assume this means that connectivity cannot change unless it is a non-zero value to begin with'''
                Jmm.data = Jmm.data * Jmmbin
                Jmd.data = Jmd.data * Jmdbin
                Jmo.data = Jmo.data * Jmobin
                Jdm.data = Jdm.data * Jdmbin
                Jdd.data = Jdd.data * Jddbin
                Jdo.data = Jdo.data * Jdobin
                Jom.data = Jom.data * Jombin
                Jod.data = Jod.data * Jodbin
                Joo.data = Joo.data * Joobin
                
                # Excitatory/inhibitory neurons
                #   MBON-MBON
                Jmm.data[:,mexc] = torch.relu(Jmm.data[:,mexc])
                Jmm.data[:,minh] = -torch.relu(-Jmm.data[:,minh])
                #   DAN-MBON
                Jdm.data[:,mexc] = torch.relu(Jdm.data[:,mexc])
                Jdm.data[:,minh] = -torch.relu(-Jdm.data[:,minh])
                #   FBN-MBOM
                Jom.data[:,mexc] = torch.relu(Jom.data[:,mexc])
                Jom.data[:,minh] = -torch.relu(-Jom.data[:,minh])
                #   MBOM-FBN
                Jmo.data[:,oexc] = torch.relu(Jmo.data[:,oexc])
                Jmo.data[:,oinh] = -torch.relu(-Jmo.data[:,oinh])
                #   DAN-FBN
                Jdo.data[:,oexc] = torch.relu(Jdo.data[:,oexc])
                Jdo.data[:,oinh] = -torch.relu(-Jdo.data[:,oinh])
                #   FBN-FBN
                Joo.data[:,oexc] = torch.relu(Joo.data[:,oexc])
                Joo.data[:,oinh] = -torch.relu(-Joo.data[:,oinh])
                
                # Update valence biases (MBON to PI mapping)
                wrm.data[:,mpos] = torch.relu(wrm.data[:,mpos])
                wrm.data[:,mneg] = -torch.relu(-wrm.data[:,mneg])

                # Plotting
                if (epochIndex % Nepochspc) == 0:
                    if doplot:
                        plt.clf()
                        plt.subplot(311)
                        plt.semilogy(track_loss[:epochIndex]/B)
                        plt.semilogy(track_loss_da[:epochIndex]/B)
                        plt.xlim(0,Nepochs)
                        plt.subplot(312)
                        plt.plot(r[:,0,0].detach().numpy())
                        plt.plot(rtarg[:,0,0].numpy())
                        plt.ylim(-1.1,1.1)
                        plt.subplot(313)
                        ra = np.vstack([ma[:,:,0].T,da[:,:,0].T,oa[:,:,0].T])
                        plt.imshow(ra,vmin=0,vmax=1)
                        plt.pause(.0001)
                        plt.show()
                    curt = time.time()
                    print("\r" + str(int(epochIndex/Nepochspc)) + "%, ", np.round(curt-lastt,2), "seconds", end="")
                    lastt = curt

            # Save results
            lossa[ittype,icond,itrain,:] = track_loss
            torch.save((train_vars,track_loss),modelfname)
