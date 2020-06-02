#this file is part of eschbach_et_al_recurrent_2020
#Copyright (C) 2019 Ashok Litwin-Kumar
#see README for more information

# Get initial neural rates
m = torch.from_numpy(m0) # MBON
d = torch.from_numpy(d0) # DAN
o = torch.from_numpy(o0) # FBN

# Get initial weights
w = torch.from_numpy(w0)
wfast = torch.from_numpy(w0)

# Get task variables
s = torch.from_numpy(s0) # KC inpit
u = torch.from_numpy(u0) # US input
rtarg = torch.from_numpy(rtarg0) # target outputs

# ????????????????????????????????????????????????????
sbar = torch.from_numpy(sbar0) # firing rate of the KC
dastdpbar = torch.from_numpy(dastdpbar0)

# Get CS and US input times
baselineinds = (np.sum(s0,1,keepdims=True) == 0) & (np.sum(u0,1,keepdims=True) == 0)
baselineinds_d = torch.from_numpy(np.repeat(baselineinds,D,axis=1).astype(np.float32)) # copy CS/US inputs for all DAN neurons

# Initialise stores
ma = np.zeros([T,M,B]) # MBON firing rate
da = np.zeros([T,D,B]) # DAN firing rate
oa = np.zeros([T,O,B]) # FBN firing rate
wa = np.zeros([T,M,S,B]) # from KC to an MBON
r = torch.zeros(T,R,B) # output valence
dacost = torch.zeros(1) # regularization term

# Loop over time points
for ti in range(T):
    # Reset firing rates to initial values when time = reset time
    if ti in resettimes:
        mnew = torch.from_numpy(m0)
        dnew = torch.from_numpy(d0)
        onew = torch.from_numpy(o0)
        sbar = torch.from_numpy(sbar0)
        dastdpbar = torch.from_numpy(dastdpbar0)
    else: # else, apply differentialy equations to mode firing rates
        mnew = (1-dt)*m + dt*torch.relu(torch.tanh(Jmm.mm(m) + Jmo.mm(o) + torch.einsum('ijb,jb->ib',(w,s[ti,:,:])) + bm))
        dnew = (1-dt)*d + dt*torch.relu(torch.tanh(Jdm.mm(m) + Jdd.mm(d) + Jdo.mm(o) + wdu.mm(u[ti,:]) + bd))
        onew = (1-dt)*o + dt*torch.relu(torch.tanh(Jom.mm(m) + Jod.mm(d) + Joo.mm(o) + wou.mm(u[ti,:]) + bo))
    
    # Update firing rate values and save to stores
    m,d,o = mnew, dnew, onew
    ma[ti,:,:] = m.detach()
    da[ti,:,:] = d.detach()
    oa[ti,:,:] = o.detach()
    wa[ti,:,:,:] = w.detach()

    # Get mushroom body firing rates (i.e. output valence)
    r[ti,:,:] = wrm.mm(m) # matrix multiply wrm(biases) and m (i.e. MBON rates)
    
    # Equation 2
    dastdp = torch.relu(Jmd.mm(d)) # (di in paper) a weighted sum of DAN inputs according to the DAN-to-MBON connectivity matrix
    stdp_update = -torch.einsum('ib,jb->ijb',(dastdp,sbar)) + torch.einsum('ib,jb->ijb',(dastdpbar,s[ti,:,:])) # anti-Hebbian
    wfast = torch.relu(w + dt*(stdp_update - torch.relu(stdp_update - (wmax - w)))) #update that does not exceed wmax
    
    # Equation 3
    w = w + (dt/tauw)*(wfast - w)

    # ????????????????????????????????????????????????????????
    dastdpbar = (1. - dt/tau)*dastdpbar + (dt/tau)*dastdp
    sbar = (1. - dt/tau)*sbar + (dt/tau)*s[ti,:,:] # firing rate of the KC

    # Calculate regularization term
    dacost += torch.sum(torch.pow(torch.relu(d*baselineinds_d[ti,:,:]-dinit),2))
