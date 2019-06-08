
# to install - pip3 install brian2
from brian2 import * # Import Brian
from matplotlib.pyplot import *

# State sections to run
sectionToRun = 'STDP'

#------------------------------------------
# The simplest synapse
#------------------------------------------
if sectionToRun == 'simplestSynapse':

    # Once you have some neurons, the next step is to connect them up via
    # synapses. We'll start out with doing the simplest possible type of
    # synapse that causes an instantaneous change in a variable after a spike.

    eqs = '''
    dv/dt = (I-v)/tau : 1
    I : 1
    tau : second
    '''
    G = NeuronGroup(2, eqs, threshold='v>1', reset='v = 0', method='exact')
    G.I = [2, 0] # Input for neuron one = 2; two = 0
    G.tau = [10, 100]*ms # The neurons also have different tau

    # Given that neuron two has no driving input (i.e. as I = 0), this neuron
    # will not spike on its own. However, we can influence its activity by linking
    # it to neuron one via a synapse. In Brian, we can define synapses using the
    # following code:
    #   Synapses(source, target, ...) [i.e. going from source neurons to target neurons]
    # In the case below, the source and target are both the same (i.e. the group G).
    # The syntax "on_pre='v_post += 0.2'" means that when a spike occurs in the
    # presynaptic neuron (hence on_pre), it causes an instantaneous change to happen
    # v_post += 0.2. The _post means that the value of v referred to is the post-
    # synaptic value, and it is increased by 0.2. Therefore, in total, what this
    # model says is that whenever two neurons in G are connected by a synapse,
    # when the source neuron fires a spike, the target neuron will have its
    # value of v increased by 0.2.
    S = Synapses(G, G, on_pre='v_post += 0.2')

    # However, at this point we have only defined the synapse model, we haven't
    # actually created any synapses. The next line S.connect(i=0, j=1) creates
    # a synapse from neuron 0 to neuron 1.
    S.connect(i=0, j=1)

    # Run simulation and plot
    M = StateMonitor(G, 'v', record=True)
    run(100*ms)
    plot(M.t/ms, M.v[0], label='Neuron 0')
    plot(M.t/ms, M.v[1], label='Neuron 1')
    xlabel('Time (ms)')
    ylabel('v')
    legend()
    show()

#------------------------------------------
# Adding synaptic weights
#------------------------------------------
if sectionToRun == 'synapticWeights':

    eqs = '''
    dv/dt = (I-v)/tau : 1
    I : 1
    tau : second
    '''
    G = NeuronGroup(3, eqs, threshold='v>1', reset='v = 0', method='exact')
    G.I = [2, 0, 0]
    G.tau = [10, 100, 100]*ms

    # In the previous section, we hard coded the weight of the synapse to be the value
    # 0.2. However, often we would to want the weight to be different for different
    # synapses. We do that by introducing synapse equations. This is done in the code
    # below.

    # This example behaves very similarly to the previous example, but now there’s a
    # synaptic weight variable 'w'. The string 'w : 1' is an equation string, precisely
    # the same as for neurons, that defines a single dimensionless parameter 'w'. We
    # also changed the behaviour on a spike to on_pre='v_post += w' now, so that each
    # synapse can behave differently depending on the value of 'w'. To illustrate this,
    # we’ve made a third neuron which behaves precisely the same as the second neuron,
    # and connected neuron 0 to both neurons 1 and 2. We’ve also set the weights via
    # S.w = 'j*0.2'. (When i and j occur in the context of synapses, i refers to the
    # source neuron index, and j to the target neuron index). So this will give a
    # synaptic connection from 0 to 1 with weight 0.2=0.2*1 and from 0 to 2 with
    # weight 0.4=0.2*2.
    S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
    S.connect(i=0, j=[1, 2])
    S.w = 'j*0.2'

    # Run simulation and plot results
    M = StateMonitor(G, 'v', record=True)
    run(50*ms)
    plot(M.t/ms, M.v[0], label='Neuron 0')
    plot(M.t/ms, M.v[1], label='Neuron 1')
    plot(M.t/ms, M.v[2], label='Neuron 2')
    xlabel('Time (ms)')
    ylabel('v')
    legend()
    show()

#------------------------------------------
# Introducing a delay
#------------------------------------------
if sectionToRun == 'introducingDelay':

    # We can also create synapses which act with a given delay. As you will see below
    # this is as simple as adding a "line S.delay = 'j*2*ms'" (so that the synapse from
    # 0 to 1 has a delay of 2 ms, and from 0 to 2 has a delay of 4 ms).

    eqs = '''
    dv/dt = (I-v)/tau : 1
    I : 1
    tau : second
    '''
    G = NeuronGroup(3, eqs, threshold='v>1', reset='v = 0', method='exact')
    G.I = [2, 0, 0]
    G.tau = [10, 100, 100]*ms

    S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
    S.connect(i=0, j=[1, 2])
    S.w = 'j*0.2'
    S.delay = 'j*2*ms' # Synapse delay set to j*2

    M = StateMonitor(G, 'v', record=True)
    run(50*ms)
    plot(M.t/ms, M.v[0], label='Neuron 0')
    plot(M.t/ms, M.v[1], label='Neuron 1')
    plot(M.t/ms, M.v[2], label='Neuron 2')
    xlabel('Time (ms)')
    ylabel('v')
    legend()
    show()

#------------------------------------------
# More complex connectivity
#------------------------------------------
if sectionToRun == 'moreComplexConnectivity':

    # So far, we have specified the synaptic connectivity explicitly, but for
    # larger networks this isn't usually possible. For that, we usually want to
    # specify some condition. Below, we have created a dummy neuron group of N
    # neurons and a dummy synapses model that doens't actually do anything, just
    # to demonstrate the connectivity. The line "S.connect(condition='i!=j', p=0.2)"
    # will connect all pairs of neurons i and j with probability 0.2 as long as the
    # condition i!=j holds.
    N = 10
    G = NeuronGroup(N, 'v:1')
    S = Synapses(G, G)
    S.connect(condition='i!=j', p=0.2)
    # To change to 10% connection probability: "p=0.1"
    # To connect only neighboring neurons: "condition='abs(i-j)<4 and i!=j'""
    # To not connect neurons to non-existant neurons: "skip_if_invalid=True)"
    # For 1-1 connetions: "connect(j='i')"

    # We can visualise the above-defined connectivity using the function below
    def visualise_connectivity(S):
        Ns = len(S.source) # Get number of source neurons
        Nt = len(S.target) # Get number of target neurons
        # Get figure
        figure(figsize=(10, 4))
        subplot(121) # Left subplot...
        plot(zeros(Ns), arange(Ns), 'ok', ms=10)
        plot(ones(Nt), arange(Nt), 'ok', ms=10)
        for i, j in zip(S.i, S.j):
            plot([0, 1], [i, j], '-k')
        xticks([0, 1], ['Source', 'Target'])
        ylabel('Neuron index')
        xlim(-0.1, 1.1)
        ylim(-1, max(Ns, Nt))
        subplot(122)
        plot(S.i, S.j, 'ok')
        xlim(-1, Ns)
        ylim(-1, Nt)
        xlabel('Source neuron index')
        ylabel('Target neuron index')
        show()
    visualise_connectivity(S)

#------------------------------------------
# Distance-dependent connectivity
#------------------------------------------
if sectionToRun == 'distanceDependentConnectivity':

    # Building very much on the previous examples, sd can also do things 
    # like specifying the value of weights with a string. Below is an
    # example where we assign each neuron a spatial location and have a
    # distance-dependent connectivity function. We visualise the weight
    # of a synapse by the size of the marker.

    # Create neurons with location
    N = 30
    neuron_spacing = 50*umetre
    width = N/4.0*neuron_spacing
    G = NeuronGroup(N, 'x : metre') # Neuron has one variable x, its position
    G.x = 'i*neuron_spacing'

    # All synapses are connected (excluding self-connections)
    S = Synapses(G, G, 'w : 1')
    S.connect(condition='i!=j')

    # Weight varies with distance
    S.w = 'exp(-(x_pre-x_post)**2/(2*width**2))'

    # Plot results
    scatter(S.x_pre/um, S.x_post/um, S.w*20)
    xlabel('Source neuron position (um)')
    ylabel('Target neuron position (um)')
    show()


#------------------------------------------
# More complex synapse models: STDP (spike-timing dependent plasticity)
#------------------------------------------
if sectionToRun == 'STDP':

    # Brian's synapse framework is very general and can do things like short-term
    # plasticity (STP) or spike-timing dependent plasticity (STDP). Let's see how
    # that works for STDP. STDP is normally defined by an equation something like
    # this:
    #   Δ𝑤 =∑𝑡𝑝𝑟𝑒 and ∑𝑡𝑝𝑜𝑠𝑡: 𝑊(𝑡𝑝𝑜𝑠𝑡−𝑡𝑝𝑟𝑒)
    # In other words, the change in synaptic weight w is the sum over all presynaptic
    # spike times (𝑡𝑝𝑟𝑒) and postsynaptic spike times (𝑡𝑝𝑜𝑠𝑡) of some function 𝑊 of the
    # difference in these spike times:
    # 𝑊(Δ𝑡) = {𝐴𝑝𝑟𝑒^(𝑒−Δ𝑡/𝜏𝑝𝑟𝑒)    : Δ𝑡>0
    #         {𝐴𝑝𝑜𝑠𝑡^(𝑒Δ𝑡/𝜏𝑝𝑜𝑠𝑡)    : Δ𝑡<0
    # In Python code, this function looks like:
    tau_pre = tau_post = 20*ms
    A_pre = 0.01
    A_post = -A_pre*1.05
    delta_t = linspace(-50, 50, 100)*ms
    W = where(delta_t>0, A_pre*exp(-delta_t/tau_pre), A_post*exp(delta_t/tau_post))
    plot(delta_t/ms, W)
    xlabel(r'$\Delta t$ (ms)')
    ylabel('W')
    axhline(0, ls='-', c='k')
    # show()

    # We could use this implementation in neural simulations. However, this approach
    # would be very inefficient, because we would have to sum over all pairs of spikes.
    # In addition, this approach would arguably be physiologically unrealistic, because
    # the neuron cannot remember all its previous spike times. It turns out there is a
    # more efficient and physiologically more plausible way to get the same effect:

    # We define two new variables (𝑎𝑝𝑟𝑒) and (𝑎𝑝𝑜𝑠𝑡), which are "traces" of pre- and post-
    # synaptic activity, governed by the differential equations:
    # 𝜏𝑝𝑟𝑒(d/d𝑡)𝑎𝑝𝑟𝑒 = −𝑎𝑝𝑟𝑒
    # 𝜏𝑝𝑜𝑠𝑡(d/d𝑡)𝑎𝑝𝑜𝑠𝑡 = −𝑎𝑝𝑜𝑠𝑡
    # We also define the constants (𝐴𝑝𝑟𝑒) and (𝐴𝑝𝑜𝑠𝑡), which set the change in 𝑎𝑝𝑟𝑒 and 𝑎𝑝𝑜𝑠𝑡
    # after pre- and post-synaptic potentials. When a presynaptic spike occurs, the
    # presynaptic trace is updated and the weight is modified according to the rule:
    # 𝑎𝑝𝑟𝑒  →   𝑎𝑝𝑟𝑒 + 𝐴𝑝𝑟𝑒
    # 𝑤    →   𝑤 + 𝑎𝑝𝑜𝑠𝑡
    # When a postsynaptic spike occurs:
    # 𝑎𝑝𝑜𝑠𝑡  →   𝑎𝑝𝑜𝑠𝑡 + 𝐴𝑝𝑜𝑠𝑡
    # 𝑤    →   𝑤 + 𝑎𝑝𝑟𝑒

    # Now that we have a formulation that relies only on differential equations and
    # spike events, we can turn that into Brian code:


    taupre = taupost = 20*ms

    wmax = 0.01
    Apre = 0.01
    Apost = -Apre*taupre/taupost*1.05

    G = NeuronGroup(2, 'v:1', threshold='t>(1+i)*10*ms', refractory=100*ms)
    # 't>(1+i)*10*ms': to make neuron 0 fire a spike at time 10 ms, and neuron 1 at time 20 ms
    
    # There are a few things to see there. Firstly, when defining the synapses, we've
    # given a more complicated multi-line string defining three synaptic variables
    # (w, apre and apost). We've also got a new bit of syntax there, (event-driven)
    # after the definitions of apre and apost. What this means is that although these
    # two variables evolve continuously over time, Brian should only update them at
    # the time of an event (a spike). This is because we don't need the values of
    # apre and apost except at spike times, and it is more efficient to only update
    # them when needed.

    # Next, we have a on_pre=... argument. The first line is "v_post += w". This is
    # the line that actually applies the synaptic weight to the target neuron.
    # The second line is "apre += Apre", which encodes the rules described previously.
    # In the third line, we're also encoding the rules described previously, but we've
    # added one extra feature: we've clamped the synaptic weights between a minimum of
    # 0 and a maximum of wmax so that the weights can't get too large or negative. The
    # function clip(x, low, high) does this.

    # Finally, we have a on_post=... argument. This gives the statements to calculate when
    # a post-synaptic neuron fires. Note that we do not modify v in this case, only the
    # synaptic variables.

    # Notice here at "(clock-driven" has been used (rather than "(event-driven"), so that
    # these variables are sampled at every moment, rather than just when spikes occur.
    S = Synapses(G, G, '''
        w : 1
        dapre/dt = -apre/taupre : 1 (clock-driven)
        dapost/dt = -apost/taupost : 1 (clock-driven)
        ''',
        on_pre='''
        v_post += w
        apre += Apre
        w = clip(w+apost, 0, wmax)
        ''',
        on_post='''
        apost += Apost
        w = clip(w+apre, 0, wmax)
        ''', method = 'exact')
    S.connect(i=0, j=1)
    # S.apost = 1
    M = StateMonitor(S, ['w', 'apre', 'apost'], record=True)

    run(30*ms)

    figure(figsize=(4, 8))
    subplot(211)
    plot(M.t/ms, M.apre[0], label='apre')
    plot(M.t/ms, M.apost[0], label='apost')
    legend()
    subplot(212)
    plot(M.t/ms, M.w[0], label='w')
    legend(loc='best')
    xlabel('Time (ms)')
    show()
