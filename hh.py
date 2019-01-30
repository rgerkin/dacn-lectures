import warnings
warnings.filterwarnings("ignore") # Turn off distracting warning messages that you don't need to see

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)

# Importing functions and units from the Brian2 library
from brian2 import Equations, start_scope, run, NeuronGroup, Synapses, StateMonitor, SpikeMonitor, defaultclock
from brian2 import ms, second # units of time
from brian2 import ufarad # units of capacitance
from brian2 import pA # units of current
from brian2 import siemens, msiemens, nS # units of conductance
from brian2 import mV # units of voltage 
from brian2 import umetre, cm # units of length

def go(n_neurons=40, frac_e=0.5, p_e=1, p_i=1, w_e=10*nS, w_i=25*nS, VT=-60*mV, 
       sigma_s=25*pA, tau_s=10*ms, T=1*second, dt=2e-2*ms):
    """
    Sets up and runs a network simulation
    
    Inputs:
        n_neurons: Number of neurons in the network
        frac_e: Fraction of neurons that are excitatory (the rest will be inhibitory)
        p_e: Probability that a potential excitatory synaptic connection is realized
        p_i: Probability that a potential inhibitory synaptic connection is realized
        w_e: Excitatory synaptic conductance (weight)
        w_i: Inhibitory synaptic conductance (weight)
        VT: 
        sigma_s: Amount of Ornstein-Uhlenbeck (OU) noise
        tau_s: Time constant of OU noise
        T: Duration of the simulation
    Returns:
        traces: The membrane potential traces
        spikes: The spike times
    """
    start_scope() # Start a new simulation
    np.random.seed(0)  # Use the same random seed each time for reproducbility
    defaultclock.dt = dt

    # Parameters to describe the size and electrical properties of the cell
    area = 20000*umetre**2  # Surface area of the cell membrane
    Cm = (1*ufarad*cm**-2) * area  # Capacitance of the cell membrane
    El = -60*mV  # Reversal potential of "Leak" conductance
    EK = -90*mV  # Reversal potential for potassium
    ENa = 50*mV  # Reversal potential for sodium
    gl = (5e-5*siemens*cm**-2) * area  # "Leak" conductance
    g_na = (100*msiemens*cm**-2) * area  # Total conductance of sodium channels
    g_kd = (30*msiemens*cm**-2) * area  # Total conductance of potassium channels

    # Time constants to describe how synaptic conductances decay
    taue = 5*ms  # Decay time constant for excitatory synaptic conductances
    taui = 10*ms  # Decay time constant for inhibitory synaptic conductances

    # Reversal potentials (the electromotive driving force for each type of ion, basically)
    Ee = 0*mV  # Overall reversal potential for excitatory synapses
    Ei = -80*mV  # Overall reversal potential for inhibitory synapses

    # The model (as a set of ODEs)
    eqs = Equations('''
    dv/dt = (gl*(El-v)+s+ge*(Ee-v)+gi*(Ei-v)-
             g_na*(m*m*m)*h*(v-ENa)-
             g_kd*(n*n*n*n)*(v-EK))/Cm : volt
    ds/dt = sigma_s*sqrt(2/tau_s)*xi - s/tau_s : amp
    dm/dt = alpha_m*(1-m)-beta_m*m : 1
    dn/dt = alpha_n*(1-n)-beta_n*n : 1
    dh/dt = alpha_h*(1-h)-beta_h*h : 1
    dge/dt = -ge*(1./taue) : siemens
    dgi/dt = -gi*(1./taui) : siemens
    alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
             (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
    beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
            (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
    alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
    beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
    alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
             (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
    beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
    ''')

    # The network, describing how each of these (otherwise identical) neurons is connected to each other
    
    count_spike_at = -20  # Count an action potential when the membrane potential reaches this value, in units of mV
    refractory_period = 3*ms  # Minimum time between action potentials
    P = NeuronGroup(n_neurons, model=eqs, threshold='v>-%f*mV' % count_spike_at, refractory=refractory_period,
                    method='euler', dt=dt)
    n_excitatory = int(n_neurons * frac_e)  # Number of excitatory neurons
    n_inhibitory = n_neurons - n_excitatory
    if n_excitatory:
        Pe = P[:n_excitatory]  # Neurons which will be labeled excitatory
        Ce = Synapses(Pe, P, on_pre='ge+=w_e')  # Set up rules for connections in which the presynaptic cells are excitatory
        Ce.connect(p=p_e) # Make the excitatory connections
    if n_inhibitory:
        Pi = P[n_excitatory:]  # Neurons which will be labeled inhibitory
        Ci = Synapses(Pi, P, on_pre='gi+=w_i')  # Set up rules for connections in which the presynaptic cells are inhibitory
        Ci.connect(p=p_i) # Make the inhibitory connections
    
    # Initial conditions for some of the state variables, including random numbers
    P.v = 'El + (randn() * 5 - 5)*mV'
    P.ge = '(randn() * 1.5 + 4) * 10.*nS'
    P.gi = '(randn() * 12 + 20) * 10.*nS'
    

    # Run this simulation and record both the membrane potential and the action potential (spike) times of the neurons 
    traces = StateMonitor(P, 'v', record=range(n_neurons))
    spikes = SpikeMonitor(P)
    run(T, report='text')
    phenotypes = [1 if i<n_excitatory else -1 for i in range(n_neurons)]
    return traces, spikes, phenotypes


def plot_traces(traces, phenotypes, cell_indices=[0,1]):
    # Plot the output for two randomly chosen neurons
    # The results are random in each simulation (see above), so if you don't like the ones here, 
    # pick different neuron indices (instead of 11 and 31), or run the simulation again.  
    plt.figure(figsize=(10,5))
    for i in cell_indices:
        if i < len(traces):
            phenotype = "Excitatory" if phenotypes[i] > 0 else "Inhibitory"
            plt.plot(traces.t/ms, traces[i].v/mV, label="Cell %d (%s)" % (i, phenotype))
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)');
    plt.legend(loc=1)