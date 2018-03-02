# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:15:26 2018

@author: Imaris
"""

from brian2 import *
import numpy
import random 
import itertools

defaultclock.dt = 0.01*ms

Cm = 1*uF # /cm**2
Iapp = 0*uA
gL = 0.1*msiemens
EL = -65*mV
ENa = 55*mV
EK = -90*mV
gNa = 35*msiemens
gK = 52*msiemens
tausyn = 10*ms
taugsyn = 50*ms

eqs = '''
dv/dt = (-gNa*m**3*h*(v-ENa)-gK*n**4*(v-EK)-gL*(v-EL)+Iapp - Isyn - gIsyn)/Cm : volt
Isyn = gsyn*(v - ENa) : amp
dgsyn/dt = -gsyn/tausyn: siemens
gIsyn = esyn*(v - EK) : amp
desyn/dt = -esyn/taugsyn : siemens
m = alpha_m/(alpha_m+beta_m) : 1
alpha_m = -0.1/mV*(v+35*mV)/(exp(-0.1/mV*(v+35*mV))-1)/ms : Hz
beta_m = 4*exp(-(v+60*mV)/(18*mV))/ms : Hz
dh/dt = 5*(alpha_h*(1-h)-beta_h*h) : 1
alpha_h = 0.07*exp(-(v+58*mV)/(20*mV))/ms : Hz
beta_h = 1./(exp(-0.1/mV*(v+28*mV))+1)/ms : Hz
dn/dt = 5*(alpha_n*(1-n)-beta_n*n) : 1
alpha_n = -0.01/mV*(v+34*mV)/(exp(-0.1/mV*(v+34*mV))-1)/ms : Hz
beta_n = 0.125*exp(-(v+44*mV)/(80*mV))/ms : Hz
'''
#choose a multiple of 3 please.
rt = 300
gap = 10 #2 gaps
skip = 100
total_rt = rt + (gap*2) + skip
#Experiment: all in one run: A1 - 1000, choose 300, every 5/50? ms diff group of 50 fire.
#                            A2 -       same 300 "                                      "
#                            B1 -       diff 300 "                                      "
#neuron groups-----------------------------------------------------------------
n1 = NeuronGroup(300, eqs, threshold = 'v > -40*mV', refractory = 'v >= -40*mV', method='exponential_euler')
n1.v = -80*mV
n1.h = 1

n2 = NeuronGroup(300, eqs, threshold = 'v > -40*mV', refractory = 'v >= -40*mV', method='exponential_euler')
n2.v = -80*mV
n2.h = 1

n3 = NeuronGroup(300, eqs, threshold = 'v > -40*mV', refractory = 'v >= -40*mV', method='exponential_euler')
n3.v = -80*mV
n3.h = 1

#inhibitory group--------------------------------------------------------------
I = NeuronGroup(100, eqs, method='exponential_euler')
wa = (600)#inhibitory
we = (650)#excitatory
#firing------------------------------------------------------------------------
#next time: change this to spike generator group. 1000, choose 300, each time, randomly choose 50 of 300 to fire every 5 ms.
#PG = PoissonGroup(1000, 1*Hz)

#==============================================================================
time_break1 = int((rt)/3)
time_break2 = time_break1 * 2
#------------------------------A1----------------------------------------------
ind = list(random.sample(range(0, 1000), 300))

t = []
temp = []
for idx in range(0 + skip, time_break1 + skip, 5):
    temp = [idx] * 50
    for idx2 in temp:
        t.append(idx2)

ind2 = []
for idx in range(0 + skip, time_break1 + skip, 5):
    temp = random.sample(ind, 50)
    for idx2 in temp:
        ind2.append(idx2)
        
#print('ind1: %s' %len(ind2))
#------------------------------A2----------------------------------------------
        
for idx in range(time_break1 + gap + skip, time_break2 + gap + skip, 5):
    temp = [idx + gap] * 50
    for idx2 in temp:
        t.append(idx2)

for idx in range(time_break1 + gap + skip, time_break2 + gap + skip, 5):
    temp = random.sample(ind, 50)
    for idx2 in temp:
        ind2.append(idx2)
        
#print('ind2: %s' %len(ind2))
#------------------------------B1----------------------------------------------
ind_B1 = list(random.sample(range(0, 1000), 300))

for idx in range(time_break2 + (gap*2) + skip, total_rt + skip, 5):
    temp = [idx + (gap*2)] * 50
    for idx2 in temp:
        t.append(idx2)

for idx in range(time_break2 + (gap*2) + skip, total_rt + skip, 5):
    temp = random.sample(ind_B1, 50)
    for idx2 in temp:
        ind2.append(idx2)


#print('ind3: %s' %len(ind2))
#print('time3: %s' %len(t))
#print('ind: %s' %ind2)
#print('time: %s' %t)


indicies = numpy.array(ind2)
times = numpy.array(t) * ms
PG = SpikeGeneratorGroup(1000, indicies, times) 
#==============================================================================

#firing + neuron group---------------------------------------------------------
S1 = Synapses(PG, n1, on_pre='gsyn += we*nsiemens')
S1.connect(p = .2)

S2 = Synapses(PG, n2, on_pre='gsyn += we*nsiemens')
S2.connect(p = .2)

S3 = Synapses(PG, n3, on_pre='gsyn += we*nsiemens')
S3.connect(p = .2)

#neuron group + inhibitory-----------------------------------------------------
S4 = Synapses(n1, I, on_pre='gsyn += we*nsiemens')
S4.connect(p = .2)

S5 = Synapses(n2, I, on_pre='gsyn += we*nsiemens')
S5.connect(p = .1)

#S6 = Synapses(n3, I, on_pre='gsyn += wa*nsiemens')
#S6.connect(p = .1)

#inhibitoty + neuron group-----------------------------------------------------
S4 = Synapses(n1, I, on_pre='esyn += wa*nsiemens')
S4.connect(p = .2)

S4 = Synapses(n1, I, on_pre='esyn += wa*nsiemens')
S4.connect(p = .1)

#S4 = Synapses(n1, I, on_pre='esyn += wa*nsiemens')
#S4.connect(p = .1)

#spikemonitor------------------------------------------------------------------
M1 = SpikeMonitor(n1)
M2 = SpikeMonitor(n2)
M3 = SpikeMonitor(n3)

#stateMonitor------------------------------------------------------------------
M4 = StateMonitor(n1, True, record = True)

#------------------------------------------------------------------------------
run(total_rt*ms)

#==============================================================================
#also next time ignore first few ms because many fire for some weird reason....
plt.figure(0)
plt.plot(M1.t/ms, M1.i, '.k')
plt.show
xlabel('Time (ms)')
ylabel('Neuron index')

plt.figure(1)
plt.plot(M2.t/ms, M2.i, '.k')
plt.show
xlabel('Time (ms)')
ylabel('Neuron index')

plt.figure(2)
plt.plot(M3.t/ms, M3.i, '.k')
plt.show
xlabel('Time (ms)')
ylabel('Neuron index')
#==============================================================================
#prints out the total neurons fired in a group, average firing rate of each neuron in the group, and standard deviation.
def neuro_spike_info(M, n):
    n_idx = {}
   
    for i in range(0, 300):
        n_idx[i] = 0
        
    for j in M.i:
        n_idx[j] += 1
        
    list1 = []
    
    for k in n_idx:
        list1.append(n_idx[k])
        
    average = numpy.mean(list1)
    std = numpy.std(list1)
    
    total = 0
    for l in n_idx:
        if n_idx[l] > 0:
            total += n_idx[l]
    print('total neurons fired in group %s: %d' %(n, total))
    print('average firing rate: %d' %average)
    print('std: %d' %std)

neuro_spike_info(M1, 1)
neuro_spike_info(M2, 2)
neuro_spike_info(M3, 3)
#==============================================================================
def sort_spikes(M, n):

    list_A1 = []
    list_A2 = []
    list_B1 = []

    for i in range(0, 300):
        list_A1.append(0.0)
        list_A2.append(0.0)
        list_B1.append(0.0)
    

    for i, t in itertools.zip_longest(M.i, M.t):
        if t < time_break1*ms:
            list_A1[i] += 1.0
        elif t >= time_break2*ms:
            list_B1[i] += 1.0
        else:
            list_A2[i] += 1.0
        
    total_A1 = sum(list_A1)
    total_A2 = sum(list_A2)
    total_B1 = sum(list_B1)

    percent_A1 = []
    percent_A2 = []
    percent_B1 = []

    for a1, a2, b1 in zip(list_A1, list_A2, list_B1):
        percent_A1.append(a1/total_A1)
        percent_A2.append(a2/total_A2)
        percent_B1.append(b1/total_B1)

    occur_A1 = []
    occur_A2 = []
    occur_B1 = []

    for l_a1, l_a2, l_b1, p_a1, p_a2, p_b1 in zip(list_A1, list_A2, list_B1, percent_A1, percent_A2, percent_B1):
        occur_A1.append(l_a1 * p_a1)
        occur_A2.append(l_a2 * p_a2)
        occur_B1.append(l_b1 * p_b1)
    
#==============================================================================
#     sum_A1 = 0.0
#     sum_A2 = 0.0
#     sum_B1 = 0.0
#     
#     for o_a1, o_a2, o_b1 in itertools.zip_longest(occur_A1, occur_A2, occur_B1):
#         sum_A1 += o_a1
#         sum_A2 += o_a2
#         sum_B1 += o_b1
#==============================================================================
        
        
    print('--------For Neuron Group: %d--------' %n)
    print('A1 total: %d' %sum(occur_A1))
    print('A2 total: %d' %sum(occur_A2))
    print('B1 total: %d' %sum(occur_B1))
   

sort_spikes(M1, 1)
sort_spikes(M2, 2)
sort_spikes(M3, 3)



show()