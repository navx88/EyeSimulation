from numpy import fft
import numpy as np
import matplotlib.pyplot as plt
import neuron
import gaussnoise as gn
import cv2

import nengo
from nengo.dists import Uniform

N = 2
D = 1
S = 1000
x = np.linspace(-2,2,S)
x_int = np.array([0.3, 0.3])
e = np.zeros((N,1))
max_rates = np.array([115, 115])
x = np.linspace(-2,2,S)
# x_int = np.zeros(N)
# e = np.zeros((N,1))
# max_rates = np.zeros(N)
t_ref = 0.002
t_rc = 0.02
radius = 2
dt = 0.001

# for i in range (0, max_rates.shape[0]):
# 	max_rates[i] = np.random.randint(20,50)

e[0][0] = 1
e[1][0] = -1

# print(A)
# x_int[0] = 0.302317
# x_int[1] = -0.
x_int[0] = 1
x_int[1] = -1

# max_rates[0] = 116
# max_rates[1] = 116
max_rates[0] = 150
max_rates[1] = 150
# for i in range(0, x_int.shape[0]):
# 	x_int[i] = 0.2*i-1.9

T = 1
max_freq = 5
rms = 1
dt = 0.001

xt = neuron.generate_signal(T, max_freq, rms, dt)
# x_5Hz, w_5Hz = gn.generate_signal(T, dt, 1, 5, 1)
# xt_ = np.fft.fftshift(np.fft.fft(xt))
print("N: " + str(N))
print("D: " + str(D))
print("encoders: " + str(e.shape))
print("max_rates: " + str(max_rates.shape))
print("intercepts: " + str(x_int.shape))
print("tau ref: " + str(t_ref))
print("tau rc: " + str(t_rc))
print("radius: " + str(radius))
print("xt.shape: " + str(xt.shape))
print("dt: " + str(dt))
print("\n")

T = 1.0
max_freq = 5

model = nengo.Network()

with model:
    stim = nengo.Node(output=nengo.processes.WhiteSignal(period=T, high=max_freq, rms=1))
    ensA = nengo.Ensemble(8, dimensions=1)
    ensB = nengo.Ensemble(256, dimensions=1)

    nengo.Connection(stim, ensA)
    nengo.Connection(stim, ensB)

    stim_p = nengo.Probe(stim)
    ensA_p = nengo.Probe(ensA, synapse=None)
    ensB_p = nengo.Probe(ensB, synapse=None)

    ensA_spikes_p = nengo.Probe(ensA.neurons, 'spikes')
    ensB_spikes_p = nengo.Probe(ensB.neurons, 'spikes')

sim = nengo.Simulator(model, seed=4)
sim.run(T)

t = sim.trange()
plt.figure(figsize=(8, 6))
plt.subplot(2,1,1)
# plt.ax = gca()
plt.plot(t, sim.data[stim_p],'b')
plt.plot(t, sim.data[ensA_p],'g')
plt.ylabel("Output")
# plt.rasterplot(t, sim.data[ensA_spikes_p], ax=ax.twinx(), colors=['k']*25, use_eventplot=True)
plt.ylabel("Neuron")

plt.subplot(2,1,2)
# plt.ax = gca()
plt.plot(t, sim.data[stim_p],'b')
plt.plot(t, sim.data[ensB_p],'g')
plt.ylabel("Output")
plt.xlabel("Time");
# plt.rasterplot(t, sim.data[ensB_spikes_p], ax=ax.twinx(), colors=['k']*23, use_eventplot=True)
plt.ylabel("Neuron");

plt.show()
