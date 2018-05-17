import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pylab

import random
import neuron
import gaussnoise as gn

import nengo

##### important set up and constants #####

tau_rc = 0.02
tau_ref = 0.002

latMax = 0.7854 
verMax = 0.6545 
velMax = 17.45 
sps = 300

N = 70

tau = 0.05
synC = 0.02 
sct = 0.15

##### model #####

eye_ctrl = nengo.Network()

with eye_ctrl:
	ip1 = nengo.Node(nengo.processes.Piecewise({
		0:latMax/4,
		1:latMax/4*-1,
		2:latMax/4,
		3:latMax/4*-1,
		4:0
		}))
	ip2 = nengo.Node(nengo.processes.Piecewise({
		0:verMax/4,
		2:verMax/4*-1,
		4:0
		}))

	currPosX = nengo.Ensemble(n_neurons = N,
						dimensions = 1,
						max_rates = nengo.dists.Uniform(sps/2,sps),
						neuron_type = nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref),
						radius=latMax)
	currPosY = nengo.Ensemble(n_neurons = N,
						dimensions = 1,
						max_rates = nengo.dists.Uniform(sps/2,sps),
						neuron_type = nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref),
						radius=verMax)
	targetPosX = nengo.Ensemble(n_neurons = N,
						dimensions = 1,
						max_rates = nengo.dists.Uniform(sps/2,sps),
						neuron_type = nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref),
						radius=latMax)
	targetPosY = nengo.Ensemble(n_neurons = N,
						dimensions = 1,
						max_rates = nengo.dists.Uniform(sps/2,sps),
						neuron_type = nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref),
						radius=verMax)
	targetVelX = nengo.Ensemble(n_neurons = N,
						dimensions = 1,
						max_rates = nengo.dists.Uniform(sps/2,sps),
						neuron_type = nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref),
						radius=velMax)
	targetVelY = nengo.Ensemble(n_neurons = N,
						dimensions = 1,
						max_rates = nengo.dists.Uniform(sps/2,sps),
						neuron_type = nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref),
						radius=velMax)

	def limitX(x):
		if(x>latMax):
			return latMax
		elif(x<-latMax):
			return -latMax
		else:
			return x

	def limitY(y):
		if(y>verMax):
			return verMax
		elif(y<-verMax):
			return -verMax
		else:
			return y

	nengo.Connection(ip1,targetPosX,function=limitX)
	nengo.Connection(ip2,targetPosY,function=limitY)
	nengo.Connection(targetPosX,targetVelX,transform=1/sct,synapse=synC)
	nengo.Connection(targetPosY,targetVelY,transform=1/sct,synapse=synC)
	nengo.Connection(currPosX,targetVelX,transform=-1/sct,synapse=synC)
	nengo.Connection(currPosY,targetVelY,transform=-1/sct,synapse=synC)
	nengo.Connection(targetVelX,currPosX,transform=tau,synapse=tau/10)
	nengo.Connection(currPosX,currPosX,synapse=tau)
	nengo.Connection(targetVelY,currPosY,transform=tau,synapse=tau/10)
	nengo.Connection(currPosY,currPosY,synapse=tau)

	synP = synC #0.01
	prb_ip1 = nengo.Probe(ip1,synapse=0.01)
	prb_ip2 = nengo.Probe(ip2,synapse=0.01)
	prb_posX = nengo.Probe(currPosX,synapse=synP)
	prb_posY = nengo.Probe(currPosY,synapse=synP)
	prb_velX = nengo.Probe(targetVelX,synapse=synP)
	prb_velY = nengo.Probe(targetVelY,synapse=synP)
	prb_posXn = nengo.Probe(currPosX.neurons)
	prb_posYn = nengo.Probe(currPosY.neurons)
	prb_velXn = nengo.Probe(targetVelX.neurons)
	prb_velYn = nengo.Probe(targetVelY.neurons)

with nengo.Simulator(eye_ctrl) as sim:
	sim.run(5)

t = sim.trange()

plt.figure()
plt.plot(t,sim.data[prb_posX],label='posX')
plt.plot(t,sim.data[prb_posY],label='posX')
plt.plot(t,sim.data[prb_ip1],label='inputX')
plt.plot(t,sim.data[prb_ip2],label='inputY')
plt.xlabel("time(s)")
plt.ylabel("rad")
plt.title("Eye Position")

plt.figure()
plt.plot(t,sim.data[prb_posX],label='posX')
plt.plot(t,sim.data[prb_ip1],label='inputX')
plt.xlabel("time(s)")
plt.ylabel("rad")
plt.title("Horizontal Eye Position")

plt.figure()
plt.plot(t,sim.data[prb_posY],label='posY')
plt.plot(t,sim.data[prb_ip2],label='inputY')
plt.xlabel("time(s)")
plt.ylabel("rad")
plt.title("Vertical Eye Position")

plt.figure()
plt.plot(t,sim.data[prb_velX],label='velX')
plt.plot(t,sim.data[prb_velY],label='velY')
plt.xlabel("time(s)")
plt.ylabel("rad/s")
plt.title("Eye Velocity (hor and vert)")

print("DONE")
plt.show()