import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import nengo
import numpy as np

def compute_lif_decoder(n_neurons, dimensions, encoders, max_rates, intercepts, tau_ref, tau_rc, radius, x_pts, function):
    """
    Parameters:
        n_neurons: number of neurons (integer)
        dimensions: number of dimensions (integer)
        encoders: the encoders for the neurons (array of shape (n_neurons, dimensions))
        max_rates: the maximum firing rate for each neuron (array of shape (n_neurons))
        intercepts: the x-intercept for each neuron (array of shape (n_neurons))
        tau_ref: refractory period for the neurons (float)
        tau_rc: membrane time constant for the neurons (float)
        radius: the range of values the neurons are optimized over (float)
        x_pts: the x-values to use to solve for the decoders (array of shape (S, dimensions))
        function: the function to approximate
    Returns:
        A (the tuning curve matrix)
        dec (the decoders)
    """
    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(n_neurons=n_neurons,
                             dimensions=dimensions,
                             encoders=encoders,
                             max_rates=max_rates,
                             intercepts=[x/radius for x in intercepts],
                             neuron_type=nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref),
                             radius=radius)
    sim = nengo.Simulator(model)
    
    x_pts = np.array(x_pts)
    if len(x_pts.shape) == 1:
        x_pts.shape = x_pts.shape[0], 1
    _, A = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=x_pts)

    target = [function(xx) for xx in x_pts]
    solver = nengo.solvers.LstsqL2()
    dec, info = solver(A, target)
    return A, dec

def generate_signal(T, max_freq, rms, dt):
    """
    Parameters:
        T: the period of time to generate a random signal for
        max_freq: the highest frequency in the signal
        rms: the RMS power of the signal
        dt: the time step (usually 0.001)
    Returns:
        signal (an array of length T/dt containing the random signal)
    """
    signal = nengo.processes.WhiteSignal(period=T, high=max_freq, rms=rms)
    return signal.run(T, dt=dt)

def generate_spikes(n_neurons, dimensions, encoders, max_rates, intercepts, tau_ref, tau_rc, radius, x, dt):
    """
    Parameters:
        n_neurons: number of neurons (integer)
        dimensions: number of dimensions (integer)
        encoders: the encoders for the neurons (array of shape (n_neurons, dimensions))
        max_rates: the maximum firing rate for each neuron (array of shape (n_neurons))
        intercepts: the x-intercept for each neuron (array of shape (n_neurons))
        tau_ref: refractory period for the neurons (float)
        tau_rc: membrane time constant for the neurons (float)
        radius: the range of values the neurons are optimized over (float)
        x: the input signal to feed into the neurons (array of shape (T, dimensions))
        dt: the time step of the simulation (usually 0.001)
    Returns:
        spikes (a (timesteps, n_neurons) array of the spiking outputs)
    """
    # print("1 " + str(len(x)))
    model = nengo.Network()
    # print("2 " + str(len(x)))
    with model:
    	print("hi")
        stim = nengo.Node(lambda t: x[int(t/dt)-1])
        # print("3 " + str(len(x)))
        ens = nengo.Ensemble(n_neurons=n_neurons,
                             dimensions=dimensions,
                             encoders=encoders,
                             max_rates=max_rates,
                             intercepts=[x/radius for x in intercepts],
                             neuron_type=nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref),
                             radius=radius)
        # print("4 " + str(len(x)))
        nengo.Connection(stim, ens, synapse=None)
        # print("5 " + str(len(x)))
        p = nengo.Probe(ens.neurons)

    # print(stim)
    # plt.figure()
    # plt.plot(stim)

    # print("6 " + str(len(x)))
    sim = nengo.Simulator(model, dt=dt)
    # print("7 " + str(len(x)))
    # x = stim
    T = len(x)*dt #len(x)*dt
    # print("8 " + str(len(x)))
    sim.run(T, progress_bar=True)
    # print("9 " + str(len(x)))
    return sim.data[p]


def rectified(x, e, x_int, y_max):

	# e = np.random.randint(0,2)

	# if e==0:
	# 	e = -1

	y_int = np.array([e, y_max[1]])

	a = np.array([[x_int[0],y_int[0]], [1,1]])
	b = np.array([x_int[1], y_int[1]])
	sol = np.linalg.solve(a,b)

	alph = sol[0]
	Jb = sol[1]

	ai = np.maximum(((alph*x).dot(e) + Jb),0)

	#print(ai)

	return ai


def LIF(x, e, x_int, y_max, t_ref, t_rc):

	# this is the 1D implementation of LIF

	y_int = np.array([ e, 1/( 1 - np.exp((y_max[1]*t_ref - 1)/(y_max[1]*t_rc)) ) ])

	alph = y_int[1]/(e*e-x_int[0]*e)
	Jb = -x_int[0]*alph*e

	a = np.array([[x_int[0]*e,y_int[0]*e], [1,1]])
	b = np.array([x_int[1], y_int[1]])
	sol = np.linalg.solve(a,b)

	alph = sol[0]
	Jb = sol[1]

	J = (alph*x).dot(e) + Jb
	ai = np.array(x)
	c = 0

	for j in J:
		if j>1:
			ai[c] = 1 / ( t_ref - t_rc*np.log( 1 - 1/j ) )
		else:
			ai[c] = 0

		c = c+1


	#ai = 1 / ( t_ref - t_rc*np.log( 1 - 1/((alph*x).dot(e) + Jb) ) )

	#print(ai)

	return ai


def LIFxd(x, e, x_int, y_max, t_ref, t_rc):

	# x is multi-D (D)
	# e is multi-D (D-1)
	# x_int is multi-D (D-1)
	# y_max is 1D
	# t_ref is 1D
	# t_rc is 1D

	# print(np.array([(y_max*t_ref - 1),(y_max*t_rc)]))

	y_int = np.append(e, 1/( 1 - np.exp((y_max*t_ref - 1)/(y_max*t_rc)) ) )

	# a = np.array([[np.dot(x_int,e),np.dot(e,e)], [1,1]])
	a = np.array([[np.dot(x_int,e),np.dot(e,e)], [1,1]])
	b = np.array([0, y_int[y_int.shape[0]-1]])
	sol = np.linalg.solve(a,b)

	alph = sol[0]
	Jb = sol[1]

	ai = np.array(alph*x + Jb) 


	for j in np.nditer(ai, op_flags=['readwrite']):
		# print(j)
		if j>1:
			j[...] = 1 / ( t_ref - t_rc*np.log( 1 - 1/j ) )
		else:
			j[...] = 0

	return ai


def LIFspike(x, T, dt, e, a_min, a_max, t_ref, t_rc, rest, thresh):

	# convert a mins and maxes to y mins and maxes (from Hz to V)
	y_min = np.append(a_min[0] , 1/( 1 - np.exp((a_min[1]*t_ref - 1)/(a_min[1]*t_rc)) ) )
	y_max = np.append(a_max[0] , 1/( 1 - np.exp((a_max[1]*t_ref - 1)/(a_max[1]*t_rc)) ) )
	# initiate arrays to solve for alph and Jb
	a = np.array([[np.dot(y_min[0],e),np.dot(y_max[0],e)], [1,1]])
	b = np.array([y_min[y_min.shape[0]-1], y_max[y_max.shape[0]-1]])
	sol = np.linalg.solve(a,b)

	alph = sol[0]
	Jb = sol[1]

	# the good old J equation
	J = alph*np.dot(e,x) + Jb

	V = np.zeros(T.shape[0])
	tc = 0

	# at t=0
	V[0] = 0
	saturated = 0

	for t in range(1,V.shape[0]):

		f_prev = 1/t_rc*( J[t-1] - V[t-1] ) 
		V[t] = max(V[t-1] + dt*f_prev,rest)

		if saturated:
			V[t] = 0
			tc = tc + dt
			if tc > t_ref:
				# ti = 0
				tc = 0
				saturated = 0

		if V[t] > thresh:
			V[t] = thresh
			saturated = 1

	# for t in range(1,V.shape[0]):
	# 	if V[t] < thresh:
	# 		V[t] = 0

	return V


def LIFspike_network(N, xt, radius, t, dt, minVrange, maxVrange, t_ref, t_rc, rest, thresh):

	spikes = np.zeros((N,int(t.shape[0])))

	rt = np.zeros(int(t.shape[0]))

	for i in range(0,N):
		e = np.random.choice(np.array([0,1]))
		# print(e)
		if e == 0:
			e = -1

		e = e*radius
		maxV = np.random.randint(maxVrange[0],maxVrange[1])
		minV = np.random.randint(minVrange[0],minVrange[1])
		# print(e)
		a_min = np.array([0, minV])
		a_max = np.array([e, maxV])
		# print(a_min)
		# print(a_max[0])
		spikes[i][:] = LIFspike(xt, t, dt, e, a_min, a_max, t_ref, t_rc, rest, thresh)
		# print(spikes[i])
		# if e==1*radius:
		# 	rt = rt - spikes[i]
		# elif e==-1*radius:
		# 	rt = rt + spikes[i]


	T0 = 0.005
	# ht = np.exp(-t/T0)
	# ht = ht/np.linalg.norm(ht,1)
	# hw = np.fft.fftshift(np.fft.fft(ht))
	t_h = np.arange(1000)*dt-0.5
	ht = np.exp(-t_h/T0)/T0
	ht[np.where(t_h<0)]=0
	ht = ht/np.linalg.norm(ht,1)
	hw = np.fft.fftshift(np.fft.fft(ht))

	rt = spikes[0] - spikes[1]
	# print(rt)
	for j in range(2,N):
		rt = rt - spikes[j]

	rw = np.zeros(rt.shape,dtype=np.complex) #initialize rw
	rw = np.fft.fftshift(np.fft.fft(rt))

	xhw = hw*rw
	xht = np.fft.ifft(np.fft.ifftshift(xhw)).real

	freq = np.arange(int(1/dt)) - (1/dt)/2 #prep freq

	plt.figure()
	plt.plot(freq,np.abs(np.transpose(rw)))
	plt.xlim(-500,500)
	plt.xlabel('freq (Hz)')
	plt.ylabel('$|R(\omega)|$')

	plt.figure()
	plt.plot(freq,np.abs(hw))
	plt.xlim(-500,500)
	plt.xlabel('freq (Hz)')
	plt.ylabel('$|H(\omega)|$')

	plt.figure()
	plt.plot(freq,np.abs(xhw))
	plt.xlim(-500,500)
	plt.ylim(0,200)
	plt.xlabel('freq (Hz)')
	plt.ylabel('$|XH(\omega)|$')

	return spikes, xht

def two_neurons(x, dt, alpha, Jbias, tau_rc, tau_ref):

	J1 = alpha*np.dot(1,x) + Jbias
	J2 = alpha*np.dot(-1,x) + Jbias

	V1 = np.zeros(x.shape[0])
	V2 = np.zeros(x.shape[0])

	# at t=0
	V1[0] = 0
	V2[0] = 0
	saturated1 = 0
	saturated2 = 0
	tc1 = 0
	tc2 = 0

	for t in range(1,x.shape[0]):

		f_prev1 = 1/tau_rc*( J1[t-1] - V1[t-1] )
		f_prev2 = 1/tau_rc*( J2[t-1] - V2[t-1] )
		V1[t] = max(V1[t-1] + dt*f_prev1,0)
		V2[t] = max(V2[t-1] + dt*f_prev2,0)

		if saturated1:
			V1[t] = 0
			tc1 = tc1 + dt
			if tc1 > tau_ref:
				tc1 = 0
				saturated1 = 0

		if saturated2:
			V2[t] = 0
			tc2 = tc2 + dt
			if tc2 > tau_ref:
				tc2 = 0
				saturated2 = 0

		if V1[t] > 1:
			V1[t] = 1
			saturated1 = 1

		if V2[t] > 1:
			V2[t] = 1
			saturated2 = 1

	return np.array([V1, V2])

def network_rectified(x, S, N, fire_min, fire_max):

	pt1 = np.array([0, -0.95])
	pt2 = np.array([N-1, 0.95])

	m = (pt2[1] - pt1[1])/(pt2[0] - pt1[0])
	b = pt1[1]

	ai = (N,S)
	ai = np.zeros(ai)

	for i in range(0, N):
		x_int = np.array([m*i + b, 0]) # x-intercepts evenly spaced btwn -0.95 and 0.95
		y_int = np.array([0, np.random.randint(fire_min,fire_max)]) # randomly generate y-intercepts btwn 100 and 200 Hz
		ai[i] = rectified(x, e_1D(), x_int, y_int) # from neuron.py, generates random neuron using intercepts and x

	A = np.transpose(ai)

	return A


def network_LIF(x, S, N, fire_min, fire_max, t_ref, t_rc):

	pt1 = np.array([0, -0.95])
	pt2 = np.array([N-1, 0.95])

	m = (pt2[1] - pt1[1])/(pt2[0] - pt1[0])
	b = pt1[1]

	ai = (N,S)
	ai = np.zeros(ai)

	for i in range(0, N):
		x_int = np.array([m*i + b, 0]) # x-intercepts evenly spaced btwn -0.95 and 0.95
		y_int = np.array([0, np.random.randint(fire_min,fire_max)]) # randomly generate y-intercepts btwn 100 and 200 Hz
	#	print("X-INT: " + str(x_int) + " | Y-INT: " + str(y_int))
		e = e_1D()
		ai[i] = LIFxd(x*e, e, x_int[0], y_int[1], t_ref, t_rc) # from neuron.py, generates random neuron using intercepts and x

	A = np.transpose(ai)

	return A

def get_ro(A,S):
	return np.dot(np.transpose(A), A) / S


def get_Y(x,A,S):
	return np.dot(np.transpose(A), x) / S


def get_ro_noisy(A_noise, SD, S, N):
	return np.dot(np.transpose(A_noise), A_noise) / S + SD*np.identity(N)


def e_1D():

	e = np.random.randint(0,2)
	
	if e==0:
		e = -1
	return e

def func(x):
	return x

def calc_RMSE(x1,x2,st):
	RMSE = np.sqrt(np.average((x1-x2)**2))
	print(st + '%g' % RMSE)
	return RMSE

def calc_actual_op(ip,t):
	ip_act = np.zeros(ip.shape[0])
	ip_act[0] = ip[0]
	hold = False

	for i in range(1,ip.shape[0]):
		if abs(ip[i]-ip[i-1]) >= 0.9:
			hold = True

		if ip[i] > 0:
			ip_act[i] = ip[i]*t[i]
		elif ip[i] < 0:
			ip_act[i] = ip[i]*t[i]
		
		if ip[i] == 0 or hold:
			ip_act[i] = ip_act[i-1]

	return ip_act