import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def generate_signal(T, dt, rms, limit, seed):

	# T: the length of the signal in seconds
	# dt: the time step in seconds
	# rms: the root mean square power level of the signal. That is, the resulting signal should have sqrt(1/T*intg(x(t)^2dt))=rms
	# limit: the maximum frequency for the signal (in Hz)
	# seed: the random number seed to use (so we can regenerate the same signal again)

	# N = int(T/dt)
	# xt = np.zeros(N)
	np.random.seed(seed)

	# for i in np.nditer(xt, op_flags=['readwrite']):
	# 	i[...] = np.random.normal(limit/2,rms)

	# xw = np.conj(np.fft.fft(xt)/N)
	# xw = np.fft.fftshift(xw)
	# xr = np.real(xw)
	# xc = np.imag(xw)

	n = int(T/dt)
	# print(n)

	t = np.linspace(0,T,n)

	xt = np.zeros(n)
	xr = np.zeros(n)
	xw = np.zeros(n, dtype=np.complex)

	xw = np.fft.fftshift(xw)

	# generate random 
	for i in range( 1, limit ):
		xw[i] = (np.random.normal(0,1)+ np.random.normal(0,1)*1j)
		xw[-i] = (np.random.normal(0,1)+ np.random.normal(0,1)*1j)

	nu = np.arange(int(T/dt))/T - (T/dt)/2

	xw = np.conj(xw)
	# xw = np.fft.fftshift(xw)

	# plt.figure()
	# plt.plot(nu,np.abs(np.fft.fftshift(xw)))
	# plt.xlim(-25,25)
	# plt.show()

	xt = np.fft.ifft(xw) # get time-domain of random signal
	k = rms/(np.std(xt)) # get scaling factor
	xt = xt*k # scale signal

	# plt.figure()
	# plt.plot(t,xt)
	# # plt.xlim(-50,50)
	# plt.show()

	return np.real(xt), np.real(xw)


def generate_signal_2(T, dt, rms, b, seed):

	# T: the length of the signal in seconds
	# dt: the time step in seconds
	# rms: the root mean square power level of the signal. That is, the resulting signal should have sqrt(1/T*intg(x(t)^2dt))=rms
	# limit: the maximum frequency for the signal (in Hz)
	# seed: the random number seed to use (so we can regenerate the same signal again)

	# N = int(T/dt)
	# xt = np.zeros(N)
	np.random.seed(seed)

	# for i in np.nditer(xt, op_flags=['readwrite']):
	# 	i[...] = np.random.normal(limit/2,rms)

	# xw = np.conj(np.fft.fft(xt)/N)
	# xw = np.fft.fftshift(xw)
	# xr = np.real(xw)
	# xc = np.imag(xw)

	n = int(T/dt)
	# print(n)

	t = np.linspace(0,T,n)

	xt = np.zeros(n)
	xr = np.zeros(n)
	xw = np.zeros(n, dtype=np.complex)

	xw = np.fft.fftshift(xw)

	# generate random 
	for i in range( 1, n/2 ):
		a = np.exp(-(i*i)/(2*b*b))
		if(a>0):
			xw[i] = (np.random.normal(0,a)+ np.random.normal(0,a)*1j)
			xw[-i] = (np.random.normal(0,a)+ np.random.normal(0,a)*1j)

	nu = np.arange(int(T/dt))/T - (T/dt)/2

	xw = np.conj(xw)

	xt = np.fft.ifft(xw) # get time-domain of random signal
	k = rms/(np.std(xt)) # get scaling factor
	xt = xt*k # scale signal

	return np.real(xt), np.real(xw)