import numpy as np

from dark_news import *

def efficiency(samples, weights, xmin, xmax):

	histTOT = np.histogram(samples, 
												weights=weights, 
												bins=100, 
												density=False, 
												range=(np.min(samples),np.max(samples)))

	histCUT = np.histogram(samples, 
												weights=weights, 
												bins=100, 
												density=False, 
												range=(xmin,xmax))

	return np.sum(histCUT[0])/np.sum(histTOT[0])


def MB_efficiency(samples, weights):

	costhetaZ = np.array([ fourvec.dot3(samples[i],fourvec.kz)/np.sqrt( fourvec.dot3(samples[i],samples[i]) ) for i in range(np.shape(samples)[0])])
	EZ = np.array([ fourvec.dot4(samples[i],fourvec.k0) for i in range(np.shape(samples)[0])])
	# EZ = EZ - const.Me
	PZ =  np.array([ np.sqrt( fourvec.dot3(samples[i],samples[i])) for i in range(np.shape(samples)[0])])
	
	mee =  np.array([ np.sqrt( fourvec.dot4(samples[i],samples[i])) for i in range(np.shape(samples)[0])])
	mee_cut = 0.03203 + 0.007417*EZ + 0.02738*EZ**2

	# EnuQE = (const.mproton*EZ - const.Me**2/2.0)/(const.mproton - EZ + PZ*costhetaZ)
	EnuQE = (const.mproton*EZ)/(const.mproton - EZ*(1.0 - costhetaZ) )

	mask =  (1.5 > EnuQE) &\
				 (EnuQE > 0.2) \
				# & (mee < mee_cut)
				
					 
	return np.sum(weights*mask)/np.sum(weights)
	

def MB_smear(samples, m):
	
	if np.size(m)!=1:
		print("ERROR! Passing the wrong cariable to MB_smear.")

	print("Smearing...")
	size_samples = np.shape(samples)[0]

	E = np.array([ fourvec.dot4(samples[i],fourvec.k0) for i in range(size_samples)])
	px = np.array([ samples[i][1] for i in range(size_samples)])
	py = np.array([ samples[i][2] for i in range(size_samples)])
	pz = np.array([ samples[i][3] for i in range(size_samples)])
	P = np.array([ np.sqrt(fourvec.dot3(samples[i],samples[i])) for i in range(size_samples)])

	sigma_E = const.MB_STOCHASTIC*np.sqrt(E) + const.MB_NOISE
	sigma_angle = const.MB_ANGULAR

	T = E - m*np.ones((size_samples,))
	theta = np.arccos(pz/P)
	phi = np.arctan2(py,px)

	T = np.array([ np.random.normal(T[i], sigma_E[i]) for i in range(size_samples)])
	theta = np.array([ np.random.normal(theta[i], sigma_angle) for i in range(size_samples)])
	phi = np.array([ np.random.normal(phi[i], sigma_angle) for i in range(size_samples)])
	E = T + m*np.ones((size_samples,))
	P = np.sqrt(E**2 - m**2)

	smeared = np.array([ [E[i], 
						P[i]*np.sin(theta[i])*np.cos(phi[i]),
						P[i]*np.sin(theta[i])*np.sin(phi[i]),
						P[i]*np.cos(theta[i])] for i in range(size_samples)])

	return smeared

