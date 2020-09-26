import numpy as np
import vegas as vg
import gvar as gv

import random

from scipy import interpolate
import scipy.stats

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

import fourvec
import hist_plot

from scipy.integrate import quad


#### MINIBOONE PARAMTERS #####
POTS = 12.84*1e20

NINT = 40
NEVAL = 100000


# Enu > Mn**2/2.0/MA + Mn  371.2 MeV for p / 324.26 MeV for 12C
EMIN = 0.1
EMAX = 7

mproton = 1.0

############### CONSTANTS ############################
s2w = 0.231
cw = np.sqrt(1.0 - s2w)
e = np.sqrt(4.0*np.pi/137.0)

gvP = 1.0
Gf = 1.16e-5 # GeV^-2
Mw = 80.35 # GeV
g = np.sqrt(Gf*8/np.sqrt(2)*Mw*Mw)

gA = 1.26
tau3 = 1

MAG_N = -1.913
MAG_P = 2.792

def D(Q2):
	return 1.0/((1+Q2/0.84/0.84)**2)


def H1_p(Q2):
	tau = -Q2/4.0/mproton/mproton
	F1 = (D(Q2) - tau*MAG_P*D(Q2))/(1-tau)
	F2 = (MAG_P*D(Q2) - D(Q2))/(1-tau)
	return F1**2 - tau * F2**2
def H2_p(Q2):
	tau = -Q2/4.0/mproton/mproton
	F1 = (D(Q2) - tau*MAG_P*D(Q2))/(1-tau)
	F2 = (MAG_P*D(Q2) - D(Q2))/(1-tau)
	return (F1+F2)*(F1+F2)



def H1_n(Q2):
	tau = -Q2/4.0/mproton/mproton
	F1 = (- tau*MAG_N*D(Q2))/(1-tau)
	F2 = (MAG_N*D(Q2))/(1-tau)
	return F1**2 - tau * F2**2

def H2_n(Q2):
	tau = -Q2/4.0/mproton/mproton
	F1 = (-tau*MAG_N*D(Q2))/(1-tau)
	F2 = (MAG_N*D(Q2))/(1-tau)
	return (F1+F2)*(F1+F2)



def F1_EM(Q2):
	tau = -Q2/4.0/mproton/mproton
	return (D(Q2) - tau*MAG_P*D(Q2))/(1-tau)

def F2_EM(Q2):
	tau = -Q2/4.0/mproton/mproton
	return (MAG_P*D(Q2) - D(Q2))/(1-tau)



def F1_WEAK(Q2):
	tau = -Q2/4.0/mproton/mproton
	f = (0.5 - s2w)*(tau3)*(1-tau*(1+MAG_P-MAG_N))/(1-tau) - s2w*(1-tau*(1+MAG_P+MAG_N))/(1-tau)  
	return f*D(Q2)

def F2_WEAK(Q2):
	tau = -Q2/4.0/mproton/mproton
	f = (0.5 - s2w)*(tau3)*(MAG_P-MAG_N)/(1-tau) - s2w*(MAG_P+MAG_N)/(1-tau)  
	return f*D(Q2)

def F3_WEAK(Q2):
	f = gA*tau3/2.0/(1+Q2/1.02/1.02)**2
	return f



def Fcoh(Q,MA):
	a = 0.523/0.197 # GeV^-1
	r0 = 1.126*(MA**(1.0/3.0))/0.197 # GeV^-1
	return 3.0*np.pi*a/(r0**2 + np.pi**2 * a**2) * (np.pi*a *(1.0/np.tanh(np.pi*a*Q))*np.sin(Q*r0) - r0*np.cos(Q*r0))/(Q*r0*np.sinh(np.pi*Q*a))

def test(theta):
	return np.sin(theta)

def Power(x,n):
	return x**n


def distribution(fluxfile, datafile, MA, Z, Mzprime, Mn):

	Elo, Ehi, numu, numub, nue, nueb = np.genfromtxt(fluxfile, unpack=True)
 	
 	Eminiboone = (Ehi+Elo)/2.0
	numu *= POTS*0.050 

	mask = (Eminiboone > Mn**2/2.0/MA + Mn)

	MiniBooNE_flux = scipy.interpolate.interp1d(Eminiboone, numu*mask)
	
	# I , bla= quad(MiniBooNE_flux_d, EMIN, EMAX)

	# print "INTEGRAL", I

	# def MiniBooNE_flux(E):
	# 	return MiniBooNE_flux_d(E)

	# plt.plot(Eminiboone, MiniBooNE_flux(Eminiboone))
	# plt.plot(Elo,numu/np.sum(numu)/0.05)
	# plt.show()

	class dsigmadQ2_batch(vg.BatchIntegrand):

		def __init__(self, dim, Emax, Emin):

			self.dim = dim
			self.Emax = Emax
			if (Emin > Mn**2/2.0/MA + Mn):
				self.Emin = Emin
			else:
				self.Emin = 1.01*(Mn**2/2.0/MA + Mn)


		def __call__(self, x):

			Enu = (self.Emax - self.Emin) * x[:, 1] + self.Emin
			# Enu = 1.0

			s = MA**2 + 2*Enu*MA

			lambda12 = (s-MA**2)**2
			lambda34 = (s-MA**2 - Mn**2)**2 - 4*MA**2*Mn**2

			beta12 = np.sqrt(lambda12)/s
			beta34 = np.sqrt(lambda34)/s


			E1CM = (s - MA**2)/2.0/np.sqrt(s)
			E2CM = (s + MA**2)/2.0/np.sqrt(s)
			E3CM = (s + Mn**2 - MA**2)/2.0/np.sqrt(s)
			E4CM = (s - Mn**2 + MA**2)/2.0/np.sqrt(s)

			p1CM = E1CM
			# p1CM = Enu * MA / np.sqrt(s)
			p2CM = np.sqrt(E2CM**2 - MA**2)
			p3CM = np.sqrt(E3CM**2 - Mn**2)
			p4CM = np.sqrt(E4CM**2 - MA**2)

			# p3CM = np.sqrt(s)*beta34/2.0
			# p4CM = np.sqrt(s)*beta34/2.0

			# t0 = (Mn)**2/4.0/s - (p1CM - p3CM)**2
			# t1 = (Mn)**2/4.0/s - (p1CM + p3CM)**2


			# tcosmin = Mn**2 - s/2.0*(1.0 - MA**2/s)*(1 + (Mn**2 - MA**2)/s) - s/2.0*beta12*beta34
			# tcosmax = Mn**2 - s/2.0*(1.0 - MA**2/s)*(1 + (Mn**2 - MA**2)/s) + s/2.0*beta12*beta34
		
			tcosmin = Mn**2 - 2 * ( E1CM*E3CM + p1CM*p3CM) 
			tcosmax = Mn**2 - 2 * ( E1CM*E3CM - p1CM*p3CM) 

			Q2min = -tcosmax
			Q2max = -tcosmin

			Q2lmin = np.log(Q2min)
			Q2lmax = np.log(Q2max)

			Q2l = (Q2lmax - Q2lmin) * x[:, 0] + Q2lmin
			Q2 = np.exp(Q2l)


			# cost = (1 + 1) * x[:, 0] - 1
			# Q2 = -Mn**2 + 2*E3CM*E1CM-2*p1CM*p3CM*cost

			# Q2 = (Q2max - Q2min) * x[:, 0] + Q2min

			# print "CHECK  Q2 = ", Q2
			# print "CHECK cost = ", (-Q2 - Mn**2 + 2 *E1CM*E3CM) / (2*p1CM*p3CM)
			# print "CHECK E1 = ", E1CM, ",  p1CM = ", p1CM
			# print "CHECK E2 = ", E2CM, ",  p2CM = ", p2CM
			# print "CHECK E3 = ", E3CM, ",  p3CM = ", p3CM
			# print "CHECK E4 = ", E4CM, ",  p4CM = ", p4CM, "\n\n"

			# print "CHECK MIN", Q2min[0]
			# print "CHECK MAX", Q2max[0]

			u = 2*MA**2 + Mn**2 - s + Q2

			dsigma = g**2 / 4.0 / cw**2 
			dsigma *= 1.0/(Q2 + Mzprime**2)**2


			if Z == 1:
				# H1d = H1_p(Q2)
				# H2d = H2_p(Q2)

				# LumuHmunu
				# dsigma *= 2*Power(e,2)*(H1d*(-((4*Power(MA,2) + Q2)*(Power(Mn,2) + Q2)) + Power(s - u,2)) + 
		  #    				H2d*(4*Power(MA,4) + Power(Q2,2) + 4*Power(MA,2)*(Power(Mn,2) - s - u) + Power(s + u,2) + Power(Mn,2)*(Q2 - 2*(s + u))))
				
				# F1 = F1_WEAK(Q2)
				# F2 = F2_WEAK(Q2)
				# FA = F3_WEAK(Q2)
							
				F1 = F1_EM(Q2)#F1_SM(Q2)
				F2 = F2_EM(Q2)#F2_SM(Q2)
				FA = 0
				# dsigma *= g**2 / 4.0 / cw**2 
				# dsigma *= 0.5*(-((Power(Mn,2) + Q2)*(16*(F1 - FA)*(F1 + FA)*Power(MA,4) - 8*F1*F2*Power(MA,2)*Q2 + Power(F2,2)*Power(Q2,2))) + 
				#      4*(-Power(MA,2) + s)*(-((4*(2*Power(F1,2) + 2*F1*F2 + Power(F2,2) - 4*(F1 + F2)*FA + 2*Power(FA,2))*Power(MA,2) + Power(F2,2)*Q2)*
				#             (Power(MA,2) + Power(Mn,2) - s))/4. - (F2*(4*(2*F1 + F2)*Power(MA,2) - F2*Q2)*(Power(MA,2) + Power(Mn,2) - u))/4.) + 
				#      (F2*(4*(2*F1 + F2)*Power(MA,2) - F2*Q2)*(Power(MA,2) + Power(Mn,2) - s) + 
				#         2*((Power(2*F1 + F2,2) + 8*(F1 + F2)*FA + 4*Power(FA,2))*Power(MA,2) + (Power(F2,2)*(2*Power(MA,2) + Q2))/2.)*
				#          (Power(MA,2) + Power(Mn,2) - u))*(Power(MA,2) - u))/Power(MA,2)


				# dsigma *= (-1-((Power(Mn,2) + Q2)*(16*(F1 - FA)*(F1 + FA)*Power(MA,4) - 8*F1*F2*Power(MA,2)*Q2 + Power(F2,2)*Power(Q2,2))) + 
				# 	     4*(-Power(MA,2) + s)*(-((4*(2*Power(F1,2) + 2*F1*F2 + Power(F2,2) - 4*(F1 + F2)*FA + 2*Power(FA,2))*Power(MA,2) + Power(F2,2)*Q2)*
				# 	            (Power(MA,2) + Power(Mn,2) - s))/4. - (F2*(4*(2*F1 + F2)*Power(MA,2) - F2*Q2)*(Power(MA,2) + Power(Mn,2) - u))/4.) + 
				# 	     (F2*(4*(2*F1 + F2)*Power(MA,2) - F2*Q2)*(Power(MA,2) + Power(Mn,2) - s) + 
				# 	        2*((Power(2*F1 + F2,2) + 8*(F1 + F2)*FA + 4*Power(FA,2))*Power(MA,2) + (Power(F2,2)*(2*Power(MA,2) + Q2))/2.)*
				# 	         (Power(MA,2) + Power(Mn,2) - u))*(Power(MA,2) - u))/Power(MA,2)
				
				# dsigma *= (-((Power(Mn,2) + Q2)*(16*(F1 - FA)*(F1 + FA)*Power(MA,4) - 8*F1*F2*Power(MA,2)*Q2 + Power(F2,2)*Power(Q2,2))) + 
    #  4*(-Power(MA,2) + s)*(-((4*(2*Power(F1,2) + 2*F1*F2 + Power(F2,2) - 4*(F1 + F2)*FA + 2*Power(FA,2))*Power(MA,2) + Power(F2,2)*Q2)*
    #         (Power(MA,2) + Power(Mn,2) - s))/4. - (F2*(4*(2*F1 + F2)*Power(MA,2) - F2*Q2)*(Power(MA,2) + Power(Mn,2) - u))/4.) + 
    #  (F2*(4*(2*F1 + F2)*Power(MA,2) - F2*Q2)*(Power(MA,2) + Power(Mn,2) - s) + 
    #     2*((Power(2*F1 + F2,2) + 8*(F1 + F2)*FA + 4*Power(FA,2))*Power(MA,2) + (Power(F2,2)*(2*Power(MA,2) + Q2))/2.)*
    #      (Power(MA,2) + Power(Mn,2) - u))*(Power(MA,2) - u))/(2.*Power(MA,2))


				dsigma *= 8*F1*FA*(2*Power(MA,2) + Power(Mn,2) - s - u)*(s - u) - 8*F2*FA*(2*Power(MA,2) + Power(Mn,2) - s - u)*(s - u) + \
				   4*Power(FA,2)*(2*Power(MA,4) + Power(s,2) + 2*Power(MA,2)*(2*Power(Mn,2) + Q2 - s - u) + Power(u,2) - Power(Mn,2)*(s + u)) + \
				   (Power(F2,2)*(16*Power(MA,6) + Q2*(-(Q2*(Power(Mn,2) + Q2)) + Power(s - u,2)) + 16*Power(MA,4)*(Power(Mn,2) - s - u) - \
				        4*Power(MA,2)*(2*Power(Mn,2) - s - u)*(s + u)))/(2.*Power(MA,2)) + \
				   4*Power(F1,2)*(2*Power(MA,4) + Power(s,2) + Power(u,2) - Power(Mn,2)*(s + u) - 2*Power(MA,2)*(Q2 + s + u)) - \
				   4*F1*F2*(4*Power(MA,4) + Power(Q2,2) + 4*Power(MA,2)*(Power(Mn,2) - s - u) + Power(s + u,2) + Power(Mn,2)*(Q2 - 2*(s + u)))

				# dsigma *= (((Power(F2,2)*(16*Power(MA,6) + Q2*(-(Q2*(Power(Mn,2) + Q2)) + Power(s - u,2)) + 16*Power(MA,4)*(Power(Mn,2) - s - u) - 
			 #            4*Power(MA,2)*(2*Power(Mn,2) - s - u)*(s + u)))/Power(MA,2) + 
				# 		      8*Power(F1,2)*(2*Power(MA,4) + Power(s,2) + Power(u,2) - Power(Mn,2)*(s + u) - 2*Power(MA,2)*(Q2 + s + u)) + 
				# 		      8*F1*F2*(4*Power(MA,4) + Power(Q2,2) + 4*Power(MA,2)*(Power(Mn,2) - s - u) + Power(s + u,2) + Power(Mn,2)*(Q2 - 2*(s + u)))))/2.

			else:
				F = Fcoh(np.sqrt(Q2),MA)
				dsigma *= Z*Z

				## LumuHmunu
				dsigma *= 4*Power(e,2)*F*F*(-((4*Power(MA,2) + Q2)*(Power(Mn,2) + Q2)) + Power(s - u,2) )

			
			betaPS = np.sqrt(1.0 - (Mn**2 + MA**2)/s + (Mn**2 - MA**2)**2/s**2)
			# dsigma /= p1CM*np.sqrt(s)
			# dsigma /= (16.0*np.pi*s**2 *beta12**2)
			dsigma *= betaPS/(32.0*4.0*np.pi**2*Enu*MA)


			## Flux convolution
			dsigma *= MiniBooNE_flux(Enu) 

			dsigma *= (self.Emax - self.Emin)
			# dsigma *= (Q2max - Q2min)
			
			# dsigma *= 2

			dsigma *= 1.0/2.0/p1CM/p3CM
			dsigma *= (Q2lmax - Q2lmin)*np.exp(Q2l)

			return dsigma
	


	batch_f = dsigmadQ2_batch(dim=2, Emin=EMIN, Emax=EMAX)

	integ = vg.Integrator([ [0.0, 1.0], [0.0, 1.0] ])
	result = integ(batch_f,  nitn = NINT, neval = NEVAL, rtol=1e-6)

	Q2points 	= []
	Epoints 	= []
	TEST 	= []
	weights 	= []
	integral = 0.0
	variance = 0.0
	for x, wgt, hcube in integ.random_batch(yield_hcube=True):
		
		wgt_fx = wgt*batch_f(x)

		Q2points = np.concatenate((Q2points,x[:,0]))
		Epoints = np.concatenate((Epoints,x[:,1]))
		TEST = np.concatenate((TEST,x[:,1]))
		weights = np.concatenate((weights,wgt_fx))

		for i in range(hcube[0], hcube[-1] + 1):
			idx = (hcube == i)
			nwf = np.sum(idx)
			wf  = wgt_fx[idx]

			sum_wf = np.sum(wf)
			sum_wf2 = np.sum(wf ** 2) # sum of (wgt * f(x)) ** 2

			integral += sum_wf
			variance += (sum_wf2 * nwf - sum_wf ** 2) / (nwf - 1.)

	final_integral = gv.gvar(integral, variance ** 0.5)
	
	## Check that I get the same VEGAS integral!
	# print final_integral
	# print "integral = %s, Q = %.2f"%(result, result.Q)
	
	Q2p = np.array(Q2points)
	Enup = np.array(Epoints)
	TEST = np.array(TEST)
	weights = np.array(weights)

	Emax = EMAX
	if (EMIN > Mn**2/2.0/MA + Mn):
		Emin = EMIN
	else:
		Emin = 1.01*(Mn**2/2.0/MA + Mn)

	Enup = Enup*(Emax - Emin) + Emin

	# print Enup
	s = MA**2 + 2*Enup*MA

	# lambda12 = (s-MA**2)**2
	# lambda34 = (s-MA**2 - Mn**2)**2 - 4*MA**2*Mn**2

	# beta12 = np.sqrt(lambda12)/s
	# beta34 = np.sqrt(lambda34)/s


	E1CM = (s - MA**2)/2.0/np.sqrt(s)
	E2CM = (s + MA**2)/2.0/np.sqrt(s)
	E3CM = (s + Mn**2 - MA**2)/2.0/np.sqrt(s)
	E4CM = (s - Mn**2 + MA**2)/2.0/np.sqrt(s)

	# p1CM = Enup * MA / np.sqrt(s)
	p1CM = E1CM
	p2CM = np.sqrt(E2CM**2 - MA**2)
	p3CM = np.sqrt(E3CM**2 - Mn**2)
	p4CM = np.sqrt(E4CM**2 - MA**2)


	# p3CM = np.sqrt(s)*beta34/2.0
	# p4CM = np.sqrt(s)*beta34/2.0

	# tcosmin = Mn**2 - s/2.0*(1.0 - MA**2/s)*(1 + (Mn**2 - MA**2)/s) - s/2.0*beta12*beta34
	# tcosmax = Mn**2 - s/2.0*(1.0 - MA**2/s)*(1 + (Mn**2 - MA**2)/s) + s/2.0*beta12*beta34

	tcosmin = Mn**2 - 2 * ( E1CM*E3CM + p1CM*p3CM) 
	tcosmax = Mn**2 - 2 * ( E1CM*E3CM - p1CM*p3CM) 

	Q2min = -tcosmax
	Q2max = -tcosmin

	Q2lmin = np.log(Q2min)
	Q2lmax = np.log(Q2max)

	# costp = 2.0* Q2p - 1.0 
	# Q2p = -Mn**2 + 2*E3CM*E1CM-2*p1CM*p3CM*costp
	# Q2p = (Q2max - Q2min) * Q2p + Q2min
	Q2l = (Q2lmax - Q2lmin) * Q2p + Q2lmin

	Q2p = np.exp(Q2l)
	
	# KINEMATICS TO LAB FRAME
	costheta_CM = ( -Q2p - Mn**2 + 2*E1CM*E3CM) / (2*p1CM*p3CM)
	# costheta_CM = costp
	# print costheta_CM
	theta_CM = np.arccos(costheta_CM)

	beta = -p2CM/E2CM
	gamma = 1.0/np.sqrt(1.0 - beta**2)

	phi_CM = [ random.random()*2.0*np.pi for i in range(0,np.size(theta_CM)) ]
	# phi_CM = [ 1.8*np.pi for i in range(0,np.size(theta_CM)) ]
	# 
	P3CM = [	[E3CM[i],
				 p3CM[i]*np.cos(phi_CM[i])*np.sin(theta_CM[i]), 
				 p3CM[i]*np.sin(phi_CM[i])*np.sin(theta_CM[i]), 
				 p3CM[i]*np.cos(theta_CM[i])] for i in range(0,np.size(theta_CM))]
	
	P4CM = [	[E4CM[i],
				 -p3CM[i]*np.cos(phi_CM[i])*np.sin(theta_CM[i]), 
				 -p3CM[i]*np.sin(phi_CM[i])*np.sin(theta_CM[i]), 
				 -p3CM[i]*np.cos(theta_CM[i])] for i in range(0,np.size(theta_CM))]

	P1CM = [	[E1CM[i], 0, 0,  p1CM[i]] for i in range(0,np.size(theta_CM))]
	P2CM = [	[E2CM[i], 0, 0, -p2CM[i]] for i in range(0,np.size(theta_CM))]


	P1LAB = [ fourvec.L(P1CM[i], beta[i]) for i in range(np.size(theta_CM))]
	P2LAB = [ fourvec.L(P2CM[i], beta[i]) for i in range(np.size(theta_CM))]
	P3LAB = [ fourvec.L(P3CM[i], beta[i]) for i in range(np.size(theta_CM))]
	P4LAB = [ fourvec.L(P4CM[i], beta[i]) for i in range(np.size(theta_CM))]

	# p3zCM = p3CM*costheta_CM
	# E3LAB = gamma*( E3CM - beta*p3zCM )
	# p3zLAB = gamma*( p3zCM - beta*E3CM  )
	# p3LAB = np.sqrt(p3CM**2 - p3zCM**2 + p3zLAB**2)

	# costheta_LAB = ( - (Q2p + Mn**2)/2.0 + Enup*E3LAB)/p3LAB/Enup
	# theta_LAB = np.arccos(costheta_LAB)
	# P3LAB = [ [1.0, np.sin(theta_LAB[i])*np.cos(phi_CM[i]),  np.sin(theta_LAB[i])*np.sin(phi_CM[i]),  np.cos(theta_LAB[i])] for i in range(np.size(theta_CM))]

	# P3LAB= [	 [E3LAB[i],
	# 			 p3LAB[i]*np.cos(phi_LAB[i])*np.sin(theta_LAB[i]), 
	# 			 p3LAB[i]*np.sin(phi_LAB[i])*np.sin(theta_LAB[i]), 
	# 			 p3LAB[i]*np.cos(theta_LAB[i])] for i in range(0,np.size(theta_LAB))]
	
	########################### SAVE THE DATA ##################################################
	data = [ [P3LAB[i][0], P3LAB[i][1], P3LAB[i][2], P3LAB[i][3], weights[i]] for i in range(np.shape(P3LAB)[0])] 

	np.savetxt(datafile, np.array(data), header="P0 Px Py Pz weights "+str(Mn)+" "+str(Mzprime)+" "+str(integral))


	# return Q2p, weights, integral*3.9204e-28
	return P3LAB, weights, integral#*3.9204e-28
