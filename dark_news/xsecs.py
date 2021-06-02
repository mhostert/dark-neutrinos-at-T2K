import numpy as np
import scipy
import vegas as vg
import random 

#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv


from . import const
from . import fourvec
from . import decay_rates
from . import pdg 
from . import MC 

def Power(x,n):
	return x**n
def lam(a,b,c):
	return a**2 + b**2 + c**2 -2*a*b - 2*b*c - 2*a*c
def Sqrt(x):
	return np.sqrt(x)


class dsigma_zprime(vg.BatchIntegrand):

			def __init__(self, dim, params, Enu, MA, Z, nu_produced=pdg.neutrino4, h_upscattered=-1):
				self.dim = dim
				self.Enu = Enu
				self.params = params
				self.h_upscattered = h_upscattered
				self.MA = MA
				self.Z  = Z

				if nu_produced==pdg.neutrino6:
					self.mh = params.m6
					###########################33
					# THESE ENTER INTO SCATTERING
					self.cij=params.Umu6*const.gweak/2.0/const.cw
					self.cji=self.cij
					self.dij=params.UD6*params.Umu6*params.gprime
					self.dji=self.dij

				elif nu_produced==pdg.neutrino5:
					self.mh = params.m5

					###########################33
					# THESE ENTER INTO SCATTERING
					self.cij=params.Umu5*const.gweak/2.0/const.cw
					self.cji=self.cij
					self.dij=params.UD5*params.Umu5*params.gprime
					self.dji=self.dij

				elif nu_produced==pdg.neutrino4:
					self.mh = params.m4

					###########################33
					# THESE ENTER INTO SCATTERING
					self.cij=params.Umu4*const.gweak/2.0/const.cw
					self.cji=self.cij
					self.dij=params.UD4*params.Umu4*params.gprime
					self.dji=self.dij


			def __call__(self, x):

				g = const.g			
				gweak = const.g	
				MZBOSON = const.Mz
				cw = const.cw
				MW = const.Mw
				pi = np.pi

				cij = self.cij
				cji = self.cji
				dij = self.dij
				dji = self.dji

				MA = self.MA
				Z = self.Z
				params = self.params
				Mn = self.mh
				
				Enu = self.Enu

				#######################
				# # Upscattering N decay
				# alphaD = params.gprime**2/4.0/np.pi
				# Umu4 = params.Umu4
				# epsilon = params.chi * const.cw
				# UD4     = params.UD4				
				# Mzprime = params.Mzprime

				# s = MA**2 + 2*Enu*MA

				# E1CM = (s - MA**2)/2.0/np.sqrt(s)
				# E3CM = (s + Mn**2 - MA**2)/2.0/np.sqrt(s)

				# p1CM = E1CM # massless
				# p3CM = np.sqrt(E3CM**2 - Mn**2)

				# # Q2lmin = np.log(-(Mn**2 - 2 *  E1CM*(E3CM - p3CM) ))
				# # Q2lmax = np.log(-(Mn**2 - 2 *  E1CM*(E3CM + p3CM)))
				# Q2lmin = np.log(-(Mn*Mn- 2 * ( E1CM*E3CM - p1CM*p3CM) ))
				# Q2lmax = np.log(-(Mn*Mn- 2 * ( E1CM*E3CM + p1CM*p3CM)))

				# Q2l = (Q2lmax - Q2lmin) * x[:, 0] + Q2lmin
				# Q2 = np.exp(Q2l)

				# u = 2*MA**2 + Mn**2 - s + Q2

				# # Start sigma
				# dsigma = 1.0

				# # BSM parameters
				# dsigma *= Umu4**2*alphaD*4.0*np.pi/4.0
				# dsigma *= epsilon**2

				# # Z' propagator
				# dsigma *= 1.0/(Q2 + Mzprime**2)**2

				# # Hadronic tensor coupling
				# dsigma *= Power(const.eQED,2)

				# if Z == 1:
				# 	F1 = const.F1_EM(Q2)#F1_SM(Q2)
				# 	F2 = const.F2_EM(Q2)#F2_SM(Q2)
				# 	FA = 0
				# 	dsigma *= 16*Power(Enu,2)*(4*(Power(F1,2) + Power(FA,2))*Power(MA,2) + Power(F2,2)*Q2) + ((Power(Mn,2) + Q2)*(-16*(F1 - FA)*(F1 + FA)*Power(MA,4) - 4*F2*(2*F1 + F2)*Power(MA,2)*Power(Mn,2) + 4*(2*Power(F1,2) + Power(F2,2) + 4*F2*FA + 2*Power(FA,2) + 4*F1*(F2 + FA))*Power(MA,2)*Q2 + Power(F2,2)*Power(Mn,2)*Q2))/Power(MA,2) - (8*Enu*(8*F1*FA*Power(MA,2)*Q2 + 8*F2*FA*Power(MA,2)*Q2 + 4*Power(F1,2)*Power(MA,2)*(Power(Mn,2) + Q2) + 4*Power(FA,2)*Power(MA,2)*(Power(Mn,2) + Q2) + Power(F2,2)*Q2*(Power(Mn,2) + Q2)))/MA
				# 	# dsigma *= 16*F2*FA*Q2*(2*Power(MA,2) + Power(Mn,2) + Q2 - 2*s) + 8*F1*(4*FA*Power(MA,2)*Q2 - (Power(Mn,2) + Q2)*(F2*Power(Mn,2) - 2*(F2 + FA)*Q2) - 4*FA*Q2*s) + 8*Power(FA,2)*(2*Power(MA,4) + Q2*(Power(Mn,2) + Q2) + 4*Power(MA,2)*(Power(Mn,2) + Q2 - s) - 2*(Power(Mn,2) + Q2)*s + 2*Power(s,2)) + 8*Power(F1,2)*(Q2*(Power(Mn,2) + Q2) + 2*(Power(MA,4) - (2*Power(MA,2) + Power(Mn,2) + Q2)*s + Power(s,2))) + (Power(F2,2)*(4*Power(MA,4)*Q2 - 4*Power(MA,2)*(Power(Mn,4) - Power(Mn,2)*Q2 + 2*Q2*(-Q2 + s)) + Q2*(Power(Mn,4) + Power(Mn,2)*(Q2 - 4*s) + 4*s*(-Q2 + s))))/Power(MA,2)

				# else:
				# 	F1 = const.FEMcoh(np.sqrt(Q2),MA)
				# 	F2 = 0
				# 	FA = 0
				# 	dsigma *= Z*Z*4.0*Power(F1,2)*(4*Power(MA,4) + Power(Mn,4) + Power(Mn,2)*Q2 - 4*(2*Power(MA,2) + Power(Mn,2) + Q2)*s + 4*Power(s,2))
				# 	# dsigma *= Z*Z*4*Power(F1,2)*(4*Power(MA,4) + Power(Mn,4) + Power(Mn,2)*Q2 - 4*(2*Power(MA,2) + Power(Mn,2) + Q2)*s + 4*Power(s,2))
					
				# betaPS = np.sqrt(1.0 - 2*(Mn**2 + MA**2)/s + (Mn**2 - MA**2)**2/s**2)
				# dsigma *= betaPS/(32.0*np.pi**2)
				
				# dsigma *= 1.0/(4.0*Enu*MA) 
				# dsigma *= 1.0/2.0/p1CM/p3CM

				# # Integrated over phi 
				# dsigma *= 2*np.pi
				# dsigma *= (Q2lmax - Q2lmin)*np.exp(Q2l)








				s = MA**2 + 2*Enu*MA
				E1CM = (s - MA**2)/2.0/np.sqrt(s)
				E3CM = (s + Mn**2 - MA**2)/2.0/np.sqrt(s)

				p1CM = E1CM # massless
				p3CM = np.sqrt(E3CM**2 - Mn**2)

				Q2lmin = np.log(-(Mn**2 - 2 * ( E1CM*E3CM - p1CM*p3CM) ))
				Q2lmax = np.log(-(Mn**2 - 2 * ( E1CM*E3CM + p1CM*p3CM)))

				Q2l = (Q2lmax - Q2lmin) * x[:, 0] + Q2lmin
				Q2 = np.exp(Q2l)

				if Z == 1:

					M=MA
					mHNL=Mn
					Mzprime=params.Mzprime
					g = const.g
					pi=np.pi
					mzprime=Mzprime
					MZPRIME=Mzprime

					# FFf1=const.F1_EM(Q2)#const.F1_WEAK(Q2)
					# FFf2=const.F2_EM(Q2)#const.F2_WEAK(Q2)
					# FFga=0.0#const.F3_WEAK(Q2)
					# FFgp=0.0
					EMf1=const.F1_EM(Q2)#const.F1_WEAK(Q2)
					EMf2=const.F2_EM(Q2)#const.F2_WEAK(Q2)
					EMga=0.0#const.F3_WEAK(Q2)
					EMgp=0.0

					Wf1=const.F1_WEAK(Q2)
					Wf2=const.F2_WEAK(Q2)
					Wga=const.F3_WEAK(Q2)
					Wgp=0.0

					deV = const.eQED*params.epsilon
					dWV = (1.0-4.0*const.s2w)*Z-(MA/const.MAVG - Z)

					t = -Q2
					h = self.h_upscattered

					# Dirac
					if params.Dirac == 1:
						dsigma = (cij*cji*(params.epsilon*params.epsilon)*(const.eQED*const.eQED)*Sqrt((-4*(M*M)*(mHNL*mHNL) + (M*M + mHNL*mHNL - s)*(M*M + mHNL*mHNL - s))/(s*s))*(2*(FFf2*FFf2)*(M*M)*((mHNL*mHNL - t)*(mHNL*mHNL - t)) + ((mHNL*mHNL - t)*(-16*(FFf1*FFf1)*(M*M*M*M) + 16*(FFga*FFga)*(M*M*M*M) - 6*(FFf2*FFf2)*(M*M)*(mHNL*mHNL) - 8*(FFgp*FFgp)*(M*M)*(mHNL*mHNL) + 8*FFf1*FFf2*(M*M)*(-(M*M) - mHNL*mHNL + s) + 8*FFf2*FFga*(M*M)*(-(M*M) - mHNL*mHNL + s) - 4*FFf1*FFf2*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) - 8*FFf2*FFga*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) - 8*FFga*FFgp*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) - FFf2*FFf2*(mHNL*mHNL)*(2*(M*M) - t) + 4*(FFgp*FFgp)*(mHNL*mHNL)*(2*(M*M) - t) - 8*FFf1*FFf2*(M*M)*(-(M*M) + s + t) + 8*FFf2*FFga*(M*M)*(-(M*M) + s + t) + 2*(FFf2*FFf2)*(-(M*M) - mHNL*mHNL + s)*(-(M*M) + s + t) - FFf2*FFf2*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(-(M*M) + s + t) - 2*FFf2*FFgp*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(-(M*M) + s + t) - 8*FFf1*FFf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t) + 8*FFf2*FFga*(M*M)*(-(M*M) - mHNL*mHNL + s + t) - 2*FFf2*(-(M*M) + s)*(-4*(FFf1 + FFga)*(M*M) + ((FFf2 + 2*FFgp)*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))))/2. - FFf2*(-(M*M) - mHNL*mHNL + s + t)) - 8*(FFf2*FFf2)*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) + 8*FFf1*FFf2*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 16*FFf2*FFga*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) + 16*FFga*FFgp*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 2*(FFf2*FFf2)*h*mHNL*(-(M*M) - mHNL*mHNL + s)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) + 4*FFf2*FFgp*h*mHNL*(-(M*M) - mHNL*mHNL + s)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 2*(FFf2*FFf2)*h*mHNL*(-(M*M) - mHNL*mHNL + s + t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) + 4*FFf2*FFgp*h*mHNL*(-(M*M) - mHNL*mHNL + s + t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))))/2. + 2*(h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*(8*(FFf1*FFf1)*(M*M*M*M) - 8*(FFga*FFga)*(M*M*M*M) + FFf2*FFf2*(M*M)*(mHNL*mHNL) - 4*(FFgp*FFgp)*(M*M)*(mHNL*mHNL) - 2*FFf1*FFf2*(M*M)*(-(M*M) - mHNL*mHNL + s) + 4*FFga*FFgp*(M*M)*(-(M*M) - mHNL*mHNL + s) - (FFf2*FFf2*(mHNL*mHNL)*(2*(M*M) - t))/2. + 2*(FFgp*FFgp)*(mHNL*mHNL)*(2*(M*M) - t) + 2*FFf1*FFf2*(M*M)*(-(M*M) + s + t) - 4*FFga*FFgp*(M*M)*(-(M*M) + s + t) + 4*FFf1*FFf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t) - 4*FFf2*FFga*(M*M)*(-(M*M) - mHNL*mHNL + s + t) + (FFf2*FFf2*(-(M*M) + s + t)*(-(M*M) - mHNL*mHNL + s + t))/2. - FFf2*FFgp*(-(M*M) + s + t)*(-(M*M) - mHNL*mHNL + s + t) + FFf2*(-(M*M) + s)*(-4*(FFf1 + FFga)*(M*M) + ((FFf2 + 2*FFgp)*(-(M*M) - mHNL*mHNL + s))/2. - FFf2*(-(M*M) - mHNL*mHNL + s + t))) - 2*(((-(M*M) + s)*(-2*((FFf1 + FFga)*(FFf1 + FFga))*(M*M)*(-(M*M) - mHNL*mHNL + s) + 2*((FFf1 + FFga)*(FFf1 + FFga))*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) + mHNL*mHNL*(2*(FFf1*FFf2 + 2*FFga*(FFf2 + FFgp))*(M*M) - (FFf2*(FFf2 + 2*FFgp)*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))))/2.)))/2. + ((-(M*M) - mHNL*mHNL + s + t)*(-2*(mHNL*mHNL)*((FFf1*FFf2 + 2*FFga*(-FFf2 + FFgp))*(M*M) - (FFf2*FFf2*(-(M*M) + s))/2.) - 2*((FFf1 - FFga)*(FFf1 - FFga))*(M*M)*(-(M*M) + s + t) + h*mHNL*(4*((FFf1 - FFga)*(FFf1 - FFga))*(M*M) - FFf2*(FFf2 - 2*FFgp)*(mHNL*mHNL))*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))))/2.))))/(64.*(M*M)*pi*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt((1 - (M*M)/s)*(1 - (M*M)/s))*Sqrt((M*M - s)*(M*M - s))*s*((mzprime*mzprime - t)*(mzprime*mzprime - t)))
					# Majorana
					else:
						# dsigma = (cij*cji*(params.epsilon*params.epsilon)*(const.eQED*const.eQED)*Sqrt((-4*(M*M)*(mHNL*mHNL) + (M*M + mHNL*mHNL - s)*(M*M + mHNL*mHNL - s))/(s*s))*(2*(FFf2*FFf2)*(M*M)*((mHNL*mHNL - t)*(mHNL*mHNL - t)) + ((mHNL*mHNL - t)*(-16*(FFf1*FFf1)*(M*M*M*M) + 16*(FFga*FFga)*(M*M*M*M) - 6*(FFf2*FFf2)*(M*M)*(mHNL*mHNL) - 8*(FFgp*FFgp)*(M*M)*(mHNL*mHNL) + 8*FFf1*FFf2*(M*M)*(-(M*M) - mHNL*mHNL + s) + 8*FFf2*FFga*(M*M)*(-(M*M) - mHNL*mHNL + s) - 4*FFf1*FFf2*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) - 8*FFf2*FFga*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) - 8*FFga*FFgp*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) - FFf2*FFf2*(mHNL*mHNL)*(2*(M*M) - t) + 4*(FFgp*FFgp)*(mHNL*mHNL)*(2*(M*M) - t) - 8*FFf1*FFf2*(M*M)*(-(M*M) + s + t) + 8*FFf2*FFga*(M*M)*(-(M*M) + s + t) + 2*(FFf2*FFf2)*(-(M*M) - mHNL*mHNL + s)*(-(M*M) + s + t) - FFf2*FFf2*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(-(M*M) + s + t) - 2*FFf2*FFgp*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(-(M*M) + s + t) - 8*FFf1*FFf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t) + 8*FFf2*FFga*(M*M)*(-(M*M) - mHNL*mHNL + s + t) - 2*FFf2*(-(M*M) + s)*(-4*(FFf1 + FFga)*(M*M) + ((FFf2 + 2*FFgp)*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))))/2. - FFf2*(-(M*M) - mHNL*mHNL + s + t)) - 8*(FFf2*FFf2)*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) + 8*FFf1*FFf2*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 16*FFf2*FFga*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) + 16*FFga*FFgp*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 2*(FFf2*FFf2)*h*mHNL*(-(M*M) - mHNL*mHNL + s)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) + 4*FFf2*FFgp*h*mHNL*(-(M*M) - mHNL*mHNL + s)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 2*(FFf2*FFf2)*h*mHNL*(-(M*M) - mHNL*mHNL + s + t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) + 4*FFf2*FFgp*h*mHNL*(-(M*M) - mHNL*mHNL + s + t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))))/2. + 2*(h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*(8*(FFf1*FFf1)*(M*M*M*M) - 8*(FFga*FFga)*(M*M*M*M) + FFf2*FFf2*(M*M)*(mHNL*mHNL) - 4*(FFgp*FFgp)*(M*M)*(mHNL*mHNL) - 2*FFf1*FFf2*(M*M)*(-(M*M) - mHNL*mHNL + s) + 4*FFga*FFgp*(M*M)*(-(M*M) - mHNL*mHNL + s) - (FFf2*FFf2*(mHNL*mHNL)*(2*(M*M) - t))/2. + 2*(FFgp*FFgp)*(mHNL*mHNL)*(2*(M*M) - t) + 2*FFf1*FFf2*(M*M)*(-(M*M) + s + t) - 4*FFga*FFgp*(M*M)*(-(M*M) + s + t) + 4*FFf1*FFf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t) - 4*FFf2*FFga*(M*M)*(-(M*M) - mHNL*mHNL + s + t) + (FFf2*FFf2*(-(M*M) + s + t)*(-(M*M) - mHNL*mHNL + s + t))/2. - FFf2*FFgp*(-(M*M) + s + t)*(-(M*M) - mHNL*mHNL + s + t) + FFf2*(-(M*M) + s)*(-4*(FFf1 + FFga)*(M*M) + ((FFf2 + 2*FFgp)*(-(M*M) - mHNL*mHNL + s))/2. - FFf2*(-(M*M) - mHNL*mHNL + s + t))) - 2*(((-(M*M) + s)*(-2*((FFf1 + FFga)*(FFf1 + FFga))*(M*M)*(-(M*M) - mHNL*mHNL + s) + 2*((FFf1 + FFga)*(FFf1 + FFga))*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) + mHNL*mHNL*(2*(FFf1*FFf2 + 2*FFga*(FFf2 + FFgp))*(M*M) - (FFf2*(FFf2 + 2*FFgp)*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))))/2.)))/2. + ((-(M*M) - mHNL*mHNL + s + t)*(-2*(mHNL*mHNL)*((FFf1*FFf2 + 2*FFga*(-FFf2 + FFgp))*(M*M) - (FFf2*FFf2*(-(M*M) + s))/2.) - 2*((FFf1 - FFga)*(FFf1 - FFga))*(M*M)*(-(M*M) + s + t) + h*mHNL*(4*((FFf1 - FFga)*(FFf1 - FFga))*(M*M) - FFf2*(FFf2 - 2*FFgp)*(mHNL*mHNL))*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))))/2.))))/(64.*(M*M)*pi*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt((1 - (M*M)/s)*(1 - (M*M)/s))*Sqrt((M*M - s)*(M*M - s))*s*((mzprime*mzprime - t)*(mzprime*mzprime - t)))					
						dsigma = -((4*(deV*deV)*dij*dji*(-(EMf2*EMf2*(M*M)*((mHNL*mHNL - t)*(mHNL*mHNL - t))) - h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*(8*(EMf1*EMf1)*(M*M*M*M) + EMf2*EMf2*(M*M)*(mHNL*mHNL) - 2*EMf1*EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s) - (EMf2*EMf2*(mHNL*mHNL)*(2*(M*M) - t))/2. + 2*EMf1*EMf2*(M*M)*(-(M*M) + s + t) + 4*EMf1*EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t) + (EMf2*EMf2*(-(M*M) + s + t)*(-(M*M) - mHNL*mHNL + s + t))/2. + EMf2*(-(M*M) + s)*(-4*EMf1*(M*M) + (EMf2*(-(M*M) - mHNL*mHNL + s))/2. - EMf2*(-(M*M) - mHNL*mHNL + s + t))) + ((mHNL*mHNL - t)*(8*(EMf1*EMf1)*(M*M*M*M) + 3*(EMf2*EMf2)*(M*M)*(mHNL*mHNL) - 4*EMf1*EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s) + 2*EMf1*EMf2*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) + (EMf2*EMf2*(mHNL*mHNL)*(2*(M*M) - t))/2. + 4*EMf1*EMf2*(M*M)*(-(M*M) + s + t) - EMf2*EMf2*(-(M*M) - mHNL*mHNL + s)*(-(M*M) + s + t) + (EMf2*EMf2*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(-(M*M) + s + t))/2. + 4*EMf1*EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t) - EMf2*(-(M*M) + s)*(4*EMf1*(M*M) - (EMf2*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))))/2. + EMf2*(-(M*M) - mHNL*mHNL + s + t)) + 4*(EMf2*EMf2)*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 4*EMf1*EMf2*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) + EMf2*EMf2*h*mHNL*(-(M*M) - mHNL*mHNL + s)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) + EMf2*EMf2*h*mHNL*(-(M*M) - mHNL*mHNL + s + t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))))/2. - 2*(((-(M*M) + s)*(2*(EMf1*EMf1)*(M*M)*(-(M*M) - mHNL*mHNL + s) - 2*(EMf1*EMf1)*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) + EMf2*(mHNL*mHNL)*(-2*EMf1*(M*M) + (EMf2*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))))/2.)))/2. + ((-(M*M) - mHNL*mHNL + s + t)*(2*EMf2*(mHNL*mHNL)*(EMf1*(M*M) - (EMf2*(-(M*M) + s))/2.) + 2*(EMf1*EMf1)*(M*M)*(-(M*M) + s + t) - h*mHNL*(4*(EMf1*EMf1)*(M*M) - EMf2*EMf2*(mHNL*mHNL))*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))))/2.)))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (4*deV*(cji*dij + cij*dji)*g*(-(EMf2*(M*M)*((mHNL*mHNL - t)*(mHNL*mHNL - t))*Wf2) - h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*(8*EMf1*(M*M*M*M)*Wf1 - EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s)*Wf1 + EMf2*(M*M)*(-(M*M) + s + t)*Wf1 + 2*EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t)*Wf1 + EMf2*(M*M)*(mHNL*mHNL)*Wf2 - EMf1*(M*M)*(-(M*M) - mHNL*mHNL + s)*Wf2 - (EMf2*(mHNL*mHNL)*(2*(M*M) - t)*Wf2)/2. + EMf1*(M*M)*(-(M*M) + s + t)*Wf2 + 2*EMf1*(M*M)*(-(M*M) - mHNL*mHNL + s + t)*Wf2 + (EMf2*(-(M*M) + s + t)*(-(M*M) - mHNL*mHNL + s + t)*Wf2)/2. - 2*EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t)*Wga - (EMf2*(-(M*M) + s + t)*(-(M*M) - mHNL*mHNL + s + t)*Wgp)/2. - (-(M*M) + s)*(EMf2*(-(M*M) - mHNL*mHNL + s + t)*Wf2 + 2*(M*M)*(EMf1*Wf2 + EMf2*(Wf1 + Wga)) - (EMf2*(-(M*M) - mHNL*mHNL + s)*(Wf2 + Wgp))/2.)) + ((mHNL*mHNL - t)*(8*EMf1*(M*M*M*M)*Wf1 - 2*EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s)*Wf1 + EMf2*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*Wf1 + 2*EMf2*(M*M)*(-(M*M) + s + t)*Wf1 + 2*EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t)*Wf1 - 2*EMf2*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wf1 + 3*EMf2*(M*M)*(mHNL*mHNL)*Wf2 - 2*EMf1*(M*M)*(-(M*M) - mHNL*mHNL + s)*Wf2 + EMf1*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*Wf2 + (EMf2*(mHNL*mHNL)*(2*(M*M) - t)*Wf2)/2. + 2*EMf1*(M*M)*(-(M*M) + s + t)*Wf2 - EMf2*(-(M*M) - mHNL*mHNL + s)*(-(M*M) + s + t)*Wf2 + (EMf2*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(-(M*M) + s + t)*Wf2)/2. + 2*EMf1*(M*M)*(-(M*M) - mHNL*mHNL + s + t)*Wf2 + 4*EMf2*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wf2 - 2*EMf1*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wf2 + EMf2*h*mHNL*(-(M*M) - mHNL*mHNL + s)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wf2 + EMf2*h*mHNL*(-(M*M) - mHNL*mHNL + s + t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wf2 - 2*EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s)*Wga + 2*EMf2*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*Wga - 2*EMf2*(M*M)*(-(M*M) + s + t)*Wga - 2*EMf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t)*Wga + 4*EMf2*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wga + (EMf2*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(-(M*M) + s + t)*Wgp)/2. - EMf2*h*mHNL*(-(M*M) - mHNL*mHNL + s)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wgp - EMf2*h*mHNL*(-(M*M) - mHNL*mHNL + s + t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wgp - (-(M*M) + s)*(EMf2*(-(M*M) - mHNL*mHNL + s + t)*Wf2 + 2*(M*M)*(EMf1*Wf2 + EMf2*(Wf1 + Wga)) - (EMf2*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(Wf2 + Wgp))/2.)))/2. + 2*(-((-(M*M) - mHNL*mHNL + s + t)*(mHNL*mHNL*(-(EMf2*(-(M*M) + s)*Wf2) + M*M*(EMf1*Wf2 + EMf2*(Wf1 - 2*Wga))) + 2*EMf1*(M*M)*(-(M*M) + s + t)*(Wf1 - Wga) - h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*(4*EMf1*(M*M)*(Wf1 - Wga) + EMf2*(mHNL*mHNL)*(-Wf2 + Wgp))))/2. + ((-(M*M) + s)*(-2*EMf1*(M*M)*(-(M*M) - mHNL*mHNL + s)*(Wf1 + Wga) + 2*EMf1*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(Wf1 + Wga) + mHNL*mHNL*(M*M*(EMf1*Wf2 + EMf2*(Wf1 + 2*Wga)) - (EMf2*h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(Wf2 + Wgp))/2.)))/2.)))/(cw*(MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) + (cij*cji*(g*g)*(-(M*M*((mHNL*mHNL - t)*(mHNL*mHNL - t))*(Wf2*Wf2)) - h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*(8*(M*M*M*M)*(Wf1*Wf1) - 2*(M*M)*(-(M*M) - mHNL*mHNL + s)*Wf1*Wf2 + 2*(M*M)*(-(M*M) + s + t)*Wf1*Wf2 + 4*(M*M)*(-(M*M) - mHNL*mHNL + s + t)*Wf1*Wf2 + M*M*(mHNL*mHNL)*(Wf2*Wf2) - (mHNL*mHNL*(2*(M*M) - t)*(Wf2*Wf2))/2. + ((-(M*M) + s + t)*(-(M*M) - mHNL*mHNL + s + t)*(Wf2*Wf2))/2. - 4*(M*M)*(-(M*M) - mHNL*mHNL + s + t)*Wf2*Wga - 8*(M*M*M*M)*(Wga*Wga) - (-(M*M) + s + t)*(-(M*M) - mHNL*mHNL + s + t)*Wf2*Wgp + 4*(M*M)*(-(M*M) - mHNL*mHNL + s)*Wga*Wgp - 4*(M*M)*(-(M*M) + s + t)*Wga*Wgp - 4*(M*M)*(mHNL*mHNL)*(Wgp*Wgp) + 2*(mHNL*mHNL)*(2*(M*M) - t)*(Wgp*Wgp) + (-(M*M) + s)*Wf2*(-((-(M*M) - mHNL*mHNL + s + t)*Wf2) - 4*(M*M)*(Wf1 + Wga) + ((-(M*M) - mHNL*mHNL + s)*(Wf2 + 2*Wgp))/2.)) + ((mHNL*mHNL - t)*(8*(M*M*M*M)*(Wf1*Wf1) - 4*(M*M)*(-(M*M) - mHNL*mHNL + s)*Wf1*Wf2 + 2*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*Wf1*Wf2 + 4*(M*M)*(-(M*M) + s + t)*Wf1*Wf2 + 4*(M*M)*(-(M*M) - mHNL*mHNL + s + t)*Wf1*Wf2 - 4*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wf1*Wf2 + 3*(M*M)*(mHNL*mHNL)*(Wf2*Wf2) + (mHNL*mHNL*(2*(M*M) - t)*(Wf2*Wf2))/2. - (-(M*M) - mHNL*mHNL + s)*(-(M*M) + s + t)*(Wf2*Wf2) + (h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(-(M*M) + s + t)*(Wf2*Wf2))/2. + 4*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*(Wf2*Wf2) + h*mHNL*(-(M*M) - mHNL*mHNL + s)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*(Wf2*Wf2) + h*mHNL*(-(M*M) - mHNL*mHNL + s + t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*(Wf2*Wf2) - 4*(M*M)*(-(M*M) - mHNL*mHNL + s)*Wf2*Wga + 4*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*Wf2*Wga - 4*(M*M)*(-(M*M) + s + t)*Wf2*Wga - 4*(M*M)*(-(M*M) - mHNL*mHNL + s + t)*Wf2*Wga + 8*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wf2*Wga - 8*(M*M*M*M)*(Wga*Wga) + h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(-(M*M) + s + t)*Wf2*Wgp - 2*h*mHNL*(-(M*M) - mHNL*mHNL + s)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wf2*Wgp - 2*h*mHNL*(-(M*M) - mHNL*mHNL + s + t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wf2*Wgp + 4*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*Wga*Wgp - 8*h*(M*M)*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*Wga*Wgp + 4*(M*M)*(mHNL*mHNL)*(Wgp*Wgp) - 2*(mHNL*mHNL)*(2*(M*M) - t)*(Wgp*Wgp) - (-(M*M) + s)*Wf2*((-(M*M) - mHNL*mHNL + s + t)*Wf2 + 4*(M*M)*(Wf1 + Wga) - (h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*(Wf2 + 2*Wgp))/2.)))/2. - 2*(((-(M*M) - mHNL*mHNL + s + t)*(2*(M*M)*(-(M*M) + s + t)*((Wf1 - Wga)*(Wf1 - Wga)) - h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))*(4*(M*M)*((Wf1 - Wga)*(Wf1 - Wga)) - mHNL*mHNL*Wf2*(Wf2 - 2*Wgp)) + 2*(mHNL*mHNL)*(-((-(M*M) + s)*(Wf2*Wf2))/2. + M*M*(Wf1*Wf2 + 2*Wga*(-Wf2 + Wgp)))))/2. + ((-(M*M) + s)*(2*(M*M)*(-(M*M) - mHNL*mHNL + s)*((Wf1 + Wga)*(Wf1 + Wga)) - 2*h*(M*M)*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*((Wf1 + Wga)*(Wf1 + Wga)) - mHNL*mHNL*(-(h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s)))*Wf2*(Wf2 + 2*Wgp))/2. + 2*(M*M)*(Wf1*Wf2 + 2*Wga*(Wf2 + Wgp)))))/2.)))/(cw*cw*((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t))))/(128.*(M*M)*pi*Sqrt((1 - (M*M)/s)*(1 - (M*M)/s))*Sqrt((M*M - s)*(M*M - s))*s)
						#averaged
						# dsigma = (cij*cji*(params.epsilon*params.epsilon)*(const.eQED*const.eQED)*Sqrt((mHNL*mHNL*mHNL*mHNL + (M*M - s)*(M*M - s) - 2*(mHNL*mHNL)*(M*M + s))/(s*s))*(2*(FFf2*FFf2)*(M*M)*((mHNL*mHNL - t)*(mHNL*mHNL - t)) - 2*(2*FFf1*FFf2*(M*M)*(mHNL*mHNL)*(-(M*M) + s) + 4*FFf2*FFga*(M*M)*(mHNL*mHNL)*(-(M*M) + s) + 4*FFga*FFgp*(M*M)*(mHNL*mHNL)*(-(M*M) + s) - 2*(FFf1*FFf1)*(M*M)*(-(M*M) + s)*(-(M*M) - mHNL*mHNL + s) - 4*FFf1*FFga*(M*M)*(-(M*M) + s)*(-(M*M) - mHNL*mHNL + s) - 2*(FFga*FFga)*(M*M)*(-(M*M) + s)*(-(M*M) - mHNL*mHNL + s) + (-(M*M) - mHNL*mHNL + s + t)*(mHNL*mHNL*(-2*(FFf1*FFf2 + 2*FFga*(-FFf2 + FFgp))*(M*M) + FFf2*FFf2*(-(M*M) + s)) - 2*((FFf1 - FFga)*(FFf1 - FFga))*(M*M)*(-(M*M) + s + t))) + ((mHNL*mHNL - t)*(-16*(FFf1*FFf1)*(M*M*M*M) + 16*(FFga*FFga)*(M*M*M*M) - 6*(FFf2*FFf2)*(M*M)*(mHNL*mHNL) - 8*(FFgp*FFgp)*(M*M)*(mHNL*mHNL) + 8*FFf1*FFf2*(M*M)*(-(M*M) - mHNL*mHNL + s) + 8*FFf2*FFga*(M*M)*(-(M*M) - mHNL*mHNL + s) - FFf2*FFf2*(mHNL*mHNL)*(2*(M*M) - t) + 4*(FFgp*FFgp)*(mHNL*mHNL)*(2*(M*M) - t) - 8*FFf1*FFf2*(M*M)*(-(M*M) + s + t) + 8*FFf2*FFga*(M*M)*(-(M*M) + s + t) + 2*(FFf2*FFf2)*(-(M*M) - mHNL*mHNL + s)*(-(M*M) + s + t) - 8*FFf1*FFf2*(M*M)*(-(M*M) - mHNL*mHNL + s + t) + 8*FFf2*FFga*(M*M)*(-(M*M) - mHNL*mHNL + s + t) + 2*FFf2*(-(M*M) + s)*(4*(FFf1 + FFga)*(M*M) + FFf2*(-(M*M) - mHNL*mHNL + s + t))))/2.))/(32.*(M*M)*pi*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt((1 - (M*M)/s)*(1 - (M*M)/s))*Sqrt((M*M - s)*(M*M - s))*s*((mzprime*mzprime - t)*(mzprime*mzprime - t)))
					
					# Pauli blocking effects
					dsigma *= const.Fpauli_blocking(Q2,MA)

				elif Z > 1:

					M=MA
					mHNL=Mn
					Mzprime=params.Mzprime
					mzprime=Mzprime
					MZPRIME=Mzprime
					cw = const.cw

					FCOH = const.FEMcoh(np.sqrt(Q2),MA)
					FEM = const.FEMcoh(np.sqrt(Q2),MA)
					FWEAK = const.FWEAKcoh(np.sqrt(Q2),MA)
					t = -Q2

					deV = Z*const.eQED*params.epsilon
					dWV = 0*const.gweak/2.0/cw*((1.0-4.0*const.s2w)*Z-(MA/const.MAVG - Z))

					h = self.h_upscattered

					# Dirac
					if params.D_or_M == 'dirac':
						dsigma = (cij*cji*(params.epsilon*params.epsilon)*(const.eQED*const.eQED)*(FCOH*FCOH)*Sqrt((-4*(M*M)*(mHNL*mHNL) + (M*M + mHNL*mHNL - s)*(M*M + mHNL*mHNL - s))/(s*s))*(-((4*(M*M) - t)*(mHNL*mHNL - t)) + 2*h*mHNL*(4*(M*M) - t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 2*((-(M*M) + s)/2. + (-(M*M) - mHNL*mHNL + s + t)/2.)*(2*(M*M) + mHNL*mHNL - 2*s + h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) - t + 2*h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))))*(Z*Z))/(64.*pi*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt((1 - (M*M)/s)*(1 - (M*M)/s))*Sqrt((M*M - s)*(M*M - s))*s*((mzprime*mzprime - t)*(mzprime*mzprime - t)))
					# Majorana
					elif params.D_or_M == 'majorana':
						dsigma = -((deV*deV*dij*dji*(FEM*FEM)*((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + 2*deV*(cji*dij + cij*dji)*dWV*FEM*FWEAK*(MZBOSON*MZBOSON - t)*(MZPRIME*MZPRIME - t) + cij*cji*(dWV*dWV)*(FWEAK*FWEAK)*((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)))*(((4*(M*M) - t)*(mHNL*mHNL - t))/2. - h*mHNL*(4*(M*M) - t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 2*((-(M*M) + s)/2. + (-(M*M) - mHNL*mHNL + s + t)/2.)*((-(M*M) - mHNL*mHNL + s)/2. - (h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))))/2. + (-(M*M) + s + t)/2. - h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)))))/(16.*pi*Sqrt((1 - (M*M)/s)*(1 - (M*M)/s))*Sqrt((M*M - s)*(M*M - s))*s*((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t))*((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)))
					else:
						print("Could not identify Dirac or Majorana nature of HNL.")
				
					# # Dirac
					# if params.Dirac == 1:
					# 	dsigma = (cij*cji*(params.epsilon*params.epsilon)*(const.eQED*const.eQED)*(FCOH*FCOH)*Sqrt((-4*(M*M)*(mHNL*mHNL) + (M*M + mHNL*mHNL - s)*(M*M + mHNL*mHNL - s))/(s*s))*(-((4*(M*M) - t)*(mHNL*mHNL - t)) + 2*h*mHNL*(4*(M*M) - t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 2*((-(M*M) + s)/2. + (-(M*M) - mHNL*mHNL + s + t)/2.)*(2*(M*M) + mHNL*mHNL - 2*s + h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) - t + 2*h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))))*(Z*Z))/(64.*pi*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt((1 - (M*M)/s)*(1 - (M*M)/s))*Sqrt((M*M - s)*(M*M - s))*s*((mzprime*mzprime - t)*(mzprime*mzprime - t)))
					# # Majorana
					# else:
					# 	# dsigma = (cij*cji*(params.epsilon*params.epsilon)*(const.eQED*const.eQED)*(FCOH*FCOH)*Sqrt((-4*(M*M)*(mHNL*mHNL) + (M*M + mHNL*mHNL - s)*(M*M + mHNL*mHNL - s))/(s*s))*(-((4*(M*M) - t)*(mHNL*mHNL - t)) + 2*h*mHNL*(4*(M*M) - t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 2*((-(M*M) + s)/2. + (-(M*M) - mHNL*mHNL + s + t)/2.)*(2*(M*M) + mHNL*mHNL - 2*s + h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))) - t + 2*h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s))))*(Z*Z))/(32.*pi*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt((1 - (M*M)/s)*(1 - (M*M)/s))*Sqrt((M*M - s)*(M*M - s))*s*((mzprime*mzprime - t)*(mzprime*mzprime - t)))
					# 	dsigma = -((deV*deV*dij*dji*(FEM*FEM)*((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + 2*deV*(cji*dij + cij*dji)*dWV*FEM*FWEAK*(MZBOSON*MZBOSON - t)*(MZPRIME*MZPRIME - t) + cij*cji*(dWV*dWV)*(FWEAK*FWEAK)*((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)))*(((4*(M*M) - t)*(mHNL*mHNL - t))/2. - h*mHNL*(4*(M*M) - t)*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(-(M*M) + s))/(4.*mHNL) - ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)) - 2*((-(M*M) + s)/2. + (-(M*M) - mHNL*mHNL + s + t)/2.)*((-(M*M) - mHNL*mHNL + s)/2. - (h*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt(s)*((-(M*M) + s)/(2.*Sqrt(s)) + (M*M + s)/(2.*Sqrt(s))))/2. + (-(M*M) + s + t)/2. - h*mHNL*((Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*(M*M + s))/(4.*mHNL) + ((-(M*M) + mHNL*mHNL + s)*(-(mHNL*mHNL) + ((-(M*M) + s)*(-(M*M) + mHNL*mHNL + s))/(2.*s) + t))/(2.*mHNL*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*s)))))/(16.*pi*Sqrt((1 - (M*M)/s)*(1 - (M*M)/s))*Sqrt((M*M - s)*(M*M - s))*s*((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t))*((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)))
					# 	# averaged
					# 	# dsigma = (cij*cji*(params.epsilon*params.epsilon)*(const.eQED*const.eQED)*(FCOH*FCOH)*Sqrt((mHNL*mHNL*mHNL*mHNL + (M*M - s)*(M*M - s) - 2*(mHNL*mHNL)*(M*M + s))/(s*s))*(-((4*(M*M) - t)*(mHNL*mHNL - t)) - 2*(((-(M*M) + s)*(2*(M*M) + mHNL*mHNL - 2*s - t))/2. + ((2*(M*M) + mHNL*mHNL - 2*s - t)*(-(M*M) - mHNL*mHNL + s + t))/2.))*(Z*Z))/(16.*pi*Sqrt((1 - (M*M)/s - (mHNL*mHNL)/s)*(1 - (M*M)/s - (mHNL*mHNL)/s) - (4*(M*M)*(mHNL*mHNL))/(s*s))*Sqrt((1 - (M*M)/s)*(1 - (M*M)/s))*Sqrt((M*M - s)*(M*M - s))*s*((mzprime*mzprime - t)*(mzprime*mzprime - t)))

					dsigma *= (Q2lmax - Q2lmin)*np.exp(Q2l)

					# print(dsigma*1e10)
					####################################
					return dsigma

def get_sigma_zprime(params, Enu, MA, Z, nu_produced=pdg.neutrino4, h_upscattered=-1, neval=20000, nitn=20):
	
	if nu_produced ==pdg.neutrino4:
		Mn = params.m4
	elif nu_produced ==pdg.neutrino5:
		Mn = params.m5
	elif nu_produced ==pdg.neutrino6:
		Mn = params.m6
	
	#############################
	# THRESHOLD
	if (Enu > Mn**2/2.0/MA + Mn):
		dim = 1
		batch_f = dsigma_zprime(dim=dim, params=params, Enu=Enu, MA=MA, Z=Z, nu_produced=nu_produced, h_upscattered=h_upscattered)

		integ = vg.Integrator(dim*[[0.0, 1.0]])

		integ(batch_f,nitn = nitn, neval = neval)
		result = integ(batch_f,nitn = nitn, neval = neval)

		integral = result.mean
	else:
		integral = 0.0
	# ##########################################################################
	# # READ SAMPLES
	# Q2points 	= []
	# weights 	= []
	# my_integral = 0.0
	# variance = 0.0
	# for x, wgt, hcube in integ.random_batch(yield_hcube=True):
		
	# 	wgt_fx = wgt*batch_f(x)

	# 	Q2points = np.concatenate((Q2points,x[:,0]))
	# 	weights = np.concatenate((weights,wgt_fx))

	# 	for i in range(hcube[0], hcube[-1] + 1):
	# 		idx = (hcube == i)
	# 		nwf = np.sum(idx)
	# 		wf  = wgt_fx[idx]

	# 		sum_wf = np.sum(wf)
	# 		sum_wf2 = np.sum(wf ** 2) # sum of (wgt * f(x)) ** 2

	# 		my_integral += sum_wf
	# 		variance += (sum_wf2 * nwf - sum_wf ** 2) / (nwf - 1.)
	
	# Q2p = np.array(Q2points)
	# print(my_integral*const.GeV2_to_cm2, variance)


	# s = MA**2 + 2*Enu*MA
	# E1CM = (s - MA**2)/2.0/np.sqrt(s)
	# E3CM = (s + Mn**2 - MA**2)/2.0/np.sqrt(s)

	# p1CM = E1CM # massless
	# p3CM = np.sqrt(E3CM**2 - Mn**2)

	# Q2lmin = np.log(-(Mn**2 - 2 * ( E1CM*E3CM - p1CM*p3CM) ))
	# Q2lmax = np.log(-(Mn**2 - 2 * ( E1CM*E3CM + p1CM*p3CM)))

	# Q2l = (Q2lmax - Q2lmin) * x[:, 0] + Q2lmin
	# Q2 = np.exp(Q2l)

	return integral*const.GeV2_to_cm2#, Q2

