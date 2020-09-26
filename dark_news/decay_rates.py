import numpy as np
from scipy import interpolate
import scipy.stats
import matplotlib.pyplot as plt
from scipy.integrate import quad

from . import const
from . import pdg

def tau_GeV_to_s(decay_rate):
	return 1./decay_rate/1.52/1e24

def L_GeV_to_cm(decay_rate):
	return 1./decay_rate/1.52/1e24*2.998e10

def lam(a,b,c):
	return a**2 + b**2 + c**2 -2*a*b - 2*b*c - 2*a*c

def I1_2body(x,y):
	return ((1+x-y)*(1+x) - 4*x)*np.sqrt(lam(1.0,x,y))


class HeavyNu:
	def __init__(self,params,particle):

		self.params=params
		self.particle = particle

		self.R_total = 0.0

		self.R_nu_nu_nu = 0.0								
		self.R_nu4_nu_nu = 0.0								
		self.R_nu4_nu4_nu = 0.0								
		self.R_nu4_nu4_nu4 = 0.0								
		self.R_nu5_nu_nu = 0.0								
		self.R_nu5_nu5_nu = 0.0								
		self.R_nu5_nu5_nu5 = 0.0								
		self.R_nu_e_e = 0.0										
		self.R_nu_e_e_SM = 0.0								
		self.R_nu_e_mu = 0.0								
		self.R_nu_mu_mu = 0.0								
		self.R_e_pi = 0.0								
		self.R_e_K = 0.0								
		self.R_mu_pi = 0.0								
		self.R_mu_K = 0.0								
		self.R_nu_pi = 0.0								
		self.R_nu_eta = 0.0								
		self.R_nu_rho = 0.0								
		self.R_nu4_e_e = 0.0								
		self.R_nu4_e_mu = 0.0								
		self.R_nu4_mu_mu = 0.0								
		self.R_nu4_pi = 0.0								
		self.R_nu4_eta = 0.0							
		self.R_nu4_rho = 0.0							
		self.R_nu5_e_e = 0.0								
		self.R_nu5_e_mu = 0.0								
		self.R_nu5_mu_mu = 0.0								
		self.R_nu5_pi = 0.0								
		self.R_nu5_eta = 0.0							
		self.R_nu5_rho = 0.0	
		self.R_nu_gamma = 0.0							
		self.R_nu4_gamma = 0.0							
		self.R_nu5_gamma = 0.0							

		# self.BR_nu_nu_nu = 0.0								
		# self.BR_nu4_nu_nu = 0.0								
		# self.BR_nu4_nu4_nu = 0.0								
		# self.BR_nu4_nu4_nu4 = 0.0								
		# self.BR_nu_e_e = 0.0										
		# self.BR_nu_e_e_SM = 0.0								
		# self.BR_nu_e_mu = 0.0								
		# self.BR_nu_mu_mu = 0.0								
		# self.BR_e_pi = 0.0								
		# self.BR_e_K = 0.0								
		# self.BR_mu_pi = 0.0								
		# self.BR_mu_K = 0.0								
		# self.BR_nu_pi = 0.0								
		# self.BR_nu_eta = 0.0								
		# self.BR_nu_rho = 0.0								
		# self.BR_nu4_e_e = 0.0								
		# self.BR_nu4_e_mu = 0.0								
		# self.BR_nu4_mu_mu = 0.0								
		# self.BR_nu4_pi = 0.0								
		# self.BR_nu4_eta = 0.0		
		# self.BR_nu4_rho = 0.0		
		# self.BR_nu_gamma = 0.0		
		# self.BR_nu4_gamma = 0.0		


	def compute_rates(self):
		params = self.params
		##################
		# Neutrino 4
		if self.particle==pdg.neutrino4:
			mh = self.params.m4
			
			# nuSM limit does not match -- FIX ME 
			self.R_nu_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino4, const.neutrino_light, const.neutrino_light, const.neutrino_light)
			self.R_nu_gamma = nui_nu_gamma(params, pdg.neutrino4, const.neutrino_light)

			# dileptons -- already contain the Delta L = 2 channel
			if mh > 2*const.Me:
				self.R_nu_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino4, const.neutrino_light, const.electron, const.electron)
				self.R_nu_e_e_SM = nui_nuj_ell1_ell2(params, pdg.neutrino4, const.neutrino_light, const.electron, const.electron, SM=True)
			if mh > const.Me + const.Mmu:
				self.R_nu_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino4, const.neutrino_light, const.electron, const.muon)
			if mh > 2*const.Mmu:
				self.R_nu_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino4, const.neutrino_light, const.muon, const.muon)
			# pseudoscalar -- factor of 2 for delta L=2 channel 
			if mh > const.Me+const.Mcharged_pion:
				self.R_e_pi = 2*nui_l_P(params, pdg.neutrino4, const.electron, const.charged_pion)
			if mh > const.Me+const.charged_kaon:
				self.R_e_K = 2*nui_l_P(params, pdg.neutrino4, const.electron, const.charged_kaon)
			# pseudoscalar -- already contain the Delta L = 2 channel
			if mh > const.Mmu+const.Mcharged_pion:
				self.R_mu_pi = 2*nui_l_P(params, pdg.neutrino4, const.muon, const.charged_pion)
			if mh > const.Mmu+const.charged_kaon:
				self.R_mu_K = 2*nui_l_P(params, pdg.neutrino4, const.muon, const.charged_kaon)
			if mh > const.Mneutral_pion:
				self.R_nu_pi = nui_nu_P(params, pdg.neutrino4, const.neutrino_light, const.neutral_pion)
			if mh > const.neutral_eta:
				self.R_nu_eta = nui_nu_P(params, pdg.neutrino4, const.neutrino_light, const.neutral_eta)

			# vector mesons
			if mh > const.Mneutral_rho:
				self.R_nu_rho = nui_nu_V(params, pdg.neutrino5, const.neutrino_light, const.neutral_rho)
		


		##################
		# Neutrino 5
		if self.particle==pdg.neutrino5:
			mh = self.params.m5
			
			# nuSM limit does not match -- FIX ME 
			self.R_nu_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino5, const.neutrino_light, const.neutrino_light, const.neutrino_light)
			self.R_nu_gamma = nui_nu_gamma(params, pdg.neutrino5, const.neutrino_light)

			if mh > self.params.m4:
		 			self.R_nu4_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino5, pdg.neutrino4, const.neutrino_light, const.neutrino_light)
		 			self.R_nu4_gamma = nui_nu_gamma(params, pdg.neutrino5, pdg.neutrino4)
			if mh > 2*self.params.m4:
		 			self.R_nu4_nu4_nu = nuh_nui_nuj_nuk(params, pdg.neutrino5, pdg.neutrino4, pdg.neutrino4, const.neutrino_light)
			if mh > 3*self.params.m4:
		 			self.R_nu4_nu4_nu4 = nuh_nui_nuj_nuk(params, pdg.neutrino5, pdg.neutrino4, pdg.neutrino4, pdg.neutrino4)

			# dileptons -- already contain the Delta L = 2 channel
			if mh > 2*const.Me:
				self.R_nu_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino5, const.neutrino_light, const.electron, const.electron)
				self.R_nu_e_e_SM = nui_nuj_ell1_ell2(params, pdg.neutrino5, const.neutrino_light, const.electron, const.electron, SM=True)
			if mh > const.Me + const.Mmu:
				self.R_nu_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino5, const.neutrino_light, const.electron, const.muon)
			if mh > 2*const.Mmu:
				self.R_nu_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino5, const.neutrino_light, const.muon, const.muon)
			# pseudoscalar -- factor of 2 for delta L=2 channel 
			if mh > const.Me+const.Mcharged_pion:
				self.R_e_pi = 2*nui_l_P(params, pdg.neutrino5, const.electron, const.charged_pion)
			if mh > const.Me+const.charged_kaon:
				self.R_e_K = 2*nui_l_P(params, pdg.neutrino5, const.electron, const.charged_kaon)
			
			# pseudoscalar -- already contain the Delta L = 2 channel
			if mh > const.Mmu+const.Mcharged_pion:
				self.R_mu_pi = 2*nui_l_P(params, pdg.neutrino5, const.muon, const.charged_pion)
			if mh > const.Mmu+const.charged_kaon:
				self.R_mu_K = 2*nui_l_P(params, pdg.neutrino5, const.muon, const.charged_kaon)
			if mh > const.Mneutral_pion:
				self.R_nu_pi = nui_nu_P(params, pdg.neutrino5, const.neutrino_light, const.neutral_pion)
			if mh > const.neutral_eta:
				self.R_nu_eta = nui_nu_P(params, pdg.neutrino5, const.neutrino_light, const.neutral_eta)

			# vector mesons
			if mh > const.Mneutral_rho:
				self.R_nu_rho = nui_nu_V(params, pdg.neutrino5, const.neutrino_light, const.neutral_rho)
		

			# dileptons -- already contain the Delta L = 2 channel
			if mh > params.m4+2*const.Me:
				self.R_nu4_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino5, pdg.neutrino4, const.electron, const.electron)
			if mh > params.m4+const.Me + const.Mmu:
				self.R_nu4_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino5, pdg.neutrino4, const.electron, const.muon)
			if mh > params.m4+2*const.Mmu:
				self.R_nu4_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino5, pdg.neutrino4, const.muon, const.muon)
			if mh > params.m4+const.Mneutral_pion:
				self.R_nu4_pi = nui_nu_P(params, pdg.neutrino5, pdg.neutrino4, const.neutral_pion)
			if mh > params.m4+const.Mneutral_eta:
				self.R_nu4_eta = nui_nu_P(params, pdg.neutrino5, pdg.neutrino4, const.neutral_eta)
			if mh > params.m4+const.Mneutral_rho:
				self.R_nu4_rho = nui_nu_V(params, pdg.neutrino5, pdg.neutrino4, const.neutral_rho)

		self.array_R = [   self.R_nu_nu_nu,
							self.R_nu4_nu_nu,
							self.R_nu4_nu4_nu,
							self.R_nu4_nu4_nu4,
							self.R_nu_e_e,
							self.R_nu_e_mu,
							self.R_nu_mu_mu,
							self.R_e_pi,
							self.R_e_K,
							self.R_mu_pi,
							self.R_mu_K,
							self.R_nu_pi,
							self.R_nu_eta,
							self.R_nu4_e_e,
							self.R_nu4_e_mu,
							self.R_nu4_mu_mu,
							self.R_nu4_pi,
							self.R_nu4_eta,
							self.R_nu_rho,
							self.R_nu4_rho,
							self.R_nu_gamma,
							self.R_nu4_gamma,
							self.R_nu_e_e_SM]			

		##################
		# Neutrino 5
		if self.particle==pdg.neutrino6:
			mh = self.params.m6
			
			# nuSM limit does not match -- FIX ME 
			self.R_nu_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino6, const.neutrino_light, const.neutrino_light, const.neutrino_light)
			self.R_nu_gamma = nui_nu_gamma(params, pdg.neutrino6, const.neutrino_light)

			if mh > self.params.m5:
		 			self.R_nu5_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino5, const.neutrino_light, const.neutrino_light)
		 			self.R_nu5_gamma = nui_nu_gamma(params, pdg.neutrino6, pdg.neutrino5)
			if mh > 2*self.params.m5:
		 			self.R_nu5_nu5_nu = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino5, pdg.neutrino5, const.neutrino_light)
			if mh > 3*self.params.m5:
		 			self.R_nu5_nu5_nu5 = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino5, pdg.neutrino5, pdg.neutrino5)


			if mh > self.params.m4:
		 			self.R_nu4_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino4, const.neutrino_light, const.neutrino_light)
		 			self.R_nu4_gamma = nui_nu_gamma(params, pdg.neutrino6, pdg.neutrino4)
			if mh > 2*self.params.m4:
		 			self.R_nu4_nu4_nu = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino4, pdg.neutrino4, const.neutrino_light)
			if mh > 3*self.params.m4:
		 			self.R_nu4_nu4_nu4 = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino4, pdg.neutrino4, pdg.neutrino4)


		 	###################################3
		 	# FIX ME 
		 	# NEED TO IMPLEMENT MIXED FINAL STATE DECAYS!!!!!!!


			# dileptons -- already contain the Delta L = 2 channel
			if mh > 2*const.Me:
				self.R_nu_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino6, const.neutrino_light, const.electron, const.electron)
				self.R_nu_e_e_SM = nui_nuj_ell1_ell2(params, pdg.neutrino6, const.neutrino_light, const.electron, const.electron, SM=True)
			if mh > const.Me + const.Mmu:
				self.R_nu_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, const.neutrino_light, const.electron, const.muon)
			if mh > 2*const.Mmu:
				self.R_nu_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, const.neutrino_light, const.muon, const.muon)
			# pseudoscalar -- factor of 2 for delta L=2 channel 
			if mh > const.Me+const.Mcharged_pion:
				self.R_e_pi = 2*nui_l_P(params, pdg.neutrino6, const.electron, const.charged_pion)
			if mh > const.Me+const.charged_kaon:
				self.R_e_K = 2*nui_l_P(params, pdg.neutrino6, const.electron, const.charged_kaon)
			
			# pseudoscalar -- already contain the Delta L = 2 channel
			if mh > const.Mmu+const.Mcharged_pion:
				self.R_mu_pi = 2*nui_l_P(params, pdg.neutrino6, const.muon, const.charged_pion)
			if mh > const.Mmu+const.charged_kaon:
				self.R_mu_K = 2*nui_l_P(params, pdg.neutrino6, const.muon, const.charged_kaon)
			if mh > const.Mneutral_pion:
				self.R_nu_pi = nui_nu_P(params, pdg.neutrino6, const.neutrino_light, const.neutral_pion)
			if mh > const.neutral_eta:
				self.R_nu_eta = nui_nu_P(params, pdg.neutrino6, const.neutrino_light, const.neutral_eta)

			# vector mesons
			if mh > const.Mneutral_rho:
				self.R_nu_rho = nui_nu_V(params, pdg.neutrino6, const.neutrino_light, const.neutral_rho)
		

			# dileptons -- already contain the Delta L = 2 channel
			if mh > params.m5+2*const.Me:
				self.R_nu5_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino5, const.electron, const.electron)
			if mh > params.m5+const.Me + const.Mmu:
				self.R_nu5_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino5, const.electron, const.muon)
			if mh > params.m5+2*const.Mmu:
				self.R_nu5_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino5, const.muon, const.muon)
			if mh > params.m5+const.Mneutral_pion:
				self.R_nu5_pi = nui_nu_P(params, pdg.neutrino6, pdg.neutrino5, const.neutral_pion)
			if mh > params.m5+const.Mneutral_eta:
				self.R_nu5_eta = nui_nu_P(params, pdg.neutrino6, pdg.neutrino5, const.neutral_eta)
			if mh > params.m5+const.Mneutral_rho:
				self.R_nu5_rho = nui_nu_V(params, pdg.neutrino6, pdg.neutrino5, const.neutral_rho)

			# dileptons -- already contain the Delta L = 2 channel
			if mh > params.m4+2*const.Me:
				self.R_nu4_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino4, const.electron, const.electron)
			if mh > params.m4+const.Me + const.Mmu:
				self.R_nu4_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino4, const.electron, const.muon)
			if mh > params.m4+2*const.Mmu:
				self.R_nu4_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino4, const.muon, const.muon)
			if mh > params.m4+const.Mneutral_pion:
				self.R_nu4_pi = nui_nu_P(params, pdg.neutrino6, pdg.neutrino4, const.neutral_pion)
			if mh > params.m4+const.Mneutral_eta:
				self.R_nu4_eta = nui_nu_P(params, pdg.neutrino6, pdg.neutrino4, const.neutral_eta)
			if mh > params.m4+const.Mneutral_rho:
				self.R_nu4_rho = nui_nu_V(params, pdg.neutrino6, pdg.neutrino4, const.neutral_rho)

		self.array_R = np.array([   self.R_nu_nu_nu,
							self.R_nu4_nu_nu,
							self.R_nu4_nu4_nu,
							self.R_nu4_nu4_nu4,
							self.R_nu5_nu_nu,
							self.R_nu5_nu5_nu,
							self.R_nu5_nu5_nu5,
							self.R_nu_e_e,
							self.R_nu_e_mu,
							self.R_nu_mu_mu,
							self.R_e_pi,
							self.R_e_K,
							self.R_mu_pi,
							self.R_mu_K,
							self.R_nu_pi,
							self.R_nu_eta,
							self.R_nu4_e_e,
							self.R_nu4_e_mu,
							self.R_nu4_mu_mu,
							self.R_nu4_pi,
							self.R_nu4_eta,
							self.R_nu5_e_e,
							self.R_nu5_e_mu,
							self.R_nu5_mu_mu,
							self.R_nu5_pi,
							self.R_nu5_eta,
							self.R_nu_rho,
							self.R_nu4_rho,
							self.R_nu5_rho,
							self.R_nu_gamma,
							self.R_nu4_gamma,
							self.R_nu5_gamma,
							self.R_nu_e_e_SM])

	def total_rate(self):
		# self.R_total =  self.R_nu_nu_nu+self.R_nu4_nu_nu+self.R_nu4_nu4_nu+self.R_nu4_nu4_nu4+self.R_nu_e_e+self.R_nu_e_mu\
		# 								+self.R_nu_mu_mu+self.R_e_pi+self.R_e_K+self.R_mu_pi+self.R_mu_K+self.R_nu_pi+self.R_nu_eta+self.R_nu4_e_e\
		# 								+self.R_nu4_e_mu+self.R_nu4_mu_mu+self.R_nu4_pi+self.R_nu4_eta+self.R_nu4_rho+self.R_nu_rho+self.R_nu_gamma+self.R_nu4_gamma
		self.R_total =  np.sum(self.array_R[:-1])

		return self.R_total

	def compute_BR(self):
		
		self.BR_nu_nu_nu = self.R_nu_nu_nu/self.R_total								
		self.BR_nu4_nu_nu = self.R_nu4_nu_nu/self.R_total								
		self.BR_nu4_nu4_nu = self.R_nu4_nu4_nu/self.R_total								
		self.BR_nu4_nu4_nu4 = self.R_nu4_nu4_nu4/self.R_total								
		self.BR_nu_e_e = self.R_nu_e_e/self.R_total								
		self.BR_nu_e_mu = self.R_nu_e_mu/self.R_total								
		self.BR_nu_mu_mu = self.R_nu_mu_mu/self.R_total								
		self.BR_e_pi = self.R_e_pi/self.R_total								
		self.BR_e_K = self.R_e_K/self.R_total								
		self.BR_mu_pi = self.R_mu_pi/self.R_total								
		self.BR_mu_K = self.R_mu_K/self.R_total								
		self.BR_nu_pi = self.R_nu_pi/self.R_total								
		self.BR_nu_eta = self.R_nu_eta/self.R_total								
		self.BR_nu4_e_e = self.R_nu4_e_e/self.R_total								
		self.BR_nu4_e_mu = self.R_nu4_e_mu/self.R_total								
		self.BR_nu4_mu_mu = self.R_nu4_mu_mu/self.R_total								
		self.BR_nu4_pi = self.R_nu4_pi/self.R_total								
		self.BR_nu4_eta = self.R_nu4_eta/self.R_total				
		self.BR_nu4_rho = self.R_nu4_rho/self.R_total				
		self.BR_nu_rho = self.R_nu_rho/self.R_total				
		self.BR_nu4_gamma = self.R_nu4_gamma/self.R_total				
		self.BR_nu_gamma = self.R_nu_gamma/self.R_total				
		
		# self.array_BR = [   self.BR_nu_nu_nu,
		# 					self.BR_nu4_nu_nu,
		# 					self.BR_nu4_nu4_nu,
		# 					self.BR_nu4_nu4_nu4,
		# 					self.BR_nu_e_e,
		# 					self.BR_nu_e_mu,
		# 					self.BR_nu_mu_mu,
		# 					self.BR_e_pi,
		# 					self.BR_e_K,
		# 					self.BR_mu_pi,
		# 					self.BR_mu_K,
		# 					self.BR_nu_pi,
		# 					self.BR_nu_eta,
		# 					self.BR_nu4_e_e,
		# 					self.BR_nu4_e_mu,
		# 					self.BR_nu4_mu_mu,
		# 					self.BR_nu4_pi,
		# 					self.BR_nu4_eta,
		# 					self.BR_nu_rho,
		# 					self.BR_nu4_rho,	
		# 					self.BR_nu_gamma,	
		# 					self.BR_nu4_gamma]	

		self.array_BR = self.array_R[:-1]/self.R_total

		return self.array_BR

def nui_nu_gamma(params, initial_neutrino, final_neutrino):

	if (initial_neutrino==pdg.neutrino6):
		mh = params.m6
		if (final_neutrino==const.neutrino_tau):
			CC_mixing = params.Utau6
			mf=0.0
		elif(final_neutrino==const.neutrino_muon):
			CC_mixing = params.Umu6
			mf=0.0
		elif(final_neutrino==const.neutrino_electron):
			CC_mixing = params.Ue6
			mf=0.0
		elif(final_neutrino==const.neutrino_light):
			CC_mixing = np.sqrt(params.Ue6*params.Ue6+params.Umu6*params.Umu6+params.Utau6*params.Utau6)
			mf=0.0
		elif(final_neutrino==pdg.neutrino5):
			CC_mixing = params.c56
			mf=params.m5
		elif(final_neutrino==pdg.neutrino4):
			CC_mixing = params.c46
			mf=params.m4
	
	elif (initial_neutrino==pdg.neutrino5):
		mh = params.m5
		if (final_neutrino==const.neutrino_tau):
			CC_mixing = params.Utau5
			mf=0.0
		elif(final_neutrino==const.neutrino_muon):
			CC_mixing = params.Umu5
			mf=0.0
		elif(final_neutrino==const.neutrino_electron):
			CC_mixing = params.Ue5
			mf=0.0
		elif(final_neutrino==const.neutrino_light):
			CC_mixing = np.sqrt(params.Ue5*params.Ue5+params.Umu5*params.Umu5+params.Utau5*params.Utau5)
			mf=0.0
		elif(final_neutrino==pdg.neutrino4):
			CC_mixing = params.c45
			mf=params.m4
	

	elif(initial_neutrino==pdg.neutrino4):
		mh=params.m4
		mf=0.0
		if (final_neutrino==const.neutrino_tau):
			CC_mixing = params.Utau4
		elif(final_neutrino==const.neutrino_muon):
			CC_mixing = params.Umu4
		elif(final_neutrino==const.neutrino_electron):
			CC_mixing = params.Ue4
		elif(final_neutrino==const.neutrino_light):
			CC_mixing = np.sqrt(params.Ue4*params.Ue4+params.Umu4*params.Umu4+params.Utau4*params.Utau4)
		else:
			print('ERROR! Wrong inital neutrino')

	return (const.Gf*CC_mixing)*(const.Gf*CC_mixing)*mh**5/(192.0*np.pi*np.pi*np.pi)*(27.0*const.alphaQED/32.0/np.pi)*np.sqrt(lam(1.0, mf/mh*mf/mh,0.0))


def nui_l_P(params, initial_neutrino, final_lepton, final_hadron):


	if (initial_neutrino==pdg.neutrino6):
		mh = params.m6
		if (final_lepton==const.tau):
			ml = const.Mtau
			CC_mixing = params.Utau6
		elif(final_lepton==const.muon):
			ml = const.Mmu
			CC_mixing = params.Umu6
		elif(final_lepton==const.electron):
			ml = const.Me
			CC_mixing = params.Ue6

	elif (initial_neutrino==pdg.neutrino5):
		mh = params.m5
		if (final_lepton==const.tau):
			ml = const.Mtau
			CC_mixing = params.Utau5
		elif(final_lepton==const.muon):
			ml = const.Mmu
			CC_mixing = params.Umu5
		elif(final_lepton==const.electron):
			ml = const.Me
			CC_mixing = params.Ue5

	elif(initial_neutrino==pdg.neutrino4):
		mh = params.m4
		if (final_lepton==const.tau):
			ml = const.Mtau
			CC_mixing = params.Utau4
		elif(final_lepton==const.muon):
			ml = const.Mmu
			CC_mixing = params.Umu4
		elif(final_lepton==const.electron):
			ml = const.Me
			CC_mixing = params.Ue4
	else:
			print('ERROR! Wrong inital neutrino')

	if (final_hadron==const.charged_pion):
		mp = const.Mcharged_pion
		Vqq = const.Vud
		fp  = const.Fcharged_pion
	elif(final_hadron==const.charged_kaon):
		mp = const.Mcharged_kaon
		Vqq = const.Vus
		fp  = const.Fcharged_kaon
	# elif(final_hadron==const.charged_rho):
	# 	mp = const.Mcharged_rho
	# 	Vqq = params.Vud

	return (const.Gf*fp*CC_mixing*Vqq)**2 * mh**3/(16*np.pi) * I1_2body((ml/mh)**2, (mp/mh)**2)



def nui_nu_P(params, initial_neutrino, final_neutrino, final_hadron):

	if (initial_neutrino==pdg.neutrino6):
		mh = params.m6
		if (final_neutrino==const.neutrino_tau):
			CC_mixing = params.Utau6
		elif(final_neutrino==const.neutrino_muon):
			CC_mixing = params.Umu6
		elif(final_neutrino==const.neutrino_electron):
			CC_mixing = params.Ue6
		elif(final_neutrino==const.neutrino_light):
			CC_mixing = np.sqrt(params.Ue6*params.Ue6+params.Umu6*params.Umu6+params.Utau6*params.Utau6)
		elif(final_neutrino==pdg.neutrino5):
			CC_mixing = params.c56
		elif(final_neutrino==pdg.neutrino4):
			CC_mixing = params.c46

	elif (initial_neutrino==pdg.neutrino5):
		mh = params.m5
		if (final_neutrino==const.neutrino_tau):
			CC_mixing = params.Utau5
		elif(final_neutrino==const.neutrino_muon):
			CC_mixing = params.Umu5
		elif(final_neutrino==const.neutrino_electron):
			CC_mixing = params.Ue5
		elif(final_neutrino==const.neutrino_light):
			CC_mixing = np.sqrt(params.Ue5*params.Ue5+params.Umu5*params.Umu5+params.Utau5*params.Utau5)
		elif(final_neutrino==pdg.neutrino4):
			CC_mixing = params.c45
	

	elif(initial_neutrino==pdg.neutrino4):
		mh = params.m4
		if (final_neutrino==const.neutrino_tau):
			CC_mixing = params.Utau4
		elif(final_neutrino==const.neutrino_muon):
			CC_mixing = params.Umu4
		elif(final_neutrino==const.neutrino_electron):
			CC_mixing = params.Ue4
		elif(final_neutrino==const.neutrino_light):
			CC_mixing = np.sqrt(params.Ue4*params.Ue4+params.Umu4*params.Umu4+params.Utau4*params.Utau4)
		else:
			print('ERROR! Wrong inital neutrino')
	
	if (final_hadron==const.neutral_pion):
		mp = const.Mneutral_pion
		fp  = const.Fneutral_pion
	elif(final_hadron==const.neutral_eta):
		mp = const.Mneutral_eta
		fp  = const.Fneutral_eta


	return (const.Gf*fp*CC_mixing)**2*mh**3/(64*np.pi)*(1-(mp/mh)**2)**2


def nui_nu_V(params, initial_neutrino, final_neutrino, final_hadron):

	if (initial_neutrino==pdg.neutrino6):
		mh = params.m6
		mix = params.A6
	elif (initial_neutrino==pdg.neutrino5):
		mh = params.m5
		mix = params.A5
	elif(initial_neutrino==pdg.neutrino4):
		mh = params.m4
		mix = params.A4

	if (final_hadron==const.neutral_rho):
		mp = const.Mneutral_rho
		fp  = const.Fneutral_rho
	else:
		print('ERROR! Final state not recognized.')

	rp  = (mp/mh)*(mp/mh)
	bsm = (const.eQED*params.chi*const.cw)**2*params.alphaD*mh*mh*mh*fp*fp*(1-rp)*(1-rp)*(0.5+rp)/4.0/params.Mzprime/params.Mzprime/params.Mzprime/params.Mzprime
	sm  = const.Gf*const.Gf*mh*mh*mh*fp*fp*(1-rp)*(1-rp)*(0.5+rp)/16.0/np.pi
	return (sm+bsm)*mix


###############################
# New containing all terms!
def nui_nuj_ell1_ell2(params, initial_neutrino, final_neutrino, final_lepton1, final_lepton2, SM=False):
	
	################################
	# COUPLINGS

	# Is neutral current possible?
	if final_lepton2==final_lepton1:
		NCflag=1
		if initial_neutrino==pdg.neutrino6:
			mh = params.m6

			# Which outgoing neutrino?
			if final_neutrino==const.neutrino_electron:
				Cih = params.ce6
				Dih = params.de6
				mf = 0.0
			if final_neutrino==const.neutrino_muon:
				Cih = params.cmu6
				Dih = params.dmu6
				mf = 0.0
			if final_neutrino==const.neutrino_tau:
				Cih = params.ctau6
				Dih = params.dtau6
				mf = 0.0
			if final_neutrino==const.neutrino_light:
				Cih = params.clight6
				Dih = params.dlight6
				mf = 0.0
			if final_neutrino==pdg.neutrino5:
				Cih = params.c56*0
				Dih = params.d56*0
				mf = params.m5
			if final_neutrino==pdg.neutrino4:
				Cih = params.c46
				Dih = params.d46
				mf = params.m4

		elif initial_neutrino==pdg.neutrino5:
			mh = params.m5

			# Which outgoing neutrino?
			if final_neutrino==const.neutrino_electron:
				Cih = params.ce5
				Dih = params.de5
				mf = 0.0
			if final_neutrino==const.neutrino_muon:
				Cih = params.cmu5
				Dih = params.dmu5
				mf = 0.0
			if final_neutrino==const.neutrino_tau:
				Cih = params.ctau5
				Dih = params.dtau5
				mf = 0.0
			if final_neutrino==const.neutrino_light:
				Cih = params.clight5
				Dih = params.dlight5
				mf = 0.0
			if final_neutrino==pdg.neutrino4:
				Cih = params.c45
				Dih = params.d45
				mf = params.m4

		elif initial_neutrino==pdg.neutrino4:
			mh = params.m4
			# Which outgoing neutrino?
			if final_neutrino==const.neutrino_electron:
				Cih = params.ce4
				Dih = params.de4
				mf = 0.0
			if final_neutrino==const.neutrino_muon:
				Cih = params.cmu4
				Dih = params.dmu4
				mf = 0.0
			if final_neutrino==const.neutrino_tau:
				Cih = params.ctau4
				Dih = params.dtau4
				mf = 0.0
			if final_neutrino==const.neutrino_light:
				Cih = params.clight4
				Dih = params.dlight4
				mf = 0.0
			if final_neutrino==pdg.neutrino4:
				print('ERROR! (nu4 -> nu4 l l) is kinematically not allowed!')
		
		if final_neutrino/10==final_lepton1:
			# Mixing required for CC N-like
			if (final_lepton1==const.tau):
				CC_mixing1 = params.Utau4
			elif(final_lepton1==const.muon):
				CC_mixing1 = params.Umu4
			elif(final_lepton1==const.electron):
				CC_mixing1 = params.Ue4
			else:
				print("WARNING! Unable to set CC mixing parameter for decay. Assuming 0.")
				CC_mixing1 = 0

			# Mixing required for CC Nbar-like
			if (final_lepton2==const.tau):
				CC_mixing2 = params.Utau4
			elif(final_lepton2==const.muon):
				CC_mixing2 = params.Umu4
			elif(final_lepton2==const.electron):
				CC_mixing2 = params.Ue4
			else:
				print("WARNING! Unable to set CC mixing parameter for decay. Assuming 0.")
				CC_mixing2 = 0
		else:
			CC_mixing1 = 0
			CC_mixing2 = 0

	# Only CC is possible
	# FIX ME!
	# NEED TO INCLUDE ALL HNL MIXINGS
	else:
		NCflag=0
		Cih = 0
		Dih = 0
		if initial_neutrino==pdg.neutrino6:
			mh = params.m6
			# Which outgoing neutrino?
			if final_neutrino==pdg.neutrino5:
				mf = params.m5
				# Which outgoin leptons?
				if final_lepton1==const.electron and final_lepton2==const.muon:
					CC_mixing1 = params.Umu6 * params.Ue5
					CC_mixing2 = params.Ue6 * params.Umu5
				if final_lepton1==const.electron and final_lepton2==const.tau:
					CC_mixing1 = params.Utau6 * params.Ue5
					CC_mixing2 = params.Ue6 * params.Utau5
				if final_lepton1==const.muon and final_lepton2==const.tau:
					CC_mixing1 = params.Umuon6 * params.Utau5
					CC_mixing2 = params.Utau6 * params.Umuon5

			elif final_neutrino==pdg.neutrino4:
				mf = params.m4
				# Which outgoin leptons?
				if final_lepton1==const.electron and final_lepton2==const.muon:
					CC_mixing1 = params.Umu6 * params.Ue4
					CC_mixing2 = params.Ue6 * params.Umu4
				if final_lepton1==const.electron and final_lepton2==const.tau:
					CC_mixing1 = params.Utau6 * params.Ue4
					CC_mixing2 = params.Ue6 * params.Utau4
				if final_lepton1==const.muon and final_lepton2==const.tau:
					CC_mixing1 = params.Umuon6 * params.Utau4
					CC_mixing2 = params.Utau6 * params.Umuon4

					
			elif final_neutrino==const.neutrino_light:
				mf = 0.0
				if final_lepton1==const.electron and final_lepton2==const.muon:
					CC_mixing1 = params.Umu5 
					CC_mixing2 = params.Ue5 
				if final_lepton1==const.electron and final_lepton2==const.tau:
					CC_mixing1 = params.Utau5
					CC_mixing2 = params.Ue5
				if final_lepton1==const.muon and final_lepton2==const.tau:
					CC_mixing1 = params.Umuon5 
					CC_mixing2 = params.Utau5 		

		if initial_neutrino==pdg.neutrino5:
			mh = params.m5
			# Which outgoing neutrino?
			if final_neutrino==pdg.neutrino4:
				mf = params.m4
				# Which outgoin leptons?
				if final_lepton1==const.electron and final_lepton2==const.muon:
					CC_mixing1 = params.Umu5 * params.Ue4
					CC_mixing2 = params.Ue5 * params.Umu4
				if final_lepton1==const.electron and final_lepton2==const.tau:
					CC_mixing1 = params.Utau5 * params.Ue4
					CC_mixing2 = params.Ue5 * params.Utau4
				if final_lepton1==const.muon and final_lepton2==const.tau:
					CC_mixing1 = params.Umuon5 * params.Utau4
					CC_mixing2 = params.Utau5 * params.Umuon4
			if final_neutrino==const.neutrino_light:
				mf = 0.0
				if final_lepton1==const.electron and final_lepton2==const.muon:
					CC_mixing1 = params.Umu5 
					CC_mixing2 = params.Ue5 
				if final_lepton1==const.electron and final_lepton2==const.tau:
					CC_mixing1 = params.Utau5
					CC_mixing2 = params.Ue5
				if final_lepton1==const.muon and final_lepton2==const.tau:
					CC_mixing1 = params.Umuon5 
					CC_mixing2 = params.Utau5 		


		if initial_neutrino==pdg.neutrino4:
			mh = params.m4
			mf = 0.0
			if final_neutrino==const.neutrino_light:
				if final_lepton1==const.electron and final_lepton2==const.muon:
					CC_mixing1 = params.Umu4 
					CC_mixing2 = params.Ue4 
				if final_lepton1==const.electron and final_lepton2==const.tau:
					CC_mixing1 = params.Utau4
					CC_mixing2 = params.Ue4
				if final_lepton1==const.muon and final_lepton2==const.tau:
					CC_mixing1 = params.Umuon4 
					CC_mixing2 = params.Utau4
			else:
				print('ERROR! Unable to find outgoing neutrino.')


	#######################################
	#######################################
	### WATCH OUT FOR THE MINUS SIGN HERE -- IMPORTANT FOR INTERFERENCE
	## Put requires mixings in CCflags
	CCflag1 = CC_mixing1
	CCflag2 = -CC_mixing2

	##############################
	# CHARGED LEPTON MASSES 

	if (final_lepton1==const.tau):
		mm = const.Mtau
	elif(final_lepton1==const.muon):
		mm = const.Mmu
	elif(final_lepton1==const.electron):
		mm = const.Me
	else:
		print("WARNING! Unable to set charged lepton mass. Assuming massless.")
		mm = 0

	if (final_lepton2==const.tau):
		mp = const.Mtau
	elif(final_lepton2==const.muon):
		mp = const.Mmu
	elif(final_lepton2==const.electron):
		mp = const.Me
	else:
		print("WARNING! Unable to set charged lepton mass. Assuming massless.")
		mp = 0


	# couplings and masses LEADING ORDER!
	Cv = params.ceV
	Ca = params.ceA
	Dv = params.deV
	Da = params.deA
	MZBOSON = const.Mz
	MZPRIME = params.Mzprime
	MW = const.Mw

	xZBOSON=MZBOSON/mh
	xZPRIME=MZPRIME/mh
	xWBOSON=MW/mh
	m1=mh
	x2=mf/mh
	x3=mm/mh
	x4=mp/mh

	gweak=const.gweak

	if SM==True:
		Dv=0
		Da=0
		MZPRIME=0
		gprime=0

	def DGammaDuDt(x23,x24,m1,x2,x3,x4,NCflag,CCflag1,CCflag2,Cv,Ca,Dv,Da,Cih,Dih,MZBOSON,MZPRIME,MW):
		pi = np.pi

		# return (u*((-256*(Ca*Ca)*(Cih*Cih)*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (256*(Cih*Cih)*(Cv*Cv)*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (64*(Ca*Ca)*(Cih*Cih)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Cih*Cih)*(Cv*Cv)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (256*(Da*Da)*(Dih*Dih)*mf*mh*mm*mp*(NCflag*NCflag))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (256*(Dih*Dih)*(Dv*Dv)*mf*mh*mm*mp*(NCflag*NCflag))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (64*(Da*Da)*(Dih*Dih)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Dih*Dih)*(Dv*Dv)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Ca*Ca)*(Cih*Cih)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Cih*Cih)*(Cv*Cv)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Da*Da)*(Dih*Dih)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Dih*Dih)*(Dv*Dv)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (512*Ca*Cih*Da*Dih*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (512*Cih*Cv*Dih*Dv*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) + (128*Ca*Cih*Da*Dih*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (128*Cih*Cv*Dih*Dv*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (128*Ca*Cih*Da*Dih*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (128*Cih*Cv*Dih*Dv*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (32*Ca*CCflag1*Cih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (32*CCflag1*Cih*Cv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(MW*MW - u)) - (8*Ca*CCflag1*Cih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) - (32*CCflag1*Da*Dih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (32*CCflag1*Dih*Dv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(MW*MW - u)) - (8*CCflag1*Da*Dih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*Ca*CCflag1*Cih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Da*Dih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (32*(Ca*Ca)*(Cih*Cih)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Cih*Cih)*(Cv*Cv)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Da*Da)*(Dih*Dih)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (32*(Dih*Dih)*(Dv*Dv)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (64*Ca*Cih*Da*Dih*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (64*Cih*Cv*Dih*Dv*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/(2.*((MW*MW - u)*(MW*MW - u))) + (4*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (4*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (32*(Ca*Ca)*(Cih*Cih)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Cih*Cih)*(Cv*Cv)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Da*Da)*(Dih*Dih)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (32*(Dih*Dih)*(Dv*Dv)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (64*Ca*Cih*Da*Dih*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (64*Cih*Cv*Dih*Dv*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/(2.*((MW*MW - u)*(MW*MW - u))) + (4*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (4*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/(2.*((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))) + (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/(2.*((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))) + (32*Ca*CCflag2*Cih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (32*CCflag2*Cih*Cv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*Ca*CCflag2*Cih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Cih*Cv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (32*CCflag2*Da*Dih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (32*CCflag2*Dih*Dv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Da*Dih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Dih*Dv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*Ca*CCflag2*Cih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Cih*Cv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Da*Dih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Dih*Dv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (2*CCflag1*CCflag2*(const.g*const.g*const.g*const.g)*mf*mh*(-(mm*mm) - mp*mp + t))/((MW*MW - u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MW*MW - u)*(MW*MW - u)) + (8*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) - (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))))/(512.*mh*(pi*pi*pi)*((t + u)*(t + u)))	
		return -(CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x2*x2))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) + (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*x24)/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) + (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x2*x2)*x24)/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) - (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x24*x24))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) - (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x2*x2)*(x3*x3))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) + (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*x24*(x3*x3))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) - (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x4*x4))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) + (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*x24*(x4*x4))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) - (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x3*x3)*(x4*x4))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) - (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x2*x2))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) + (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*x23)/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) + (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x2*x2)*x23)/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) - (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x23*x23))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) - (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x3*x3))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) + (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*x23*(x3*x3))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) - (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x2*x2)*(x4*x4))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) + (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*x23*(x4*x4))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) - (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x3*x3)*(x4*x4))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) - (CCflag1*CCflag2*(gweak*gweak*gweak*gweak)*m1*x2)/(256.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON)) - (CCflag1*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x2*x2*x2))/(256.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON)) + (CCflag1*CCflag2*(gweak*gweak*gweak*gweak)*m1*x2*x23)/(256.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON)) + (CCflag1*CCflag2*(gweak*gweak*gweak*gweak)*m1*x2*x24)/(256.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON)) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x2)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x2)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x2*x23)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x2*x23)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2)*x23)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2)*x23)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x2*x24)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x2*x24)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2)*x24)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2)*x24)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x2*x3*x4)/(8.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x2*x3*x4)/(8.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x23)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x23)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x23*x23))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x23*x23))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x24)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x24)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x24*x24))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x24*x24))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x23*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x23*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x24*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x24*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x23*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x23*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x24*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x24*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x23*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x23*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x24*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x24*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x2*x2)*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2)*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x24*x24))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x24*x24))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x2*x2)*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2)*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x24*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x24*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x24*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x24*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x2*x2)*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2)*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x23*x23))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x23*x23))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x23*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x23*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x2*x2)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x23*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x23*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x2)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x2)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x2*x23)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x2*x23)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2)*x23)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2)*x23)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x2*x24)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x2*x24)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2)*x24)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2)*x24)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x2*x3*x4)/(8.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x2*x3*x4)/(8.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x2)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x2)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2*x2))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2*x2))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x2*x23)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x2*x23)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2)*x23)/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2)*x23)/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x2*x24)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x2*x24)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2)*x24)/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2)*x24)/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x3*x3))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x3*x3))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x2*x3*x4)/(4.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x2*x3*x4)/(4.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x4*x4))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x4*x4))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x23)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x23)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x23*x23))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x23*x23))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x24)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x24)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x24*x24))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x24*x24))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x23*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x23*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x24*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x24*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x23*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x23*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x24*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x24*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x23*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x23*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x24*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x24*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2)*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2)*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x24*x24))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x24*x24))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2)*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2)*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x24*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x24*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x24*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x24*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2)*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2)*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x23*x23))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x23*x23))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x23*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x23*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x23*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x23*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x23)/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x23)/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x23*x23))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x23*x23))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x24)/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x24)/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x24*x24))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x24*x24))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x23*(x3*x3))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x23*(x3*x3))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x24*(x3*x3))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x24*(x3*x3))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x23*x3*x4)/(16.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x23*x3*x4)/(16.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x24*x3*x4)/(16.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x24*x3*x4)/(16.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x23*(x4*x4))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x23*(x4*x4))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x24*(x4*x4))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x24*(x4*x4))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))

	def Sqrt(x):
		return np.sqrt(x)

	x23min = lambda x24: x2*x2 + x3*x3 - ((-1 + x24 + x3*x3)*(x2*x2 + x24 - x4*x4))/(2.*x24) - (Sqrt((-1 + x24)*(-1 + x24) - 2*(1 + x24)*(x3*x3) + x3*x3*x3*x3)*Sqrt(x2*x2*x2*x2 + (x24 - x4*x4)*(x24 - x4*x4) - 2*(x2*x2)*(x24 + x4*x4)))/(2.*x24)	
	x23max = lambda x24: x2*x2 + x3*x3 - ((-1 + x24 + x3*x3)*(x2*x2 + x24 - x4*x4))/(2.*x24) + (Sqrt((-1 + x24)*(-1 + x24) - 2*(1 + x24)*(x3*x3) + x3*x3*x3*x3)*Sqrt(x2*x2*x2*x2 + (x24 - x4*x4)*(x24 - x4*x4) - 2*(x2*x2)*(x24 + x4*x4)))/(2.*x24)
	

	integral, error = scipy.integrate.dblquad(	DGammaDuDt,
												(x2+x4)**2,
												(1-x3)**2, 
												x23min,
												x23max,
												args=(mh,x2,x3,x4,NCflag,CCflag1,CCflag2,Cv,Ca,Dv,Da,Cih,Dih,MZBOSON,MZPRIME,MW),\
												epsabs=1.49e-08, epsrel=1.49e-08)

	return integral




###############################
# New containing all terms!
def nuh_nui_nuj_nuk(params, initial_neutrino, final_neutrinoi, final_neutrinoj, final_neutrinok):

	################################
	# COUPLINGS
	# FIX ME
	# NEED TO RE-DO THIS WHOLE NU6 SECTION...
	if initial_neutrino==pdg.neutrino6:
		mh = params.m6
		# Which outgoing neutrino?
		Ah = params.A6
		Dh = params.D6
		
		##############
		# NU5 ->  nu nu nu 
		if final_neutrinoi==const.neutrino_light:

			aBSM = Dh*(params.A4+params.A5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
			bBSM = aBSM
			cBSM = aBSM
			dBSM = aBSM
			eBSM = aBSM
			fBSM = aBSM

			aSM = Ah*(1-Ah)*(1+(1-Ah)*(1-Ah))/6.0
			bSM = aSM
			cSM = aSM
			dSM = aSM
			eSM = aSM
			fSM = aSM

			aINT = -2*Dh *(params.D4+params.D5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
			bINT = aINT
			cINT = aINT
			dINT = aINT
			eINT = aINT
			fINT = aINT

			x2 = 0.0
			x3 = 0.0
			x4 = 0.0

			S=6

		##############
		# NU5 ->  NU4 nu nu 
		elif (final_neutrinoj==const.neutrino_light):

			aSM = params.C45SQR*(2+(1-params.A4-params.A5)*(1-params.A4-params.A5) +2*params.C45SQR - 2*params.A4*params.A5)
			bSM = (params.A4-params.A4*params.A4-params.A5*params.A5)*(params.A5-params.A4*params.A4-params.A5*params.A5)
			cSM = bSM
			dSM = params.C45SQR*((1-params.A4-params.A5)*(1-params.A4-params.A5)+params.C45SQR-params.A4*params.A5 )
			eSM = dSM
			fSM = params.C45SQR*(1-params.A4-params.A5)*(1-params.A4-params.A5)

			aBSM = Dh*params.D4*(params.A4+params.A5)*(params.A4+params.A5)
			bBSM = aBSM
			cBSM = aBSM
			dBSM = aBSM
			eBSM = aBSM
			fBSM = aBSM

			##########
			# ADD INTERFERENCE WHEN YOU CAN
			aINT = 0.0
			bINT = aINT
			cINT = aINT
			dINT = aINT
			eINT = aINT
			fINT = aINT

			x2 = params.m4/mh
			x3 = 0.0
			x4 = 0.0

			S=2*6

		##############
		# NU5 ->  NU4 NU4 nu 
		elif (final_neutrinok==const.neutrino_light):

			aSM = params.C45SQR*(params.A4-params.A4*params.A4-params.A5*params.A5)
			bSM = aSM
			cSM = params.A4*params.A4*(params.A5-params.A4*params.A4-params.A5*params.A5)
			dSM = aSM
			eSM = params.C45SQR*(1-params.A4-params.A5)
			fSM = eSM

			aBSM = Dh*params.D4*params.D4*(params.A4+params.A5)
			bBSM = aBSM
			cBSM = aBSM
			dBSM = aBSM
			eBSM = aBSM
			fBSM = aBSM

			##########
			# ADD INTERFERENCE WHEN YOU CAN
			aINT = 0.0
			bINT = aINT
			cINT = aINT
			dINT = aINT
			eINT = aINT
			fINT = aINT

			x2 = params.m4/mh
			x3 = params.m4/mh
			x4 = 0.0

			S=2*6

		##############
		# NU5 ->  NU4 NU4 NU4 
		elif (final_neutrinok==pdg.neutrino4):

			aSM = params.C45SQR*params.A4*params.A4/6.0
			bSM = aSM
			cSM = aSM
			dSM = aSM
			eSM = aSM
			fSM = aSM

			aBSM = Dh*params.D4*params.D4*params.D4/6.0
			bBSM = aBSM
			cBSM = aBSM
			dBSM = aBSM
			eBSM = aBSM
			fBSM = aBSM

			##########
			# ADD INTERFERENCE WHEN YOU CAN
			aINT = 0.0
			bINT = aINT
			cINT = aINT
			dINT = aINT
			eINT = aINT
			fINT = aINT

			x2 = params.m4/mh
			x3 = params.m4/mh
			x4 = params.m4/mh

			S=6

	################################
	# COUPLINGS
	elif initial_neutrino==pdg.neutrino5:
		mh = params.m5
		# Which outgoing neutrino?
		Ah = params.A5
		Dh = params.D5
		
		##############
		# NU5 ->  nu nu nu 
		if final_neutrinoi==const.neutrino_light:

			aBSM = Dh*(params.A4+params.A5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
			bBSM = aBSM
			cBSM = aBSM
			dBSM = aBSM
			eBSM = aBSM
			fBSM = aBSM

			aSM = Ah*(1-Ah)*(1+(1-Ah)*(1-Ah))/6.0
			bSM = aSM
			cSM = aSM
			dSM = aSM
			eSM = aSM
			fSM = aSM

			aINT = -2*Dh *(params.D4+params.D5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
			bINT = aINT
			cINT = aINT
			dINT = aINT
			eINT = aINT
			fINT = aINT

			x2 = 0.0
			x3 = 0.0
			x4 = 0.0

			S=6

		##############
		# NU5 ->  NU4 nu nu 
		elif (final_neutrinoj==const.neutrino_light):

			aSM = params.C45SQR*(2+(1-params.A4-params.A5)*(1-params.A4-params.A5) +2*params.C45SQR - 2*params.A4*params.A5)
			bSM = (params.A4-params.A4*params.A4-params.A5*params.A5)*(params.A5-params.A4*params.A4-params.A5*params.A5)
			cSM = bSM
			dSM = params.C45SQR*((1-params.A4-params.A5)*(1-params.A4-params.A5)+params.C45SQR-params.A4*params.A5 )
			eSM = dSM
			fSM = params.C45SQR*(1-params.A4-params.A5)*(1-params.A4-params.A5)

			aBSM = Dh*params.D4*(params.A4+params.A5)*(params.A4+params.A5)
			bBSM = aBSM
			cBSM = aBSM
			dBSM = aBSM
			eBSM = aBSM
			fBSM = aBSM

			##########
			# ADD INTERFERENCE WHEN YOU CAN
			aINT = 0.0
			bINT = aINT
			cINT = aINT
			dINT = aINT
			eINT = aINT
			fINT = aINT

			x2 = params.m4/mh
			x3 = 0.0
			x4 = 0.0

			S=2*6

		##############
		# NU5 ->  NU4 NU4 nu 
		elif (final_neutrinok==const.neutrino_light):

			aSM = params.C45SQR*(params.A4-params.A4*params.A4-params.A5*params.A5)
			bSM = aSM
			cSM = params.A4*params.A4*(params.A5-params.A4*params.A4-params.A5*params.A5)
			dSM = aSM
			eSM = params.C45SQR*(1-params.A4-params.A5)
			fSM = eSM

			aBSM = Dh*params.D4*params.D4*(params.A4+params.A5)
			bBSM = aBSM
			cBSM = aBSM
			dBSM = aBSM
			eBSM = aBSM
			fBSM = aBSM

			##########
			# ADD INTERFERENCE WHEN YOU CAN
			aINT = 0.0
			bINT = aINT
			cINT = aINT
			dINT = aINT
			eINT = aINT
			fINT = aINT

			x2 = params.m4/mh
			x3 = params.m4/mh
			x4 = 0.0

			S=2*6

		##############
		# NU5 ->  NU4 NU4 NU4 
		elif (final_neutrinok==pdg.neutrino4):

			aSM = params.C45SQR*params.A4*params.A4/6.0
			bSM = aSM
			cSM = aSM
			dSM = aSM
			eSM = aSM
			fSM = aSM

			aBSM = Dh*params.D4*params.D4*params.D4/6.0
			bBSM = aBSM
			cBSM = aBSM
			dBSM = aBSM
			eBSM = aBSM
			fBSM = aBSM

			##########
			# ADD INTERFERENCE WHEN YOU CAN
			aINT = 0.0
			bINT = aINT
			cINT = aINT
			dINT = aINT
			eINT = aINT
			fINT = aINT

			x2 = params.m4/mh
			x3 = params.m4/mh
			x4 = params.m4/mh

			S=6


	elif initial_neutrino==pdg.neutrino4:
		mh = params.m4
		# Which outgoing neutrino?
		Ah = params.A4
		Dh = params.D4

		##############
		# NU4 ->  nu nu nu 
		if final_neutrinoi==const.neutrino_light:

			aBSM = Dh*(params.A4+params.A5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
			bBSM = aBSM
			cBSM = aBSM
			dBSM = aBSM
			eBSM = aBSM
			fBSM = aBSM

			aSM = Ah*(1-Ah)*(1+(1-Ah)*(1-Ah))/6.0
			bSM = aSM
			cSM = aSM
			dSM = aSM
			eSM = aSM
			fSM = aSM

			aINT = -2*Dh *(params.D4+params.D5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
			bINT = aINT
			cINT = aINT
			dINT = aINT
			eINT = aINT
			fINT = aINT

			x2 = 0.0
			x3 = 0.0
			x4 = 0.0

			S=6

	# couplings and masses 	LEADING ORDER!
	MZBOSON = const.Mz
	MZPRIME = params.Mzprime
	MW = const.Mw
	cw = const.cw
	xZBOSON=MZBOSON/mh
	xZPRIME=MZPRIME/mh
	xWBOSON=MW/mh	
	m1=mh

	gweak=const.gweak
	gprime=params.gprime

	pi = np.pi

	def DGammaDuDt(x23,x24,m1,x2,x3,x4):
		# return (u*((-256*(Ca*Ca)*(Cih*Cih)*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (256*(Cih*Cih)*(Cv*Cv)*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (64*(Ca*Ca)*(Cih*Cih)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Cih*Cih)*(Cv*Cv)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (256*(Da*Da)*(Dih*Dih)*mf*mh*mm*mp*(NCflag*NCflag))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (256*(Dih*Dih)*(Dv*Dv)*mf*mh*mm*mp*(NCflag*NCflag))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (64*(Da*Da)*(Dih*Dih)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Dih*Dih)*(Dv*Dv)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Ca*Ca)*(Cih*Cih)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Cih*Cih)*(Cv*Cv)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Da*Da)*(Dih*Dih)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (64*(Dih*Dih)*(Dv*Dv)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (512*Ca*Cih*Da*Dih*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (512*Cih*Cv*Dih*Dv*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) + (128*Ca*Cih*Da*Dih*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (128*Cih*Cv*Dih*Dv*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (128*Ca*Cih*Da*Dih*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (128*Cih*Cv*Dih*Dv*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (32*Ca*CCflag1*Cih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (32*CCflag1*Cih*Cv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(MW*MW - u)) - (8*Ca*CCflag1*Cih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) - (32*CCflag1*Da*Dih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (32*CCflag1*Dih*Dv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(MW*MW - u)) - (8*CCflag1*Da*Dih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*Ca*CCflag1*Cih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Da*Dih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (32*(Ca*Ca)*(Cih*Cih)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Cih*Cih)*(Cv*Cv)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Da*Da)*(Dih*Dih)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (32*(Dih*Dih)*(Dv*Dv)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (64*Ca*Cih*Da*Dih*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (64*Cih*Cv*Dih*Dv*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/(2.*((MW*MW - u)*(MW*MW - u))) + (4*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (4*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (32*(Ca*Ca)*(Cih*Cih)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Cih*Cih)*(Cv*Cv)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Da*Da)*(Dih*Dih)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) + (32*(Dih*Dih)*(Dv*Dv)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MZPRIME*MZPRIME - t)) - (64*Ca*Cih*Da*Dih*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) - (64*Cih*Cv*Dih*Dv*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(MZPRIME*MZPRIME) + t)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/(2.*((MW*MW - u)*(MW*MW - u))) + (4*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (4*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/(2.*((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))) + (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/(2.*((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))) + (32*Ca*CCflag2*Cih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (32*CCflag2*Cih*Cv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*Ca*CCflag2*Cih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Cih*Cv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (32*CCflag2*Da*Dih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (32*CCflag2*Dih*Dv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Da*Dih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Dih*Dv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*Ca*CCflag2*Cih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Cih*Cv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Da*Dih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Dih*Dv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (2*CCflag1*CCflag2*(const.g*const.g*const.g*const.g)*mf*mh*(-(mm*mm) - mp*mp + t))/((MW*MW - u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MW*MW - u)*(MW*MW - u)) + (8*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(MW*MW - u)) - (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZPRIME*MZPRIME - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))))/(512.*mh*(pi*pi*pi)*((t + u)*(t + u)))	
		x34=1.0-x23-x24+x2*x2+x3*x3+x4*x4
		return (gweak*gweak*gweak*gweak*m1*(-((cSM*(x24*x24 + x3*x3 - x34 - x3*x3*x34 + x34*x34 - 2*x23*x4 + 2*(x3*x3)*x4 + 2*(x4*x4) + x3*x3*(x4*x4) - x34*(x4*x4) - x24*(1 + x3*x3 + x4*x4) + x2*x2*(1 - x24 + 2*(x3*x3) - x34 + 2*x4 + x4*x4) + 2*x2*x3*(1 - x23 + 4*x4 + x4*x4)))/((x23 - xZBOSON*xZBOSON)*(x23 - xZBOSON*xZBOSON))) - (bSM*(x23*x23 - 2*x24*x3 + 2*(x3*x3) - x34 - x3*x3*x34 + x34*x34 + 2*x2*(1 - x24 + 4*x3 + x3*x3)*x4 + x4*x4 + 2*x3*(x4*x4) + x3*x3*(x4*x4) - x34*(x4*x4) - x23*(1 + x3*x3 + x4*x4) + x2*x2*(1 - x23 + 2*x3 + x3*x3 - x34 + 2*(x4*x4))))/((x24 - xZBOSON*xZBOSON)*(x24 - xZBOSON*xZBOSON)) - (aSM*(x23*x23 - x24 + x24*x24 + x3*x3 - x24*(x3*x3) + 2*x3*x4 - 2*x3*x34*x4 + x4*x4 - x24*(x4*x4) + 2*(x3*x3)*(x4*x4) - x23*(1 + x3*x3 + x4*x4) + x2*x2*(2 - x23 - x24 + x3*x3 + 2*x3*x4 + x4*x4) + 2*x2*(x3*x3 - x34 + 4*x3*x4 + x4*x4)))/((x34 - xZBOSON*xZBOSON)*(x34 - xZBOSON*xZBOSON)) + (2*fSM*(-(x24*x3) + x3*x3 - x34 - x3*x3*x34 + x34*x34 - x23*x4 + x3*x4 + x3*x3*x4 - x3*x34*x4 + x4*x4 + x3*(x4*x4) - x34*(x4*x4) + x2*x2*(x3 + x3*x3 - x34 + x4 + x3*x4 + x4*x4) + x2*(-x34 + x4 - x24*x4 + x4*x4 + x3*x3*(1 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x23 - xZBOSON*xZBOSON)*(-x24 + xZBOSON*xZBOSON)) + (2*eSM*(x24*x24 + x2*x2*(1 - x24 + x3 + x3*x3 + x4 + x3*x4) - x24*(1 + x3 + x3*x3 + x4*x4) + x4*(-x23 + x4 + x3*x3*(1 + x4) + x3*(1 - x34 + x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x23 - xZBOSON*xZBOSON)*(-x34 + xZBOSON*xZBOSON)) + (2*dSM*(x23*x23 - x23*(1 + x3*x3 + x4 + x4*x4) + x2*x2*(1 - x23 + x3 + x4 + x3*x4 + x4*x4) + x3*(-x24 + x4*(1 - x34 + x4) + x3*(1 + x4 + x4*x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x24 - xZBOSON*xZBOSON)*(-x34 + xZBOSON*xZBOSON))))/(1024.*(cw*cw*cw*cw)*(pi*pi*pi)) + (gprime*gprime*gprime*gprime*m1*(-((cBSM*(x24*x24 + x3*x3 - x34 - x3*x3*x34 + x34*x34 - 2*x23*x4 + 2*(x3*x3)*x4 + 2*(x4*x4) + x3*x3*(x4*x4) - x34*(x4*x4) - x24*(1 + x3*x3 + x4*x4) + x2*x2*(1 - x24 + 2*(x3*x3) - x34 + 2*x4 + x4*x4) + 2*x2*x3*(1 - x23 + 4*x4 + x4*x4)))/((x23 - xZPRIME*xZPRIME)*(x23 - xZPRIME*xZPRIME))) - (bBSM*(x23*x23 - 2*x24*x3 + 2*(x3*x3) - x34 - x3*x3*x34 + x34*x34 + 2*x2*(1 - x24 + 4*x3 + x3*x3)*x4 + x4*x4 + 2*x3*(x4*x4) + x3*x3*(x4*x4) - x34*(x4*x4) - x23*(1 + x3*x3 + x4*x4) + x2*x2*(1 - x23 + 2*x3 + x3*x3 - x34 + 2*(x4*x4))))/((x24 - xZPRIME*xZPRIME)*(x24 - xZPRIME*xZPRIME)) - (aBSM*(x23*x23 - x24 + x24*x24 + x3*x3 - x24*(x3*x3) + 2*x3*x4 - 2*x3*x34*x4 + x4*x4 - x24*(x4*x4) + 2*(x3*x3)*(x4*x4) - x23*(1 + x3*x3 + x4*x4) + x2*x2*(2 - x23 - x24 + x3*x3 + 2*x3*x4 + x4*x4) + 2*x2*(x3*x3 - x34 + 4*x3*x4 + x4*x4)))/((x34 - xZPRIME*xZPRIME)*(x34 - xZPRIME*xZPRIME)) + (2*fBSM*(-(x24*x3) + x3*x3 - x34 - x3*x3*x34 + x34*x34 - x23*x4 + x3*x4 + x3*x3*x4 - x3*x34*x4 + x4*x4 + x3*(x4*x4) - x34*(x4*x4) + x2*x2*(x3 + x3*x3 - x34 + x4 + x3*x4 + x4*x4) + x2*(-x34 + x4 - x24*x4 + x4*x4 + x3*x3*(1 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x23 - xZPRIME*xZPRIME)*(-x24 + xZPRIME*xZPRIME)) + (2*eBSM*(x24*x24 + x2*x2*(1 - x24 + x3 + x3*x3 + x4 + x3*x4) - x24*(1 + x3 + x3*x3 + x4*x4) + x4*(-x23 + x4 + x3*x3*(1 + x4) + x3*(1 - x34 + x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x23 - xZPRIME*xZPRIME)*(-x34 + xZPRIME*xZPRIME)) + (2*dBSM*(x23*x23 - x23*(1 + x3*x3 + x4 + x4*x4) + x2*x2*(1 - x23 + x3 + x4 + x3*x4 + x4*x4) + x3*(-x24 + x4*(1 - x34 + x4) + x3*(1 + x4 + x4*x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x24 - xZPRIME*xZPRIME)*(-x34 + xZPRIME*xZPRIME))))/(64.*(pi*pi*pi)) - (gprime*gprime*(gweak*gweak)*m1*(aINT*(x23*x23 - x24 + x24*x24 + x3*x3 - x24*(x3*x3) + 2*x3*x4 - 2*x3*x34*x4 + x4*x4 - x24*(x4*x4) + 2*(x3*x3)*(x4*x4) - x23*(1 + x3*x3 + x4*x4) + x2*x2*(2 - x23 - x24 + x3*x3 + 2*x3*x4 + x4*x4) + 2*x2*(x3*x3 - x34 + 4*x3*x4 + x4*x4))*(x23 - xZBOSON*xZBOSON)*(x24 - xZBOSON*xZBOSON)*(x23 - xZPRIME*xZPRIME)*(x24 - xZPRIME*xZPRIME) - bINT*(-(x23*x23) + 2*x24*x3 - 2*(x3*x3) + x34 + x3*x3*x34 - x34*x34 + 2*x2*(-1 + x24 - 4*x3 - x3*x3)*x4 - x4*x4 - 2*x3*(x4*x4) - x3*x3*(x4*x4) + x34*(x4*x4) + x2*x2*(-1 + x23 - 2*x3 - x3*x3 + x34 - 2*(x4*x4)) + x23*(1 + x3*x3 + x4*x4))*(x23 - xZBOSON*xZBOSON)*(x34 - xZBOSON*xZBOSON)*(x23 - xZPRIME*xZPRIME)*(x34 - xZPRIME*xZPRIME) - cINT*(-(x24*x24) - x3*x3 + x34 + x3*x3*x34 - x34*x34 + 2*x23*x4 - 2*(x3*x3)*x4 - 2*(x4*x4) - x3*x3*(x4*x4) + x34*(x4*x4) + 2*x2*x3*(-1 + x23 - 4*x4 - x4*x4) + x2*x2*(-1 + x24 - 2*(x3*x3) + x34 - 2*x4 - x4*x4) + x24*(1 + x3*x3 + x4*x4))*(x24 - xZBOSON*xZBOSON)*(x34 - xZBOSON*xZBOSON)*(x24 - xZPRIME*xZPRIME)*(x34 - xZPRIME*xZPRIME) - fINT*(-(x24*x3) + x3*x3 - x34 - x3*x3*x34 + x34*x34 - x23*x4 + x3*x4 + x3*x3*x4 - x3*x34*x4 + x4*x4 + x3*(x4*x4) - x34*(x4*x4) + x2*x2*(x3 + x3*x3 - x34 + x4 + x3*x4 + x4*x4) + x2*(-x34 + x4 - x24*x4 + x4*x4 + x3*x3*(1 + x4) + x3*(1 - x23 + 4*x4 + x4*x4)))*(x34 - xZBOSON*xZBOSON)*(x34 - xZPRIME*xZPRIME)*(-2*(xZBOSON*xZBOSON)*(xZPRIME*xZPRIME) + x24*(xZBOSON*xZBOSON + xZPRIME*xZPRIME) + x23*(-2*x24 + xZBOSON*xZBOSON + xZPRIME*xZPRIME)) - eINT*(x24*x24 + x2*x2*(1 - x24 + x3 + x3*x3 + x4 + x3*x4) - x24*(1 + x3 + x3*x3 + x4*x4) + x4*(-x23 + x4 + x3*x3*(1 + x4) + x3*(1 - x34 + x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4)))*(x24*x24 + xZBOSON*xZBOSON*(xZPRIME*xZPRIME) - x24*(xZBOSON*xZBOSON + xZPRIME*xZPRIME))*(-2*(xZBOSON*xZBOSON)*(xZPRIME*xZPRIME) + x34*(xZBOSON*xZBOSON + xZPRIME*xZPRIME) + x23*(-2*x34 + xZBOSON*xZBOSON + xZPRIME*xZPRIME)) - dINT*(x23*x23 - x23*(1 + x3*x3 + x4 + x4*x4) + x2*x2*(1 - x23 + x3 + x4 + x3*x4 + x4*x4) + x3*(-x24 + x4*(1 - x34 + x4) + x3*(1 + x4 + x4*x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4)))*(x23*x23 + xZBOSON*xZBOSON*(xZPRIME*xZPRIME) - x23*(xZBOSON*xZBOSON + xZPRIME*xZPRIME))*(-2*(xZBOSON*xZBOSON)*(xZPRIME*xZPRIME) + x34*(xZBOSON*xZBOSON + xZPRIME*xZPRIME) + x24*(-2*x34 + xZBOSON*xZBOSON + xZPRIME*xZPRIME))))/(128.*(cw*cw)*(pi*pi*pi)*(x23 - xZBOSON*xZBOSON)*(-x24 + xZBOSON*xZBOSON)*(-x34 + xZBOSON*xZBOSON)*(x23 - xZPRIME*xZPRIME)*(x24 - xZPRIME*xZPRIME)*(x34 - xZPRIME*xZPRIME))
	def Sqrt(x):
		return np.sqrt(x)

	x23min = lambda x24: x2*x2 + x3*x3 - ((-1 + x24 + x3*x3)*(x2*x2 + x24 - x4*x4))/(2.*x24) - (Sqrt((-1 + x24)*(-1 + x24) - 2*(1 + x24)*(x3*x3) + x3*x3*x3*x3)*Sqrt(x2*x2*x2*x2 + (x24 - x4*x4)*(x24 - x4*x4) - 2*(x2*x2)*(x24 + x4*x4)))/(2.*x24)	
	x23max = lambda x24: x2*x2 + x3*x3 - ((-1 + x24 + x3*x3)*(x2*x2 + x24 - x4*x4))/(2.*x24) + (Sqrt((-1 + x24)*(-1 + x24) - 2*(1 + x24)*(x3*x3) + x3*x3*x3*x3)*Sqrt(x2*x2*x2*x2 + (x24 - x4*x4)*(x24 - x4*x4) - 2*(x2*x2)*(x24 + x4*x4)))/(2.*x24)
	

	integral, error = scipy.integrate.dblquad(	DGammaDuDt,
												(x2+x4)**2,
												(1-x3)**2, 
												x23min,
												x23max,
												args=(mh,x2,x3,x4),\
												epsabs=1.49e-08, epsrel=1.49e-08)

	return integral/S


def nu4_to_nualpha_l_l(params, final_lepton):
	if (final_lepton==const.tau):
		m_ell = const.Mtau
	elif(final_lepton==const.muon):
		m_ell = const.Mmu
	elif(final_lepton==const.electron):
		m_ell = const.Me
	else:
		print("WARNING! Unable to set charged lepton mass. Assuming massless.")
		m_ell = 0

	if (final_lepton==const.tau):
		CC_mixing = params.Utau4
	elif(final_lepton==const.muon):
		CC_mixing = params.Umu4
	elif(final_lepton==const.electron):
		CC_mixing = params.Ue4
	else:
		print("WARNING! Unable to set CC mixing parameter for decay. Assuming 0.")
		CC_mixing = 0

	mi = params.m4
	m0 = 0.0
	def func(u,t):
		gv = (const.g/const.cw)**2/2.0 *( params.cmu4*params.ceV/const.Mz**2 - params.dmu4*params.deV/(t-params.Mzprime**2) ) \
						- const.g**2/4.0*CC_mixing/const.Mw**2
		ga = (const.g/const.cw)**2/2.0 *(-params.cmu4*params.ceA/const.Mz**2 + params.dmu4*params.deA/(t-params.Mzprime**2) ) \
						+ const.g**2/4.0*CC_mixing/const.Mw**2
		# print "gv, ga: ", gv, ga
		return 4.0*((gv + ga)**2 *(mi**2 + m_ell**2 - u)*(u - m0**2 -m_ell**2)
						+ (gv - ga)**2*(mi**2 - m0**2 - m_ell**2)*(mi**2 + m_ell**2 - mi**2)
							+ (gv**2 - ga**2)*m_ell**2/2.0*(mi**2 + m0**2 - t))
	
	uminus = lambda t: (mi**2 - m0**2)**2/4.0/t - t/4.0*(np.sqrt(lam(1, mi**2/t, m0**2/t)) + np.sqrt(lam(1,m_ell**2/t, m_ell**2/t)))**2
	uplus = lambda t: (mi**2 - m0**2)**2/4.0/t - t/4.0*(np.sqrt(lam(1, mi**2/t, m0**2/t)) - np.sqrt(lam(1,m_ell**2/t, m_ell**2/t)))**2

	integral, error = scipy.integrate.dblquad(	func,
												(mi-m0)**2, 
												4*m_ell**2,
												uplus,
												uminus,
												args=(), epsabs=1.49e-08, epsrel=1.49e-08)

	return integral*1.0/(2.0*np.pi)**3 / 32.0 / mi**3


############### HEAVY NEUTRINO ############################
def N_to_Z_nu(params):
	Mn = params.m4
	return params.alphaD/2.0 * params.UD4**2 * (params.Ue4**2 + params.Umu4**2 + params.Utau4**2) * Mn**3/params.Mzprime**2 *(1.0 - params.Mzprime**2/Mn**2)*(1 + params.Mzprime**2/Mn**2 - 2.0 * params.Mzprime**4/Mn**4) * ( 1 + params.Dirac*(-1/2.0) )

def N_total(params):
	return N_to_Z_nu(params)


############### Z PRIME ############################
def Z_to_ll(params, ml):
	if 2*ml < params.Mzprime:
		### includes lepton mass effects
		return (const.alphaQED*(params.epsilon*params.epsilon)*np.sqrt(-4.0*(ml*ml) + params.Mzprime*params.Mzprime)*(5*(ml*ml) + 2*(params.Mzprime*params.Mzprime)))/(6.*(params.Mzprime*params.Mzprime))
	elif 2*ml >= params.Mzprime:
		return 0.0

def Z_to_nunu(params):
	return params.alphaD/3.0 * (params.Ue4**2 + params.Umu4**2 + params.Utau4**2)**2 * params.Mzprime

def Z_total(params):
	return Z_to_nunu(params) + Z_to_ll(params, const.Me) + Z_to_ll(params, const.Mmu)

