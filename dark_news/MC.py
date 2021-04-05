import numpy as np
import vegas as vg
import gvar as gv

import random

from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D
#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv
from . import C_MC

from . import fourvec
from . import fluxes
from . import decay_rates
from . import const 
from . import model 
from . import integrands
from . import pdg 

# Integration parameters
NINT = 100
NEVAL = 1e5

NINT_warmup = 10
NEVAL_warmup = 1000

def Power(x,n):
	return x**n
def lam(a,b,c):
	return a**2 + b**2 + c**2 -2*a*b - 2*b*c - 2*a*c


class MC_events:
	def __init__(self,
				params, 
				exp, 
				datafile=None, 
				MA=1*const.MAVG, 
				Z=6, 
				HNLtype= const.MAJORANA, 
				nu_scatterer=pdg.numu,
				nu_produced=pdg.neutrino4, 
				nu_outgoing=pdg.numu, 
				final_lepton=pdg.electron,
				h_upscattered=-1):

		self.params = params

		if nu_produced ==pdg.neutrino4:
			self.Mn = params.m4
		elif nu_produced ==pdg.neutrino5:
			self.Mn = params.m5
		elif nu_produced ==pdg.neutrino6:
			self.Mn = params.m6
		
		if nu_outgoing ==pdg.neutrino5:
			self.Mn_outgoing = params.m5
		elif nu_outgoing ==pdg.neutrino4:
			self.Mn_outgoing = params.m4
		elif nu_outgoing ==pdg.nutau:
			self.Mn_outgoing = 0.0
		elif nu_outgoing ==pdg.numu:
			self.Mn_outgoing = 0.0
		elif nu_outgoing ==pdg.nue:
			self.Mn_outgoing = 0.0
		else:
			print("ERROR! Unable to set intermediate neutrino mass.")

		# Get the experiment active neutrino flux from file
		# self.flux, self.EMIN, self.EMAX, self.DET_SIZE = fluxes.get_exp_params(fluxfile, flavour=nu_scatterer)
		
		#########################################3
		# Some experimental definitions
		self.exp = exp
		self.flux = exp.get_flux_func(flavour=nu_scatterer)
		self.EMIN = exp.EMIN
		self.EMAX = exp.EMAX

		if (self.EMIN < 1.05*(self.Mn**2/2.0/MA + self.Mn)):
			self.EMIN = 1.05*(self.Mn**2/2.0/MA + self.Mn)


		# set target properties
		self.MA = MA
		self.Z = Z
		self.final_lepton = final_lepton
		self.nu_produced = nu_produced
		self.nu_outgoing = nu_outgoing

		# Needed to compute the CC contribution
		# set final state charged lepton mass and BSM parameters
		if (final_lepton==pdg.tau):
			self.m_ell = const.Mtau
		elif(final_lepton==pdg.muon):
			self.m_ell = const.Mmu
		elif(final_lepton==pdg.electron):
			self.m_ell = const.Me
		else:
			print("WARNING! Unable to set charged lepton mass. Assuming massless.")
			self.m_ell = 0


		###############################
		# set helicity here
		self.h_upscattered = h_upscattered

		###############################
		# Set hierarchy here
		self.hierarchy = params.hierarchy
		# if self.hierarchy == const.HM: # else, very simple expressions can be used.
			#############################
			# DECAY PROPERTIES
			# self.decay_prop = decay_rates.HeavyNu(params,self.nu_produced)
			# self.decay_prop.compute_rates() # -- uses quad integrate -- slow!!
			# self.decay_prop.total_rate()
			# self.decay_prop.compute_BR()
		
		# Dirac or Majorana
		self.HNLtype = HNLtype


	def get_MC_events(self):

		params = self.params
		Mn = self.Mn
		Mn_outgoing = self.Mn_outgoing
		Mzprime = params.Mzprime
		MA = self.MA
		m_ell = self.m_ell
		flux = self.flux
		final_lepton=self.final_lepton
		prod=self.final_lepton


		if self.hierarchy == const.LM:
			#########################################################################
			# BATCH SAMPLE INTEGRAN OF INTEREST
			DIM =3
			if params.scan:
				DIM += params.number_of_scanned_params

			batch_f = integrands.cascade(dim=DIM, Emin=self.EMIN, Emax=self.EMAX, MC_case=self)
			integ = vg.Integrator(DIM*[[0.0, 1.0]])

		elif self.hierarchy == const.HM:
			#########################################################################
			# BATCH SAMPLE INTEGRAN OF INTEREST
			DIM = 6
			if params.scan:
				DIM += params.number_of_scanned_params
				
			batch_f = integrands.threebody(dim=DIM, Emin=self.EMIN, Emax=self.EMAX, MC_case=self)
			integ = vg.Integrator(DIM*[[0.0, 1.0]])
		
		##########################################################################
		# COMPUTE TOTAL INTEGRAL
		# Sample the integrand to adapt integrator
		# nstrat = DIM*[1]
		integ(batch_f, nitn=NINT_warmup, neval=NEVAL_warmup)
		# if params.scan:
		# 	nstrat = integ.nstrat
		# 	nstrat[-1] = 1
		# 	nstrat[-2] = 1
		# 	warmup = integ(batch_f, nitn=NINT_warmup, neval=NEVAL_warmup, nstrat=nstrat)

		# Sample again, now saving result
		result = integ(batch_f, nitn=NINT, neval=NEVAL)
		# if params.scan:
		# 	nstrat = integ.nstrat
		# 	nstrat[:-2] = int((NEVAL)**(1/(DIM-2)))
		# 	nstrat[-1] = 1
		# 	nstrat[-2] = 1
		# 	result = integ(batch_f, nitn=NINT, neval=NEVAL, nstrat=nstrat)

		#########################
		# Get the int variables and weights
		SAMPLES,weights = C_MC.get_samples(DIM, integ, batch_f)
		
		if params.hierarchy == const.LM:
			dic = integrands.cascade_phase_space(samples=SAMPLES, MC_case=self)
			decay_rates=result['decay rate N'] # the Zprime width is trivially accounted for
		
		elif params.hierarchy == const.HM:			
			dic = integrands.three_body_phase_space(samples=SAMPLES, MC_case=self)
			decay_rates=result['decay rate N']
            
			weights['full integrand'] = weights['full integrand']/dic['mzprime_scan']**8 * dic['m4_scan']**5
			weights['decay rate N'] = weights['decay rate N']/dic['mzprime_scan']**4 * dic['m4_scan']**5

		integral = result['full integrand']

		# Append the two kinds of weights and the total integral
		dic['w'] = weights['full integrand']
		dic['w_decay'] = weights['decay rate N']
		dic['I'] =  np.sum(weights['full integrand'])
		dic['I_decay'] = np.sum(weights['decay rate N'])

		# print(integ.random_batch())
		# for x, wgt, hcube in integ.random_batch(yield_hcube=True):
		# 	print(np.shape(x))
		# 	wgt_fx = wgt * batch_f(x)['full integrand']

		# 	for i in range(hcube[0], hcube[-1] + 1):
		# 		idx = (hcube == i)          # select array items for h-cube i
		# 		nwf = np.sum(idx)           # number of points in h-cube i
				# print(nwf)
				# print(np.shape(wgt_fx[idx]))
		# print("testing each case:    I=", np.sum(weights['full integrand']), integral, "Gamma = ", np.sum(weights['decay rate N']), decay_rates)

		return dic


def Combine_MC_output(cases, Ifactors=None, flags=None):
    
	# merged dic
	dic ={}
	
	# initialize with first case
	for x in cases[0]:
		if (x=='w' or x=='w_decay'):
			dic[x] = cases[0][x]*Ifactors[0]
		elif (x=='I' or x=='I_decay'):
			dic[x] = cases[0][x]*Ifactors[0]
		else:
			dic[x]= cases[0][x]
		
	dic['flags'] = np.ones(np.shape(cases[0]['w'])[0])*flags[0]


	# append all subsequent ones
	for i in range(1,np.shape(cases)[0]):
		for x in cases[0]:
			if (x=='w' or x=='w_decay'):
				dic[x] = np.array( np.append(dic[x], cases[i][x]*Ifactors[i], axis=0) )
			elif (x=='I' or x=='I_decay'):
				dic[x] = dic[x] + cases[i][x]*Ifactors[i]
			else:
				dic[x]=np.array( np.append(dic[x], cases[i][x], axis=0) )
			
		dic['flags'] = np.append(dic['flags'], np.ones(np.shape(cases[i]['w'])[0])*flags[i] )

	dic['w_decay'] /= np.size(cases)
	dic['I_decay'] /= np.size(cases)

	return dic

# THIS FUNCTION NEEDS SOME OPTIMIZING... currently setting event flags by hand.
def run_MC(BSMparams, exp, FLAVOURS, INCLUDE_HC=True, INCLUDE_HF=False, INCLUDE_COH=True, INCLUDE_DIF=False):
	cases = []
	flags = []
	for flavour in FLAVOURS:
		for i in range(np.size(exp.MATERIALS_A)):

			# include helicity conserving scattering
			if INCLUDE_HC:
				if INCLUDE_DIF:
					cases.append(MC_events(BSMparams,
											exp,
											None,
											MA=const.MAVG,
											Z=1,
											nu_scatterer=flavour,
											nu_produced=pdg.neutrino4,
											nu_outgoing=pdg.numu,
											final_lepton=pdg.electron,
											h_upscattered=-1))
					flags.append(const.DIFLH)
				if INCLUDE_COH:
					cases.append(MC_events(BSMparams,
											exp,
											None,
											MA=exp.MATERIALS_A[i]*const.MAVG,
											Z=exp.MATERIALS_Z[i],
											nu_scatterer=flavour,
											nu_produced=pdg.neutrino4,
											nu_outgoing=pdg.numu,
											final_lepton=pdg.electron,
											h_upscattered=-1))
					flags.append(const.COHLH)

			# include helicity flipping scattering
			if INCLUDE_HF:
				if INCLUDE_DIF:
					cases.append(MC_events(BSMparams,
											exp,
											None,
											MA=const.MAVG,
											Z=1,
											nu_scatterer=flavour,
											nu_produced=pdg.neutrino4,
											nu_outgoing=pdg.numu,
											final_lepton=pdg.electron,
											h_upscattered=+1))
					flags.append(const.DIFRH)
				if INCLUDE_COH:
					cases.append(MC_events(BSMparams,
											exp,
											None,
											MA=exp.MATERIALS_A[i]*const.MAVG,
											Z=exp.MATERIALS_Z[i],
											nu_scatterer=flavour,
											nu_produced=pdg.neutrino4,
											nu_outgoing=pdg.numu,
											final_lepton=pdg.electron,
											h_upscattered=+1))
					flags.append(const.COHRH)

	cases_events = [cases[i].get_MC_events() for i in range(np.size(cases))]
	
	Ifactors = np.ones((np.size(cases)))

	# Combine all cases into one object
	all_events = Combine_MC_output(cases_events, Ifactors=Ifactors, flags=flags)

	return all_events