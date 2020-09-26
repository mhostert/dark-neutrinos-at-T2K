import numpy as np
import vegas as vg
import gvar as gv

import random

from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

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
from . import hist_plot
from . import fluxes
from . import decay_rates
from . import const 
from . import model 
from . import integrands
from . import pdg 

# Integration parameters
NINT = 100
NEVAL = 1e4

NINT_warmup = 10
NEVAL_warmup = 1000

def Power(x,n):
	return x**n
def lam(a,b,c):
	return a**2 + b**2 + c**2 -2*a*b - 2*b*c - 2*a*c


class MC_events:
	def __init__(self,
				params, 
				fluxfile='None', 
				datafile=None, 
				MA=1*const.MAVG, 
				Z=6, 
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
		self.flux, self.EMIN, self.EMAX, self.DET_SIZE = fluxes.get_exp_params(fluxfile, flavour=nu_scatterer)

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

		#############################
		# DECAY PROPERTIES
		self.decay_prop = decay_rates.HeavyNu(params,self.nu_produced)
		self.decay_prop.compute_rates()
		self.decay_prop.total_rate()
		self.decay_prop.compute_BR()
		# print(self.decay_prop.array_R)

		###############################
		# set helicity here
		self.h_upscattered = h_upscattered


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


		if Mn - Mn_outgoing > Mzprime:

			print("M_4 = %s GeV\nM_5 = %s GeV\nm_zprime = %s GeV\nm_Had = %s GeV\nUsing cascade of 2-body decays."%(params.m4,params.m5,params.Mzprime,MA))

			#########################################################################
			# BATCH SAMPLE INTEGRAN OF INTEREST
			DIM =3
			batch_f = integrands.cascade(dim=DIM, Emin=self.EMIN, Emax=self.EMAX, MC_case=self)
			integ = vg.Integrator(DIM*[[0.0, 1.0]])
			##########################################################################
			# COMPUTE TOTAL INTEGRAL
			# Sample the integrand to adapt integrator
			integ(batch_f, nitn=NINT_warmup, neval=NEVAL_warmup)

			# Sample again, now saving result
			result = integ(batch_f,  nitn = NINT, neval = NEVAL, minimize_mem = False)
			
			# final integral
			integral = result.mean/self.decay_prop.total_rate()/decay_rates.Z_total(params)
			##########################################################################
		elif Mn - Mn_outgoing < Mzprime:

			#########################################################################
			# BATCH SAMPLE INTEGRAN OF INTEREST
			DIM = 6
			batch_f = integrands.threebody(dim=DIM, Emin=self.EMIN, Emax=self.EMAX, MC_case=self)
			integ = vg.Integrator(DIM*[[0.0, 1.0]])
			##########################################################################
			# COMPUTE TOTAL INTEGRAL
			# Sample the integrand to adapt integrator
			integ(batch_f, nitn=NINT_warmup, neval=NEVAL_warmup)

			# Sample again, now saving result
			result = integ(batch_f,  nitn = NINT, neval = NEVAL, minimize_mem = False)
			
			# final integral
			####################################################################
			# print("integral:", result.mean, ", decay rate: ", self.decay_prop.total_rate(), ", rates: ",self.decay_prop.array_R[:-1])
			integral = result.mean/self.decay_prop.total_rate()
			####################################################################

		#########################
		# Get the int variables and weights
		SAMPLES,weights = C_MC.get_samples(DIM, integ, batch_f)

		if Mn - Mn_outgoing > Mzprime:
			weights *= 1.0/self.decay_prop.total_rate()/decay_rates.Z_total(params)
			return integrands.cascade_phase_space(samples=SAMPLES, MC_case=self, w=weights*const.GeV2_to_cm2, I=result.mean*const.GeV2_to_cm2)
		elif Mn - Mn_outgoing < Mzprime:
			weights *= 1.0/self.decay_prop.total_rate()
			return integrands.three_body_phase_space(samples=SAMPLES, MC_case=self, w=weights*const.GeV2_to_cm2, I=result.mean*const.GeV2_to_cm2)



def Combine_MC_output(cases, Ifactors=None, flags=None):
    
	# merged dic
	dic ={}

	
	# initialize with first case
	for x in cases[0]:
		if x=='w':
			dic[x] = cases[0][x]*Ifactors[0]
		elif x=='I':
			dic[x]	= cases[0][x]*Ifactors[0]
		else:
			dic[x]= cases[0][x]
		
	dic['flags'] = np.ones(np.shape(cases[0]['w'])[0])*flags[0]


	# append all subsequent ones
	for i in range(1,np.shape(cases)[0]):
		for x in cases[0]:
			if x=='w':
				dic[x] = np.array( np.append(dic[x], cases[i][x]*Ifactors[i], axis=0) )
			elif x=='I':
				dic[x]	= dic[x] + cases[i][x]*Ifactors[i]
			else:
				dic[x]=np.array( np.append(dic[x], cases[i][x], axis=0) )
			
		dic['flags'] = np.append(dic['flags'], np.ones(np.shape(cases[i]['w'])[0])*flags[i] )

	return dic
