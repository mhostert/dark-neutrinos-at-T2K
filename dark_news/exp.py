import numpy as np
import os
import sys, getopt
from scipy import interpolate

from . import const
from . import pdg



# experiment 
CHARMII     = "charmii"
MINERVA_LE  = "minerva_le"
MINERVA_ME  = "minerva_me"
MINIBOONE   = "miniboone"
uBOONE      = "uboone"
ND280_nu    = "nd280_nu"
ND280_nubar = "nd280_nubar"


class experiment():

	def __init__(self,EXP_FLAG):

		self.EXP_FLAG = EXP_FLAG

		#########################
		# Experimental Parameters
		if self.EXP_FLAG == MINIBOONE:
			self.FLUXFILE="fluxes/MiniBooNE_nu_mode_flux.dat"
			self.FLUX_NORM=1.0/0.05
			self.EMIN = 0.1
			self.EMAX = 9
			# Detector targets
			self.Nucleons_per_target = 14.0
			self.P_per_target = 8.0
			self.TARGETS = (818e6) * const.NAvo / self.Nucleons_per_target
			self.POTS = 18.75e20
			self.MATERIALS_A = [12.0]
			self.MATERIALS_Z = [6.0]
		
		elif self.EXP_FLAG == uBOONE:
			self.FLUXFILE="fluxes/MiniBooNE_nu_mode_flux.dat"
			self.FLUX_NORM=1.0/0.05*(541.0/463.0)**2
			self.EMIN = 0.1
			self.EMAX = 9
			# Detector targets
			self.Nucleons_per_target = 40.0
			self.P_per_target = 18.0
			self.TARGETS = 60.0e6 * const.NAvo / self.Nucleons_per_target
			self.POTS = 13.2e20 
			self.A_NUMBER = 40.0
			self.Z_NUMBER = 18.0
		
		elif self.EXP_FLAG == MINERVA_LE:
			self.FLUXFILE="fluxes/MINERVA_LE_numu_flux.dat"
			self.FLUX_NORM=(1e-4)*(1e-6)*2
			self.EMIN = 0.1
			self.EMAX = 19
			# Detector targets
			self.Nucleons_per_target = 13.0
			self.P_per_target = 7.0
			self.TARGETS = 6.10e6*const.NAvo / self.Nucleons_per_target
			self.POTS = 3.43e20*0.73
			self.A_NUMBER = 12.0
			self.Z_NUMBER = 6.0
		
		elif self.EXP_FLAG == MINERVA_ME:
			self.FLUXFILE="fluxes/MINERVA_ME_numu_flux.dat"
			self.FLUX_NORM=1e-4			
			self.EMIN = 0.10
			self.EMAX = 19.0
			self.DET_SIZE = 2.5/2.0 # meters
			# Detector targets
			self.Nucleons_per_target = 13.0
			self.P_per_target = 7.0
			self.TARGETS = 6.10e6*const.NAvo / self.Nucleons_per_target
			self.POTS = 1.16e21*0.73
			self.A_NUMBER = 12.0
			self.Z_NUMBER = 6.0

		elif self.EXP_FLAG == CHARMII:
			self.FLUXFILE="fluxes/CHARMII.dat"
			self.FLUX_NORM=1.0/(370)**2 *1e-13
			self.EMIN = 1.5
			self.EMAX = 198
			self.DET_SIZE = 35.67/2.0 # meters
			# Detector targets
			self.Nucleons_per_target = 20.7
			self.P_per_target = 11.0
			self.TARGETS = 574e6*const.NAvo / self.Nucleons_per_target
			self.POTS = 2.5e19*0.79
			self.A_NUMBER = 12.0
			self.Z_NUMBER = 6.0

		elif self.EXP_FLAG == ND280_nu:
			self.FLUXFILE="fluxes/T2Kflux2016/t2kflux_2016_nd280_plus250kA.txt"
			self.FLUX_NORM=1.0/1e21/0.05
			self.EMIN = 0.05
			self.EMAX = 20.0
			# Detector targets
			self.POTS = 12.34e20
			self.MATERIALS_NAME=['H1', 'C12', 'O16', 'Cu29', 'Zn30', 'Pb208']
			self.MATERIALS_A=[1, 12, 16, 63.546, 65.38, 207.2]
			self.MATERIALS_Z=[1, 6, 8, 29, 30, 82]
			self.MATERIALS_MASS =[
								2.9e6*2.0/18.0,
								0,
								2.9e6*16.0/18.0,
								0,
								12.9e6] # grams
			self.TOT_MASS =np.sum(self.MATERIALS_MASS)

		elif self.EXP_FLAG == ND280_nubar:
			self.FLUXFILE="fluxes/T2Kflux2016/t2kflux_2016_nd280_minus250kA.txt"
			self.FLUX_NORM=1.0/1e21/0.05
			self.EMIN = 0.05
			self.EMAX = 20.0
			# Detector targets
			self.POTS = 12.34e20
			self.MATERIALS_NAME=['H1', 'C12', 'O16', 'Cu29', 'Zn30', 'Pb208']
			self.MATERIALS_A=[1, 12, 16, 63.546, 65.38, 207.2]
			self.MATERIALS_Z=[1, 6, 8, 29, 30, 82]
			self.MATERIALS_MASS =[
								2.9e6*2.0/18.0,
								0,
								2.9e6*16.0/18.0,
								0,
								12.9e6] # grams
			self.TOT_MASS =np.sum(self.MATERIALS_MASS)

		
		else:
			print('ERROR! No experiment chosen.')


	######################################################
	# ALL FLUXES ARE NORMALIZED SO THAT THE UNITS ARE    
	#   nus/cm^2/GeV/POT      
	######################################################
	def get_flux_func(self, flavour = pdg.numu):
		if (self.FLUXFILE == "fluxes/MiniBooNE_nu_mode_flux.dat" and (flavour==pdg.numu or flavour==pdg.numubar)):
			Elo, Ehi, numu, numub, nue, nueb = np.loadtxt(self.FLUXFILE, unpack=True)
			E = (Ehi+Elo)/2.0
			if flavour==pdg.numu:
				nf = numu
			if flavour==pdg.numubar:
				nf = numub
		elif (self.FLUXFILE == "fluxes/MINERVA_ME_numu_flux.dat" and flavour==pdg.numu):
			E, nf = np.loadtxt(self.FLUXFILE, unpack=True)
		elif (self.FLUXFILE == "fluxes/MINERVA_LE_numu_flux.dat" and flavour==pdg.numu):
			E, nf = np.loadtxt(self.FLUXFILE, unpack=True)
		elif (self.FLUXFILE == "fluxes/CHARMII.dat" and flavour==pdg.numu):
			E, nf = np.loadtxt(self.FLUXFILE, unpack=True)
		elif (self.FLUXFILE=="fluxes/T2Kflux2016/t2kflux_2016_nd280_minus250kA.txt" or 
				self.FLUXFILE=="fluxes/T2Kflux2016/t2kflux_2016_nd280_plus250kA.txt"):
			data = np.genfromtxt(self.FLUXFILE,unpack=True,skip_header=3)
			E = (data[1]+data[2])/2
			if flavour==pdg.numu:
				nf = data[3]
			elif flavour==pdg.numubar:
				nf = data[4]
			elif flavour==pdg.nue:
				nf = data[5]
			elif flavour==pdg.nuebar:
				nf = data[6]
		else:
			print("No flux was found!")
			sys.exit(0)

		flux = interpolate.interp1d(E, nf*self.FLUX_NORM, fill_value=0.0, bounds_error=False)
		return flux