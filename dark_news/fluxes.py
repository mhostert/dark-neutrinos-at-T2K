import numpy as np
import sys 

from scipy import interpolate

from . import const 
from . import pdg
###########################
# ALL FLUXES UNITS ARE    #
#   nus/cm^2/GeV/POT      #
###########################

def get_exp_params(fluxfile, flavour = pdg.numu):

	if (fluxfile == "fluxes/MiniBooNE_nu_mode_flux.dat"):
		Elo, Ehi, numu, numub, nue, nueb = np.loadtxt(fluxfile, unpack=True)
		Eminiboone = (Ehi+Elo)/2.0
		if flavour==pdg.numu:
			nf = numu
		if flavour==pdg.numubar:
			nf = numub
		nf *= 1.0/0.05
		flux = interpolate.interp1d(Eminiboone, nf, fill_value=0.0, bounds_error=False)
		EMIN = 0.1
		EMAX = 9
		DET_SIZE = 6.1 # meters
		# print "Running MiniBooNE fluxes"
		return flux, EMIN, EMAX, DET_SIZE


	elif (fluxfile == "fluxes/MINERVA_ME_numu_flux.dat"):
		E, numu = np.loadtxt(fluxfile, unpack=True)
		numu *= (1e-4)
		flux = interpolate.interp1d(E, numu, fill_value=0.0, bounds_error=False)
		EMIN = 0.10
		EMAX = 19.0
		DET_SIZE = 2.5/2.0 # meters
		return flux, EMIN, EMAX, DET_SIZE
	elif (fluxfile == "fluxes/MINERVA_LE_numu_flux.dat"):
		E, numu = np.loadtxt(fluxfile, unpack=True)
		numu *=  (1e-4)*(1e-6)*2
		flux = interpolate.interp1d(E, numu, fill_value=0.0, bounds_error=False)
		EMIN = 0.1
		EMAX = 19
		DET_SIZE = 2.5/2.0 # meters
		return flux, EMIN, EMAX, DET_SIZE
	elif (fluxfile == "fluxes/CHARMII.dat"):
		E, numu = np.loadtxt(fluxfile, unpack=True)
		numu *= 1.0/(370)**2 *1e-13
		flux = interpolate.interp1d(E, numu, fill_value=0.0, bounds_error=False)
		EMIN = 1.5
		EMAX = 198
		DET_SIZE = 35.67/2.0 # meters
		return flux, EMIN, EMAX, DET_SIZE
	else:
		print("No flux was found!")
		sys.exit(0)


flux, EMIN, EMAX, DET_SIZE = get_exp_params("fluxes/MiniBooNE_nu_mode_flux.dat")