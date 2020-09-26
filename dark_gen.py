#!/usr/bin/python
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys, getopt

# Dark Neutrino and MC stuff
from dark_news import *


def main(argv):

	####################
	# Default options
	# 3+1 model unless otherwise specified by passing mheavy argument
	
	MLIGHT = 0.100 #nu4
	MHEAVY = 1e5 #nu5
	MHEAVIEST = 0.220 #nu6
	UMU4 = np.sqrt(13.6e-8)
	UMU5 = np.sqrt(26.5e-8)
	UMU6 = np.sqrt(123.0e-8*0.0629)
	GPRIME = np.sqrt(4*np.pi*0.32)

	MZPRIME = 1.25

	l_decay_proper = 0.05 # cm

	MODEL = const.THREEPLUSONE
	# MODEL = const.THREEPLUSTWO
	
	TOT_EVENTS = 100
	EXP_FLAG = const.MINIBOONE
	COH_FLAG = 1 # 1 - include diffractive events 0 - exclude it
	DIF_FLAG = 1 # 1 - include diffractive events 0 - exclude it
	PLOT_FLAG = 1 # 1 - plot 0 -- dont plot
	HNLtype = const.MAJORANA # 1 - DIRAC 0 -- MAJORANA 

	# Other standard parameters that control normalization mainly
	UTAU4 = np.sqrt(7.8e-4)*0
	UTAU5 = np.sqrt(7.8e-4)*0
	UTAU6 = np.sqrt(7.8e-4)*0
	CHI  = np.sqrt(5.94e-4)

	UD4 = 1.0
	UD5 = 1.0
	UD6 = 1.0
	
	####################
	# User specified
	try:
  		opts, args = getopt.getopt(argv,"h",["help","mzprime=","mlight=","mheavy=","ldecay=","nevents=","exp=","nodif","nocoh","noplot"])
	except getopt.GetoptError:
		with open("README.md", 'r') as fin:
			print(fin.read())
			sys.exit(2)
		for opt, arg in opts:
			if opt == '-h':
				with open("README.md", 'r') as fin:
					print(fin.read())
					sys.exit(2)
			elif opt in ("--mzprime"):
				MZPRIME = np.float64(arg)
			elif opt in ("--mlight"):
		 		MLIGHT = np.float64(arg)
			elif opt in ("--mheavy"):
		 		MHEAVY = np.float64(arg)
			elif opt in ("--ldecay"):
		 		l_decay_proper = np.float64(arg)
			elif opt in ("--nevents"):
				TOT_EVENTS = np.int(arg)
			elif opt in ("--nodif"):
				# no diffractive events
				DIF_FLAG = 0
			elif opt in ("--nocoh"):
				# no coherent events
				COH_FLAG = 0	  
			elif opt in ("--exp"):
				# choose experiment
				EXP_FLAG = arg
			elif opt in ("--noplot"):
				# choose experiment
				PLOT_FLAG = 0
			elif opt in ("--HNLtype"):
				# choose experiment
				HNLtype = np.int(arg)

	##########################
	# MC evaluations and iterations
	MC.NEVAL_warmup = 1e3
	MC.NINT_warmup = 100
	MC.NEVAL = 1e4
	MC.NINT  = 100

	# Set the model to use
	if (MHEAVY > MLIGHT and MHEAVY < MZPRIME):
		MODEL  = const.THREEPLUSTWO
	elif ((MLIGHT < MZPRIME) and (MHEAVY > MZPRIME) or ( (MLIGHT > MZPRIME) and (MHEAVY > 1e5) )):
		MODEL = const.THREEPLUSONE
	else:
		print('ERROR! Mass spectrum not allowed.')
		
	#########################
	# Set BSM parameters
	BSMparams = model.model_params()

	BSMparams.gprime = GPRIME
	BSMparams.chi = CHI
	BSMparams.Ue4 = 0.0
	BSMparams.Umu4 = UMU4 
	BSMparams.Utau4 = UTAU4
	BSMparams.Ue5 = 0.0
	BSMparams.Umu5 = UMU5
	BSMparams.Utau5 = UTAU5
	BSMparams.Ue6 = 0.0
	BSMparams.Umu6 = UMU6
	BSMparams.Utau6 = UTAU6
	BSMparams.UD4 = UD4
	BSMparams.UD5 = UD5
	BSMparams.UD6 = UD6
	BSMparams.m4 = MLIGHT
	BSMparams.m5 = MHEAVY
	BSMparams.m6 = MHEAVIEST
	BSMparams.Mzprime = MZPRIME
	BSMparams.Dirac = HNLtype

	BSMparams.set_high_level_variables()

	#########################
	# Choose experimental parameters
	if EXP_FLAG == const.MINIBOONE:
		FLUXFILE="fluxes/MiniBooNE_nu_mode_flux.dat"
		Nucleons_per_target = 14.0
		P_per_target = 8.0
		TARGETS = (818e6) * const.NAvo / Nucleons_per_target
		POTS = 18.75e20
		A_NUMBER = 12.0
		Z_NUMBER = 6.0
	elif EXP_FLAG == const.uBOONE:
		FLUXFILE="fluxes/MiniBooNE_nu_mode_flux.dat"
		Nucleons_per_target = 40.0
		P_per_target = 18.0
		TARGETS = 60.0e6 * const.NAvo / Nucleons_per_target
		flux_miniboone_scaled = (541.0/463.0)**2
		POTS = 13.2e20 * flux_miniboone_scaled
		A_NUMBER = 40.0
		Z_NUMBER = 18.0
	elif EXP_FLAG == const.MINERVA_LE:
		FLUXFILE="fluxes/MINERVA_LE_numu_flux.dat"
		Nucleons_per_target = 13.0
		P_per_target = 7.0
		TARGETS = 6.10e6*const.NAvo / Nucleons_per_target
		POTS = 3.43e20*0.73
		A_NUMBER = 12.0
		Z_NUMBER = 6.0
	elif EXP_FLAG == const.MINERVA_ME:
		FLUXFILE="fluxes/MINERVA_ME_numu_flux.dat"
		Nucleons_per_target = 13.0
		P_per_target = 7.0
		TARGETS = 6.10e6*const.NAvo / Nucleons_per_target
		POTS = 1.16e21*0.73
		A_NUMBER = 12.0
		Z_NUMBER = 6.0
	elif EXP_FLAG == const.CHARMII:
		FLUXFILE="fluxes/CHARMII.dat"
		Nucleons_per_target = 20.7
		P_per_target = 11.0
		TARGETS = 574e6*const.NAvo / Nucleons_per_target
		POTS = 2.5e19*0.79
		A_NUMBER = 12.0
		Z_NUMBER = 6.0
	else:
		print('ERROR! No experiment chosen.')

	##########################
	# Run MC and get events
  
  #####################################################################
	if MODEL==const.THREEPLUSONE:
		coh_events = MC.MC_events(BSMparams,
								 FLUXFILE,
								None,
								MA=A_NUMBER*const.MAVG,
								Z=Z_NUMBER,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino4,
								nu_outgoing=pdg.numu,
								final_lepton=pdg.electron,
								h_upscattered=-1)
		dif_events = MC.MC_events(BSMparams,
								FLUXFILE, 
								None,
								MA=1.0*const.mproton,
								Z=1,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino4,
								nu_outgoing=pdg.numu,
								final_lepton=pdg.electron,
								h_upscattered=-1)
		
		coh_eventsRH = MC.MC_events(BSMparams,
								 FLUXFILE,
								None,
								MA=A_NUMBER*const.MAVG,
								Z=Z_NUMBER,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino4,
								nu_outgoing=pdg.numu,
								final_lepton=pdg.electron,
								h_upscattered=+1)
		dif_eventsRH = MC.MC_events(BSMparams,
								FLUXFILE, 
								None,
								MA=1.0*const.mproton,
								Z=1,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino4,
								nu_outgoing=pdg.numu,
								final_lepton=pdg.electron,
								h_upscattered=+1)
		a,b,c,d = coh_events.get_MC_events(),dif_events.get_MC_events(),coh_eventsRH.get_MC_events(),dif_eventsRH.get_MC_events()
		cases = [a,b,c,d]
		
		Ifactors = [1.0*COH_FLAG,
					1*P_per_target*DIF_FLAG,
					1.0*COH_FLAG,
					1*P_per_target*DIF_FLAG]

		flags = [const.COHLH,const.DIFLH,
				const.COHRH,const.DIFRH]

		bag = MC.Combine_MC_output(cases, Ifactors=Ifactors, flags=flags)


		### NAMING 
		## HEPEVT Event file name
		PATH_data = 'data/'+EXP_FLAG+'/3plus1/m4_'+str(round(MLIGHT,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
		PATH = 'plots/'+EXP_FLAG+'/3plus1/m4_'+str(round(MLIGHT,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
		
		# title for plots
		power=int(np.log10(UMU4**2))-1
		print(power)
		title = r"$m_{4} = \,$"+str(round(MLIGHT,4))+r" GeV, $M_{Z^\prime} = \,$"+str(round(MZPRIME,4))+r" GeV, $|U_{D4}|^2=%1.1g$, $|U_{\mu 4}|^2=%1.1f \times 10^{-%i}$"%(UD4**2,UMU4**2/10**(power),-power)
	
	#####################################################################
	elif MODEL==const.THREEPLUSTWO:
		coh_events = MC.MC_events(BSMparams,
								 FLUXFILE,
								None,
								MA=A_NUMBER*const.MAVG,
								Z=Z_NUMBER,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino5,
								nu_outgoing=pdg.neutrino4,
								final_lepton=pdg.electron,
								h_upscattered=-1)
		dif_events = MC.MC_events(BSMparams,
								FLUXFILE, 
								None,
								MA=1.0*const.mproton,
								Z=1,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino5,
								nu_outgoing=pdg.neutrino4,
								final_lepton=pdg.electron,
								h_upscattered=-1)
		
		coh_eventsRH = MC.MC_events(BSMparams,
								 FLUXFILE,
								None,
								MA=A_NUMBER*const.MAVG,
								Z=Z_NUMBER,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino6,
								nu_outgoing=pdg.neutrino4,
								final_lepton=pdg.electron,
								h_upscattered=-1)
		dif_eventsRH = MC.MC_events(BSMparams,
								FLUXFILE, 
								None,
								MA=1.0*const.mproton,
								Z=1,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino6,
								nu_outgoing=pdg.neutrino4,
								final_lepton=pdg.electron,
								h_upscattered=-1)
		
		cases = [
				coh_events.get_MC_events(),
				dif_events.get_MC_events(),
				coh_eventsRH.get_MC_events(),
				dif_eventsRH.get_MC_events()
				]
		
		Ifactors = [1.0*COH_FLAG,
					P_per_target*DIF_FLAG,
					1.0*COH_FLAG,
					P_per_target*DIF_FLAG]

		flags = [const.COHLH,const.DIFLH,
				const.COHRH,const.DIFRH]

		bag = MC.Combine_MC_output(cases, Ifactors=Ifactors, flags=flags)

		pN   = bag['P3']
		pZ   = bag['P3_decay']+bag['P4_decay']
		plm  = bag['P3_decay']
		plp  = bag['P4_decay']
		pHad = bag['P4']
		w = bag['w']
		I = bag['I']
		regime = bag['flags']


		### NAMING 
		## HEPEVT Event file name
		if EXP_FLAG == const.uBOONE:
			PATH_data = 'data/'+EXP_FLAG+'/3plus2/mlight_'+str(round(MLIGHT,4))+'_mheavy_'+str(round(MHEAVY,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
			PATH = 'plots/'+EXP_FLAG+'/3plus2/mlight_'+str(round(MLIGHT,4))+'_mheavy_'+str(round(MHEAVY,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
		elif EXP_FLAG == const.MINIBOONE:
			PATH_data = 'data/'+EXP_FLAG+'/3plus2/mlight_'+str(round(MLIGHT,4))+'_mheavy_'+str(round(MHEAVY,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
			PATH = 'plots/'+EXP_FLAG+'/3plus2/mlight_'+str(round(MLIGHT,4))+'_mheavy_'+str(round(MHEAVY,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
		else:
			print("Error! No experiment chosen.")

		# efile_name = 'data/ubooneHEPevt/uboone_Zpmass_'+format(MZPRIME,'0.8g')+"_nu4mass_"+format(MHEAVY,'.8g')+".dat"
		efile_name = PATH_data+EXP_FLAG+"_mlight_"+format(MLIGHT,'.8g')+"_mheavy_"+format(MHEAVY,'.8g')+"_mzprime_"+format(MZPRIME,'.8g')+".dat"
		
		# title for plots
		title = r"$m_{4} = \,$"+str(int(round(MLIGHT*1000,4)))+r" MeV,\quad $m_{5} = \,$"+str(int(round(MHEAVY*1000,4)))+r" MeV,\quad $m_{6} = \,$"+str(int(round(MHEAVIEST*1000,4)))+r" MeV"


	############################################################################
	# HEPEVT 
	#############################################################################
	print(MODEL)
	print(PATH_data)
	hepevt.generate_muboone_HEPEVT_files(PATH_data, bag, TOT_EVENTS, BSMparams, l_decay_proper=l_decay_proper)

if __name__ == "__main__":
	try:
		main(sys.argv[1:])
	except (KeyboardInterrupt, SystemExit):
		raise
