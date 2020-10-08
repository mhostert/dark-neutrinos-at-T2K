#!/usr/bin/python
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys, getopt

# Dark Neutrino and MC stuff
from dark_news import *
from exp_analysis import *


def main(argv):

	####################
	# Default options
	# 3+1 model unless otherwise specified by passing M5 argument
	
	M4 = 0.100 # GeV nu4
	M5 = 1e5 # GeV nu5
	M6 = 1e5 # GeV nu6
	UMU4 = np.sqrt(13.6e-8)
	UMU5 = np.sqrt(26.5e-8)
	UMU6 = np.sqrt(123.0e-8*0.0629)
	GPRIME = np.sqrt(4*np.pi*0.32)

	MZPRIME = 1.25

	l_decay_proper = 0.05 # cm

	MODEL = const.THREEPLUSONE
	# MODEL = const.THREEPLUSTWO
	
	TOT_EVENTS = 100
	EXP_FLAG = exp.ND280_nu
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
  		opts, args = getopt.getopt(argv,"h",["help","mzprime=","M4=","M5=","ldecay=","nevents=","exp=","nodif","nocoh","noplot"])
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
			elif opt in ("--M4"):
		 		M4 = np.float64(arg)
			elif opt in ("--M5"):
		 		M5 = np.float64(arg)
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
	if (M5 > M4 and M5 < MZPRIME):
		MODEL  = const.THREEPLUSTWO
	elif ((M4 < MZPRIME) and (M5 > MZPRIME) or ( (M4 > MZPRIME) and (M5 > 1e5) )):
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
	BSMparams.m4 = M4
	BSMparams.m5 = M5
	BSMparams.m6 = M6
	BSMparams.Mzprime = MZPRIME
	BSMparams.Dirac = HNLtype

	BSMparams.set_high_level_variables()

	####################################################
	# CHOOSE EXPERIMENTAL FLUXES, NORMALIZATION, ETC...
	myexp = exp.experiment(EXP_FLAG)

	#####################################################################
	# Run MC and get events
	if MODEL==const.THREEPLUSONE:

		bag = MC.run_MC(BSMparams,myexp,[pdg.numu])

		### NAMING 
		## HEPEVT Event file name
		PATH_data = 'data/'+EXP_FLAG+'/3plus1/m4_'+str(round(M4,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
		PATH = 'plots/'+EXP_FLAG+'/3plus1/m4_'+str(round(M4,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
		
		# title for plots
		power=int(np.log10(UMU4**2))-1
		title = r"$m_{4} = \,$"+str(round(M4,4))+r" GeV, $M_{Z^\prime} = \,$"+str(round(MZPRIME,4))+r" GeV, $|U_{D4}|^2=%1.1g$, $|U_{\mu 4}|^2=%1.1f \times 10^{-%i}$"%(UD4**2,UMU4**2/10**(power),-power)
	
	#####################################################################
	elif MODEL==const.THREEPLUSTWO:
		coh_events = MC.MC_events(BSMparams,
								 myexp,
								None,
								MA=A_NUMBER*const.MAVG,
								Z=Z_NUMBER,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino5,
								nu_outgoing=pdg.neutrino4,
								final_lepton=pdg.electron,
								h_upscattered=-1)
		dif_events = MC.MC_events(BSMparams,
								myexp, 
								None,
								MA=1.0*const.mproton,
								Z=1,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino5,
								nu_outgoing=pdg.neutrino4,
								final_lepton=pdg.electron,
								h_upscattered=-1)
		
		coh_eventsRH = MC.MC_events(BSMparams,
								 myexp,
								None,
								MA=A_NUMBER*const.MAVG,
								Z=Z_NUMBER,
								nu_scatterer=pdg.numu,
								nu_produced=pdg.neutrino6,
								nu_outgoing=pdg.neutrino4,
								final_lepton=pdg.electron,
								h_upscattered=-1)
		dif_eventsRH = MC.MC_events(BSMparams,
								myexp, 
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


		# ### NAMING 
		# ## HEPEVT Event file name
		# if EXP_FLAG == exp.uBOONE:
		# 	PATH_data = 'data/'+EXP_FLAG+'/3plus2/M4_'+str(round(M4,4))+'_M5_'+str(round(M5,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
		# 	PATH = 'plots/'+EXP_FLAG+'/3plus2/M4_'+str(round(M4,4))+'_M5_'+str(round(M5,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
		# elif EXP_FLAG == exp.MINIBOONE:
		# 	PATH_data = 'data/'+EXP_FLAG+'/3plus2/M4_'+str(round(M4,4))+'_M5_'+str(round(M5,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
		# 	PATH = 'plots/'+EXP_FLAG+'/3plus2/M4_'+str(round(M4,4))+'_M5_'+str(round(M5,4))+'_mzprime_'+str(round(MZPRIME,4))+'/'
		# else:
		# 	print("Error! No experiment chosen.")

		# efile_name = 'data/ubooneHEPevt/uboone_Zpmass_'+format(MZPRIME,'0.8g')+"_nu4mass_"+format(M5,'.8g')+".dat"
		# efile_name = PATH_data+EXP_FLAG+"_M4_"+format(M4,'.8g')+"_M5_"+format(M5,'.8g')+"_mzprime_"+format(MZPRIME,'.8g')+".dat"
		
		# # title for plots
		# title = r"$m_{4} = \,$"+str(int(round(M4*1000,4)))+r" MeV,\quad $m_{5} = \,$"+str(int(round(M5*1000,4)))+r" MeV,\quad $m_{6} = \,$"+str(int(round(M6*1000,4)))+r" MeV"


	############################################################################
	# Print events to file -- currently in data/exp/m4____mzprime____.dat 
	#############################################################################
	hepevt.print_events_to_file(PATH_data, bag, TOT_EVENTS, BSMparams, l_decay_proper=l_decay_proper)

if __name__ == "__main__":
	try:
		main(sys.argv[1:])
	except (KeyboardInterrupt, SystemExit):
		raise
