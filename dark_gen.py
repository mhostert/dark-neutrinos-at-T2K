#!/usr/bin/python
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse


# Dark Neutrino and MC stuff
from dark_news import *
from exp_analysis import *


def main(argv):

	####################
	# Default options
	# 3+1 model unless otherwise specified by passing M5 argument
	
	####################
	# User specified
	parser = argparse.ArgumentParser(description="Generate dark nu events")
	parser.add_argument("--mzprime", type=float, help="Z' mass", default=1.25)
	parser.add_argument("--M4", type=float, help="mass of the fourth neutrino", default=0.100)
	parser.add_argument("--M5", type=float, help="mass of the fifth neutrino", default=1e5)
	parser.add_argument("--M6", type=float, help="mass of the sixth neutrino", default=1e5)

	parser.add_argument("--UMU4", type=float, help="Umu4", default=np.sqrt(13.6e-8))
	parser.add_argument("--UMU5", type=float, help="Umu5", default=np.sqrt(26.5e-8))
	parser.add_argument("--UMU6", type=float, help="Umu6", default=np.sqrt(123.0e-8*0.0629))
	parser.add_argument("--GPRIME", type=float, help="gprime", default=np.sqrt(4*np.pi*0.32))

	parser.add_argument("--UTAU4", type=float, help="UTAU4", default=np.sqrt(7.8e-4)*0)
	parser.add_argument("--UTAU5", type=float, help="UTAU5", default=np.sqrt(7.8e-4)*0)
	parser.add_argument("--UTAU6", type=float, help="UTAU6", default=np.sqrt(7.8e-4)*0)
	parser.add_argument("--CHI", type=float, help="CHI", default=np.sqrt(5.94e-4))

	parser.add_argument("--UD4", type=float, help="UD4", default=1.0)
	parser.add_argument("--UD5", type=float, help="UD5", default= 1.0)
	parser.add_argument("--UD6", type=float, help="UD6", default= 1.0)

	parser.add_argument("--ldecay", type=float, help="ctau of the fourth neutrino", default=0.05)
	parser.add_argument("--exp", type=str, help="experiment", choices=["charmii",
																		"minerva_le",
																		"minerva_me",
																		"miniboone",
																		"uboone",
																		"nd280_nu",
																		"nd280_nubar"],
																		default="nd280_nu")

	parser.add_argument("--nodif", help="remove diffractive events", action="store_true")
	parser.add_argument("--nocoh", help="remove coherent events", action="store_true")
	parser.add_argument("--noplot", help="no plot", action="store_true")
	parser.add_argument("--HNLtype", type=int, help="HNLtype: 1 is DIRAC, 0 is MAJORANA", choices=[0, 1], default=0)
	
	parser.add_argument("--nevents", type=int, help="number of events to generate", default=100)
	args = parser.parse_args()

	##########################
	# MC evaluations and iterations
	MC.NEVAL_warmup = 1e3
	MC.NINT_warmup = 100
	MC.NEVAL = 1e4
	MC.NINT  = 100
		
	#########################
	# Set BSM parameters
	BSMparams = model.model_params()

	BSMparams.gprime = args.GPRIME
	BSMparams.chi = args.CHI
	BSMparams.Ue4 = 0.0
	BSMparams.Umu4 = args.UMU4 
	BSMparams.Utau4 = args.UTAU4
	BSMparams.Ue5 = 0.0
	BSMparams.Umu5 = args.UMU5
	BSMparams.Utau5 = args.UTAU5
	BSMparams.Ue6 = 0.0
	BSMparams.Umu6 = args.UMU6
	BSMparams.Utau6 = args.UTAU6
	BSMparams.UD4 = args.UD4
	BSMparams.UD5 = args.UD5
	BSMparams.UD6 = args.UD6
	BSMparams.m4 = args.M4
	BSMparams.m5 = args.M5
	BSMparams.m6 = args.M6
	BSMparams.Mzprime = args.mzprime
	BSMparams.Dirac = args.HNLtype

	BSMparams.set_high_level_variables()

	####################################################
	# CHOOSE EXPERIMENTAL FLUXES, NORMALIZATION, ETC...
	myexp = exp.experiment(args.exp)

	####################################################
	# Set the model to use
	if (args.M5 > args.M4 and args.M5 < args.mzprime):
		MODEL = const.THREEPLUSTWO
		print('Model used 3+2')
	elif (((args.M4 < args.mzprime) and (args.M5 > args.mzprime)) or ((args.M4 > args.mzprime) and (args.M5 >= 1e5))):
		MODEL = const.THREEPLUSONE
		print('Model used 3+1')
	else:
		print('ERROR! Mass spectrum not allowed.')

	####################################################
	# Run MC and get events
	if MODEL==const.THREEPLUSONE:

		bag = MC.run_MC(BSMparams, myexp, [pdg.numu])

		### NAMING 
		## HEPEVT Event file name
		PATH_data = f'data/{args.exp}/3plus1/m4_{args.M4:.4g}_mzprime_{args.mzprime:.4g}/'
		PATH = f'plots/{args.exp}/3plus1/m4_{args.M4:.4g}_mzprime_{args.mzprime:.4g}/'
		
		# title for plots
		power = int(np.log10(args.UMU4**2))-1
		title = r"$m_{4} = \,$"+str(round(args.M4,4))+r" GeV, $M_{Z^\prime} = \,$"+str(round(args.mzprime,4))+r" GeV, $|U_{D4}|^2=%1.1g$, $|U_{\mu 4}|^2=%1.1f \times 10^{-%i}$"%(args.UD4**2,args.UMU4**2/10**(power),-power)
	
	#####################################################################
	# elif MODEL==const.THREEPLUSTWO:
	# 	coh_events = MC.MC_events(BSMparams,
	# 							 myexp,
	# 							None,
	# 							MA=A_NUMBER*const.MAVG,
	# 							Z=Z_NUMBER,
	# 							nu_scatterer=pdg.numu,
	# 							nu_produced=pdg.neutrino5,
	# 							nu_outgoing=pdg.neutrino4,
	# 							final_lepton=pdg.electron,
	# 							h_upscattered=-1)
	# 	dif_events = MC.MC_events(BSMparams,
	# 							myexp, 
	# 							None,
	# 							MA=1.0*const.mproton,
	# 							Z=1,
	# 							nu_scatterer=pdg.numu,
	# 							nu_produced=pdg.neutrino5,
	# 							nu_outgoing=pdg.neutrino4,
	# 							final_lepton=pdg.electron,
	# 							h_upscattered=-1)
		
	# 	coh_eventsRH = MC.MC_events(BSMparams,
	# 							 myexp,
	# 							None,
	# 							MA=A_NUMBER*const.MAVG,
	# 							Z=Z_NUMBER,
	# 							nu_scatterer=pdg.numu,
	# 							nu_produced=pdg.neutrino6,
	# 							nu_outgoing=pdg.neutrino4,
	# 							final_lepton=pdg.electron,
	# 							h_upscattered=-1)
	# 	dif_eventsRH = MC.MC_events(BSMparams,
	# 							myexp, 
	# 							None,
	# 							MA=1.0*const.mproton,
	# 							Z=1,
	# 							nu_scatterer=pdg.numu,
	# 							nu_produced=pdg.neutrino6,
	# 							nu_outgoing=pdg.neutrino4,
	# 							final_lepton=pdg.electron,
	# 							h_upscattered=-1)
		
		# cases = [
		# 		coh_events.get_MC_events(),
		# 		dif_events.get_MC_events(),
		# 		coh_eventsRH.get_MC_events(),
		# 		dif_eventsRH.get_MC_events()
		# 		]
		
		# Ifactors = [1.0*COH_FLAG,
		# 			P_per_target*DIF_FLAG,
		# 			1.0*COH_FLAG,
		# 			P_per_target*DIF_FLAG]

		# flags = [const.COHLH,const.DIFLH,
		# 		const.COHRH,const.DIFRH]

		# bag = MC.Combine_MC_output(cases, Ifactors=Ifactors, flags=flags)

		# pN   = bag['P3']
		# pZ   = bag['P3_decay']+bag['P4_decay']
		# plm  = bag['P3_decay']
		# plp  = bag['P4_decay']
		# pHad = bag['P4']
		# w = bag['w']
		# I = bag['I']
		# regime = bag['flags']


	print(np.sum(bag['w']))
	print(np.size(bag['w']))
	print(Cfv.dot4(bag['P2_decay'],bag['P2_decay']))
	############################################################################
	# Print events to file -- currently in data/exp/m4____mzprime____.dat 
	#############################################################################
	printer.print_events_to_pandas(PATH_data, bag, args.nevents, BSMparams, l_decay_proper=args.ldecay)

if __name__ == "__main__":
	try:
		main(sys.argv[1:])
	except (KeyboardInterrupt, SystemExit):
		raise
