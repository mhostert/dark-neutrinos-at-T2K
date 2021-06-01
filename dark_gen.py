#!/usr/bin/python3
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse


# Dark Neutrino and MC stuff
from dark_news import *
# from exp_analysis import *


def main(argv):

	####################
	# Default options
	# 3+1 model unless otherwise specified by passing M5 argument
	
	####################
	# User specified
	parser = argparse.ArgumentParser(description="Generate dark nu events")
	parser.add_argument("--mzprime", type=float, help="Z' mass", default=0.03)
	parser.add_argument("--M4", type=float, help="mass of the fourth neutrino", default=0.100)
	parser.add_argument("--M5", type=float, help="mass of the fifth neutrino", default=1e5)
	parser.add_argument("--M6", type=float, help="mass of the sixth neutrino", default=1e5)

	parser.add_argument("--UMU4", type=float, help="Umu4 square", default=2.2e-7)
	parser.add_argument("--UMU5", type=float, help="Umu5 square", default=0*26.5e-8)
	parser.add_argument("--UMU6", type=float, help="Umu6 square", default=0*123.0e-8*0.0629)
	# parser.add_argument("--GPRIME", type=float, help="gprime", default=np.sqrt(4*np.pi*1/4.0))
	# parser.add_argument("--CHI", type=float, help="CHI", default=np.sqrt(2e-10/const.alphaQED)/const.cw)
	parser.add_argument("--alpha_dark", type=float, help="alpha_dark", default=0.4)
	parser.add_argument("--alpha_epsilon2", type=float, help="Product of alpha QED times epsilon^2", default=4.6e-4*const.alphaQED)
	parser.add_argument("--epsilon2", type=float, help="epsilon^2")

	parser.add_argument("--UTAU4", type=float, help="UTAU4", default=0)
	parser.add_argument("--UTAU5", type=float, help="UTAU5", default=0)
	parser.add_argument("--UTAU6", type=float, help="UTAU6", default=0)
	

	parser.add_argument("--UD4", type=float, help="UD4", default=1.0)
	parser.add_argument("--UD5", type=float, help="UD5", default=0*1.0)
	parser.add_argument("--UD6", type=float, help="UD6", default=0*1.0)

	parser.add_argument("--ldecay", type=float, help="ctau of the fourth neutrino in cm", default=0.00)
	parser.add_argument("--exp", type=str, help="experiment", choices=["charmii",
																		"minerva_le",
																		"minerva_me",
																		"miniboone",
																		"uboone",
																		"nd280_nu",
																		"nd280_nubar"],
																		default="nd280_nubar")

	parser.add_argument("--nodif", help="remove diffractive events", action="store_true")
	parser.add_argument("--nocoh", help="remove coherent events", action="store_true")
	parser.add_argument("--noplot", help="no plot", action="store_true")
	parser.add_argument("--D_or_M", help="D_or_M: dirac or majorana", choices=["dirac", "majorana"], default="majorana")
	
	parser.add_argument("--neval", type=int, help="number of evaluations of integrand", default=1e4)
	parser.add_argument("--nint", type=int, help="number of adaptive iterations", default=20)
	parser.add_argument("--neval_warmup", type=int, help="number of evaluations of integrand in warmup", default=1e3)
	parser.add_argument("--nint_warmup", type=int, help="number of adaptive iterations in warmup", default=10)

	parser.add_argument("--hepevt_events", type=int, help="number of events to accept in HEPEVT format", default=1e2)

	parser.add_argument("--hierarchy",
						 type=str,
						 help="light or heavy Z' case",
						 default='light',
						 choices=['light', 'heavy'])

	args = parser.parse_args()

	##########################
	# MC evaluations and iterations
	MC.NEVAL_warmup = args.neval_warmup
	MC.NINT_warmup = args.nint_warmup
	MC.NEVAL = args.neval
	MC.NINT = args.nint
		
	#########################
	# Set BSM parameters
	BSMparams = model.model_params()

	BSMparams.gprime = np.sqrt(4*np.pi*args.alpha_dark)

	if args.epsilon2:
		BSMparams.chi = np.sqrt(args.epsilon2)/const.cw
	else:
		BSMparams.chi = np.sqrt(args.alpha_epsilon2/const.alphaQED)/const.cw

	BSMparams.Ue4 = 0.0
	BSMparams.Umu4 = np.sqrt(args.UMU4)
	BSMparams.Utau4 = np.sqrt(args.UTAU4)
	BSMparams.Ue5 = 0.0
	BSMparams.Umu5 = np.sqrt(args.UMU5)
	BSMparams.Utau5 = np.sqrt(args.UTAU5)
	BSMparams.Ue6 = 0.0
	BSMparams.Umu6 = np.sqrt(args.UMU6)
	BSMparams.Utau6 = np.sqrt(args.UTAU6)
	BSMparams.UD4 = args.UD4
	BSMparams.UD5 = args.UD5
	BSMparams.UD6 = args.UD6
	BSMparams.m4 = args.M4
	BSMparams.m5 = args.M5
	BSMparams.m6 = args.M6
	BSMparams.Mzprime = args.mzprime
	BSMparams.D_or_M = args.D_or_M

	BSMparams.set_high_level_variables()
	BSMparams.hierarchy=args.hierarchy

	####################################################
	# CHOOSE EXPERIMENTAL FLUXES, NORMALIZATION, ETC...
	myexp = exp.experiment(args.exp)

	####################################################
	# Set the model to use
	wrong_order=((args.M4 <= args.mzprime) & (args.hierarchy == 'light'))|((args.M4 >= args.mzprime) & (args.hierarchy == 'heavy'))
	if (wrong_order):
		print(f'Error! Bad choice of m4 and mzprime for {args.hierarchy} Zprime and {args.D_or_M} case.') 	
		return 1
	####################################################
	# Run MC and get events

	print(f"Using: m4 = {args.M4} mzprime={args.mzprime}")
	bag = MC.run_MC(BSMparams, myexp, INCLUDE_DIF=False)

	### NAMING 
	## HEPEVT Event file name
	PATH_data = f'data/{args.exp}/3plus1/m4_{args.M4:.4g}_mzprime_{args.mzprime:.4g}_{args.hierarchy}_{args.D_or_M}/'
	PATH = f'plots/{args.exp}/3plus1/m4_{args.M4:.4g}_mzprime_{args.mzprime:.4g}/'
	
	# title for plots
	power = int(np.log10(args.UMU4**2))-1
	title = r"$m_{4} = \,$"+str(round(args.M4,4))+r" GeV, $M_{Z^\prime} = \,$"+str(round(args.mzprime,4))+r" GeV, $|U_{D4}|^2=%1.1g$, $|U_{\mu 4}|^2=%1.1f \times 10^{-%i}$"%(args.UD4**2,args.UMU4**2/10**(power),-power)


	# print(bag['I']/bag['I_decay']*const.NAvo*1e6*13e20*818*0.05/12)
	# print(bag['I_decay'])
	############################################################################
	# Print events to file -- currently in data/exp/m4____mzprime____.dat 
	#############################################################################
	printer.print_events_to_pandas(PATH_data, bag, BSMparams,
									l_decay_proper=args.ldecay,
									out_file_name=f"MC_m4_{BSMparams.m4:.8g}_mzprime_{BSMparams.Mzprime:.8g}.pckl")

if __name__ == "__main__":
	try:
		main(sys.argv[1:])
	except (KeyboardInterrupt, SystemExit):
		raise
