import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import datetime, time
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

	parser.add_argument("--mzprime_min", type=float, help="Z' mass min", default=0.01)
	parser.add_argument("--mzprime_max", type=float, help="Z' mass max", default=0.05)
	
	parser.add_argument("--M4", type=float, help="fourth neutrino mass", default=0.100)
	parser.add_argument("--M5", type=float, help="fifth neutrino mass", default=1e6)
	parser.add_argument("--M6", type=float, help="sixth neutrino mass", default=1e6)

	parser.add_argument("--M4_min", type=float, help="fourth neutrino mass min", default=0.1)
	parser.add_argument("--M4_max", type=float, help="fourth neutrino mass max", default=0.5)

	parser.add_argument("--M5_min", type=float, help="fifth neutrino mass min", default=None)
	parser.add_argument("--M5_max", type=float, help="fifth neutrino mass max", default=None)
	parser.add_argument("--M6_min", type=float, help="sixth neutrino mass min", default=None)
	parser.add_argument("--M6_max", type=float, help="sixth neutrino mass max", default=None)

	parser.add_argument("--UMU4", type=float, help="Umu4 square", default=8.0e-9)
	parser.add_argument("--UMU5", type=float, help="Umu5 square", default=0*26.5e-8)
	parser.add_argument("--UMU6", type=float, help="Umu6 square", default=0*123.0e-8*0.0629)

	parser.add_argument("--UMU4_min", type=float, help="Umu4 square min", default=None)
	parser.add_argument("--UMU4_max", type=float, help="Umu4 square max", default=None)
	parser.add_argument("--UMU5_min", type=float, help="Umu5 square min", default=None)
	parser.add_argument("--UMU5_max", type=float, help="Umu5 square max", default=None)
	parser.add_argument("--UMU6_min", type=float, help="Umu6 square min", default=None)
	parser.add_argument("--UMU6_max", type=float, help="Umu6 square max", default=None)

	parser.add_argument("--hierarchy",
						 type=str,
						 help="light or heavy Z' case",
						 default=const.HM,
						 choices=['light_mediator', 'heavy_mediator'])

	# parser.add_argument("--GPRIME", type=float, help="gprime", default=np.sqrt(4*np.pi*1/4.0))
	# parser.add_argument("--CHI", type=float, help="CHI", default=np.sqrt(2e-10/const.alphaQED)/const.cw)
	parser.add_argument("--alpha_dark", type=float, help="alpha_dark", default=0.25)
	parser.add_argument("--alpha_epsilon2", type=float, help="Product of alpha QED times epsilon^2", default=2e-10)
	parser.add_argument("--epsilon2", type=float, help="epsilon^2")

	parser.add_argument("--UTAU4", type=float, help="UTAU4", default=0)
	parser.add_argument("--UTAU5", type=float, help="UTAU5", default=0)
	parser.add_argument("--UTAU6", type=float, help="UTAU6", default=0)

	parser.add_argument("--UD4", type=float, help="UD4", default=1.0)
	parser.add_argument("--UD5", type=float, help="UD5", default=0*1.0)
	parser.add_argument("--UD6", type=float, help="UD6", default=0*1.0)

	parser.add_argument("--ldecay", type=float, help="ctau of the fourth neutrino in cm", default=0.1)
	
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
	
	parser.add_argument("--neval", type=int, help="number of evaluations of integrand", default=1e5)
	parser.add_argument("--nint", type=int, help="number of adaptive iterations", default=20)
	parser.add_argument("--neval_warmup", type=int, help="number of evaluations of integrand in warmup", default=1e3)
	parser.add_argument("--nint_warmup", type=int, help="number of adaptive iterations in warmup", default=10)

	parser.add_argument("--hepevt_events", type=int, help="number of events to accept in HEPEVT format", default=1e2)

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
	BSMparams.Utau4 = args.UTAU4
	BSMparams.Ue5 = 0.0
	BSMparams.Umu5 = np.sqrt(args.UMU5)
	BSMparams.Utau5 = args.UTAU5
	BSMparams.Ue6 = 0.0
	BSMparams.Umu6 = np.sqrt(args.UMU6)
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

	BSMparams.hierarchy=args.hierarchy
	BSMparams.set_KDEscan_mode_on(args)
    
	print(f"Umu4 = {BSMparams.Umu4**2}")
	print(f"epsilon2 = {args.epsilon2}")
	print(f"alpha_dark = {args.alpha_dark}")
	print(f"chi = {BSMparams.chi}, {np.sqrt(args.epsilon2)/const.cw}")
    
	####################################################
	# CHOOSE EXPERIMENTAL FLUXES, NORMALIZATION, ETC...
	myexp = exp.experiment(args.exp)

	####################################################
	# Set the model to use
	MODEL = const.THREEPLUSONE

	####################################################
	# Run MC and get events
	if MODEL==const.THREEPLUSONE:

		bag = MC.run_MC(BSMparams, myexp, [pdg.numu], INCLUDE_DIF=False)

		### NAMING 
		## HEPEVT Event file name
		PATH_data = f'data/{args.exp}/3plus1/scan/{args.hierarchy}/'
		PATH = f'plots/{args.exp}/3plus1/scan/{args.hierarchy}/'
		
	


	############################################################################
	# Print events to file -- currently in data/exp/m4____mzprime____.dat 
	#############################################################################
	print("TEST -- sum of weights = ", np.sum(bag['w'])/const.GeV2_to_cm2)
	print("TEST -- events/12t /13e20 POT = ", np.sum(bag['w'])*const.NAvo*818e6*18e20/12)
	out_file_name = f'{args.M4_min:.3g}_m4_{args.M4_max:.3g}_{args.mzprime_min:.3g}_mzprime_{args.mzprime_max:.3g}_nevt_{args.neval}.pckl'
	printer.print_events_to_pandas(PATH_data, bag, BSMparams, 
									l_decay_proper=args.ldecay,
									out_file_name=out_file_name)
									# out_file_name=datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%Hh-%Mmin-%Ss.pckl'))
	# HEPEVT_events = args.hepevt_events
	# printer.print_unweighted_events_to_HEPEVT(PATH_data, bag, HEPEVT_events, BSMparams, l_decay_proper=args.ldecay)

if __name__ == "__main__":
	try:
		main(sys.argv[1:])
	except (KeyboardInterrupt, SystemExit):
		raise
