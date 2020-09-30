import numpy as np
import vegas as vg
import gvar as gv
import random
import scipy 

from . import cuts
from dark_news import *


#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

OVERLAPPING_TYPE = 1
ASYMMETRIC_TYPE  = 2
BOTH_TYPE        = 3
SEPARATED_TYPE   = 4



def compute_MB_spectrum(bag):
	
	pN   = bag['P3']
	pnu   = bag['P2_decay']
	pZ   = bag['P3_decay']+bag['P4_decay']
	plm  = bag['P3_decay']
	plp  = bag['P4_decay']
	pHad = bag['P4']
	w = bag['w']
	I = bag['I']
	regime = bag['flags']

	sample_size = np.shape(plp)[0]

	########################## PROCESS FOR DISTRIBUTIONS ##################################################
	plp = cuts.MB_smear(plp,const.Me)
	plm = cuts.MB_smear(plm,const.Me)


	costhetaN = pN[:,3]/np.sqrt( Cfv.dot3(pN,pN) )
	costhetanu = pnu[:,3]/np.sqrt( Cfv.dot3(pnu,pnu) )
	costhetaHad = pHad[:,3]/np.sqrt( Cfv.dot3(pHad,pHad) )
	invmass = np.sqrt( Cfv.dot4(plm + plp, plm + plp) )
	EN   = pN[:,0] 
	EZ = pZ[:,0]
	Elp  = plp[:,0]
	Elm  = plm[:,0]
	EHad = pHad[:,0]

	Mhad = np.sqrt( Cfv.dot4(pHad, pHad) )
	Mn = np.sqrt( Cfv.dot4(pN,pN))
	Q2 = -(2*Mhad*Mhad-2*EHad*Mhad)
	Q = np.sqrt(Q2)

	costheta_sum = (plm+plp)[:,3]/np.sqrt( Cfv.dot3(plm+plp,plm+plp) )
	costhetalp = plp[:,3]/np.sqrt( Cfv.dot3(plp,plp) )

	costhetalm = plm[:,3]/np.sqrt( Cfv.dot3(plm,plm) )
	Delta_costheta = Cfv.dot3(plm,plp)/np.sqrt(Cfv.dot3(plm,plm))/np.sqrt(Cfv.dot3(plp,plp))



	############################
	# Decay
	# gammabeta = np.sqrt(EN**2-Mn**2)/Mn
	#####################
	# # *PROPER* decay length -- BY HAND AT THE MOMENT!
	# ctau = 10.# const.c_LIGHT/(NgammaTOT*1.52e24)
	# ######################
	# scale_exp = 200.0/gammabeta/ctau
	# # w = (1.0 - np.exp(-scale_exp))*w # centimeters
	# print(1.0 - np.exp(-scale_exp))

	Evis, theta_beam, w, eff_s, ind1 = signal_events(plp,plm, w, 0.030,13,BOTH_TYPE)
	print("Signal spoofing efficiency at MB: ", eff_s)
	# print(Evis,theta_beam,w)
	Evis, theta_beam, w, eff_c, ind2 = MB_expcuts(Evis, theta_beam, w)
	print("Analysis cuts efficiency at MB: ", eff_c)
	# print(Evis,theta_beam,w)
	my_eff = eff_c*eff_s
	print("My efficiency at MB: ", my_eff)


	Enu = const.mproton * (Evis) / ( const.mproton - (Evis)*(1.0 - np.cos(theta_beam)))

	eff = np.array([0.0,0.089,0.135,0.139,0.131,0.123,0.116,0.106,0.102,0.095,0.089,0.082,0.073,0.067,0.052,0.026])
	enu = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.3,1.5,1.7,1.9,2.1])
	enu_c =  enu[:-1]+(enu[1:] - enu[:-1])/2
	eff_func = scipy.interpolate.interp1d(enu_c,eff,fill_value=(eff[0],eff[-1]),bounds_error=False,kind='nearest')


	# Total nueCCQE efficiency 
	E_final_eff, nueCCQE_final_eff = np.genfromtxt('digitized/Patterson_miniboone/nueCCQE_final_eff.dat',unpack=True)
	nueCCQE_final_eff_func = scipy.interpolate.interp1d(E_final_eff*1e-3,nueCCQE_final_eff,fill_value=(nueCCQE_final_eff[0],nueCCQE_final_eff[-1]),bounds_error=False,kind='nearest')

	# nueCCQE efficiency of pi/e & mu/e cuts 
	E_mupiPID_eff, nueCCQE_mupiPID_eff = np.genfromtxt('digitized/Patterson_miniboone/nueCCQE_mu_pi_eff.dat',unpack=True)
	nueCCQE_mupiPID_eff_func = scipy.interpolate.interp1d(E_mupiPID_eff*1e-3,nueCCQE_mupiPID_eff,fill_value=(nueCCQE_mupiPID_eff[0],nueCCQE_mupiPID_eff[-1]),bounds_error=False,kind='nearest')

	# nueCCQE efficiency of mu/e PID cut
	E_muPID_eff, nueCCQE_muPID_eff = np.genfromtxt('digitized/Patterson_miniboone/nueCCQE_mu_eff.dat',unpack=True)
	nueCCQE_muPID_eff_func = scipy.interpolate.interp1d(E_muPID_eff*1e-3,nueCCQE_muPID_eff,fill_value=(nueCCQE_muPID_eff[0],nueCCQE_muPID_eff[-1]),bounds_error=False,kind='nearest')



	# Computing the efficiency of mu PID cut TIMES PMT efficiency which is inside eff_fuc
	# Assumes eff_func = eff_PMT * nueCCQE_final_eff_func
	eff_miniboone = np.sum(w*nueCCQE_muPID_eff_func(Enu)*eff_func(Enu)/nueCCQE_final_eff_func(Enu))/np.sum(w)
	print("MB efficiency: ", eff_miniboone)

	eff_final = eff_miniboone*my_eff
	print("FINAL efficiency: ", eff_final)

	w = eff_func(Enu)*w
	regime = regime[ind1][ind2]

	observables = {'Enu': Enu, 'Evis' : Evis, 'theta_beam' : theta_beam, 'w' : w, 'eff' : eff_final, 'regime' : regime}
	return observables


####
# Clears the momenta of interest of any NaNs (usually from numerical errors)
def clean_events(p1, p2, p3, p4, p5, w):
	
			# mask = ~np.isnan(x[:,0])
			# for i in range(1,DIM):
			# 	mask *=(~np.isnan(x[:,i]))
			# mask *= (~np.isnan(wgt))
			# wgt = wgt[mask]
			# print 'before',np.shape(x)
			# for i in range(0,DIM):
			# 	x[:,i] = x[:,i][mask]
			# print 'after',np.shape(x)
			# print 'mask',np.sum(mask)
				
	pnew = []
	p2new = []
	p3new = []
	p4new = []
	p5new = []
	wnew = []
	# print fourvec.dot4(p[3], p[3])
	for i in range(np.size(w)):
		x = p1[i]
		x2 = p2[i]
		x3 = p3[i]
		x4 = p4[i]
		x5 = p5[i]
		if( np.isfinite(x[0]) and np.isfinite(x[1]) and np.isfinite(x[2]) and np.isfinite(x[3]) \
					and np.isfinite(x2[0]) and np.isfinite(x2[1]) and np.isfinite(x2[2]) and np.isfinite(x2[3])\
					 and np.isfinite(x3[0]) and np.isfinite(x3[1]) and np.isfinite(x3[2]) and np.isfinite(x3[3])\
					  and np.isfinite(x4[0]) and np.isfinite(x4[1]) and np.isfinite(x4[2]) and np.isfinite(x4[3])\
					  and np.isfinite(x5[0]) and np.isfinite(x5[1]) and np.isfinite(x5[2]) and np.isfinite(x5[3])) :
			pnew.append(x)
			p2new.append(x2)
			p3new.append(x3)
			p4new.append(x4)
			p5new.append(x5)
			wnew.append(w[i])

	return np.array(pnew), np.array(p2new), np.array(p3new), np.array(p4new), np.array(p5new), np.array(wnew)


def signal_events(pep, pem, w, THRESHOLD, ANGLE_MAX, TYPE):

	################### PROCESS KINEMATICS ##################
	size_samples = np.shape(pep)[0]

	# electron kinematics
	Eep = np.array([ fourvec.dot4(pep[i],fourvec.k0) for i in range(size_samples)])

	Eem = np.array([ fourvec.dot4(pem[i],fourvec.k0) for i in range(size_samples)])

	# angle of separation between ee
	cosdelta_ee = np.array([ fourvec.dot3(pep[i],pem[i])/np.sqrt( fourvec.dot3(pem[i],pem[i]))/np.sqrt( fourvec.dot3(pep[i],pep[i]) ) for i in range(size_samples)])
	theta_ee = np.arccos(cosdelta_ee)*180.0/np.pi

	# two individual angles
	costheta_ep = np.array([ fourvec.dot3(pep[i],fourvec.kz)/np.sqrt( fourvec.dot3(pep[i],pep[i])) for i in range(size_samples)])
	theta_ep = np.arccos(costheta_ep)*180.0/np.pi

	costheta_em = np.array([ fourvec.dot3(pem[i],fourvec.kz)/np.sqrt( fourvec.dot3(pem[i],pem[i])) for i in range(size_samples)])
	theta_em = np.arccos(costheta_em)*180.0/np.pi

	# this is the angle of the combination of ee with the neutrino beam
	costheta_comb = np.array([ fourvec.dot3(pem[i]+pep[i],fourvec.kz)/np.sqrt( fourvec.dot3(pem[i]+pep[i],pem[i]+pep[i])) for i in range(size_samples)])
	theta_comb = np.arccos(costheta_comb)*180.0/np.pi

	########################################
	w_asym = 0.0
	w_ovl = 0.0
	w_sep = 0.0
	w_inv = 0.0
	indices_asym_p=[]
	indices_asym_m=[]
	indices_ovl=[]
	indices_sep=[]
	indices_inv=[]
	w_tot = np.sum(w)

	## All obey experimental threshold
	## 
	for i in range(size_samples):
		# asymmetric positive one
		if ((Eem[i] - const.Me < THRESHOLD) and 
				(Eep[i] - const.Me > THRESHOLD)):
			w_asym += w[i]
			indices_asym_p.append(i)

		# asymmetric minus one
		if ((Eep[i] - const.Me < THRESHOLD) and 
				(Eem[i] - const.Me > THRESHOLD)):
			w_asym += w[i]
			indices_asym_m.append(i)

		# overlapping
		if ((Eep[i] - const.Me > THRESHOLD) and 
				(Eem[i] - const.Me > THRESHOLD) and
		 		(np.arccos(cosdelta_ee[i])*180.0/np.pi < ANGLE_MAX)):
			w_ovl += w[i]
			indices_ovl.append(i)

		# separated
		if ((Eep[i] - const.Me > THRESHOLD) and \
				(Eem[i] - const.Me > THRESHOLD) and \
		 	 (np.arccos(cosdelta_ee[i])*180.0/np.pi > ANGLE_MAX)):
			w_sep += w[i]
			indices_sep.append(i)

		# invisible
		if ((Eep[i] - const.Me < THRESHOLD) and \
				(Eem[i] - const.Me < THRESHOLD)):
			w_inv += w[i]
			indices_inv.append(i)


	indices_asym_p = np.array(indices_asym_p)
	indices_asym_m = np.array(indices_asym_m)
	indices_ovl = np.array(indices_ovl)
	indices_sep = np.array(indices_sep)
	indices_inv = np.array(indices_inv)

	eff_asym = w_asym/w_tot	
	eff_ovl 	= w_ovl/w_tot	
	eff_sep 	= w_sep/w_tot	
	eff_inv 	= w_inv/w_tot	

	# print "Efficiency for asym -> ", eff_asym*100,"%"
	# print "Efficiency for ovl -> ", eff_ovl*100,"%"
	# print "Efficiency for sep -> ", eff_sep*100,"%"
	# print "Efficiency for inv -> ", eff_inv*100,"%"

	if TYPE==OVERLAPPING_TYPE:
		######################### FINAL OBSERVABLES ##########################################
		# visible energy
		Evis = Eep[indices_ovl] + Eem[indices_ovl] + const.Me

		# angle to the beam
		theta_beam = theta_comb[indices_ovl]

		# weights
		weights = w[indices_ovl]

		return Evis, theta_beam, weights, eff_ovl, indices_ovl
	elif TYPE==ASYMMETRIC_TYPE:
		######################### FINAL OBSERVABLES ##########################################
		# visible energy
		Evis = np.append(Eep[indices_asym_p], Eem[indices_asym_m])

		# angle to the beam
		theta_beam =np.append(theta_ep[indices_asym_p], theta_em[indices_asym_m])

		# weights
		weights = np.append(w[indices_asym_p], w[indices_asym_m])

		return Evis, theta_beam, weights, eff_asym, np.append(indices_asym_p, indices_asym_m)
	elif TYPE==BOTH_TYPE:
		######################### FINAL OBSERVABLES ##########################################
		# visible energy
		Evis = np.append(np.append(Eep[indices_asym_p], Eem[indices_asym_m]), Eep[indices_ovl] + Eem[indices_ovl] + const.Me)

		# angle to the beam
		theta_beam = np.append(np.append(theta_ep[indices_asym_p], theta_em[indices_asym_m]), theta_comb[indices_ovl])

		# weights
		weights = np.append(np.append(w[indices_asym_p], w[indices_asym_m]), w[indices_ovl])

		return Evis, theta_beam, weights, eff_asym+eff_ovl, np.append(np.append(indices_asym_p, indices_asym_m), indices_ovl)

	elif TYPE==SEPARATED_TYPE:
		######################### FINAL OBSERVABLES ##########################################
		# visible energy
		Eplus = Eep[indices_sep]
		Eminus= Eem[indices_sep]

		# angle to the beam
		theta_beam_plus = theta_ep[indices_sep]
		theta_beam_minus = theta_em[indices_sep]
		theta_sep = theta_ee[indices_sep]

		# weights
		weights = w[indices_sep]

		return Eplus, Eminus, theta_beam_plus, theta_beam_minus, theta_sep, weights, eff_sep



def true_events(pep, pem, w, THRESHOLD, ANGLE_MAX, TYPE):

	################### PROCESS KINEMATICS ##################
	size_samples = np.shape(pep)[0]

	# electron kinematics
	Eep = pep[0,:]
	Eem = pem[0,:]

	# angle of separation between ee
	cosdelta_ee = Cfv.dot3(pep,pem)/np.sqrt( Cfv.dot3(pem,pem))/np.sqrt( Cfv.dot3(pep,pep) )
	theta_ee = np.arccos(cosdelta_ee)*180.0/np.pi

	# two individual angles
	costheta_ep = Cfv.get_cosTheta(pep)
	theta_ep = np.arccos(costheta_ep)*180.0/np.pi

	costheta_em = Cfv.get_cosTheta(pem)
	theta_em = np.arccos(costheta_em)*180.0/np.pi

	# this is the angle of the combination of ee with the neutrino beam
	costheta_comb = Cfv.get_cosTheta(pem+pep)
	theta_comb = np.arccos(costheta_comb)*180.0/np.pi

	mee =  np.sqrt( Cfv.dot4(samples[i],samples[i]))
	mee_cut = 0.03203 + 0.007417*(Eem+Eep) + 0.02738*(Eem+Eep)**2


	########################################
	w_asym = 0.0
	w_ovl = 0.0
	w_sep = 0.0
	w_inv = 0.0
	indices_asym_p=[]
	indices_asym_m=[]
	indices_ovl=[]
	indices_sep=[]
	indices_inv=[]
	w_tot = np.sum(w)

	## All obey experimental threshold
	## 
	for i in range(size_samples):
		# asymmetric positive one
		if ((Eem[i] - const.Me < THRESHOLD) and 
				(Eep[i] - const.Me > THRESHOLD) and 
					 mee[i] < mee_cut[i]):
			w_asym += w[i]
			indices_asym_p.append(i)

		# asymmetric minus one
		if ((Eep[i] - const.Me < THRESHOLD) and 
				(Eem[i] - const.Me > THRESHOLD) and 
				 	mee[i] < mee_cut[i]):
			w_asym += w[i]
			indices_asym_m.append(i)

		# overlapping
		if ((Eep[i] - const.Me > THRESHOLD) and 
				(Eem[i] - const.Me > THRESHOLD) and
		 			(np.arccos(cosdelta_ee[i])*180.0/np.pi < ANGLE_MAX) and 
						 mee[i] < mee_cut[i]):
			w_ovl += w[i]
			indices_ovl.append(i)

		# separed
		if ((Eep[i] - const.Me > THRESHOLD) and \
				(Eem[i] - const.Me > THRESHOLD) and \
		 	 (np.arccos(cosdelta_ee[i])*180.0/np.pi > ANGLE_MAX) and 
					 mee[i] < mee_cut[i]):
			w_sep += w[i]
			indices_sep.append(i)

		# invisible
		if ((Eep[i] - const.Me < THRESHOLD) and \
				(Eem[i] - const.Me < THRESHOLD)):
			w_inv += w[i]
			indices_inv.append(i)


	indices_asym_p = np.array(indices_asym_p)
	indices_asym_m = np.array(indices_asym_m)
	indices_ovl = np.array(indices_ovl)
	indices_sep = np.array(indices_sep)
	indices_inv = np.array(indices_inv)

	eff_asym = w_asym/w_tot	
	eff_ovl 	= w_ovl/w_tot	
	eff_sep 	= w_sep/w_tot	
	eff_inv 	= w_inv/w_tot	

	if TYPE==OVERLAPPING_TYPE:
		return indices_ovl, eff_ovl
	elif TYPE==ASYMMETRIC_TYPE:
		return indices_asym_p, indices_asym_m, eff_ovl
	elif TYPE==SEPARATED_TYPE:
		return indices_sep, eff_sep




def MB_expcuts(Evis, theta, weights):

	## Experimental cuts
	w = 0.0
	indices = []
	w_tot = np.sum(weights)

	Pe = np.sqrt(Evis**2 - const.Me**2)
	Enu = (const.mneutron*Evis - 0.5*const.Me**2)/(const.mneutron - Evis + Pe*np.cos(theta*np.pi/180.0))
	Q2 = 2*const.mneutron*(Enu - Evis)

	for i in range(np.size(Evis)):
		# asymmetric positive one
		if ((Evis[i] > const.MB_Evis_MIN) and \
				(Evis[i] < const.MB_Evis_MAX)):
			w += weights[i]
			indices.append(i)



	indices = np.array(indices)
	eff = w/w_tot	
	return Evis[indices], theta[indices], weights[indices], eff, indices

def MV_expcuts(Evis, theta, weights):

	## Experimental cuts
	w = 0.0
	indices = []
	w_tot = np.sum(weights)

	Pe = np.sqrt(Evis**2 - const.Me**2)
	Enu = (const.mneutron*Evis - 0.5*const.Me**2)/ (const.mneutron - Evis + Pe*np.cos(theta*np.pi/180.0))
	Q2 = 2*const.mneutron*(Enu - Evis)

	for i in range(np.size(Evis)):
		# asymmetric positive one
		if ((Evis[i] > const.MV_ANALYSIS_TH) and \
				(Evis[i]*(theta[i]*np.pi/180.0)**2 < const.MV_ETHETA2) and \
				(Q2[i] < const.MV_Q2)):
			w += weights[i]
			indices.append(i)

	indices = np.array(indices)
	eff = w/w_tot	
	return Evis[indices], theta[indices], weights[indices], eff

def CH_expcuts(Evis, theta, weights):

	## Experimental cuts
	w = 0.0
	indices = []
	w_tot = np.sum(weights)

	Pe = np.sqrt(Evis**2 - const.Me**2)
	Enu = (const.mneutron*Evis - 0.5*const.Me**2)/ (const.mneutron - Evis + Pe*np.cos(theta*np.pi/180.0))
	Q2 = 2*const.mneutron*(Enu - Evis)

	for i in range(np.size(Evis)):
		# asymmetric positive one
		if ((Evis[i] > const.CH_ANALYSIS_TH) and (Evis[i] < const.CH_ANALYSIS_EMAX) and \
				(Evis[i]*(theta[i]*np.pi/180.0)**2 < const.CH_ETHETA2)):
			w += weights[i]
			indices.append(i)

	indices = np.array(indices)
	eff = w/w_tot	
	return Evis[indices], theta[indices], weights[indices], eff
