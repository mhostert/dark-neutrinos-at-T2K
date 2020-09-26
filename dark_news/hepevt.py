import numpy as np
import os
import sys, getopt

from . import const
from . import pdg
#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

def generate_muboone_HEPEVT_files(PATH_data, bag, TOT_EVENTS, BSMparams,l_decay_proper=0.0):

	# events
	pN   = bag['P3']
	pnu   = bag['P2_decay']
	pZ   = bag['P3_decay']+bag['P4_decay']
	plm  = bag['P3_decay']
	plp  = bag['P4_decay']
	pHad = bag['P4']
	w = bag['w']
	I = bag['I']
	regime = bag['flags']

	# Accept/reject method -- samples distributed according to their weights
	AllEntries = np.array(range(np.shape(plm)[0]))
	AccEntries = np.random.choice(AllEntries, size=TOT_EVENTS, replace=True, p=w/np.sum(w))

	pN, plp, plm, pnu, pHad, w, regime  = pN[AccEntries], plp[AccEntries], plm[AccEntries], pnu[AccEntries], pHad[AccEntries], w[AccEntries], regime[AccEntries]

	size = np.shape(plm)[0]

	# decay the heavy nu
	M4 = np.sqrt(Cfv.dot4(pN,pN))
	MZPRIME = np.sqrt(Cfv.dot4(pZ,pZ))
	Mhad = np.sqrt(Cfv.dot4(pHad,pHad))
	gammabeta_inv = M4/(np.sqrt(pN[:,0]**2 -  M4*M4 ))
	######################
	# *PROPER* decay length -- BY HAND AT THE MOMENT!
	ctau = l_decay_proper
	######################
	d_decay = np.random.exponential(scale=ctau/gammabeta_inv)*1e2 # centimeters


	########################## HEPevt format
	# Detector geometry -- choose random position
	xmin=0;xmax=256. # cm
	ymin=-115.;ymax=115. # cm
	zmin=0.;zmax=1045. # cm

	########
	# Using initial time of the det only! 
	# CROSS CHECK THIS VALUE
	tmin=3.200e3;tmax=3.200e3 # ticks? ns?


	#####################
	# scaling it to a smaller size around the central value
	restriction = 0.3 

	xmax = xmax - restriction*(xmax -xmin)
	ymax = ymax - restriction*(ymax -ymin)
	zmax = zmax - restriction*(zmax -zmin)
	tmax = tmax - restriction*(tmax -tmin)

	xmin = xmin + restriction*(xmax-xmin)
	ymin = ymin + restriction*(ymax-ymin)
	zmin = zmin + restriction*(zmax-zmin)
	tmin = tmin + restriction*(tmax-tmin)

	# generating entries
	x = xmin + (xmax -xmin)*np.random.rand(size) 
	y = ymin + (ymax -ymin)*np.random.rand(size) 
	z = zmin + (zmax -zmin)*np.random.rand(size) 
	t = tmin + (tmax -tmin)*np.random.rand(size) 


	# direction of N
	x_decay = x + Cfv.get_3direction(pN)[:,0]*d_decay
	y_decay = y + Cfv.get_3direction(pN)[:,1]*d_decay
	z_decay = z + Cfv.get_3direction(pN)[:,2]*d_decay
	t_decay = t + d_decay/const.c_LIGHT/np.sqrt(pN[:,0]**2 - (M4)**2)*M4


	# Create target Directory if it doesn't exist
	if not os.path.exists(PATH_data):
	    os.makedirs(PATH_data)


	###############################################
	# SAVE ALL EVENTS AS AN ARRAY TO A BINARY FILE 
	npy_file_name = PATH_data+"MC_m4_"+format(BSMparams.m4,'.8g')+"_mzprime_"+format(BSMparams.Mzprime,'.8g')
	X = np.array([
		plm,
		plp,
		pnu,
		pHad,
		np.array([t,x,y,z]).T,
		np.array([t_decay,x_decay,y_decay,z_decay]).T])
	np.save(npy_file_name,X,allow_pickle=True)

	###############################################
	# SAVE ALL EVENTS AS A HEPEVT .dat file
	hepevt_file_name = PATH_data+"MC_m4_"+format(BSMparams.m4,'.8g')+"_mzprime_"+format(BSMparams.Mzprime,'.8g')+".dat"
	# Open file in write mode
	f = open(hepevt_file_name,"w+") 

	# f.write("%i\n",TOT_EVENTS)
	# loop over events
	for i in range(TOT_EVENTS):
		f.write("%i 4\n" % i)
		f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.electron,plm[i][1], plm[i][2], plm[i][3], plm[i][0], const.Me, x_decay[i], y_decay[i], z_decay[i],t_decay[i]))
		f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.positron,plp[i][1], plp[i][2], plp[i][3], plp[i][0], const.Me, x_decay[i], y_decay[i], z_decay[i],t_decay[i]))	
		f.write("2 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.numu,pnu[i][1], pnu[i][2], pnu[i][3], pnu[i][0], const.Me, x_decay[i], y_decay[i], z_decay[i], t_decay[i]))
		if (regime[i] == const.COHRH or regime[i] == const.COHLH):
			f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.Argon40,pHad[i][1], pHad[i][2], pHad[i][3], pHad[i][0], Mhad[i],x[i],y[i],z[i],t[i]))
		elif (regime[i] == const.DIFRH or regime[i] == const.DIFLH):
			f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.proton,pHad[i][1], pHad[i][2], pHad[i][3], pHad[i][0], Mhad[i],x[i],y[i],z[i],t[i]))
		else:
			print('Error! Cannot find regime of event ', i)
	f.close()