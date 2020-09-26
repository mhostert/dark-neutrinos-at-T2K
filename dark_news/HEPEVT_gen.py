import numpy as np=
from . import const

def generate_muboone_HEPEVT_files(efile_name,bag,BSMparams,l_decay_proper=0.0):

	# events
	pN   = bag['P3']
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

	plp, plm, pnu, pHad, w, regime  = plp[AccEntries], plm[AccEntries], pnu[AccEntries], pHad[AccEntries], w[AccEntries], regime[AccEntries]

	size = np.shape(plm)[0]


	# decay the heavy nu
	NgammaTOT = decayrates.N_total(BSMparams)
	MHEAVY = BSMparams.m5
	gammabeta_inv = MHEAVY/(np.sqrt(pN[:,0]**2 -  MHEAVY*MHEAVY ))
	######################
	# *PROPER* decay length -- BY HAND AT THE MOMENT!
	ctau = l_decay_proper# const.c_LIGHT/(NgammaTOT*1.52e24)
	######################
	scale_exp = gammabeta_inv/ctau
	d_decay = np.random.exponential(scale=1.0/scale_exp)*1e2 # centimeters
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
	x_decay = np.array([ x[i] + fourvec.get_direction(pN[i])[0]*d_decay[i] for i in range(TOT_EVENTS)])
	y_decay = np.array([ y[i] + fourvec.get_direction(pN[i])[1]*d_decay[i] for i in range(TOT_EVENTS)])
	z_decay = np.array([ z[i] + fourvec.get_direction(pN[i])[2]*d_decay[i] for i in range(TOT_EVENTS)])
	t_decay = np.array([ t[i] + d_decay[i]/const.c_LIGHT/np.sqrt(pN[i,0]**2 - (MHEAVY)**2)*MHEAVY for i in range(TOT_EVENTS)])


	# Create target Directory if it doesn't exist
	if not os.path.exists(PATH_data):
	    os.makedirs(PATH_data)

	# Open file in write mode
	f = open(efile_name,"w+") 

	# f.write("%i\n",TOT_EVENTS)
	# loop over events
	for i in range(TOT_EVENTS):
		f.write("%i 4\n" % i)
		f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.PDG_electron,plm[i][1], plm[i][2], plm[i][3], plm[i][0], const.Me, x_decay[i], y_decay[i], z_decay[i],t_decay[i]))
		f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.PDG_positron,plp[i][1], plp[i][2], plp[i][3], plp[i][0], const.Me, x_decay[i], y_decay[i], z_decay[i],t_decay[i]))	
		f.write("2 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.PDG_numu,pnu[i][1], pnu[i][2], pnu[i][3], pnu[i][0], const.Me, x_decay[i], y_decay[i], z_decay[i], t_decay[i]))
		if (regime[i] == 0):
			f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.PDG_Argon40,pHad[i][1], pHad[i][2], pHad[i][3], pHad[i][0], A_NUMBER*const.MAVG,x[i],y[i],z[i],t[i]))
		elif (regime[i]==1):
			f.write("1 %i 0 0 0 0 %f %f %f %f %f %f %f %f %f\n"%(pdg.PDG_proton,pHad[i][1], pHad[i][2], pHad[i][3], pHad[i][0], const.mproton,x[i],y[i],z[i],t[i]))
		else:
			print('Error! Cannot find regime of event ', i)
	f.close()