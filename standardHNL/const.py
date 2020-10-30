import numpy as np
import scipy
from scipy import interpolate

## FLAGS
electron = 1
muon = 2
tau = 3

charged_pion = 400
charged_kaon = 500

neutral_pion = 411
neutral_kaon = 511
neutral_eta = 611


neutrino0 = 99
neutrino1 = 11
neutrino2 = 22
neutrino3 = 33
neutrino4 = 44
neutrino5 = 55

neutrino_light 	= 999
neutrino_electron 	= 10
neutrino_muon 		= 20
neutrino_tau 		= 30
neutrino_sterile	= 40
neutrino_dark 		= 50



## MASSES in GeV

higgsvev = 246 # GeV

Me  = 511e-6 
Mmu = 0.105
Mtau = 1.777 

mproton = 0.938
mneutron = 0.939
MAVG = (mproton + mneutron)/2.0


Mw = 80.35 
Mz = 91
higgsvev = 246.22 

## COUPLINGS 
s2w = 0.231
sw = np.sqrt(0.231)
cw = np.sqrt(1.0 - s2w)

gl_lepton = -0.5 + s2w
gr_lepton = s2w


eQED = np.sqrt(4.0*np.pi/137.0)
alphaQED = 1./137.0359991

gvP = 1.0
Gf = 1.16e-5 # GeV^-2
g = np.sqrt(Gf*8/np.sqrt(2)*Mw*Mw)
gweak = np.sqrt(Gf*8/np.sqrt(2)*Mw*Mw)

# FORM FACTOR CONSTANTS
gA = 1.26
tau3 = 1

MAG_N = -1.913
MAG_P = 2.792

# charged hadrons
Mcharged_pion = 0.1396
Mcharged_kaon = 0.4937
Mcharged_rho = 0.7758

Fcharged_pion = 0.1307
Fcharged_kaon = 0.1598
Fcharged_rho = 0.220

# neutral hadrons
# charged hadrons
Mneutral_pion = 0.135
Mneutral_eta = 0.5478
Mneutral_rho = 0.7755

Fneutral_pion = 0.130
Fneutral_kaon = 0.1647
Fneutral_eta = 0.210



#########
# CKM elements
Vud = 0.97420
Vus = 0.2243
Vcd = 0.218
Vcs = 0.997
Vcb = 42.2e-3
Vub = 3.94e-3
Vtd = 8.1e-3 
Vts = 39.4e-3
Vtb = 1

################

#Avogadro's number
NAvo = 6.022*1e23
# from GeV^-2 to cm^2
GeV2_to_cm2 = 3.9204e-28

# speed of light (PDG) m/s
c_LIGHT = 299792458

## FORM FACTORS
def D(Q2):
	return 1.0/((1+Q2/0.84/0.84)**2)


def H1_p(Q2):
	tau = -Q2/4.0/mproton/mproton
	F1 = (D(Q2) - tau*MAG_P*D(Q2))/(1-tau)
	F2 = (MAG_P*D(Q2) - D(Q2))/(1-tau)
	return F1**2 - tau * F2**2
def H2_p(Q2):
	tau = -Q2/4.0/mproton/mproton
	F1 = (D(Q2) - tau*MAG_P*D(Q2))/(1-tau)
	F2 = (MAG_P*D(Q2) - D(Q2))/(1-tau)
	return (F1+F2)*(F1+F2)



def H1_n(Q2):
	tau = -Q2/4.0/mproton/mproton
	F1 = (- tau*MAG_N*D(Q2))/(1-tau)
	F2 = (MAG_N*D(Q2))/(1-tau)
	return F1**2 - tau * F2**2

def H2_n(Q2):
	tau = -Q2/4.0/mproton/mproton
	F1 = (-tau*MAG_N*D(Q2))/(1-tau)
	F2 = (MAG_N*D(Q2))/(1-tau)
	return (F1+F2)*(F1+F2)



def F1_EM(Q2):
	tau = -Q2/4.0/mproton/mproton
	return (D(Q2) - tau*MAG_P*D(Q2))/(1-tau)

def F2_EM(Q2):
	tau = -Q2/4.0/mproton/mproton
	return (MAG_P*D(Q2) - D(Q2))/(1-tau)



def F1_WEAK(Q2):
	tau = -Q2/4.0/mproton/mproton
	f = (0.5 - s2w)*(tau3)*(1-tau*(1+MAG_P-MAG_N))/(1-tau) - s2w*(1-tau*(1+MAG_P+MAG_N))/(1-tau)  
	return f*D(Q2)

def F2_WEAK(Q2):
	tau = -Q2/4.0/mproton/mproton
	f = (0.5 - s2w)*(tau3)*(MAG_P-MAG_N)/(1-tau) - s2w*(MAG_P+MAG_N)/(1-tau)  
	return f*D(Q2)

def F3_WEAK(Q2):
	f = gA*tau3/2.0/(1+Q2/1.02/1.02)**2
	return f



def Fcoh(Q,MA):
	a = 0.523/0.197 # GeV^-1
	r0 = 1.03*(MA**(1.0/3.0))/0.197 # GeV^-1
	return 3.0*np.pi*a/(r0**2 + np.pi**2 * a**2) * (np.pi*a *(1.0/np.tanh(np.pi*a*Q))*np.sin(Q*r0) - r0*np.cos(Q*r0))/(Q*r0*np.sinh(np.pi*Q*a))




#########################################
# EXPERIMENTAL PARAMETERS

############### MiniBooNE ###############
# signal def
MB_THRESHOLD = 0.03 # GeV
MB_ANGLE_MAX = 8 # degrees
# cuts
MB_ENU_MIN = 0.2 # GeV
MB_ENU_MAX = 1.5 # GeV
MB_Q2   = 1e10 # GeV^2
MB_ANALYSIS_TH = 0.2 # GeV
# resolutions
MB_STOCHASTIC = 0.12
MB_NOISE = 0.01
MB_ANGULAR = 3*np.pi/180.0

############### MINERVA ###############
# signal def
MV_THRESHOLD = 0.030 # GeV
MV_ANGLE_MAX = 8 # degrees
# cuts
MV_ETHETA2     = 0.0032 # GeV
MV_Q2          = 0.02 # GeV^2
MV_ANALYSIS_TH = 0.8 # GeV
# resolutions
MV_STOCHASTIC = 0.034
MV_NOISE      = 0.059
MV_ANGULAR    = 1*np.pi/180.0

########### CHARM-II ###############
# signal def
CH_THRESHOLD = 0.2 # GeV
CH_ANGLE_MAX = 4 # degrees

# cuts
CH_ANALYSIS_TH = 3.0 # GeV
CH_ANALYSIS_EMAX = 24.0 # GeV
CH_ETHETA2   = 0.03 # GeV
# resolutions
CH_STOCHASTIC = 0.09
CH_NOISE = 0.15
# For Charm's angular resolution, see CH_smear function in cuts.py
CH_ANGULAR = 2*np.pi/180.0



##### dEdX tables

# E, D1, D2, D3, delta = np.loadtxt("NIST_tables/electron_dedx_minerva.dat", skiprows=8, unpack = True)
# dedx_coll = scipy.interpolate.interp1d(E*1e-3, D1, kind='linear', bounds_error=False, fill_value=-1)
# dedx_rad = scipy.interpolate.interp1d(E*1e-3, D2, kind='linear', bounds_error=False, fill_value=-1)
# dedx_tot = scipy.interpolate.interp1d(E*1e-3, D3, kind='linear', bounds_error=False, fill_value=-1)
