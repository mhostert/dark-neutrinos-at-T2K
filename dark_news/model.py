import numpy as np

from . import const 

class model_params():

	def __init__(self):

		self.gprime		= 1.0
		self.chi		= 1.0
		self.Ue4		= 0.0
		self.Umu4		= 0.0
		self.Utau4		= 0.0
		self.Ue5		= 0.0
		self.Umu5		= 0.0
		self.Utau5		= 0.0
		self.Ue6		= 0.0
		self.Umu6		= 0.0
		self.Utau6		= 0.0
		self.UD4		= 1.0
		self.UD5		= 1.0
		self.UD6		= 1.0
		self.m4			= 1.0
		self.m5			= 1.0
		self.m6			= 1.0
		self.Mzprime	= 1.0
		self.Dirac		= 0.0

		self.scan      = False

	def set_KDEscan_mode_on(self,args,hierarchy=const.LM):
		self.scan = True
		
		self.M4_min	= args.M4_min		
		self.M4_max	= args.M4_max

		self.mzprime_min = args.mzprime_min
		self.mzprime_max = args.mzprime_max

		self.number_of_scanned_params = 2
		
		print('** Running in scan mode **\n')


	def set_high_level_variables(self):
		self.Ue1 = np.sqrt(1.0-self.Ue4*self.Ue4-self.Ue5*self.Ue5)
		self.Umu1 = np.sqrt(1.0-self.Umu4*self.Umu4-self.Umu5*self.Umu5)
		self.Utau1 = np.sqrt(1.0-self.Utau4*self.Utau4-self.Utau5*self.Utau5)
		self.UD1 = np.sqrt(self.Ue4*self.Ue4+self.Umu4*self.Umu4+self.Utau4*self.Utau4)

		self.Uactive4SQR = self.Ue4*self.Ue4+self.Umu4*self.Umu4+self.Utau4*self.Utau4
		self.Uactive5SQR = self.Ue5*self.Ue5+self.Umu5*self.Umu5+self.Utau5*self.Utau5
		self.Uactive6SQR = self.Ue6*self.Ue6+self.Umu6*self.Umu6+self.Utau6*self.Utau6

		self.R = self.m4/self.m5
		self.A4 = self.Ue4*self.Ue4+self.Umu4*self.Umu4+self.Utau4*self.Utau4
		self.A5 = self.Ue5*self.Ue5+self.Umu5*self.Umu5+self.Utau5*self.Utau5
		self.A6 = self.Ue6*self.Ue6+self.Umu6*self.Umu6+self.Utau6*self.Utau6
		
		self.D4 = self.UD4*self.UD4#(1.0 - self.A4 - self.A5)/(1.0+self.R)
		self.D5 = self.UD5*self.UD5#(1.0 - self.A4 - self.A5)/(1.0+1.0/self.R)
		self.D6 = self.UD6*self.UD6#(1.0 - self.A4 - self.A5)/(1.0+1.0/self.R)
		#########
		# FIX ME -- this expression depends on a large cancellation -- can we make more stable?
		self.C45SQR = self.A5*(1.0 - self.A5) - self.D5*(1.0-self.D4-self.D5)
		self.D45SQR = self.D4*self.D5 

		# self.UD4 = (1.0 - self.A4 - self.A5)/(1.0+self.R)
		# self.UD5 = (1.0 - self.A4 - self.A5)/(1.0+1.0/self.R)

		self.alphaD = self.gprime*self.gprime/4.0/np.pi
	########################################################
		# all the following is true to leading order in chi


		# Neutrino couplings ## CHECK THE SIGN IN THE SECOND TERM????
		self.ce6 = const.g/2/const.cw* (self.Ue5) + self.UD5*(-self.UD4*self.Ue4 -self.UD5*self.Ue5)*self.gprime*const.sw*self.chi
		self.cmu6 = const.g/2/const.cw* (self.Umu5) + self.UD5*(-self.UD4*self.Umu4 -self.UD5*self.Umu5)*self.gprime*const.sw*self.chi
		self.ctau6 = const.g/2/const.cw* (self.Utau5) + self.UD5*(-self.UD4*self.Utau4 -self.UD5*self.Utau5)*self.gprime*const.sw*self.chi
			
		self.de6 = self.UD6*(-self.UD4*self.Ue4 - self.UD5*self.Ue5 - self.UD6*self.Ue6)*self.gprime
		self.dmu6 = self.UD6*(-self.UD4*self.Umu4 - self.UD5*self.Umu5 - self.UD6*self.Umu6)*self.gprime
		self.dtau6 = self.UD6*(-self.UD4*self.Utau4 - self.UD5*self.Utau5 - self.UD6*self.Utau6)*self.gprime

		self.ce5 = const.g/2/const.cw* (self.Ue5) + self.UD5*(-self.UD4*self.Ue4 -self.UD5*self.Ue5)*self.gprime*const.sw*self.chi
		self.cmu5 = const.g/2/const.cw* (self.Umu5) + self.UD5*(-self.UD4*self.Umu4 -self.UD5*self.Umu5)*self.gprime*const.sw*self.chi
		self.ctau5 = const.g/2/const.cw* (self.Utau5) + self.UD5*(-self.UD4*self.Utau4 -self.UD5*self.Utau5)*self.gprime*const.sw*self.chi
			
		self.de5 = self.UD5*(-self.UD4*self.Ue5 - self.UD5*self.Ue5)*self.gprime
		self.dmu5 = self.UD5*(-self.UD4*self.Umu5 - self.UD5*self.Umu5)*self.gprime
		self.dtau5 = self.UD5*(-self.UD4*self.Utau5 - self.UD5*self.Utau5)*self.gprime

		self.ce4 = const.g/2/const.cw* (self.Ue4) + self.UD4*(-self.UD4*self.Ue4 -self.UD5*self.Ue5)*self.gprime*const.sw*self.chi
		self.cmu4 = const.g/2/const.cw* (self.Umu4) + self.UD4*(-self.UD4*self.Umu4 -self.UD5*self.Umu5)*self.gprime*const.sw*self.chi
		self.ctau4 = const.g/2/const.cw* (self.Utau4) + self.UD4*(-self.UD4*self.Utau4 -self.UD5*self.Utau5)*self.gprime*const.sw*self.chi
		
		self.de4 = self.UD4*(-self.UD4*self.Ue4 - self.UD5*self.Ue4)*self.gprime
		self.dmu4 = self.UD4*(-self.UD4*self.Umu4 - self.UD5*self.Umu4)*self.gprime
		self.dtau4 = self.UD4*(-self.UD4*self.Utau4 - self.UD5*self.Utau4)*self.gprime

		self.clight4 = np.sqrt(self.ce4**2+self.cmu4**2+self.ctau4**2)
		self.dlight4 = np.sqrt(self.de4**2+self.dmu4**2+self.dtau4**2)

		self.clight5 = np.sqrt(self.ce5**2+self.cmu5**2+self.ctau5**2)
		self.dlight5 = np.sqrt(self.de5**2+self.dmu5**2+self.dtau5**2)

		self.clight6 = np.sqrt(self.ce6**2+self.cmu6**2+self.ctau6**2)
		self.dlight6 = np.sqrt(self.de6**2+self.dmu6**2+self.dtau6**2)

		self.c46 = const.g/2.0/const.cw*(np.sqrt(self.Uactive4SQR*self.Uactive6SQR)) + self.UD6*self.UD4*self.gprime*const.sw*self.chi
		self.c45 = const.g/2.0/const.cw*(np.sqrt(self.Uactive4SQR*self.Uactive5SQR)) + self.UD5*self.UD4*self.gprime*const.sw*self.chi
		self.c44 = const.g/2.0/const.cw*(np.sqrt(self.Uactive4SQR*self.Uactive4SQR)) + self.UD4*self.UD4*self.gprime*const.sw*self.chi
		self.c55 = const.g/2.0/const.cw*(np.sqrt(self.Uactive5SQR*self.Uactive5SQR)) + self.UD5*self.UD5*self.gprime*const.sw*self.chi
		self.c56 = const.g/2.0/const.cw*(np.sqrt(self.Uactive5SQR*self.Uactive6SQR)) + self.UD6*self.UD5*self.gprime*const.sw*self.chi
		
		self.d56 = self.UD6*self.UD5*self.gprime
		self.d46 = self.UD6*self.UD4*self.gprime
		self.d45 = self.UD5*self.UD4*self.gprime
		self.d44 = self.UD4*self.UD4*self.gprime
		self.d55 = self.UD5*self.UD5*self.gprime

		self.clight = const.g/2/const.cw
		self.dlight = 0.0


		# BSM-like couplings
		# self.de4 = self.UD4*self.Ue1*self.UD1 * (self.gprime/const.g)
		# self.dmu4 = self.UD4*self.Umu1*self.UD1 * (self.gprime/const.g)
		# self.dtau4 = self.UD4*self.Utau1*self.UD1 * (self.gprime/const.g)

		# self.de5 = self.UD5*self.Ue1*self.UD1* self.gprime/const.g
		# self.dmu5 = self.UD5*self.Umu1*self.UD1* self.gprime/const.g
		# self.dtau5 = self.UD5*self.Utau1*self.UD1* self.gprime/const.g

		# SM-like couplings
		# self.ce4 = self.UD4*self.Ue1*self.UD1 * (0.5 - self.gprime/const.g*const.sw*const.cw*self.chi)
		# self.cmu4 = self.UD4*self.Umu1*self.UD1 * (0.5 - self.gprime/const.g*const.sw*const.cw*self.chi)
		# self.ctau4 = self.UD4*self.Utau1*self.UD1 * (0.5 - self.gprime/const.g*const.sw*const.cw*self.chi)

		# self.ce5 = self.UD5*self.Ue1*self.UD1 * (0.5 - self.gprime/const.g*const.sw*const.cw*self.chi)
		# self.cmu5 = self.UD5*self.Umu1*self.UD1 * (0.5 - self.gprime/const.g*const.sw*const.cw*self.chi)
		# self.ctau5 = self.UD5*self.Utau1*self.UD1 * (0.5 - self.gprime/const.g*const.sw*const.cw*self.chi)


		# Kinetic mixing
		##############
		tanchi = np.tan(self.chi)

		sinof2chi  = 2*tanchi/(1.0+tanchi*tanchi)
		cosof2chi  = (1.0 - tanchi*tanchi)/(1.0+tanchi*tanchi)

		s2chi = (1.0 - cosof2chi)/2.0

		tanof2beta = np.sqrt(const.s2w) *  sinof2chi / ( (self.Mzprime/const.higgsvev)**2 - cosof2chi - (1.0-const.s2w)*s2chi )

		self.epsilon = const.cw*self.chi

		######################
		# CHECK ME!
		tbeta = (-1 + np.sign(tanof2beta) * np.sqrt( 1 + tanof2beta*tanof2beta) ) / tanof2beta
		sinof2beta = 2 * tbeta/(1.0+tbeta*tbeta)
		cosof2beta = (1.0-tbeta*tbeta)/(1.0+tbeta*tbeta)
		######################

		sbeta = np.sqrt( (1 - cosof2beta)/2.0)
		cbeta = np.sqrt( (1 + cosof2beta)/2.0)

		# Charged leptons
		# self.deV = 3.0/2.0 * cbeta * const.s2w * tanchi + sbeta*(0.5 + 2*const.s2w)
		# self.deA = (-sbeta - cbeta * const.s2w * tanchi)/2.0
		self.deV = const.g/(2*const.cw) * 2*const.sw*const.cw**2*self.chi
		self.deA = const.g/(2*const.cw) * 0

		# self.ceV = cbeta*(2*const.s2w - 0.5) - 3.0/2.0*sbeta*const.sw*tanchi
		# self.ceA = -(cbeta - sbeta*const.sw*tanchi)/2.0
		self.ceV = const.g/(2*const.cw) * (2*const.s2w - 0.5)
		self.ceA = const.g/(2*const.cw) * (-1.0/2.0)

		# quarks
		self.cuV = cbeta*(0.5 - 4*const.s2w/3.0) + 5.0/6.0*sbeta*const.sw*tanchi
		self.cuA = (cbeta + sbeta*const.sw*tanchi)/2.0

		self.cdV = cbeta*(-0.5 + 2*const.s2w/3.0) - 1.0/6.0*sbeta*const.sw*tanchi
		self.cdA = -(cbeta + sbeta*const.sw*tanchi)/2.0

		self.duV = sbeta*(0.5 - 4*const.s2w/3.0) - 5.0/6.0*cbeta*const.sw*tanchi
		self.duA = (sbeta + cbeta*const.sw*tanchi)/2.0

		self.ddV = sbeta*(-0.5 + 2*const.s2w/3.0) + 1.0/6.0*cbeta*const.sw*tanchi
		self.ddA = -(sbeta + cbeta*const.sw*tanchi)/2.0

		self.gVproton = -3.0/2.0*const.sw*self.chi*(1-8.0/9.0*const.s2w)#2*self.duV +self.ddV
		self.gAproton = 2*self.duA + self.ddA
		self.gVneutron = 2*self.ddV + self.duV
		self.gAneutron = 2*self.ddA + self.duA
