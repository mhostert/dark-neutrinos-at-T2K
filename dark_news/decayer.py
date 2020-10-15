import os
import sys
import numpy as np
import pandas as pd

from . import const
from . import pdg

#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

def decay_position(pN, l_decay_proper_cm=5):

	# decay the particle
	M4 = np.sqrt(Cfv.dot4(pN,pN))
	gammabeta_inv = M4/(np.sqrt(pN[:,0]**2 -  M4*M4 ))
	######################
	# Sample from decay propability
	d_decay = np.random.exponential(scale=l_decay_proper_cm/gammabeta_inv) # centimeters

	# direction of N
	t = d_decay/const.c_LIGHT/np.sqrt(pN[:,0]**2 - (M4)**2)*M4
	x = Cfv.get_3direction(pN)[:,0]*d_decay
	y = Cfv.get_3direction(pN)[:,1]*d_decay
	z = Cfv.get_3direction(pN)[:,2]*d_decay

	return t,x,y,z