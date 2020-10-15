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

def old_geometry_muboone(size):
	######################### HEPevt format
	# Detector geometry -- choose random position
	xmin=0;xmax=256. # cm
	ymin=-115.;ymax=115. # cm
	zmin=0.;zmax=1045. # cm

	#######
	# Using initial time of the det only! 
	# CROSS CHECK THIS VALUE
	tmin=3.200e3;
	tmax=3.200e3 # ticks? ns?

	####################
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
	x = (xmin + (xmax -xmin)*np.random.rand(size))
	y = (ymin + (ymax -ymin)*np.random.rand(size))
	z = (zmin + (zmax -zmin)*np.random.rand(size))
	t = (tmin + (tmax -tmin)*np.random.rand(size))

	return t,x,y,z