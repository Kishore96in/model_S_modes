"""
References:
	BirKosDuv04: Birch, Kosovichev, Duvall 2004 - Sensitivity of Acoustic Wave Travel Times to Sound-Speed Perturbations in the Solar Interior
"""

import numpy as np
from solar_model import solar_model

def rhs(z, y, p, k, model):
	"""
	Equations A10 and A1 of [BirKosDuv04].
	Note that the scale heights etc are themselves functions of depth determined by Model S.
	
	Arguments:
		z, y, p: see scipy.integrate.solve_bvp
		k: horizontal wavenumber
		model: instance of solar_model
	"""
	dydz = np.zeros_like(y)
	
	y1, y2 = y
	assert len(p) == 1
	omega = p[0]
	
	c = model.c(z)
	H = model.H(z)
	N2 = model.N2(z)
	
	dydz[0] = (-1/c)*(
		- (c/H)*y1
		+ (1/omega)*(omega**2/Gamma - N2)*y2
		)
	
	dydz[1] = (-1/c)*(
		- (1/omega)*(omega**2 - Gamma*c**2*k**2)*y1
		+ (c/H)*y2
		)
	
	return dydz

def bc(y_bot, y_top, p, k, model):
	"""
	Equations A20 and A21 of [BirKosDuv04].
	
	Arguments:
		y_bot, y_top, p: see scipy.integrate.solve_bvp
		k: horizontal wavenumber
		model: instance of solar_model
	"""
	
	y1,y2 = y
	assert len(p) == 1
	omega = p[0]
	
	c = model.c(z)
	g = model.g(z)
	
	return np.array([
		y2,
		omega*y1 + (g/c)*y2,
		*np.zeros_like(p),
		])

