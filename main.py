"""
References:
	BirKosDuv04: Birch, Kosovichev, Duvall 2004 - Sensitivity of Acoustic Wave Travel Times to Sound-Speed Perturbations in the Solar Interior
"""

import numpy as np
import scipy.integrate
import scipy.interpolate

class read_model_file():
	def __init__(self):
		self.r, self.c, self.rho, self.P, self.gamma, self.T = np.loadtxt('solar_model_S_cptrho.l5bi.d.15c', unpack=True)

class solar_model():
	def __init__(self, filename):
		d = read_model_file(filename)
		
		z = (1 - d.r)*R_sun
		
		R = d.P/(d.rho*d.T)
		CP = R/(1-1/d.gamma)
		CV = CP-R
		entropy = CV*np.log(d.T*d.rho**(1-d.gamma)) #Note that this formula is correct even if CP, CV, and R vary.
		
		H = np.gradient(np.log(d.rho), z)
		N2 = - np.gradient(entropy, z)
		m = scipy.integrate.cumulative_trapezoid(4*np.pi*d.r**2*d.rho, d.r)
		g = G*m/d.r**2
		
		self.c = make_spline(z, d.c)
		self.H = make_spline(z, H)
		self.N2 = make_spline(z, N2)
		self.g = make_spline(z, g)
	
	def make_spline(self, x, y):
		"""
		Wrapper around scipy.interpolate.UnivariateSpline
		"""
		return scipy.interpolate.UnivariateSpline(x, y, check_finite=True, ext='raise')

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

