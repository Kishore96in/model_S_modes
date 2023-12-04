"""
Read data from Solar Model S (limited set of variables, filename `solar_model_S_cptrho.l5bi.d.15c`, downloaded from <https://users-phys.au.dk/~jcd/solar_models/> on 21 July 2020, 2:59 PM IST).

References:
	Christensen-Dalsgaard (1996) - The Current State of Solar Modeling
"""

import numpy as np
import scipy.integrate
import scipy.interpolate

class read_model_file():
	def __init__(self, filename):
		r, self.c, self.rho, self.P, self.gamma, self.T = np.loadtxt(filename, unpack=True)
		
		self.R_sun = 700e8 #cm
		self.G = 6.67408e-11 * 1e2**3 * 1e-3 #CGS
		self.r = r*self.R_sun
		
		self.sort_by(r)
	
	def sort_by(self, z):
		"""
		Assume z = z(r), and sort all arrays in order of increasing z.
		"""
		assert np.shape(z) == np.shape(self.r)
		argsort = np.argsort(z)
		
		for attr in self.__dict__.keys():
			val = getattr(self, attr)
			if (
				isinstance(val, np.ndarray) and
				np.shape(val) == np.shape(self.r)
				):
				setattr(self, attr, val[argsort])

class solar_model():
	def __init__(self, filename):
		d = read_model_file(filename)
		
		z = d.R_sun - d.r
		
		R = d.P/(d.rho*d.T)
		CP = R/(1-1/d.gamma)
		CV = CP-R
		entropy = CV*np.log(d.T*d.rho**(1-d.gamma)) #Note that this formula is correct even if CP, CV, and R vary.
		
		H = np.gradient(np.log(d.rho), z)
		N2 = - np.gradient(entropy, z)
		m = scipy.integrate.cumulative_trapezoid(4*np.pi*d.r**2*d.rho, d.r, initial=0)
		g = d.G*m/d.r**2
		
		self.c = self.make_spline(z, d.c)
		self.H = self.make_spline(z, H)
		self.N2 = self.make_spline(z, N2)
		self.g = self.make_spline(z, g)
		
		self.z_max = max(z)
		self.z_min = min(z)
	
	def make_spline(self, x, y):
		"""
		Wrapper around scipy.interpolate.UnivariateSpline
		"""
		assert np.all(x == np.sort(x))
		return scipy.interpolate.UnivariateSpline(x, y, check_finite=True, ext='raise')
