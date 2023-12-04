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
		self.r, self.c, self.rho, self.P, self.gamma, self.T = np.loadtxt(filename, unpack=True)
		
		self.R_sun = 700e8 #cm

class solar_model():
	def __init__(self, filename):
		d = read_model_file(filename)
		
		z = (1 - d.r)*d.R_sun
		
		R = d.P/(d.rho*d.T)
		CP = R/(1-1/d.gamma)
		CV = CP-R
		entropy = CV*np.log(d.T*d.rho**(1-d.gamma)) #Note that this formula is correct even if CP, CV, and R vary.
		
		r = d.r*d.R_sun
		H = np.gradient(np.log(d.rho), z)
		N2 = - np.gradient(entropy, z)
		m = scipy.integrate.cumulative_trapezoid(4*np.pi*r**2*d.rho, r)
		g = G*m/r**2
		
		self.c = make_spline(z, d.c)
		self.H = make_spline(z, H)
		self.N2 = make_spline(z, N2)
		self.g = make_spline(z, g)
	
	def make_spline(self, x, y):
		"""
		Wrapper around scipy.interpolate.UnivariateSpline
		"""
		return scipy.interpolate.UnivariateSpline(x, y, check_finite=True, ext='raise')
