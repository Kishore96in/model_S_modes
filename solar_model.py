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
