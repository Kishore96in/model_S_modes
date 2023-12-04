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
		
		d.z = d.R_sun - d.r
		
		R = d.P/(d.rho*d.T)
		CP = R/(1-1/d.gamma)
		CV = CP-R
		d.entropy = CV*np.log(d.T*d.rho**(1-d.gamma)) #Note that this formula is correct even if CP, CV, and R vary.
		
		d.H = 1/np.gradient(np.log(d.rho), d.z)
		assert np.all(d.H > 0)
		d.N2 = - np.gradient(d.entropy, d.z)
		d.m = scipy.integrate.cumulative_trapezoid(4*np.pi*d.r**2*d.rho, d.r, initial=0)
		assert np.all(d.m >= 0)
		d.g = np.where(d.m != 0, d.G*d.m/d.r**2, 0)
		
		d.sort_by(d.z)
		self.c = self.make_spline(d.z, d.c)
		self.H = self.make_spline(d.z, d.H)
		self.N2 = self.make_spline(d.z, d.N2)
		self.g = self.make_spline(d.z, d.g)
		
		self.z_max = max(d.z)
		self.z_min = min(d.z)
	
	def make_spline(self, x, y):
		"""
		Wrapper around scipy.interpolate.UnivariateSpline
		"""
		assert np.all(x == np.sort(x))
		return scipy.interpolate.UnivariateSpline(x, y, check_finite=True, ext='raise')

def plot(model, var, logy=False):
	z = np.linspace(model.z_min, model.z_max, 100)
	
	fig,ax = plt.subplots()
	ax.plot(z, getattr(model, var)(z))
	
	if logy:
		ax.set_yscale('log')
	
	ax.set_xlim(min(z), max(z))
	ax.set_xlabel("z")
	ax.set_ylabel(var)
	fig.tight_layout()
	
	return fig, ax

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	
	model = solar_model("solar_model_S_cptrho.l5bi.d.15c")
	
	plot(model, 'c', logy=True)
	plot(model, 'H', logy=True)
	plot(model, 'g', logy=True)
	
	_, ax = plot(model, 'N2')
	ax.axhline(0, ls=':', c='k')
	
	plt.show()
