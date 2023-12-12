"""
Read data from Solar Model S.

References:
	Christensen-Dalsgaard (1996) - The Current State of Solar Modeling
"""

import numpy as np
import scipy.integrate
import scipy.interpolate

from itertools import chain

class model_reader():
	def sort_by(self, z):
		"""
		Assume z = z(self.r), and sort all arrays in order of increasing z.
		"""
		if not (np.shape(z) == np.shape(self.r)):
			raise ValueError(f"z (shape: {np.shape(z)}) must be of the same shape as r (shape: {np.shape(r)}).")
		
		argsort = np.argsort(z)
		
		for attr in self.__dict__.keys():
			val = getattr(self, attr)
			if (
				isinstance(val, np.ndarray) and
				np.shape(val) == np.shape(self.r)
				):
				setattr(self, attr, val[argsort])

class read_limited_model(model_reader):
	"""
	Read limited set of variables available at <https://users-phys.au.dk/~jcd/solar_models/>.
	
	E.g. filename `solar_model_S_cptrho.l5bi.d.15c`, downloaded from <https://users-phys.au.dk/~jcd/solar_models/> on 21 July 2020, 2:59 PM IST.
	"""
	def __init__(self, filename):
		r, self.c, self.rho, self.P, self.gamma, self.T = np.loadtxt(filename, unpack=True)
		
		self.R_sun = 700e8 #cm
		self.G = 6.67408e-11 * 1e2**3 * 1e-3 #CGS
		self.r = r*self.R_sun
		
		#NOTE: these are only correct for an ideal gas.
		R = self.P/(self.rho*self.T)
		CP = R/(1-1/self.gamma)
		self.CV = CP-R
		self.CP = CP
		
		self.sort_by(self.r)

class read_extensive_model(model_reader):
	"""
	Read the extensive solar model in GONG format, downloaded from <https://users-phys.au.dk/~jcd/solar_models/>.
	"""
	
	#Override lu, tu, and mu in subclasses to change the units used.
	lu = 1 #1 cm in the desired length unit
	tu = 1 #1 second in the desired time unit
	mu = 1 #1 gram in the desired mass unit
	
	gnames = {
		#Indices of the global parameters in the file
		'R_sun': 1,
		}
	
	vnames = {
		#Indices of the model variables at each mesh point
		'r': 0,
		'T': 2,
		'P': 3,
		'rho': 4,
		'Gamma_1': 9,
		'delta': 11,
		'CP': 12,
		}
	
	G_cgs = 6.67408e-11 * 1e2**3 * 1e-3 #cm^3 g^{-1} s^{-2}
	
	def __init__(self, filename):
		glob, var = self.read_extensive_solar_model(filename)
		
		self._populate_unit_conversion_factors()
		
		#Unit conversion
		for k in self.gnames.keys():
			assert k in self.gunits.keys()
			fac = self.gunits[k]
			setattr(self, k, fac*glob[self.gnames[k]])
		
		#Unit conversion
		for k in self.vnames.keys():
			assert k in self.vunits.keys()
			fac = self.vunits[k]
			setattr(self, k, fac*var[:,self.vnames[k]])
		
		self.CV = self.CP**2/( self.P*self.Gamma_1*self.delta**2/(self.rho*self.T) + self.CP )
		
		self.c = np.sqrt(self.Gamma_1*self.P/self.rho) #adiabatic sound speed
		
		self.sort_by(self.r)
	
	def read_extensive_solar_model(self, filename):
		"""
		Format: specified at https://users-phys.au.dk/~jcd/solar_models/file-format.pdf
			Record 1: Name of model (as a character string)
			Record 2 – 4: Explanatory text, in free format
			Record 5: nn, iconst, ivar, ivers
			Record 6 – 8: glob(i), i = 1, ..., iconst
			Record 9 – : var(i,n), i = 1, . . ., ivar, n = 1, . . . nn.
		Download extensive data from https://users-phys.au.dk/~jcd/solar_models/fgong.l5bi.d.15
		"""
		def split_to_words(string, wordlen):
			return [ string[i:wordlen+i] for i in range(0, len(string)-1,wordlen) ] #The -1 is because we expect a newline at the end of each string.
		
		with open(filename, 'r') as f:
			lines = f.readlines()
		model_name = lines[0]
		model_description = ''.join(lines[1:4])
		
		nn, iconst, ivar, ivers = [ int(i) for i in lines[4].split() ]
		
		word_len = 16 #Each 'word' in line[5] onwards is 16 chars long.
		ncols = len( split_to_words(lines[5],word_len) ) #Number of columns in record 6 onwards. Assume this is the same for all following records.
		nlines_each_mesh = int(ivar/ncols) #The number of lines each mesh point occupies.
		
		glob = np.array( list( chain( *(split_to_words(i,word_len) for i in lines[5:8] ) ) ) , dtype=float)
		
		#We will construct var such that the first index refers to the mesh point, and the second index refers to the variable name.
		var = []
		for i in range(8,len(lines),nlines_each_mesh):
			var.append( list( chain( *( split_to_words(i,word_len) for i in lines[i:i+nlines_each_mesh] ) ) ) )
		
		return glob, np.array(var, dtype=float)
	
	def _populate_unit_conversion_factors(self):
		lu = self.lu
		mu = self.mu
		tu = self.tu
		
		self.gunits = {
			#Multiplicative unit conversion factors for the global variables
			'R_sun': lu, #Mm
			}
		
		self.vunits = {
			#Multiplicative unit conversion factors for the mesh variables
			'r': lu, #Mm
			'T': 1, #K
			'P': mu * lu**(-1) * tu**(-2), #kg Mm^{-1} s^{-2}
			'rho': mu * lu**(-3), #kg Mm^{-3}
			'Gamma_1': 1, #dimensionless
			'delta': 1, #dimensionless
			'CP': lu**2 * tu**(-2), #Mm^{2} K^{-1} s^{-2}
			}
		
		self.G = self.G_cgs * lu**3 * mu**(-1) * tu**(-2)

class read_extensive_model_MmKS(read_extensive_model):
	"""
	Just like read_extensive_model, but changes the length unit to Mm and the mass unit to kilogram (from CGS units).
	"""
	lu = 1e-8
	mu = 1e-3

class read_extensive_model_RsunKms(read_extensive_model):
	"""
	Just like read_extensive_model, but changes the length unit to R_sun, the mass unit to kilogram, and the time unit to milliseconds (from CGS units).
	"""
	lu = 1/6.959906258e10 #cm in R_sun, as provided in Solar Model S
	tu = 1e3 #s in ms
	mu = 1e-3 #g in kg

class solar_model():
	"""
	Read a solar model and calculate the quantities that appear in the linearized modal equations.
	"""
	def __init__(self, filename, reader):
		d = reader(filename)
		
		d.z = d.r - d.R_sun
		
		d.m = scipy.integrate.cumulative_trapezoid(4*np.pi*d.r**2*d.rho, d.r, initial=0)
		assert np.all(d.m >= 0)
		d.g = np.where(d.m != 0, - d.G*d.m/d.r**2, 0)
		d.grad_lnrho = np.gradient(np.log(d.rho), d.z)
		d.grad_lnc = np.gradient(np.log(d.c), d.z)
		d.H = 1/( - d.grad_lnrho/2 - d.grad_lnc/2 + d.g/d.c**2 )
		d.rhoinv_grad_p = np.gradient(d.P, d.z)/d.rho
		
		d.grad_lnT = np.gradient(np.log(d.T), d.z)
		d.grad_entropy = d.CV*d.grad_lnT - (d.P/(d.rho*d.T))*d.grad_lnrho
		d.N2 = - d.g*d.grad_entropy/d.CP
		
		d.sort_by(d.z)
		self.c = self.make_spline(d.z, d.c)
		self.H = self.make_spline(d.z, d.H)
		self.gradlnrho = self.make_spline(d.z, d.grad_lnrho)
		self.gradlnc = self.make_spline(d.z, d.grad_lnc)
		self.N2 = self.make_spline(d.z, d.N2)
		self.g = self.make_spline(d.z, d.g)
		self.rhoinv_grad_p = self.make_spline(d.z, d.rhoinv_grad_p)
		self.rho = self.make_spline(d.z, d.rho)
		
		self.z_max = max(d.z)
		self.z_min = min(d.z)
	
	def make_spline(self, x, y):
		"""
		Wrapper around scipy.interpolate.UnivariateSpline
		"""
		assert np.all(x == np.sort(x))
		return scipy.interpolate.InterpolatedUnivariateSpline(x, y, check_finite=True, ext='raise')

def plot(model, var, logy=False, absolute=False):
	z = np.linspace(model.z_min, model.z_max, 1000)
	data = getattr(model, var)(z)
	
	if absolute:
		data = abs(data)
		ylabel = rf"$\left| \mathrm{{{var}}} \right|$"
	else:
		ylabel = var
	
	fig,ax = plt.subplots()
	ax.plot(z, data)
	
	if logy:
		ax.set_yscale('log')
	
	ax.set_xlim(min(z), max(z))
	ax.set_xlabel("z")
	ax.set_ylabel(ylabel)
	fig.tight_layout()
	
	return fig, ax

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	
	# model = solar_model("data/Model S limited/solar_model_S_cptrho.l5bi.d.15c", reader=read_limited_model)
	# model = solar_model("data/Model S extensive/fgong.l5bi.d.15", reader=read_extensive_model)
	model = solar_model("data/Model S extensive/fgong.l5bi.d.15", reader=read_extensive_model_MmKS)
	
	plot(model, 'c', logy=True)
	plot(model, 'H')
	fig, ax = plot(model, 'g', logy=True, absolute=True)
	ax.set_ylim(1e-4, 1e-2)
	
	_, ax = plot(model, 'N2')
	ax.axhline(0, ls=':', c='k')
	ax.set_ylim(-5e-5, 5e-5)
	
	plt.show()
