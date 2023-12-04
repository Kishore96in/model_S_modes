"""
Read data from Solar Model S.

References:
	Christensen-Dalsgaard (1996) - The Current State of Solar Modeling
"""

import numpy as np
import scipy.integrate
import scipy.interpolate

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
		
		R = self.P/(self.rho*self.T)
		CP = R/(1-1/self.gamma)
		self.CV = CP-R
		
		self.sort_by(self.r)

class read_extensive_model(model_reader):
	"""
	Read the extensive solar model in GONG format, downloaded from <https://users-phys.au.dk/~jcd/solar_models/>.
	"""
	gnames = {
		#Indices of the global parameters in the file
		'M_sun': 0,
		'R_sun': 1,
		}
	
	vnames = {
		#Indices of the model variables at each mesh point
		'rbyR': 0,
		'T': 2,
		'P': 3,
		'rho': 4,
		'Gamma_1': 9,
		'delta': 11,
		'CP': 12,
		}
	
	def __init__(self, filename):
		glob, var = self.read_extensive_solar_model(filename)
		
		for k in gnames.keys():
			setattr(self, k, glob[gnames[k]])
		for k in vnames.keys():
			setattr(self, k, var[:,vnames[k]])
		
		self.r = self.rbyR * self.R_sun
		self.G = 6.67408e-11 * 1e2**3 * 1e-3 #CGS
		
		self.CV = self.CP**2/( self.P*self.Gamma_1*self.delta**2/(self.rho*self.T) + self.CP )
		
		self.sort_by(self.r)
	
	def read_extensive_solar_model(filename):
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

class solar_model():
	"""
	Read a solar model and calculate the quantities that appear in the linearized modal equations.
	"""
	def __init__(self, filename, reader):
		d = reader(filename)
		
		d.z = d.R_sun - d.r
		
		d.grad_lnT = np.gradient(np.log(d.T), d.z)
		d.grad_lnrho = np.gradient(np.log(d.rho), d.z)
		d.grad_entropy = d.CV*d.grad_lnT - (d.P/(d.rho*d.T))*d.grad_lnrho
		d.N2 = - d.grad_entropy
		
		d.H = 1/np.gradient(np.log(d.rho), d.z)
		assert np.all(d.H > 0)
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
	
	model = solar_model("solar_model_S_cptrho.l5bi.d.15c", reader=read_limited_model)
	
	plot(model, 'c', logy=True)
	plot(model, 'H', logy=True)
	plot(model, 'g', logy=True)
	
	_, ax = plot(model, 'N2')
	ax.axhline(0, ls=':', c='k')
	
	plt.show()
