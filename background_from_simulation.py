"""
Generate a background model by reading simulation data (horizontal averages)

"""

import numpy as np
import pickle

from solar_model import solar_model

class solar_model_from_sim(solar_model):
	def __init__(self, filename):
		with open(filename, 'rb') as f:
			d = pickle.load(f)
		
		z = d['z']
		g = d['gravz']
		#NOTE: we assume the simulation uses an ideal gas EOS.
		c = np.sqrt(d['cp']*(d['gamma']-1)*d['TTmz'])
		H = 1/( - np.gradient(np.log(d['rhomz']), z)/2 - np.gradient(np.log(c), z)/2 - g/c**2 )
		
		grad_entropy = np.gradient(d['ssmz'], z)
		N2 = - g*grad_entropy
		
		self.c = self.make_spline(z, c)
		self.H = self.make_spline(z, H)
		self.N2 = self.make_spline(z, N2)
		self.g = self.make_spline(z, np.full_like(z, g))
		
		self.z_max = max(z)
		self.z_min = min(z)
	
	@staticmethod
	def save(savefile, simdir="."):
		import pencil as pc
		
		sim = pc.sim.get(path=simdir, quiet=True)
		param = pc.read.param(quiet=True, datadir=sim.datadir)
		grid = pc.read.grid(quiet=True, trim=True, datadir=sim.datadir)
		av = pc.read.aver(plane_list=['xy'], quiet=True, datadir=sim.datadir, simdir=sim.path)
		
		ret = {
			'z': grid.z - param.zcool, #Make sure z=0 corresponds to the 'surface'.
			'gravz': param.gravz,
			'cp': param.cp,
			'gamma': param.gamma,
			'TTmz': get_av(av, 'TTmz'),
			'rhomz': get_av(av, 'rhomz'),
			'ssmz': get_av(av, 'ssmz'),
			}
		
		with open(savefile, 'wb') as f:
			pickle.dump(ret, f)

def get_av(av, key, it=-500):
	"""
	Used to take the time average of a horizontal average.
	"""
	return np.average(getattr(av.xy, key)[it:], axis=0)

if __name__ == "__main__":
	solar_model_from_sim.save(savefile="background.pickle")
