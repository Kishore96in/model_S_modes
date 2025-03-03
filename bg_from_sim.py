"""
Generate a background model by reading simulation data (horizontal averages)
"""

import numpy as np
import pickle

from solar_model import solar_model

class solar_model_from_sim(solar_model):
	"""
	Construct a fake solar model from a Pencil simulation that has gravity in the -z direction. z=zcool is considered as the 'surface'. Ideal gas EOS is assumed.
	"""
	def __init__(self, filename):
		with open(filename, 'rb') as f:
			d = pickle.load(f)
		
		z = d['z']
		g = d['gravz']
		c = np.sqrt(d['cp']*(d['gamma']-1)*d['TTmz'])
		gradlnrho = np.gradient(np.log(d['rhomz']), z)
		gradlnc = np.gradient(np.log(c), z)
		H = 1/( - gradlnrho/2 - gradlnc/2 + g/c**2 )
		
		grad_entropy = np.gradient(d['ssmz'], z)
		N2 = - g*grad_entropy
		
		self.c = self.make_spline(z, c)
		self.H = self.make_spline(z, H)
		self.gradlnrho = self.make_spline(z, gradlnrho)
		self.gradlnc = self.make_spline(z, gradlnc)
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
			'simpath': sim.path,
			}
		
		with open(savefile, 'wb') as f:
			pickle.dump(ret, f)

def get_av(av, key, it=-500):
	"""
	Used to take the time average of a horizontal average.
	"""
	return np.average(getattr(av.xy, key)[it:], axis=0)

if __name__ == "__main__":
	import sys
	
	simdir = sys.argv[1]
	savefile = sys.argv[2]
	
	solar_model_from_sim.save(simdir=simdir, savefile=savefile)
