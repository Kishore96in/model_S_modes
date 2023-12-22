"""
Plot the dispersion relations of all found modes with the background state taken from a modified version of Solar Model S. A ceiling is applied to the  Brunt-Vaisala frequency to ensure that g modes do not interact with the f mode for k*R_sun > 1.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from solar_model import solar_model, read_extensive_model_MmKS as reader
from komega import construct_komega, plot_komega
from problem import rhs, bc
from utils import ceil_spline

if __name__ == "__main__":
	plot = True
	cachefile = "komega_from_solar_N2ceil.pickle"
	
	if not os.path.isfile(cachefile):
		model = solar_model("data/Model S extensive/fgong.l5bi.d.15", reader=reader)
		
		model.N2 = ceil_spline(model.N2, ceil=0)
		
		construct_komega(
			model = model,
			rhs = rhs,
			bc = bc,
			z_bot = -690,
			z_top = 0.45,
			k_list = np.linspace(0,0.025,35),
			omega_max = 7e-3,
			omega_min = 1e-3,
			n_omega = 1800,
			d_omega = 5e-5,
			nz = 2100, #empirically, 3 grid points per Mm seems okay.
			outputfile=cachefile,
			n_workers = 2,
			)
	else:
		print("Skipping computation as cached results already exist.")
	
	if plot:
		with open(cachefile, 'rb') as f:
			ret = pickle.load(f)
		
		model = ret['model']
		z_bot = ret['z_bot']
		z_top = ret['z_top']
		k_max = max(ret['k_list'])
		omega_max = ret['omega_max']
		
		fig,ax = plt.subplots()
		ax = plot_komega(
			cachefile,
			n_max=5,
			ax=ax,
			scatter_kwargs={'s': 3**2},
			k_scl = 6.959906258E+2, #Use k*R_sun
			omega_scl = 1e3, #Use mHz
			)
		l = ax.legend(loc='lower right')
		l.set_title("Nodes")
		ax.set_xlabel(r"$k R_\odot$")
		ax.set_ylabel(r"$\omega$ (mHz)")
		ax.set_xlim(0, k_max)
		ax.set_ylim(bottom=0, top=omega_max)
		
		# The location where the modes are expected to change order
		k = np.linspace(0, k_max, 1000)
		om = model.c(z_bot)*k
		ax.plot(k, om, ls='--', c='k')
		
		# The theoretical f mode
		om = np.sqrt(np.abs(model.g(z_top))*k)
		ax.plot(k, om, ls='-', c='k')
		
		fig.set_size_inches(4,3.5)
		fig.tight_layout()
		
		plt.show()
