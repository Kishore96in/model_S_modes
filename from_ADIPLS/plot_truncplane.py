"""
Truncated (inner 5% of radius removed) plane-parallel calculations in the Cowling approximation.
"""

import matplotlib.pyplot as plt
import numpy as np

from plot import (
	run_and_get_modes,
	plot_komega_by_nodes,
	)

import sys
sys.path.append("..")
from solar_model import solar_model, read_extensive_model_MmKS as reader #Just to plot some background quantities

if __name__ == "__main__":
	x, modes = run_and_get_modes("workingdir_truncplane")
	
	fig,ax = plt.subplots()
	plot_komega_by_nodes(ax, modes, n_max=8)
	
	model = solar_model("../data/Model S extensive/fgong.l5bi.d.15", reader=reader)
	
	#Theoretical frequency of the f mode
	ell = np.linspace(0, ax.get_xlim()[1], 1000)
	R_sun = abs(model.z_min) #Assume the given model extends to the center of the Sun
	k = np.sqrt(ell*(ell+1))/R_sun
	g = abs(model.g(0))
	ax.autoscale(False)
	ax.plot(ell, 1e3*np.sqrt(g*k), ls='-', c='k')
	
	#omega = cmax*k line
	z = np.linspace(model.z_min + min(x)*R_sun, 0, 1000)
	ax.plot(ell, 1e3*max(model.c(z))*k, ls='--', c='k')
	
	ax.set_ylabel(r"$\omega$ (mHz)")
	ax.set_xlabel(r"$\ell$")
	ax.legend(loc='lower right')
	
	plt.show()
