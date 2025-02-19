"""
Truncated (inner 5% of radius removed) calculations in the Cowling approximation.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools

from plot import run_and_get_modes, plot_komega_by_nodes

import sys
sys.path.append("..")
from solar_model import solar_model, read_extensive_model_MmKS as reader #Just to plot some background quantities

if __name__ == "__main__":
	mpl.style.use("../kishore.mplstyle")
	
	x, modes = run_and_get_modes("workingdir_trunc")
	x_hil, modes_hil = run_and_get_modes("workingdir_trunc_hil")
	
	if min(x) != min(x_hil):
		raise RuntimeError
	
	fig, axs = plt.subplots(1,2, sharey=True)
	plot_komega_by_nodes(axs, [modes, modes_hil], n_max=7)
	
	model = solar_model("../data/Model S extensive/fgong.l5bi.d.15", reader=reader)
	
	axs[0].autoscale(False)
	axs[1].autoscale(False)
	
	for ax in axs:
		#Theoretical frequency of the f mode
		ell = np.linspace(0, ax.get_xlim()[1], 1000)
		R_sun = abs(model.z_min) #Assume the given model extends to the center of the Sun
		k = np.sqrt(ell*(ell+1))/R_sun
		g = abs(model.g(0))
		ax.plot(ell, 1e3*np.sqrt(g*k), ls='-', c='k')
		
		#omega = cmax*k line
		z = np.linspace(model.z_min + min(x)*R_sun, 0, 1000)
		ax.plot(ell, 1e3*max(model.c(z))*k, ls='--', c='k')
		
		ax.set_xlabel(r"$\ell$")
	
	axs[0].set_ylabel(r"$\omega$ (mHz)")
	fig.legend(*axs[0].get_legend_handles_labels(), loc='outside right')
	fig.set_size_inches(5,3)
	
	plt.show()
