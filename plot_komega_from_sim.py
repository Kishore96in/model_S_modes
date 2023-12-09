"""
Plot the dispersion relations of all found modes with the background state taken from a simulation of convection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

from bg_from_sim import solar_model_from_sim as solar_model
from problem_BKD04 import bc_imp_both
from problem import rhs
from komega import construct_komega, plot_komega

if __name__ == "__main__":
	plot = True
	cachefile = "komega_from_sim.pickle"
	
	if not os.path.isfile(cachefile):
		model = solar_model("data/background_from_simulation/background_a6.0l.1.pickle")
		L_0 = model.c(0)**2/np.abs(model.g(0))
		omega_0 = np.abs(model.g(0))/model.c(0)
		
		construct_komega(
			model = model,
			omega_max = 1.5*omega_0,
			omega_min = 0.1*omega_0,
			k_list = np.linspace(0, 0.5/L_0, 5),
			n_omega = 100,
			d_omega = 0.1*omega_0,
			outputfile=cachefile,
			n_workers = 2,
			rhs = rhs,
			bc = bc_imp_both,
			)
	else:
		print("Skipping computation as cached results already exist.")
	
	if plot:
		with open(cachefile, 'rb') as f:
			ret = pickle.load(f)
		
		model = ret['model']
		z_bot = ret['z_bot']
		k_max = max(ret['k_list'])
		omega_max = ret['omega_max']
		
		L_0 = model.c(0)**2/np.abs(model.g(0))
		omega_0 = np.abs(model.g(0))/model.c(0)
		
		fig,ax = plt.subplots()
		plot_komega(cachefile, k_scl=L_0, omega_scl=1/omega_0, ax=ax, n_max=3, scatter_kwargs={'s': 3**2})
		l = ax.legend(loc='lower right')
		l.set_title("Nodes")
		ax.set_xlabel(r"$\widetilde{k}$")
		ax.set_ylabel(r"$\widetilde{\omega}$")
		ax.set_xlim(0, k_max*L_0)
		ax.set_ylim(bottom=0, top=omega_max/omega_0)
		
		#The line where the modes appear to change order.
		k = np.linspace(0, k_max, 100)
		om = model.c(z_bot)*k
		ax.plot(k*L_0, om/omega_0, ls='--', c='k')
		
		fig.set_size_inches(4,3.5)
		fig.tight_layout()
		
		#Plot some background quantities
		z = np.linspace(model.z_min, model.z_max, 1000)
		
		fig = plt.figure()
		gs = mpl.gridspec.GridSpec(2,2, figure=fig)
		
		axs = [
			fig.add_subplot(gs[:,1]),
			fig.add_subplot(gs[0,0]),
			fig.add_subplot(gs[1,0]),
			]
		
		axs[0].plot(z, model.N2(z))
		axs[0].axhline(0, ls=':', c='k')
		axs[0].set_ylabel("$N^2$")
		
		axs[1].plot(z, model.c(z))
		axs[1].set_ylabel("$c$")
		
		axs[2].plot(z, -1/model.gradlnrho(z))
		axs[2].set_ylabel(r"$H_\rho$")
		
		for ax in axs:
			ax.set_xlim(model.z_min, model.z_max)
		
		for ax in axs[1:-1]:
			ax.xaxis.set_ticklabels([])
		
		axs[0].set_xlabel("$z$")
		axs[-1].set_xlabel("$z$")
		
		fig.set_size_inches(6,3)
		fig.tight_layout()
		
		plt.show()
