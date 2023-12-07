"""
Plot the dispersion relations of all found modes with the background state taken from Solar Model S.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from solar_model import solar_model, read_extensive_model_MmKS as reader
from plot_komega_from_sim import construct_komega, plot_komega
from problem import rhs, bc

if __name__ == "__main__":
	plot = True
	model = solar_model("Model S extensive data/fgong.l5bi.d.15", reader=reader)
	
	k_max = 1
	omega_max = 2.5e-2
	omega_min = 5e-4
	
	if not os.path.isfile("komega_from_solar.pickle"):
		construct_komega(
			model = model,
			rhs = rhs,
			bc = bc,
			z_bot = -25,
			z_top = 0.45,
			k_max = k_max,
			n_k = 5,
			omega_max = omega_max,
			omega_min = omega_min,
			n_omega = 200,
			d_omega = omega_min,
			outputfile="komega_from_solar.pickle",
			n_workers = 2,
			)
	
	if plot:
		fig,ax = plt.subplots()
		plot_komega("komega_from_solar.pickle", n_max=3, ax=ax)
		l = ax.legend()
		l.set_title("Nodes")
		ax.set_xlabel(r"$k$")
		ax.set_ylabel(r"$\omega$")
		ax.set_xlim(0, k_max)
		ax.set_ylim(bottom=0, top=omega_max)
		
		fig.tight_layout()
		
		plt.show()
