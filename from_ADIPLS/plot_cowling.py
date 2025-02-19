"""
Full radial extent, spherical geometry, Cowling approximation.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from plot import (
	run_and_get_modes,
	plot_komega_by_nodes,
	)

import sys
sys.path.append("..")
from utils import fig_saver


if __name__ == "__main__":
	mpl.style.use("../kishore.mplstyle")
	save = fig_saver(savedir="plots")
	
	x, modes = run_and_get_modes("workingdir_cowling")
	
	fig,ax = plt.subplots()
	plot_komega_by_nodes(ax, modes, n_max=6)
	
	ax.set_ylabel(r"$\omega$ (mHz)")
	ax.set_xlabel(r"$\ell$")
	ax.legend(loc='lower right', title="Nodes")
	
	save(fig, "komega_cow.pdf")
