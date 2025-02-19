import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import subprocess
import os
import itertools

from read_output import read_modes

import sys
sys.path.append("..")
from utils import count_zero_crossings, fig_saver


def get_n_nodes(mode):
	"""
	ADIPLS' node count seems to include the node at the center, so I use my own function to count the number of zero crossings (excluding the boundaries) in the radial displacement eigenfunctions.
	"""
	# return count_zero_crossings(mode.y[:,0], z_max=1, z=x) # ignore nodes above r=R
	return count_zero_crossings(mode.y[:,0])

def run_and_get_modes(dirname):
	# https://stackoverflow.com/questions/7040592/calling-the-source-command-from-subprocess-popen/12708396#12708396
	pipe = subprocess.Popen(". ./init.sh; env", stdout=subprocess.PIPE, shell=True, text=True)
	output = pipe.communicate()[0]
	env = dict((line.split("=", 1) for line in output.splitlines()))
	
	subprocess.run(["make"], cwd=dirname, env=env)
	return read_modes(os.path.join(dirname, "amde.l9bi.d.202c.prxt3"))

def plot_komega_by_nodes(axs, modes_lists, n_max):
	"""
	Arguments
		axs: list of mpl Axes objects
		modes_lists: list of list of read_output.py::Mode instances. The outer list should be of the same length as axs. Each element of the outer list will be plotted in the corresponding element of axs.
		n_max: int. Maximum number of nodes that should be marked with a unique colour.
	"""
	
	if not np.iterable(axs):
		axs = [axs]
		modes_lists = [modes_lists]
	
	if len(axs) != len(modes_lists):
		raise ValueError("axs and modes_lists are of unequal lengths")
	
	allmodes = list(itertools.chain(*modes_lists))
	
	for mode in allmodes:
		n_this = get_n_nodes(mode)
		if n_this < 0:
			mode.n_nodes_ceil = -1
		elif get_n_nodes(mode) < n_max:
			mode.n_nodes_ceil = n_this
		else:
			mode.n_nodes_ceil = n_max
	
	n_uniq = np.sort(np.unique([mode.n_nodes_ceil for mode in allmodes]))
	
	for n, color in zip(n_uniq, mpl.cm.tab10.colors):
		if n == n_max:
			label = rf"$\geq{n}$"
		elif n == -1:
			label = rf"$<0$"
		else:
			label = rf"${n}$"
		
		for ax, modes in zip(axs, modes_lists):
			modes_this_n = [mode for mode in modes if mode.n_nodes_ceil == n]
			ax.scatter(
				[mode.l for mode in modes_this_n],
				[mode.omega*1e3 for mode in modes_this_n],
				label=label,
				s=3**2,
				color=color,
				)

if __name__ == "__main__":
	mpl.style.use("../kishore.mplstyle")
	save = fig_saver(savedir="plots")
	
	x, modes = run_and_get_modes("workingdir")
	
	fig,ax = plt.subplots()
	ax.set_ymargin(0)
	plot_komega_by_nodes(ax, modes, n_max=6)
	
	ax.set_ylabel(r"$\omega$ (mHz)")
	ax.set_xlabel(r"$\ell$")
	ax.legend(loc='lower right')
	
	save(fig, "komega_full.pdf")
