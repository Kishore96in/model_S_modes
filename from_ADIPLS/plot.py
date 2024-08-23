import matplotlib.pyplot as plt
import numpy as np

from read_output import read_modes

import sys
sys.path.append("..")
from utils import count_zero_crossings
from solar_model import solar_model, read_extensive_model_MmKS as reader #Just to plot some background quantities

x, modes = read_modes("workingdir/amde.l9bi.d.202c.prxt3")

n_max = 8

def get_n_nodes(mode):
	"""
	ADIPLS' node count seems to include the node at the center, so I use my own function to count the number of zero crossings (excluding the boundaries) in the radial displacement eigenfunctions.
	"""
	# return count_zero_crossings(mode.y[:,0], z_max=1, z=x) # ignore nodes above r=R
	return count_zero_crossings(mode.y[:,0])

for mode in modes:
	n_this = get_n_nodes(mode)
	if n_this < 0:
		mode.n_nodes_ceil = -1
	elif get_n_nodes(mode) < n_max:
		mode.n_nodes_ceil = n_this
	else:
		mode.n_nodes_ceil = n_max

n_uniq = np.sort(np.unique([mode.n_nodes_ceil for mode in modes]))

fig,ax = plt.subplots()
for n in n_uniq:
	if n == n_max:
		label = rf"$\geq{n}$"
	elif n == -1:
		label = rf"$<0$"
	else:
		label = rf"${n}$"
	
	modes_this_n = [mode for mode in modes if mode.n_nodes_ceil == n]
	ax.scatter(
		[mode.l for mode in modes_this_n],
		[mode.omega*1e3 for mode in modes_this_n],
		label=label,
		s=3**2,
		)

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
