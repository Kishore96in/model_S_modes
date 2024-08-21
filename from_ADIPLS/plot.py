import matplotlib.pyplot as plt
import numpy as np

from read_output import read_modes

x, modes = read_modes("workingdir/amde.l9bi.d.202c.prxt3")

n_max = 5

get_n_nodes = lambda mode: mode.nordp - mode.nordg

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
ax.set_ylabel(r"$\omega$ (mHz)")
ax.set_xlabel(r"$\ell$")
ax.legend(loc='lower right')

plt.show()
