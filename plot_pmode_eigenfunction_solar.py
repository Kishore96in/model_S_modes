"""
Plot the eigenfunction of the p mode of the same order at different k to show that the number of nodes changes across the omega = c_bot*k line.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from solar_model import solar_model, read_extensive_model_MmKS as reader
from komega import find_mode
from guess import make_guess_pmode
from problem import rhs, bc
from utils import sci_format, count_zero_crossings

if __name__ == "__main__":
	model = solar_model("data/Model S extensive/fgong.l5bi.d.15", reader=reader)
	form = sci_format(precision=2)
	
	z_bot = -690
	z_top = 0.45
	d_omega = 5e-4
	n_guess = 2
	
	omega_guess_list = [2.85e-3, 3e-3, 3.2e-3, 5e-3]
	k_list = [5e-3, 6.2e-3, 7e-3, 2.1e-2]
	
	c_bot = model.c(z_bot)
	print(f"omega - c_bot*k: {[om - c_bot*k for om, k in zip(omega_guess_list, k_list)]}") #debug
	
	z_guess = np.linspace(z_bot, z_top, 2100)
	
	solutions = {}
	for k, omega_guess in zip(k_list, omega_guess_list):
		guesser = lambda z_guess: make_guess_pmode(z_guess, n=n_guess)
		
		omega_sol, mode_sol, success = find_mode(
			omega_guess = omega_guess,
			k = k,
			model = model,
			z_guess = z_guess,
			guesser = guesser,
			rhs = rhs,
			bc = bc,
			)
		
		if success and (np.abs(omega_guess - omega_sol) < d_omega):
			z = np.linspace(z_bot, z_top, int(1e4))
			n_sol = count_zero_crossings(np.real(mode_sol(z)[1]), z_max=0, z=z)
			print(f"Found a mode with {n_sol = } radial nodes at {omega_sol = :.2e}, {k = :.2e}")
			solutions[k] = {
				'omega': omega_sol,
				'mode': mode_sol,
				'n': n_sol,
				}
		else:
			warnings.warn(f"Finding mode failed for {k = }, {omega_guess = }", RuntimeWarning)
	
	fig, axs = plt.subplots(len(k_list))
	
	z = np.linspace(z_bot, z_top, int(1e4))
	for i, k in enumerate(k_list):
		mode = solutions[k]['mode']
		omega = solutions[k]['omega']
		
		y1 = np.real_if_close(mode(z)[0])
		y2 = np.real_if_close(mode(z)[1])
		
		axs[i].plot(z, y1, label="$y_1$")
		axs[i].plot(z, y2, label="$y_2$")
		
		axs[i].legend()
		axs[i].set_xlim(z_bot, z_top)
		axs[i].axhline(0, ls=':', c='k')
		
		axs[i].set_title(rf"$\omega = {form(omega)}$, $k = {form(k)}$")
	
	for ax in axs[:-1]:
		ax.xaxis.set_ticklabels([])
	
	axs[-1].set_xlabel("$z$")
	
	fig.set_size_inches(4,6)
	fig.tight_layout()
	
	plt.show()
