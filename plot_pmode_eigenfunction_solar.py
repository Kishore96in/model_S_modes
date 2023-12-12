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
	
	guesses = [
		#list of tuples: (omega,k)
		(2.85e-3, 5e-3),
		(3e-3, 6.2e-3),
		(3.2e-3, 7e-3),
		(5e-3, 2.1e-2),
		]
	
	omega_scl = 1e3 #to convert omega to mHz
	k_scl = 6.959906258e2 #R_sun for scaling k
	
	omega_guess_list, k_list = zip(*guesses)
	
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
		
		axs[i].set_xlim(z_bot, z_top)
		axs[i].axhline(0, ls=':', c='k')
		
		axs[i].set_title(rf"$\omega = {omega*omega_scl:.2f}$, $k R_\odot = {k*k_scl:.2f}$")
	
	for ax in axs[:-1]:
		ax.xaxis.set_ticklabels([])
	
	axs[-1].set_xlabel("$z$")
	
	fig.legend(*axs[-1].get_legend_handles_labels(), loc='lower left')
	fig.set_size_inches(4,6)
	fig.tight_layout()
	
	#Plot wrt. k*z.
	fig, ax = plt.subplots()
	
	ax.axhline(0, ls=':', c='k')
	for i, k in enumerate(k_list):
		kz = k*z
		iz0 = np.argmin(np.abs(kz - -0.25))
		
		mode = solutions[k]['mode']
		omega = solutions[k]['omega']
		
		y2 = np.real_if_close(mode(z)[1])
		y2_norm = np.real_if_close(y2/y2[iz0])
		
		ax.plot(kz, y2_norm, label=rf"$\omega = {omega*omega_scl:.2f}$, $k R_\odot = {k*k_scl:.2f}$")
	
	ax.set_xlim(max(k_list)*z_bot, max(k_list)*z_top)
	ax.set_xlabel("$k z$")
	ax.set_ylabel("Normalized $y_2$")
	ax.legend()
	
	fig.set_size_inches(4,3)
	fig.tight_layout()
	
	plt.show()
