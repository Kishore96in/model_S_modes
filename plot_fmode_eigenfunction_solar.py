"""
Check whether the eigenfunction of the f mode falls off with depth as exp(-k*z).
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from solar_model import solar_model, read_extensive_model_MmKS as reader
from komega import find_mode
from guess import make_guess_fmode_from_k
from problem import rhs, bc
from utils import sci_format, count_zero_crossings

if __name__ == "__main__":
	model = solar_model("data/Model S extensive/fgong.l5bi.d.15", reader=reader)
	form = sci_format(precision=2)
	
	k_list = [1e-2, 0.1, 1]
	z_bot = -690
	z_top = 0.45
	d_omega = 5e-4
	
	k_scl = 6.959906258e2 #R_sun for scaling k
	
	z_guess = np.linspace(z_bot, z_top, 2100)
	
	solutions = {}
	for k in k_list:
		guesser = lambda z_guess: make_guess_fmode_from_k(z_guess, k=k)
		omega_guess = np.sqrt(np.abs(model.g(0))*k)
		
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
			n = count_zero_crossings(np.real(mode_sol(z)[1]), z_max=0, z=z)
			if n!= 0:
				raise RuntimeError(f"Found a mode with {n = } radial nodes.")
			solutions[k] = {
				'omega': omega_sol,
				'mode': mode_sol,
				}
		else:
			warnings.warn(f"Finding mode failed for {k = }, {omega_guess = }", RuntimeWarning)
	
	fig, ax = plt.subplots()
	kz = np.linspace(-4, 0, int(1e4))
	iz0 = np.argmin(np.abs(kz - 0))
	for k in k_list:
		z = kz/k
		y2 = solutions[k]['mode'](z)[1]
		uz = y2/np.sqrt(model.c(z)*model.rho(z))
		
		uz_norm = np.real_if_close(uz/uz[iz0])
		ax.plot(kz, uz_norm, label=rf"$k R_\odot = {form(k*k_scl)}$")
	
	ax.set_yscale('log')
	ax.set_xlim(min(kz), max(kz))
	ax.set_xlabel("$k z$")
	ax.set_ylabel("$u_z$")
	ax.legend()
	
	fig.set_size_inches(4,3)
	fig.tight_layout()
	
	plt.show()
