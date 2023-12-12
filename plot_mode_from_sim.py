"""
Plot the modes from a simulation of convection
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

from bg_from_sim import solar_model_from_sim as solar_model
from problem_BKD04 import bc_imp_both as bc
from guess import make_guess_pmode, make_guess_fmode_from_k
from problem import rhs
from utils import count_zero_crossings

if __name__ == "__main__":
	model = solar_model("data/background_from_simulation/background_a6.0l.1.pickle")
	
	z_bot = model.z_min
	z_top = model.z_max
	L_0 = model.c(0)**2/np.abs(model.g(0))
	omega_0 = np.abs(model.g(0))/model.c(0)
	
	k = 0/L_0
	omega_guess = 0.5*omega_0
	n_guess = 0 #number of nodes in the trial eigenmode used.
	
	z_guess = np.linspace(z_bot, z_top, 10+2*n_guess)
	y_guess = make_guess_pmode(z_guess, n=n_guess)
	# y_guess = make_guess_fmode_from_k(z_guess, k=k, model=model)
	p_guess = np.array([omega_guess])
	
	RHS = lambda z, y, p: rhs(z, y, p, k=k, model=model)
	BC = lambda y_bot, y_top, p: bc(y_bot, y_top, p, k=k, model=model, z_bot=z_bot, z_top=z_top)
	
	sol = scipy.integrate.solve_bvp(
		RHS,
		BC,
		p=p_guess,
		x=z_guess,
		y=y_guess,
		tol = 1e-6,
		max_nodes=1e6,
		)
		
	if not sol.success:
		warnings.warn(f"Solver failed for {k = }. {sol.message}", RuntimeWarning)
	
	z = np.linspace(z_bot, z_top, int(1e4))
	ruz = sol.sol(z)[1]/np.sqrt(model.c(z))
	ipbr = sol.sol(z)[0]*np.sqrt(model.c(z))
	
	print(rf"Found $\omega/\omega_0$ = {np.real_if_close(sol.p[0])/omega_0 :.2e}")
	print(f"Number of zero crossings of re(u√ρ₀) in the interior of the domain: {count_zero_crossings(np.real(ruz))}")
	
	fig,ax = plt.subplots()
	ax.plot(z, np.real(ruz), label=r"$\mathrm{re}\left( u_z \sqrt{\rho_0} \right)$")
	ax.plot(z, np.imag(ruz), label=r"$\mathrm{im}\left( u_z \sqrt{\rho_0} \right)$")
	ax.plot(z, np.real(ipbr), label=r"$\mathrm{re}\left( ip/\sqrt{\rho_0} \right)$")
	ax.plot(z, np.imag(ipbr), label=r"$\mathrm{im}\left( ip/\sqrt{\rho_0} \right)$")
	ax.set_xlim(z_bot, z_top)
	ax.axhline(0, ls=':', c='k')
	ax.set_xlabel("$z$")
	
	ax.legend()
	
	plt.show()
