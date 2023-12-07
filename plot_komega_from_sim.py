"""
Plot the dispersion relations of all found modes with the background state taken from a simulation of convection.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import pickle

from bg_from_sim import solar_model_from_sim as solar_model
from problem import rhs, bc_imp_both as bc, count_zero_crossings, make_guess_pmode, make_guess_fmode_with_omega

def find_mode(omega_guess, k, model, z_guess, guesser):
	y_guess = guesser(z_guess)
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
	
	return sol.p[0], sol.sol, sol.success

def make_guess_fmode_from_k(z_guess, k, model):
	y_guess, p_guess = make_guess_fmode_with_omega(z_guess, k=k, model=model)
	return y_guess

if __name__ == "__main__":
	model = solar_model("background_a6.0l.1.pickle")
	
	z_bot = model.z_min
	z_top = model.z_max
	L_0 = model.c(0)**2/np.abs(model.g(0))
	omega_0 = np.abs(model.g(0))/model.c(0)
	
	k_max = 0.5/L_0
	omega_max = 1.5*omega_0
	omega_min = 0.1*omega_0
	
	z = np.linspace(z_bot, z_top, int(1e3))
	k_range = np.linspace(0, k_max, 5)
	omega_range = np.linspace(omega_min, omega_max, 10)
	d_omega = omega_range[1] - omega_range[0]
	
	g = np.sqrt(np.abs(model.g(z_top))) #Used to estimate f mode frequency.
	
	solutions = {}
	for k in k_range:
		solutions_this_k = []
		omega_last = -1
		n_guess = 0
		guesser = lambda z_guess: make_guess_fmode_from_k(z_guess, k=k, model=model)
		omega_f = np.sqrt(g*k)
		
		for omega in omega_range:
			if omega < omega_last:
				continue
			
			z_guess = np.linspace(z_bot, z_top, 10+2*n_guess)
			if omega > omega_f:
				guesser = lambda z_guess: make_guess_pmode(z_guess, n=n_guess)
			
			omega_sol, mode_sol, success = find_mode(omega, k=k, model=model, z_guess=z_guess, guesser=guesser)
			
			if success and (np.abs(omega - omega_sol) < d_omega):
				solutions_this_k.append({'omega': np.real_if_close(omega_sol), 'mode': mode_sol})
				omega_last = omega_sol
				n_guess = count_zero_crossings(np.real(mode_sol(z)[1]), z_max=0, z=z) + 1
		
		solutions[k] = solutions_this_k
	
	with open("komega_from_sim.pickle", 'wb') as f:
		pickle.dump(solutions, f)
