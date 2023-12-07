"""
Plot the dispersion relations of all found modes with the background state taken from a simulation of convection.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import pickle
import multiprocessing

from bg_from_sim import solar_model_from_sim as solar_model
from problem import rhs, bc_imp_both as bc, count_zero_crossings, make_guess_pmode, make_guess_fmode_from_k

def find_mode(omega_guess, k, model, z_guess, guesser):
	y_guess = guesser(z_guess)
	p_guess = np.array([omega_guess])
	
	RHS = lambda z, y, p: rhs(z, y, p, k=k, model=model)
	BC = lambda y_bot, y_top, p: bc(y_bot, y_top, p, k=k, model=model, z_bot=min(z_guess), z_top=max(z_guess))
	
	sol = scipy.integrate.solve_bvp(
		RHS,
		BC,
		p=p_guess,
		x=z_guess,
		y=y_guess,
		tol = 1e-6,
		max_nodes=1e5,
		)
		
	if not sol.success:
		warnings.warn(f"Solver failed for {k = }, {omega_guess = }. {sol.message}", RuntimeWarning)
	
	return np.real_if_close(sol.p[0]), sol.sol, sol.success


def get_modes_at_k(
	k,
	model,
	omega_max,
	omega_min,
	n_omega,
	d_omega,
	):
	"""
	Find the modes at a single value of k.
	
	Arguments:
		k: float.
		model: instance of solar_model.solar_model. Background state for which computations should be done.
		omega_max: float. Maximum value of omega to probe.
		omega_min: float. Minimum value of omega to probe.
		n_omega: int. Number of omega values to compute for in the range omega_min, omega_max.
		d_omega: float: Only consider modes which are further apart than this.
	
	Returns:
		solutions_this_k: list of dict. Each element has the following keys.
			'omega': complex or float. Eigenfrequency
			'sol': spline describing the eigenfunction
			'n': int. Number of zero crossings of the real part of u_z below z=0.
	"""
	
	z_bot = model.z_min
	z_top = model.z_max
	z = np.linspace(z_bot, z_top, int(1e3))
	omega_range = np.linspace(omega_min, omega_max, n_omega)
	
	solutions_this_k = []
	omega_last = -np.inf
	n_guess = 0
	guesser = lambda z_guess: make_guess_fmode_from_k(z_guess, k=k)
	
	g = np.sqrt(np.abs(model.g(z_top)))
	omega_f = np.sqrt(g*k)
	
	for omega in omega_range:
		if omega < omega_last + d_omega:
			continue
		
		z_guess = np.linspace(z_bot, z_top, 10+2*n_guess)
		if omega > omega_f:
			guesser = lambda z_guess: make_guess_pmode(z_guess, n=n_guess)
		
		omega_sol, mode_sol, success = find_mode(omega, k=k, model=model, z_guess=z_guess, guesser=guesser)
		
		if success and (np.abs(omega - omega_sol) < d_omega):
			n = count_zero_crossings(np.real(mode_sol(z)[1]), z_max=0, z=z)
			solutions_this_k.append({
				'omega': omega_sol,
				'mode': mode_sol,
				'n': n,
				})
			omega_last = omega_sol
			n_guess = n+1
	
	return solutions_this_k

def construct_komega(
	model,
	k_max,
	n_k,
	omega_max,
	omega_min,
	n_omega,
	d_omega,
	outputfile,
	n_workers=1,
	):
	"""
	Construct the k-omega diagram for a specified background and store it in a pickle file.
	
	Arguments:
		model: instance of solar_model.solar_model. Background state for which computations should be done.
		k_max: float. Maximum value of k to probe.
		n_k: int. Number of k values to compute for in the range 0,k_max.
		omega_max: float. Maximum value of omega to probe.
		omega_min: float. Minimum value of omega to probe.
		n_omega: int. Number of omega values to compute for in the range omega_min, omega_max.
		d_omega: float: Only consider modes which are further apart than this.
		outputfile: str. Filename for the output.
		n_workers: int. Number of worker processes to use (each worker will do the calculations for one value of k)
	"""
	if n_workers > n_k:
		warnings.warn(f"More workers ({n_workers}). Than the number of values of k ({n_k}). Some of them will remain idle.", )
	
	with multiprocessing.Pool(n_workers) as pool:
		solutions = {}
		kwds = {
			'model': model,
			'omega_max': omega_max,
			'omega_min': omega_min,
			'n_omega': n_omega,
			'd_omega': d_omega,
			}
		for k in np.linspace(0, k_max, n_k):
			solutions[k] = pool.apply_async(get_modes_at_k, args=(k,), kwds=kwds)
		
		solutions = {k: v.get() for k, v in solutions.items()}
	
	with open(outputfile, 'wb') as f:
		pickle.dump(solutions, f)

def plot_komega(filename, k_scl, omega_scl):
	"""
	Arguments:
		filename: str. Path to pickle file in which the k-omega diagram was saved.
		k_scl: Multiplicative factor for k before plotting
		omega_scl: multiplicative factor for omega before plotting.
	"""
	with open(filename, 'rb') as f:
		solutions = pickle.load(f)
	
	k_list = list(solutions.keys())
	
	points = {
		'k': [],
		'omega': [],
		'n': [],
		}
	
	for k in k_list:
		for p in solutions[k]:
			points['k'].append(k)
			points['omega'].append(p['omega'])
			points['n'].append(p['n'])
	
	points = {key: np.array(v) for key, v in points.items()}
	
	points['k'] *= k_scl
	points['omega'] *= omega_scl
	
	plt.scatter('k', 'omega', data=points)

if __name__ == "__main__":
	plot = True
	model = solar_model("background_a6.0l.1.pickle")
	L_0 = model.c(0)**2/np.abs(model.g(0))
	omega_0 = np.abs(model.g(0))/model.c(0)
	
	if not os.path.isfile("komega_from_sim.pickle"):
		construct_komega(
			model = model,
			k_max = 0.5/L_0,
			n_k = 5,
			omega_max = 1.5*omega_0,
			omega_min = 0.1*omega_0,
			n_omega = 100,
			d_omega = 0.1*omega_0,
			outputfile="komega_from_sim.pickle",
			n_workers = 2,
			)
	
	if plot:
		plot_komega("komega_from_sim.pickle", k_scl=L_0, omega_scl=1/omega_0)
		plt.show()
