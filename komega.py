"""
Algorithm:
1. Choose a value of k
2. Choose the lowest possible value of omega and set n_guess=0
3. If omega < sqrt(g_min*k), use make_guess_fmode_with_omega; else, use make_guess_pmode with n_guess nodes.
4. If a mode is found and it its omega is close to the chosen value of omega, accept it (save for later)
5. Set n_guess to the number of radial nodes (below z=0, excluding z=z_bot) of the last accepted solution
5. Choose the larger among (the next lowest value of omega, the omega of the last accepted solution). If omega>omega_max, choose the next value of k and go to step 2. Else, go to step 3.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import pickle
import multiprocessing
import time

from guess import make_guess_pmode, make_guess_fmode_from_k
from utils import count_zero_crossings

def find_mode(omega_guess, k, model, z_guess, guesser, rhs, bc):
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
	
	return np.real_if_close(sol.p)[0], sol.sol, sol.success

def get_modes_at_k(
	k,
	model,
	omega_max,
	omega_min,
	n_omega,
	d_omega,
	rhs,
	bc,
	z_bot,
	z_top,
	nz,
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
		rhs: function. Specifies RHS for solve_bvp.
		bc: function. Specifies boundary conditions for solve_bvp.
		z_bot: float: position of the bottom of the domain.
		z_top: float: position of the top of the domain.
		nz: int. Minimum number of grid points to use for the initial grid passed to solve_bvp.
	
	Returns:
		solutions_this_k: list of dict. Each element has the following keys.
			'omega': complex or float. Eigenfrequency
			'sol': spline describing the eigenfunction
			'n': int. Number of zero crossings of the real part of u_z below z=0.
	"""
	
	omega_range = np.linspace(omega_min, omega_max, n_omega)
	
	if d_omega < np.min(np.diff(omega_range))/2:
		warnings.warn(f"get_modes_at_k: d_omega is less than the minimum spacing between omega. you will most likely miss some modes.", RuntimeWarning)
	
	solutions_this_k = []
	omega_last = -np.inf
	n_guess = 0
	guesser = lambda z_guess: make_guess_fmode_from_k(z_guess, k=k)
	
	g = np.sqrt(np.abs(model.g(z_top)))
	omega_f = np.sqrt(g*k)
	
	for omega in omega_range:
		if omega < omega_last + d_omega:
			continue
		
		z_guess = np.linspace(z_bot, z_top, nz+2*n_guess)
		if omega > omega_f:
			guesser = lambda z_guess: make_guess_pmode(z_guess, n=n_guess)
		
		omega_sol, mode_sol, success = find_mode(omega, k=k, model=model, z_guess=z_guess, guesser=guesser, rhs=rhs, bc=bc)
		
		if success and (np.abs(omega - omega_sol) < d_omega):
			z = np.linspace(z_bot, z_top, int(max(1e3, nz)))
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
	k_list,
	omega_max,
	omega_min,
	n_omega,
	d_omega,
	outputfile,
	rhs,
	bc,
	z_bot = None,
	z_top = None,
	nz = 10,
	n_workers=1,
	discard_extra = False,
	):
	"""
	Construct the k-omega diagram for a specified background and store it in a pickle file.
	
	Arguments:
		model: instance of solar_model.solar_model. Background state for which computations should be done.
		k_list: list of k at which the modes should be found.
		omega_max: float. Maximum value of omega to probe.
		omega_min: float. Minimum value of omega to probe.
		n_omega: int. Number of omega values to compute for in the range omega_min, omega_max.
		d_omega: float: Only consider modes which are further apart than this.
		outputfile: str. Filename for the output.
		n_workers: int. Number of worker processes to use (each worker will do the calculations for one value of k)
		rhs: function. Specifies RHS for solve_bvp.
		bc: function. Specifies boundary conditions for solve_bvp.
		z_bot: float: position of the bottom of the domain.
		z_top: float: position of the top of the domain.
		nz: int. Minimum number of grid points to use for the initial grid passed to solve_bvp.
		discard_extra: bool. Whether to save solutions whose eigenfrequencies are outside the specified bounds (omega_min, omega_max).
	"""
	if n_workers > len(k_list):
		warnings.warn(f"More workers ({n_workers}). Than the number of values of k ({len(k_list)}). Some of them will remain idle.", )
	
	if z_bot is None:
		z_bot = model.z_min
	if z_top is None:
		z_top = model.z_max
	
	t_start = time.time()
	
	with multiprocessing.Pool(n_workers) as pool:
		solutions = {}
		kwds = {
			'model': model,
			'omega_max': omega_max,
			'omega_min': omega_min,
			'n_omega': n_omega,
			'd_omega': d_omega,
			'rhs': rhs,
			'bc': bc,
			'z_bot': z_bot,
			'z_top': z_top,
			'nz': nz,
			}
		for k in k_list:
			solutions[k] = pool.apply_async(get_modes_at_k, args=(k,), kwds=kwds)
		
		solutions = {k: v.get() for k, v in solutions.items()}
	
	if discard_extra:
		for k in solutions.keys():
			solutions[k] = [s for s in solutions[k] if omega_min<=s['omega']<omega_max]
	
	ret = {
		'solutions': solutions,
		'model': model,
		'rhs': rhs,
		'bc': bc,
		'z_bot': z_bot,
		'z_top': z_top,
		'nz': nz,
		'k_list': k_list,
		'omega_max': omega_max,
		'omega_min': omega_min,
		'n_omega': n_omega,
		'd_omega': d_omega,
		'discard_extra': discard_extra,
		'cputime': (time.time() - t_start)*n_workers
		}
	
	with open(outputfile, 'wb') as f:
		pickle.dump(ret, f)

class ax_scaler():
	"""
	Wrapper around matplotlib.axes._axes.Axes that scales the data before plotting it.
	"""
	def __init__(self, ax, k_scl, omega_scl):
		self.__ax = ax
		self.__k_scl = k_scl
		self.__omega_scl = omega_scl
	
	def plot(self, k, omega, *args, **kwargs):
		k = k*self.__k_scl
		omega = omega*self.__omega_scl
		return self.__ax.plot(k, omega, *args, **kwargs)
	
	def set_xlim(self, left=None, right=None, *args, **kwargs):
		left = left*self.__k_scl
		right = right*self.__k_scl
		return self.__ax.set_xlim(left, right, *args, **kwargs)
	
	def set_ylim(self, bottom=None, top=None, *args, **kwargs):
		bottom = bottom*self.__omega_scl
		top = top*self.__omega_scl
		return self.__ax.set_ylim(bottom, top, *args, **kwargs)
	
	def __getattr__(self, attr):
		return getattr(self.__ax, attr)

def scale_axs(axs, k_scl, omega_scl):
	"""
	Given a list of Axes (e.g. the output of plt.subplots with nrows*ncols > 1), apply ax_scaler to each Axes object.
	"""
	if iterable(axs):
		return type(a)([scale_axs(ax, k_scl, omega_scl) for ax in axs])
	else:
		return ax_scaler(axs, k_scl, omega_scl)

def plot_komega(
	filename,
	k_scl=1,
	omega_scl=1,
	ax=None,
	n_max=None,
	scatter_kwargs=None,
	):
	"""
	Arguments:
		filename: str. Path to pickle file in which the k-omega diagram was saved.
		k_scl: Multiplicative factor for k before plotting
		omega_scl: multiplicative factor for omega before plotting.
		ax: matplotlib axes object.
		n_max: int. In the legend, do not separately indicate modes with more than this many nodes.
		scatter_kwargs: dict. Extra kwargs to be passed to plt.scatter.
	"""
	if ax is None:
		_, ax = plt.subplots()
	if scatter_kwargs is None:
		scatter_kwargs = {}
	
	with open(filename, 'rb') as f:
		solutions = pickle.load(f)['solutions']
	
	points = {
		'k': [],
		'omega': [],
		'n': [],
		}
	
	for k in solutions.keys():
		for p in solutions[k]:
			points['k'].append(k)
			points['omega'].append(p['omega'])
			points['n'].append(p['n'])
	
	points = {key: np.array(v) for key, v in points.items()}
	
	points['k'] *= k_scl
	points['omega'] *= omega_scl
	
	n_list = points['n']
	if n_max is not None:
		n_ceil = np.where(n_list < n_max, n_list, n_max)
	else:
		n_ceil = n_list
	
	n_vals = np.sort(np.unique(n_ceil))
	for n in n_vals:
		if n == n_max:
			label = rf"$\geq{n}$"
		else:
			label = rf"${n}$"
		
		inds = (n_ceil == n)
		data = {key: v[inds] for key, v in points.items()}
		ax.scatter('k', 'omega', data=data, label=label, **scatter_kwargs)
	
	return ax_scaler(ax, k_scl=k_scl, omega_scl=omega_scl)
