"""
Helpers to generate the initial guesses for the eigenfunctions.
"""

import numpy as np

def make_guess_pmode(z_guess, n):
	"""
	Generate initial guess to encourage solve_bvp to find the p mode of order n.
	
	n: number of radial nodes for the guess (excluding endpoints)
	"""
	
	z_bot = np.min(z_guess)
	z_top = np.max(z_guess)
	
	wave = np.sin((n+1)*np.pi*(z_guess-z_bot)/(z_top-z_bot))
	return np.array([
		np.zeros_like(z_guess),
		wave,
		np.linspace(0,1,len(z_guess))],
		dtype=complex,
		)

def make_guess_fmode_from_k(z_guess, k):
	"""
	Generate initial guess to encourage solve_bvp to find the f mode.
	
	Arguments:
		z_guess: 1D numpy array. Grid
		k: float, Wavenumber
	"""
	z_top = np.max(z_guess)
	
	y_guess = np.zeros((3,len(z_guess)), dtype=complex)
	y_guess[2] = np.linspace(0,1,len(z_guess))
	y_guess[1] = np.where(
		z_guess < 0,
		np.exp(-np.abs(k*z_guess)),
		1 - z_guess/z_top,
		)
	
	return y_guess

