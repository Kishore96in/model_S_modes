"""
Define the system of equations given by [BirKosDuv04] and solve it for a single k, omega.

References:
	BirKosDuv04: Birch, Kosovichev, Duvall 2004 - Sensitivity of Acoustic Wave Travel Times to Sound-Speed Perturbations in the Solar Interior
"""

import numpy as np
import scipy.integrate
import warnings
import matplotlib.pyplot as plt

from solar_model import solar_model
from solar_model import read_extensive_model_MmKS as reader

def rhs(z, y, p, k, model):
	"""
	Equations A10 and A1 of [BirKosDuv04].
	Note that the scale heights etc are themselves functions of depth determined by Model S.
	
	Arguments:
		z, y, p: see scipy.integrate.solve_bvp
		k: horizontal wavenumber
		model: instance of solar_model
	
	Implementation note:
		We would need three boundary conditions if we were to solve for y_1, y_2, and omega. The first two BCs are determined by the problem itself, but is it a priori unclear what the third boundary condition should be. To circumvent this, we add a variable y_3, which is a function such that \int y_3 d z = \int y_2^2/c d z. This allows us to impose two more BCs (y_3(z_bot) = 0 and y_3(z_top) = 1). We thus have 4 BCs for 4 variables (y_1, y_2, y_3, and omega). This trick is from <https://stackoverflow.com/questions/53053866/how-do-you-feed-scipys-bvp-only-the-bcs-you-have>.
	"""
	dydz = np.zeros_like(y, dtype=complex)
	
	y1, y2, y3 = y
	assert len(p) == 1
	omega = p[0]
	
	c = model.c(z)
	H = model.H(z)
	N2 = model.N2(z)
	
	Gamma_tilde = 1 #omega/(omega + 1j*Gamma), where Gamma specifies damping of the modes.
	
	dydz[0] = (-1/c)*(
		- (c/H)*y1
		+ (1/omega)*(omega**2/Gamma_tilde - N2)*y2
		)
	
	dydz[1] = (-1/c)*(
		- (1/omega)*(omega**2 - Gamma_tilde*c**2*k**2)*y1
		+ (c/H)*y2
		)
	
	dydz[2] = np.abs(y2)**2/c
	
	return dydz

def bc(y_bot, y_top, p, k, model, z_bot, z_top):
	"""
	Equations A20 and A21 of [BirKosDuv04].
	Top is a free surface (zero Lagrangian pressure perturbation), while the bottom is impenetrable.
	
	Arguments:
		y_bot, y_top, p: see scipy.integrate.solve_bvp
		k: horizontal wavenumber
		model: instance of solar_model
	"""
	y1_bot, y2_bot, y3_bot = y_bot
	y1_top, y2_top, y3_top = y_top
	
	assert len(p) == 1
	omega = p[0]
	
	c_top = model.c(z_top)
	g_top = model.g(z_top)
	
	return np.array([
		y2_bot,
		omega*y1_top - (g_top/c_top)*y2_top,
		y3_bot,
		y3_top - np.sign(np.real(omega)),
		])

def bc_imp_both(y_bot, y_top, p, k, model, z_bot, z_top):
	"""
	Make both top and bottom boundaries impenetrable.
	
	Arguments:
		y_bot, y_top, p: see scipy.integrate.solve_bvp
		k: horizontal wavenumber
		model: instance of solar_model
	"""
	y1_bot, y2_bot, y3_bot = y_bot
	y1_top, y2_top, y3_top = y_top
	
	assert len(p) == 1
	omega = p[0]
	
	c_top = model.c(z_top)
	g_top = model.g(z_top)
	
	return np.array([
		y2_bot,
		y2_top,
		y3_bot,
		y3_top - np.sign(np.real(omega)),
		])

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

def make_guess_fmode(z_guess):
	"""
	Generate initial guess to encourage solve_bvp to find the f mode.
	"""
	if len(z_guess) < 3:
		raise ValueError("Length of z_guess should be > 3")
	
	y_guess = np.zeros((3,len(z_guess)), dtype=complex)
	y_guess[2] = np.linspace(0,1,len(z_guess))
	y_guess[1,-2] = 1
	return y_guess


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

def count_zero_crossings(arr, z_max=None, z=None):
	"""
	Count the number of zero crossings in the array arr. The first and last points are ignored even if the values there are zero.
	
	Arguments:
		arr: 1D numpy array
		z_max: float, optional. Only count zero crossings below this depth
		z: 1D numpy array. Only required if z_max is not None. Needs to be the same shape as arr; coordinate values at the corresponding indices.
	"""
	if z_max is not None:
		if z is None:
			raise ValueError("z must be specified to use z_max")
		izmax = np.argmin(np.abs(z_max - z))
		arr = arr[1:izmax]
	else:
		arr = arr[1:-1]
	
	n = np.sum(np.sign(np.abs(np.diff(np.sign(arr)))))
	
	if int(n) != n:
		raise RuntimeError("Number of zero crossings is not an integer!")
	
	if n > 0.1*len(arr):
		warnings.warn("Number of zero crossings may be affected by Nyquist errors. Try increasing the number of grid points.", RuntimeWarning)
	
	return int(n)

if __name__ == "__main__":
	model = solar_model("data/Model S extensive/fgong.l5bi.d.15", reader=reader)
	
	z_bot = -25
	z_top = 0.45
	k = 0.5
	omega_guess = 1.87e-2
	
	z_guess = np.linspace(z_bot, z_top, 50)
	
	y_guess = make_guess_pmode(z_guess, n=1)
	
	p_guess = np.array([omega_guess])
	
	RHS = lambda z, y, p: rhs(z, y, p, k=k, model=model)
	BC = lambda y_bot, y_top, p: bc(y_bot, y_top, p, k=k, model=model, z_bot=z_bot, z_top=z_top)
	
	sol = scipy.integrate.solve_bvp(
		RHS,
		BC,
		p=p_guess,
		x=z_guess,
		y=y_guess,
		# verbose=2,
		tol = 1e-6,
		max_nodes=1e6,
		)
		
	if not sol.success:
		warnings.warn(f"Solver failed for {k = }. {sol.message}", RuntimeWarning)
	
	z = np.linspace(z_bot, z_top, int(1e4))
	ruz = sol.sol(z)[1]/np.sqrt(model.c(z))
	ipbr = sol.sol(z)[0]*np.sqrt(model.c(z))
	
	print(rf"Found $\omega$ = {np.real_if_close(sol.p[0]) :.2e}")
	print(f"Number of zero crossings of re(u√ρ₀) in the interior of the domain: {count_zero_crossings(np.real(ruz))}")
	
	fig,ax = plt.subplots()
	ax.plot(z, np.real(ruz), label=r"$\mathrm{re}\left( u_z \sqrt{\rho_0} \right)$")
	ax.plot(z, np.imag(ruz), label=r"$\mathrm{im}\left( u_z \sqrt{\rho_0} \right)$")
	ax.plot(z, np.real(ipbr), label=r"$\mathrm{re}\left( ip/\sqrt{\rho_0} \right)$")
	ax.plot(z, np.imag(ipbr), label=r"$\mathrm{im}\left( ip/\sqrt{\rho_0} \right)$")
	ax.set_xlim(z_bot, z_top)
	ax.axhline(0, ls=':', c='k')
	ax.axvline(0, ls=':', c='k')
	ax.set_xlabel("$z$")
	
	ax.legend()
	
	plt.show()
