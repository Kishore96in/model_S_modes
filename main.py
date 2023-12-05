"""
References:
	BirKosDuv04: Birch, Kosovichev, Duvall 2004 - Sensitivity of Acoustic Wave Travel Times to Sound-Speed Perturbations in the Solar Interior
"""

import numpy as np
import scipy.integrate
import warnings

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
		omega*y1_top + (g_top/c_top)*y2_top,
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

if __name__ == "__main__":
	model = solar_model("Model S extensive data/fgong.l5bi.d.15", reader=reader)
	
	z_bot = -25
	z_top = 0.45
	z_guess = np.linspace(z_bot, z_top, 10)
	
	#Initial guesses
	y_guess = make_guess_pmode(z_guess, n=0)
	# y_guess = make_guess_fmode(z_guess)
	
	p_guess = np.array([1e-3])
	
	k_list = np.linspace(0,1.3,10)
	solutions = {}
	for k in k_list:
		RHS = lambda z, y, p: rhs(z, y, p, k=k, model=model)
		BC = lambda y_bot, y_top, p: bc(y_bot, y_top, p, k=k, model=model, z_bot=z_bot, z_top=z_top)
		
		sol = scipy.integrate.solve_bvp(
			RHS,
			BC,
			p=p_guess,
			x=z_guess,
			y=y_guess,
			)
		
		if not sol.success:
			warnings.warn(f"Solver failed for {k = }. {sol.message}", RuntimeWarning)
		
		solutions[k] = sol
