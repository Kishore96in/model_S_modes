"""
Define the system of equations determining the evolution of linearized perturbations in a specific background and solve it for a single k, omega.
"""

import numpy as np
import scipy.integrate
import warnings
import matplotlib.pyplot as plt

from solar_model import solar_model
from solar_model import read_extensive_model_MmKS as reader
from guess import make_guess_pmode, make_guess_fmode_from_k
from utils import count_zero_crossings

def rhs(z, y, p, k, model):
	"""
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
	gradlnrho = model.gradlnrho(z)
	gradlnc = model.gradlnc(z)
	N2 = model.N2(z)
	g = model.g(z)
	
	beta = omega #(omega + 1j*Gamma), where Gamma specifies damping of the modes.
	
	dydz[0] = (
		(-gradlnrho/2 - gradlnc/2 + g/c**2)*y1
		+ (N2/(omega*c) - beta/c)*y2
		)
	
	dydz[1] = (
		(omega/c - c*k**2/beta)*y1
		+ (-gradlnrho/2 + gradlnc/2 + N2/g)*y2
		)
	
	dydz[2] = np.abs(y2)**2/c
	
	return dydz

def bc(y_bot, y_top, p, k, model, z_bot, z_top):
	"""
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
	rhoinv_grad_p_top = model.rhoinv_grad_p(z_top)
	g_top = model.g(z_top)
	
	return np.array([
		y2_bot,
		omega*y1_top - (y2_top/c_top)*rhoinv_grad_p_top,
		y3_bot,
		#the below is to ensure that omega remains positive
		y3_top - np.sign(np.real(omega)),
		])

if __name__ == "__main__":
	model = solar_model("data/Model S extensive/fgong.l5bi.d.15", reader=reader)
	
	z_bot = -25
	z_top = 0.45
	k = 0.5
	omega_guess = 1.83e-2
	
	z_guess = np.linspace(z_bot, z_top, 50)
	
	y_guess = make_guess_pmode(z_guess, n=2)
	
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
	y1 = sol.sol(z)[0]
	y2 = sol.sol(z)[1]
	
	uz = y2/np.sqrt(model.c(z)*model.rho(z))
	ipbrc = y1/np.sqrt(model.c(z)*model.rho(z))
	
	print(rf"Found $\omega$ = {np.real_if_close(sol.p[0]) :.2e}")
	print(f"Number of zero crossings of re(u√(cρ₀)) in the interior of the domain: {count_zero_crossings(np.real(y2))}")
	
	fig,axs = plt.subplots(nrows=3)
	
	axs[0].plot(z, np.real(y2), label=r"$\mathrm{re}(y_2)$")
	axs[0].plot(z, np.imag(y2), label=r"$\mathrm{im}(y_2)$")
	axs[0].plot(z, np.real(y1), label=r"$\mathrm{re}(y_1)$")
	axs[0].plot(z, np.imag(y1), label=r"$\mathrm{im}(y_1)$")
	axs[0].legend()
	
	axs[1].plot(z, np.real(uz*np.sqrt(model.rho(z))))
	axs[1].set_ylabel(r"$u_z \sqrt{\rho_0}$")
	
	axs[2].plot(z, np.real(uz))
	axs[2].set_ylabel(r"$\mathrm{re}(u_z)$")
	
	for ax in axs:
		ax.set_xlim(z_bot, z_top)
		ax.axhline(0, ls=':', c='k')
		ax.axvline(0, ls=':', c='k')
	
	for ax in axs[:-1]:
		ax.xaxis.set_ticklabels([])
	
	axs[-1].set_xlabel("$z$")
	fig.suptitle(f"$\omega = ${np.real_if_close(sol.p[0]) :.2e}, $k = ${k:.2e}")
	
	fig.set_size_inches(5,7)
	fig.tight_layout()
	
	plt.show()
