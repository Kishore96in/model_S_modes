"""
Plot various background quantities in Solar Model S
"""

import matplotlib.pyplot as plt
import numpy as np

from solar_model import read_extensive_model_MmKS, solar_model

if __name__ == "__main__":
	model = solar_model("data/Model S extensive/fgong.l5bi.d.15", reader=read_extensive_model_MmKS)
	d = read_extensive_model_MmKS("data/Model S extensive/fgong.l5bi.d.15")
	
	z = np.linspace(model.z_min, model.z_max, 1000)
	
	fig, ax = plt.subplots()
	ax.semilogy(z, model.c(z))
	ax.set_xlabel("$z$ (Mm)")
	ax.set_ylabel("$c$ (Mm s$^{-1}$)")
	ax.set_xlim(min(z), max(z))
	fig.set_size_inches(4,3)
	fig.tight_layout()
	
	fig,ax = plt.subplots()
	ax.semilogy(z, np.abs(model.g(z)))
	ax.set_xlabel("$z$ (Mm)")
	ax.set_ylabel(r"$\left| g \right|$ (Mm s$^{-2}$)")
	ax.set_xlim(min(z), max(z))
	ax.set_ylim(1e-4, 4e-3)
	fig.set_size_inches(4,3)
	fig.tight_layout()
	
	fig,ax = plt.subplots()
	ax.plot(z, model.N2(z))
	ax.set_xlabel("$z$ (Mm)")
	ax.set_ylabel("$N^2$ (Hz$^2$)")
	ax.axhline(0, ls=':', c='k')
	ax.set_xlim(min(z), max(z))
	ax.set_ylim(-1e-5, 1e-5)
	fig.set_size_inches(4,3)
	fig.tight_layout()
	
	fig,ax = plt.subplots()
	ax.semilogy(d.r, d.rho)
	ax.set_xlabel("$r$ (Mm)")
	ax.set_ylabel(r"$\rho$ (kg Mm$^{-3}$)")
	ax.set_xlim(min(d.r), max(d.r))
	fig.set_size_inches(4,3)
	fig.tight_layout()
	
	fig,ax = plt.subplots()
	ax.semilogy(d.r, d.T)
	ax.set_xlabel("$r$ (Mm)")
	ax.set_ylabel("$T$ (K)")
	ax.set_xlim(min(d.r), max(d.r))
	fig.set_size_inches(4,3)
	fig.tight_layout()
	
	fig,ax = plt.subplots()
	ax.semilogy(d.r, d.P)
	ax.set_xlabel("$r$ (Mm)")
	ax.set_ylabel("$P$ (kg Mm$^{-1}$ s$^{-2}$)")
	ax.set_xlim(min(d.r), max(d.r))
	fig.set_size_inches(4,3)
	fig.tight_layout()
	
	plt.show()
