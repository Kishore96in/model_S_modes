"""
References:
	BirKosDuv04: Birch, Kosovichev, Duvall 2004 - Sensitivity of Acoustic Wave Travel Times to Sound-Speed Perturbations in the Solar Interior
"""

import numpy as np

class read_model_file():
	def __init__(self):
		self.r, self.c, self.rho, self.P, self.gamma, self.T = np.loadtxt('solar_model_S_cptrho.l5bi.d.15c', unpack=True)

