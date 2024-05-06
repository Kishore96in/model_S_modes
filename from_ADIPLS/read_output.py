"""
Assumes the following ADIPLS parameters
nfmode=2
"""

import scipy.io
import numpy as np

class Mode():
	def __init__(self, csummm, y):
		self._csummm = csummm
		self.y = y
		
with scipy.io.FortranFile("amde.l9bi.d.202c.prxt3") as f:
	nnw, *_ = f.read_record("i4")

with scipy.io.FortranFile("amde.l9bi.d.202c.prxt3") as f:
	_, x = f.read_record("i4", f"({nnw},)<f8")
	
	modes = []
	while True:
		try:
			csummm, y = f.read_record("(50,)<f8", f"(2,{nnw})<f8")
			modes.append(Mode(csummm, y))
		except scipy.io.FortranEOFError:
			print("EOF")
			break


