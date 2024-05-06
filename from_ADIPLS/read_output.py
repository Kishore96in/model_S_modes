"""
Assumes the following ADIPLS parameters
nfmode=2
"""

import scipy.io
import numpy as np

class Mode():
	def __init__(self, cs, ics, y):
		self._cs = cs
		self._ics = ics[:12]
		self.y = y
		
with scipy.io.FortranFile("amde.l9bi.d.202c.prxt3") as f:
	nnw, *_ = f.read_record("i4")

with scipy.io.FortranFile("amde.l9bi.d.202c.prxt3") as f:
	_, x = f.read_record("i4", f"({nnw},)<f8")
	
	modes = []
	while True:
		try:
			cs, ics, y = f.read_record("(38,)<f8", "(24,)<i4", f"(2,{nnw})<f8")
			modes.append(Mode(cs, ics, y))
		except scipy.io.FortranEOFError:
			break


