"""
Assumes the following ADIPLS parameters
nfmode=2
"""

import scipy.io
import numpy as np

class Mode():
	def __init__(self, f, nnw):
		"""
		Arguments:
			f: a file handle generated by scipy.io.FortranFile
			nnw: int, number of mesh points used by ADIPLS
		"""
		
		cs, ics, y = f.read_record("(38,)<f8", "(24,)<i4", f"({nnw},2)<f8")
		
		self._cs = cs
		self._ics = ics[:12]
		self.y = y
		
		#NOTE: see section 8.2 for the meaning of the various elements of cs and ics
		self.l = int(cs[17])
		self.n = int(cs[18])
		self.nu = cs[36]
		
		self.nordp = ics[8]
		self.nordg = ics[9]
		self.m = ics[10]
		
		#'Derived' quantities
		self.n_nodes = self.nordp + self.nordg

with scipy.io.FortranFile("amde.l9bi.d.202c.prxt3") as f:
	nnw, *_ = f.read_record("i4")

with scipy.io.FortranFile("amde.l9bi.d.202c.prxt3") as f:
	_, x = f.read_record("i4", f"({nnw},)<f8")
	
	modes = []
	while True:
		try:
			modes.append(Mode(f, nnw))
		except scipy.io.FortranEOFError:
			break


