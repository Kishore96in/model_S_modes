import numpy as np
import matplotlib.pyplot as plt
import numbers
import warnings
import os

class sci_format():
	"""
	Format a float in scientific notation for use in matplotlib
	"""
	def __init__(self, precision=None):
		f = plt.ScalarFormatter()
		f.set_scientific(True)
		f.set_useMathText(True)
		
		self.f = f
		self.precision = precision
	
	def __call__(self, data):
		d = np.format_float_scientific(data, precision=self.precision)
		return self.f.format_data(float(d))

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

class ceil_spline():
	"""
	Applies a ceiling value to the given spline.
	"""
	def __init__(self, spline, ceil):
		self.spline = spline
		self.ceil = ceil
	
	def __call__(self, z):
		spline = self.spline(z)
		ceil = np.full_like(spline, self.ceil)
		return np.where(spline > ceil, ceil, spline)

def ralign_legend(l):
	"""
	Right-align the labels in a legend. Note that this needs to be called after plt.show(block=False) for the figure dpi to be picked up correctly.
	
	Taken from https://stackoverflow.com/questions/7936034/text-alignment-in-a-matplotlib-legend/8078114#8078114
	"""
	texts = l.get_texts()
	widths = np.array([t.get_window_extent().width for t in texts])
	shifts = np.max(widths) - widths
	for t, shift in zip(texts, shifts):
		t.set_ha('right')
		t.set_position((shift,0))

def add_arrow(line, position=None, size=15):
	"""
	Add an arrow to a line. Copied from https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot/34018322#34018322
	
	Arguments:
		line:       Line2D object
		position:   x-position of the arrow.
		size:       size of the arrow in fontsize points
	"""
	color = line.get_color()
	xdata = line.get_xdata()
	ydata = line.get_ydata()

	if position is None:
		position = np.average(xdata)
	elif position == "max":
		position = np.max(xdata)
	elif position == "min":
		position = np.min(xdata)
	elif not isinstance(position, numbers.Number):
		raise TypeError(f"Unable to handle `{position}`, of type {type(position)}")
	
	start_ind = np.argmin(np.abs(xdata - position))
	end_ind = start_ind + 1

	line.axes.annotate('',
		xytext=(xdata[start_ind], ydata[start_ind]),
		xy=(xdata[end_ind], ydata[end_ind]),
		arrowprops=dict(arrowstyle="-|>", color=color),
		size=size
	)

class fig_saver():
	"""
	Helper to save a figure in a directory, automatically creating the required paths.
	
	Arguments:
		savefig: bool
		savedir: string, path to save the figure
	"""
	def __init__(self, savefig = True, savedir = "."):
		self.savefig = savefig
		self.savedir = savedir
	
	def __call__(self, fig, name, **kwargs):
		if not self.savefig:
			return
		
		if not os.path.exists(self.savedir):
			#Create directory if it does not exist
			os.makedirs(self.savedir)
		elif not os.path.isdir(self.savedir):
			raise FileExistsError(f"Save location {self.savedir} exists but is not a directory.")
		
		loc = os.path.join(self.savedir, name)
		loc_dir = os.path.dirname(loc)
		if not os.path.exists(loc_dir):
			os.makedirs(loc_dir)
		fig.savefig(loc, **kwargs)
