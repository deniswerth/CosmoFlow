from scipy.interpolate import interp1d # import interp1d from Scipy package


class parameters():
	"""
	This class takes as inputs the parameters of the theory and creates interpolated continuous functions
	"""

	def __init__(self, N_load, H_load, g_load):

		#Importing pre-computed parameters of the theory
		self.N_load = N_load
		self.H_load = H_load
		self.g_load = g_load

		#Creating continuous functions out of the imported parameters
		self.H_f = interp1d(self.N_load, self.H_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.g_f = interp1d(self.N_load, self.g_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")

		#Creating interpolating list containing continuous functions
		self.interpolated = [self.H_f, self.g_f]

	def output(self):
		"""
		Returns the interpolated continous functions that can be evaluated at any N (time in e-folds)
		"""
		return self.interpolated
