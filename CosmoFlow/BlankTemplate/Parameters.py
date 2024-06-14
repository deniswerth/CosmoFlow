from scipy.interpolate import interp1d # import interp1d from Scipy package


class parameters():
	"""
	This class takes as inputs the parameters of the theory and creates interpolated continuous functions
	"""

	def __init__(self, N_load, parameter_load):

		#Importing pre-computed parameters of the theory
		self.N_load = N_load
		self.parameter_load = parameter_load

		#Creating continuous functions out of the imported parameters
		self.parameter_f = interp1d(self.N_load, self.parameter_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")

		#Creating interpolating list containing continuous functions
		self.interpolated = [self.parameter_f]

	def output(self):
		"""
		Returns the interpolated continous functions that can be evaluated at any N (time in e-folds)
		"""
		return self.interpolated
