from scipy.interpolate import interp1d # import interp1d from Scipy package


class parameters():
	"""
	This class takes as inputs the parameters of the theory and creates interpolated continuous functions
	"""

	def __init__(self, N_load, H_load, cs_load, m_load, rho_load, g_load):

		#Importing pre-computed parameters of the theory
		self.N_load = N_load
		self.H_load = H_load
		self.cs_load = cs_load
		self.m_load = m_load
		self.rho_load = rho_load
		self.g_load = g_load

		#Creating continuous functions out of the imported parameters
		self.H_f = interp1d(self.N_load, self.H_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.cs_f = interp1d(self.N_load, self.cs_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.m_f = interp1d(self.N_load, self.m_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.rho_f = interp1d(self.N_load, self.rho_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.g_f = interp1d(self.N_load, self.g_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")

		#Creating interpolating list containing continuous functions
		self.interpolated = [self.H_f, self.cs_f, self.m_f, self.rho_f, self.g_f]

	def output(self):
		"""
		Returns the interpolated continous functions that can be evaluated at any N (time in e-folds)
		"""
		return self.interpolated
