import numpy as np
from scipy.interpolate import interp1d

class background_inputs():
	"""
	This class takes as inputs the mimimal background arrays and gives interpolated functions.
	This class needs to be executed just once at the beginning of the numerical integration
	"""

	def __init__(self, N_load, H_load, cs_load, m_load, rho_load, lambda1_load, lambda2_load, lambda3_load, lambda4_load, lambda5_load, lambda6_load):

		#Importing pre-computed background quantities
		self.N_load       = N_load
		self.H_load       = H_load

		#Quadratic theory
		self.cs_load      = cs_load
		self.m_load       = m_load
		self.rho_load     = rho_load

		#Cubic theory
		self.lambda1_load = lambda1_load
		self.lambda2_load = lambda2_load
		self.lambda3_load = lambda3_load
		self.lambda4_load = lambda4_load
		self.lambda5_load = lambda5_load
		self.lambda6_load = lambda6_load

		#Creating continuous functions out of the imported functions
		self.H_f       = interp1d(self.N_load, self.H_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.cs_f      = interp1d(self.N_load, self.cs_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.m_f       = interp1d(self.N_load, self.m_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.rho_f     = interp1d(self.N_load, self.rho_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.lambda1_f = interp1d(self.N_load, self.lambda1_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.lambda2_f = interp1d(self.N_load, self.lambda2_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.lambda3_f = interp1d(self.N_load, self.lambda3_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.lambda4_f = interp1d(self.N_load, self.lambda4_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.lambda5_f = interp1d(self.N_load, self.lambda5_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.lambda6_f = interp1d(self.N_load, self.lambda6_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")


		#Creating interpolating list
		self.interpolated = [self.H_f, self.cs_f, self.m_f, self.rho_f, self.lambda1_f, self.lambda2_f, self.lambda3_f, self.lambda4_f, self.lambda5_f, self.lambda6_f]

	def output(self):
		"""
		Gives the interpolated continous functions that can be evaluated at any N
		"""
		return self.interpolated
