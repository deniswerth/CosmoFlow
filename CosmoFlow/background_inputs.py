
import numpy as np
from scipy.interpolate import interp1d

class background_inputs():
	"""
	This class takes as inputs the mimimal background arrays and gives interpolated functions.
	This class needs to be executed just once at the beginning of the numerical integration
	"""

	def __init__(self, N_load, H_load, cs_load, m_load, rho_load, lambda1_load, lambda2_load, mu_load, alpha_load, kappa1_load, kappa2_load):

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
		self.mu_load      = mu_load
		self.alpha_load   = alpha_load
		self.kappa1_load  = kappa1_load
		self.kappa2_load  = kappa2_load

		#Creating continuous functions out of the imported functions
		self.H_f       = interp1d(self.N_load, self.H_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.cs_f      = interp1d(self.N_load, self.cs_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.m_f       = interp1d(self.N_load, self.m_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.rho_f     = interp1d(self.N_load, self.rho_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.lambda1_f = interp1d(self.N_load, self.lambda1_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.lambda2_f = interp1d(self.N_load, self.lambda2_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.mu_f      = interp1d(self.N_load, self.mu_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.alpha_f   = interp1d(self.N_load, self.alpha_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.kappa1_f  = interp1d(self.N_load, self.kappa1_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")
		self.kappa2_f  = interp1d(self.N_load, self.kappa2_load, bounds_error = False, kind = 'cubic', fill_value = "extrapolate")

		#Creating interpolating list
		self.interpolated = [self.H_f, self.cs_f, self.m_f, self.rho_f, self.lambda1_f, self.lambda2_f, self.mu_f, self.alpha_f, self.kappa1_f, self.kappa2_f]

	def output(self):
		"""
		Gives the interpolated continous functions that can be evaluated at any N
		"""
		return self.interpolated
