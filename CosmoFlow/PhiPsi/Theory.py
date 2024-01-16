import numpy as np # import Numpy package for vectorisation
from scipy.misc import derivative # import derivative from Scipy package


class theory():
	"""
	This class defines the theory: (i) time-dependent functions, (ii) tensors Delta, M, I, A, B, C and D, and (iii) tensors u^A_B and u^A_BC
	All functions have been optimised to be computed once at each integration step.
	"""

	def __init__(self, N, Nfield, interpolated):
		self.N = N
		self.Nfield = Nfield
		self.interpolated = interpolated # (H_f, cs_f, m_f, rho_f, lambda1_f, lambda2_f, lambda3_f)

		#Evaluate the parameters at N
		N = self.N
		self.H = self.H_f(N)
		self.cs = self.cs_f(N)
		self.m = self.m_f(N)
		self.rho = self.rho_f(N)
		self.lambda1 = self.lambda1_f(N)
		self.lambda2 = self.lambda2_f(N)
		self.lambda3 = self.lambda3_f(N)

		#Additional functions and scale functions
		self.a = self.a_f(N) # Scale factor
		self.dH = self.dH_f(N) # Derivative of Hubble
		self.scale = self.scale_f(N) # Rescale function to improve performances
		self.dscale = self.dscale_f(N) # Derivative of rescale function


	#Define continuous functions
	def H_f(self, N):
		return self.interpolated[0](N)

	def cs_f(self, N):
		return self.interpolated[1](N)

	def m_f(self, N):
		return self.interpolated[2](N)

	def rho_f(self, N):
		return self.interpolated[3](N)

	def lambda1_f(self, N):
		return self.interpolated[4](N)

	def lambda2_f(self, N):
		return self.interpolated[5](N)

	def lambda3_f(self, N):
		return self.interpolated[6](N)


	#Deduced additional functions
	def a_f(self, N):
		"""
		Scale factor as a function of the number of e-folds
		"""
		return np.exp(N)

	def dH_f(self, N):
		"""
		Derivative of the Hubble rate (with respect to cosmic time)
		"""
		dHdN = derivative(self.H_f, N, dx = 1e-6)
		return self.H_f(N)*dHdN

	def k_mode(self, N_exit):
		"""
		Return the mode k horizon crossing corresponding to N_exit
		"""
		return self.a_f(N_exit) * self.H_f(N_exit)

	def scale_f(self, N):
		"""
		Define the rescaled scale factor to improve performance
		"""
		k = 1
		a = self.a_f(N)
		H = self.H_f(N)
		return a/(1. + a*H/k)/H

	def dscale_f(self, N):
		"""
		Define the derivative of scale function
		"""
		k = 1
		a = self.a_f(N)
		H = self.H_f(N)
		Hd = self.dH_f(N)
		return -Hd/H/H*a/(1. + a*H/k) + a/(1. + a*H/k) - a*(a*H*H/k + a*Hd/k)/(1. + a*H/k)/(1. + a*H/k)/H


	# Define the quadratic theory tensors
	def Delta_ab(self):
		Nfield = self.Nfield
		Deltaab = np.eye(Nfield) # Identity matrix of size Nfield.Nfield
		return Deltaab

	def I_ab(self):
		Nfield = self.Nfield
		Iab = np.zeros((Nfield, Nfield))
		Iab[0, 1] = self.rho
		return Iab

	def M_ab(self, k):
		Nfield = self.Nfield
		Mab = np.eye(Nfield)
		Mab[0, 0] = -k**2/self.a**2 * self.cs**2
		Mab[1, 1] = -k**2/self.a**2 - self.m**2 - self.rho**2
		return Mab


	# Define the cubic theory tensors
	def A_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))
		k1k2 = (k3**2 - k1**2 - k2**2)/2
		k1k3 = (k2**2 - k1**2 - k3**2)/2
		k2k3 = (k1**2 - k2**2 - k3**2)/2
		Aabc[1, 1, 1] += -self.lambda3/3 + self.rho*self.lambda2
		Aabc[0, 0, 1] += self.lambda1 * k1k2/self.a**2 /3
		Aabc[0, 1, 0] += self.lambda1 * k1k3/self.a**2 /3
		Aabc[1, 0, 0] += self.lambda1 * k2k3/self.a**2 /3
		return Aabc

	def A_abc_fast(self, k1, k2, k3): # For initial conditions
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))
		k1k2 = (k3**2 - k1**2 - k2**2)/2
		k1k3 = (k2**2 - k1**2 - k3**2)/2
		k2k3 = (k1**2 - k2**2 - k3**2)/2
		Aabc[0, 0, 1] += self.lambda1 * k1k2 /3
		Aabc[0, 1, 0] += self.lambda1 * k1k3 /3
		Aabc[1, 0, 0] += self.lambda1 * k2k3 /3
		return Aabc

	def A_abc_slow(self, k1, k2, k3):
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))
		Aabc[1, 1, 1] += -self.lambda3/3 + self.rho*self.lambda2
		return Aabc

	def B_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Babc = np.zeros((Nfield, Nfield, Nfield))
		Babc[1, 1, 0] += -self.lambda2
		return Babc

	def C_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Cabc = np.zeros((Nfield, Nfield, Nfield))
		return Cabc

	def D_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Dabc = np.zeros((Nfield, Nfield, Nfield))
		return Dabc


	# Define the u-tensors
	def u_AB(self, k):
		Nfield = self.Nfield
		H = self.H
		s = self.scale
		ds = self.dscale
		S = np.ones((Nfield, Nfield)) + (s-1)*np.eye(Nfield)
		uAB = np.zeros((2*Nfield, 2*Nfield))
		uAB[:Nfield, :Nfield] = -self.I_ab()/H
		uAB[:Nfield, Nfield:] = self.Delta_ab()/H /s
		uAB[Nfield:, :Nfield] = self.M_ab(k)/H *s
		uAB[Nfield:, Nfield:] = (self.I_ab()).T/H - 3*self.H*np.eye(Nfield)/H + ds/s*np.eye(Nfield)/H 
		return uAB

	def u_ABC(self, k1, k2, k3):
		Nfield = self.Nfield
		s = self.scale
		S = np.ones((Nfield, Nfield, Nfield)) + (s-1)*np.eye(Nfield)
		H = self.H
		uABC = np.zeros((2*Nfield, 2*Nfield, 2*Nfield))

		A123 = self.A_abc(k1, k2, k3)
		B123 = self.B_abc(k1, k2, k3)
		B132 = self.B_abc(k1, k3, k2)
		B231 = self.B_abc(k2, k3, k1)
		C123 = self.C_abc(k1, k2, k3)
		C132 = self.C_abc(k1, k3, k2)
		C321 = self.C_abc(k2, k3, k1)
		D123 = self.D_abc(k1, k2, k3)

		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					uABC[i, j, k] = -B231[j, k, i]/H
					uABC[i, Nfield+j, k] = -C123[i, j, k]/H/s
					uABC[i, j, Nfield+k] = -C132[i, k, j]/H/s
					uABC[Nfield+i, Nfield+j, Nfield+k] = C321[k, j, i]/H/s
					uABC[i, Nfield+j, Nfield+k] = 3.*D123[i, j, k]/H/s/s
					uABC[Nfield+i, j, k] = 3.*A123[i, j, k]/H*s
					uABC[Nfield+i, Nfield+j, k] = B132[i, k, j]/H
					uABC[Nfield+i, j, Nfield+k] = B123[i, j, k]/H
		return uABC




