import numpy as np
from scipy.misc import derivative



class model():
	"""
	This class defines the model (background functions, M, Delta, I, A, B, C tensors and u^A_B, u^A_BC tensors)
	where the functions have been optimized to be computed once at each integration step.
	The minimal background interpolated functions should be imported using the background_inputs class. 
	The background functions are vectorized.
	"""

	def __init__(self, N, Nfield, interpolated):
		"""
		All the functions are evaluated once at N (can be vectorized for background quantities
		or just a float for tensorial quantities)
		"""
		self.N = N
		self.Nfield = Nfield
		self.Mp = 1
		self.interpolated = interpolated #(H, cs, m, rho, lambda1, lambda2, mu, alpha, kappa1, kappa2)


		#Evaluating the background quantities at time N
		N = self.N
		self.H = self.H_f(N)

		#Quadratic theory
		self.cs = self.cs_f(N)
		self.m = self.m_f(N)
		self.rho = self.rho_f(N)

		#Cubic theory
		self.lambda1 = self.lambda1_f(N)
		self.lambda2 = self.lambda2_f(N)
		self.mu      = self.mu_f(N)
		self.alpha   = self.alpha_f(N)
		self.kappa1  = self.kappa1_f(N)
		self.kappa2  = self.kappa2_f(N)

		#Background functions and scale functions
		self.a = self.a_f(N)
		self.scale = self.scale_f(N)
		self.dscale = self.dscale_f(N)


	############################
	############################
	#Defining time-dependent background functions (as functions of the number of efolds)
	############################
	############################

	############################
	#Required as extern inputs
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

	def mu_f(self, N):
		return self.interpolated[6](N)

	def alpha_f(self, N):
		return self.interpolated[7](N)

	def kappa1_f(self, N):
		return self.interpolated[8](N)

	def kappa2_f(self, N):
		return self.interpolated[9](N)


	############################
	#Deduced background functions
	def a_f(self, N):
		"""
		Scale factor as a function of the number of efolds
		"""
		return np.exp(N)

	def dH_f(self, N):
		"""
		Derivative of the Hubble rate (with respect to cosmic time)
		"""
		dHdN = derivative(self.H_f, N, dx = 1e-6)
		return self.H_f(N)*dHdN

	############################
	############################
	#Choosing the number of efolds before horizon exit so that it fixes the mode k, 
	#and defining the power spectrum normalization
	############################
	############################

	def efold_mode(self, k_exit, N_end, N_exit):
		"""
		Function that takes k_exit and computes the number of efolds
		before inflation end at which the mode exits the horizon.
		The approximation that H is constant was made.
		"""
		return N_end - np.log(k_exit/self.H_f(N_exit))


	def k_mode(self, N_exit):
		return self.a_f(N_exit) * self.H_f(N_exit)

	def normalization(self, N_exit):
		"""
		Function that returns the single field slow-roll
		power spectrum amplitude to normalize Sigma (for the dimensionless power spectrum)
		"""
		return 1

	def scale_f(self, N):
		"""
		Function that defines the rescaled a to improve performance
		"""
		k = 1
		a = self.a_f(N)
		H = self.H_f(N)
		return a/(1. + a*H/k)/H

	def dscale_f(self, N):
		"""
		Function that defines the derivative of scale function
		"""
		k = 1
		a = self.a_f(N)
		H = self.H_f(N)
		Hd = self.dH_f(N)
		return -Hd/H/H*a/(1. + a*H/k) + a/(1. + a*H/k) - a*(a*H*H/k + a*Hd/k)/(1. + a*H/k)/(1. + a*H/k)/H



	############################
	############################
	#Defining the u_AB tensor for the power spectrum calculations
	############################
	############################

	def Delta_ab(self, k):
		Nfield = self.Nfield
		Deltaab = np.eye(Nfield)

		return Deltaab

	def I_ab(self):
		Nfield = self.Nfield
		Iab = np.zeros((Nfield, Nfield))

		Iab[0, 1] = self.rho

		return Iab

	def M_ab(self, k):
		Nfield = self.Nfield
		Mab = np.eye(Nfield)

		Mab[0, 0] = -(k**2)/(self.a**2) * self.cs**2
		Mab[1, 1] = -(k**2)/(self.a**2) - self.m**2 - self.rho**2

		return Mab

	def u_AB(self, k):
		Nfield = self.Nfield
		H = self.H
		s = self.scale
		ds = self.dscale
		S = np.ones((Nfield, Nfield)) + (s-1)*np.eye(Nfield)
		uAB = np.zeros((2*Nfield, 2*Nfield))

		uAB[:Nfield, :Nfield] = -self.I_ab()/H
		uAB[:Nfield, Nfield:] = self.Delta_ab(k)/H /s
		uAB[Nfield:, :Nfield] = self.M_ab(k)/H *s
		uAB[Nfield:, Nfield:] = (self.I_ab()).T/H - 3*self.H*np.eye(Nfield)/H + ds/s*np.eye(Nfield)/H 
		return uAB


	############################
	############################
	#Defining the u_ABC tensor for bispectrum calculations
	############################
	############################

	def A_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))
		k2k3 = (k1**2 - k2**2 - k3**2)/2
		k1k2 = (k3**2 - k1**2 - k2**2)/2
		k1k3 = (k2**2 - k1**2 - k3**2)/2

		Aabc[1, 1, 1] += -2*self.mu + self.alpha*self.rho - self.kappa2*self.rho**2 + 2*self.lambda2*self.rho**3
		Aabc[0, 0, 1] += (self.kappa1 - 2*self.lambda1*self.rho) * k1k2/self.a**2 / 3
		Aabc[0, 1, 0] += (self.kappa1 - 2*self.lambda1*self.rho) * k1k3/self.a**2 / 3
		Aabc[1, 0, 0] += (self.kappa1 - 2*self.lambda1*self.rho) * k2k3/self.a**2 / 3

		return Aabc

	def A_abc_fast(self, k1, k2, k3):
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))
		k2k3 = (k1**2 - k2**2 - k3**2)/2
		k1k2 = (k3**2 - k1**2 - k2**2)/2
		k1k3 = (k2**2 - k1**2 - k3**2)/2
		
		Aabc[0, 0, 1] += (self.kappa1 - 2*self.lambda1*self.rho) * k1k2 / 3
		Aabc[0, 1, 0] += (self.kappa1 - 2*self.lambda1*self.rho) * k1k3 / 3
		Aabc[1, 0, 0] += (self.kappa1 - 2*self.lambda1*self.rho) * k2k3 / 3

		return Aabc

	def A_abc_slow(self, k1, k2, k3):
		Nfield = self.Nfield
		Aabc = np.zeros((Nfield, Nfield, Nfield))

		Aabc[1, 1, 1] += -2*self.mu + self.alpha*self.rho - self.kappa2*self.rho**2 + 2*self.lambda2*self.rho**3

		return Aabc

	def B_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Babc = np.zeros((Nfield, Nfield, Nfield))
		k1k2 = (k3**2 - k1**2 - k2**2)/2

		Babc[1, 1, 0] += -self.alpha + 2*self.kappa2*self.rho - 6*self.lambda2*self.rho**2
		Babc[0, 0, 0] += 2*self.lambda1*k1k2/self.a**2

		return Babc

	def C_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Cabc = np.zeros((Nfield, Nfield, Nfield))

		Cabc[0, 0, 1] += - self.kappa2 + 6*self.lambda2*self.rho

		return Cabc

	def D_abc(self, k1, k2, k3):
		Nfield = self.Nfield
		Dabc = np.zeros((Nfield, Nfield, Nfield))

		Dabc[0, 0, 0] += -2*self.lambda2

		return Dabc

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