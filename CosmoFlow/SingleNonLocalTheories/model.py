import numpy as np
from scipy.misc import derivative



class model():
	"""
	This class defines the model (background functions, M, Delta, I, A, B, C tensors and u^A_B, u^A_BC tensors)
	where the functions have been optimised to be computed once at each integration step.
	The minimal background interpolated functions should be imported using the background_inputs class. 
	The background functions are vectorised.
	"""

	def __init__(self, N, interpolated):
		"""
		All the functions are evaluated once at N (can be vectorised for background quantities
		or just a float for tensorial quantities)
		"""
		self.N = N
		self.Nfield = 1
		self.interpolated = interpolated #(H, cs, m, rho, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6)

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
		self.lambda3 = self.lambda3_f(N)
		self.lambda4 = self.lambda4_f(N)
		self.lambda5 = self.lambda5_f(N)
		self.lambda6 = self.lambda6_f(N)

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

	def lambda3_f(self, N):
		return self.interpolated[6](N)

	def lambda4_f(self, N):
		return self.interpolated[7](N)

	def lambda5_f(self, N):
		return self.interpolated[8](N)

	def lambda6_f(self, N):
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
		dHdN = derivative(self.H_f, N, dx = 1e-10)
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
		inverseD = 1/(k**2/self.a**2 + self.m**2)
		Deltaab = np.eye(1)
		Deltaab[0, 0] = 1/(1 + self.rho**2 * inverseD)
		return Deltaab

	def I_ab(self):
		Iab = np.zeros((1, 1))
		return Iab

	def M_ab(self, k):
		Mab = np.eye(1)
		Mab[0, 0] = -k**2/self.a**2 * self.cs**2
		return Mab

	def u_AB(self, k):
		H = self.H
		s = self.scale
		ds = self.dscale
		S = np.ones((1, 1)) + (s-1)*np.eye(1)
		uAB = np.zeros((2*1, 2*1))
		uAB[:1, :1] = -self.I_ab()/H
		uAB[:1, 1:] = self.Delta_ab(k)/H /s
		uAB[1:, :1] = self.M_ab(k)/H *s
		uAB[1:, 1:] = (self.I_ab()).T/H - 3*self.H*np.eye(1)/H + ds/s*np.eye(1)/H 
		return uAB


	############################
	############################
	#Defining the u_ABC tensor for bispectrum calculations
	############################
	############################

	def A_abc(self, k1, k2, k3):
		Aabc = np.zeros((1, 1, 1))
		return Aabc

	def A_abc_fast(self, k1, k2, k3):
		Aabc = np.zeros((1, 1, 1))
		return Aabc

	def A_abc_slow(self, k1, k2, k3):
		Aabc = np.zeros((1, 1, 1))
		return Aabc

	def B_abc(self, k1, k2, k3):
		Babc = np.zeros((1, 1, 1))
		k1k2 = (k3**2 - k1**2 - k2**2)/2
		inverseDk3 = 1/(k3**2/self.a**2 + self.m**2)

		Babc[0, 0, 0] += 2*self.lambda1*k1k2/self.a**2 * 1/(1 + self.rho**2 * inverseDk3)
		Babc[0, 0, 0] += 2*self.lambda3*k1k2/self.a**2 * inverseDk3 * 1/(1 + self.rho**2 * inverseDk3)

		return Babc

	def B_abc_fast(self, k1, k2, k3):
		Babc = np.zeros((1, 1, 1))
		k1k2 = (k3**2 - k1**2 - k2**2)/2
		inverseDk3 = 1/(k3**2/self.a**2 + self.m**2)

		Babc[0, 0, 0] += 2*self.lambda1*k1k2 * 1/(1 + self.rho**2 * inverseDk3)
		Babc[0, 0, 0] += 2*self.lambda3*k1k2 * inverseDk3 * 1/(1 + self.rho**2 * inverseDk3)

		return Babc

	def B_abc_slow(self, k1, k2, k3):
		Babc = np.zeros((1, 1, 1))
		return Babc

	def C_abc(self, k1, k2, k3):
		Cabc = np.zeros((1, 1, 1))
		return Cabc

	def D_abc(self, k1, k2, k3):
		Dabc = np.zeros((1, 1, 1))
		inverseDk1 = 1/(k1**2/self.a**2 + self.m**2)
		inverseDk2 = 1/(k2**2/self.a**2 + self.m**2)
		inverseDk3 = 1/(k3**2/self.a**2 + self.m**2)

		Dabc[0, 0, 0] += -2*self.lambda2 * 1/(1 + self.rho**2 * inverseDk1) * 1/(1 + self.rho**2 * inverseDk2) * 1/(1 + self.rho**2 * inverseDk3)

		Dabc[0, 0, 0] += -2*self.lambda4 * 1/(1 + self.rho**2 * inverseDk1) * 1/(1 + self.rho**2 * inverseDk2) * 1/(1 + self.rho**2 * inverseDk3) * inverseDk1/3
		Dabc[0, 0, 0] += -2*self.lambda4 * 1/(1 + self.rho**2 * inverseDk1) * 1/(1 + self.rho**2 * inverseDk2) * 1/(1 + self.rho**2 * inverseDk3) * inverseDk2/3
		Dabc[0, 0, 0] += -2*self.lambda4 * 1/(1 + self.rho**2 * inverseDk1) * 1/(1 + self.rho**2 * inverseDk2) * 1/(1 + self.rho**2 * inverseDk3) * inverseDk3/3

		Dabc[0, 0, 0] += -2*self.lambda5 * 1/(1 + self.rho**2 * inverseDk1) * 1/(1 + self.rho**2 * inverseDk2) * 1/(1 + self.rho**2 * inverseDk3) * inverseDk1 * inverseDk2/3
		Dabc[0, 0, 0] += -2*self.lambda5 * 1/(1 + self.rho**2 * inverseDk1) * 1/(1 + self.rho**2 * inverseDk2) * 1/(1 + self.rho**2 * inverseDk3) * inverseDk1 * inverseDk3/3
		Dabc[0, 0, 0] += -2*self.lambda5 * 1/(1 + self.rho**2 * inverseDk1) * 1/(1 + self.rho**2 * inverseDk2) * 1/(1 + self.rho**2 * inverseDk3) * inverseDk2 * inverseDk3/3

		Dabc[0, 0, 0] += -2*self.lambda6 * 1/(1 + self.rho**2 * inverseDk1) * 1/(1 + self.rho**2 * inverseDk2) * 1/(1 + self.rho**2 * inverseDk3) * inverseDk1 * inverseDk2 * inverseDk3

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