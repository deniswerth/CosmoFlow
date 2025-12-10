import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from model import model

class solver():
	"""
	This class contains the initial conditions and the solver for the power
	spectra and the bispectra.
	"""

	def __init__(self, Nspan, Nfield, interpolated, Rtol, Atol):

		self.Nspan = Nspan
		self.Nfield = Nfield
		self.interpolated = interpolated
		self.N_init = self.Nspan[0]
		self.Rtol = Rtol
		self.Atol = Atol
		self.Mp = 1

		#Defining class for initial conditions at N = N_init
		self.initial = model(self.N_init, self.Nfield, self.interpolated)

		#Defining matrices to take into account for the difference between zeta and F
		Nfield = self.Nfield
		x = 1 #Canonnically normalised field

		Xfff = np.ones((Nfield, Nfield, Nfield))
		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					if i == 0:
						Xfff[i, j, k] *= x
					if j == 0:
						Xfff[i, j, k] *= x
					if k == 0:
						Xfff[i, j, k] *= x
		self.Xfff = Xfff

		Xffp = np.ones((Nfield, Nfield, Nfield))
		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					if i == 0:
						Xffp[i, j, k] *= x
					if j == 0:
						Xffp[i, j, k] *= x
					if k == 0:
						Xffp[i, j, k] *= 1/x
		self.Xffp = Xffp

		Xppf = np.ones((Nfield, Nfield, Nfield))
		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					if i == 0:
						Xppf[i, j, k] *= 1/x
					if j == 0:
						Xppf[i, j, k] *= 1/x
					if k == 0:
						Xppf[i, j, k] *= x
		self.Xppf = Xppf

		Xppp = np.ones((Nfield, Nfield, Nfield))
		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					if i == 0:
						Xppp[i, j, k] *= 1/x
					if j == 0:
						Xppp[i, j, k] *= 1/x
					if k == 0:
						Xppp[i, j, k] *= 1/x
		self.Xppp = Xppp


		Xpff = np.ones((Nfield, Nfield, Nfield))
		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					if i == 0:
						Xpff[i, j, k] *= 1/x
					if j == 0:
						Xpff[i, j, k] *= x
					if k == 0:
						Xpff[i, j, k] *= x
		self.Xpff = Xpff


	############################
	############################
	#Numerical solver for the power spectrum
	############################
	############################

	def Sigma_AB_Re_init(self, k):
		"""
		Function that defines the real part of the initial Sigma matrix
		"""
		Nfield = self.Nfield
		s = self.initial.scale
		S = np.ones((Nfield, Nfield)) + (s-1)*np.eye(Nfield)
		SigmaAB_Re_init = np.zeros((2*Nfield, 2*Nfield))
		SigmaAB_Re_init[:Nfield, :Nfield] = self.initial.Delta_ab(k)
		SigmaAB_Re_init[Nfield:, Nfield:] = k**2/self.initial.a**2 * np.diag(1/np.diagonal(self.initial.Delta_ab(k))) *S*S
		SigmaAB_Re_init[Nfield:, :Nfield] = - self.initial.H * np.identity(Nfield) *S
		SigmaAB_Re_init[:Nfield, Nfield:] = - self.initial.H * np.identity(Nfield) *S
		return 1/(2*k*self.initial.a**2) * SigmaAB_Re_init

	def Sigma_AB_Im_init(self, k):
		"""
		Function that defines the imaginary part of the initial Sigma matrix
		"""
		Nfield = self.Nfield
		S = np.ones((Nfield, Nfield)) + (s-1)*np.eye(Nfield)
		SigmaAB_Im_init = np.zeros((2*Nfield, 2*Nfield))
		SigmaAB_Im_init[Nfield:, :Nfield] = - np.identity(Nfield) *S
		SigmaAB_Im_init[:Nfield, Nfield:] = np.identity(Nfield) *S
		return 1/(2 * self.initial.a**3) * SigmaAB_Im_init

	def dSigma_dN(self, y, N, k, part):
		"""
		Function that defines the differential equation for the power spectrum in a flatten way
		y is a flat array of dimension 2Nfield*2Nfield
		"""
		Nfield = self.Nfield
		#X[X == 1] = 0
		Sig_AB = np.reshape(y, (2*Nfield, 2*Nfield))

		mdl = model(N = N, Nfield = self.Nfield, interpolated = self.interpolated)

		uAB = mdl.u_AB(k)
		dSig1 = np.dot(uAB, Sig_AB)
		dSig2 = np.transpose(np.dot(uAB, np.transpose(Sig_AB)))
		dSig_dN = dSig1 + dSig2

		#Impose the imaginary part to be zero because of the commutation relation (except field and its conjugate momentum)
		if part == "Im":
			for i in range(2*Nfield):
				for j in range(2*Nfield):
					if i != Nfield + j and j != Nfield + i:
						dSig_dN[i, j] = 0
		
		dSig_dN = np.reshape(dSig_dN, ((2*Nfield)**2))
		return dSig_dN

	def SigmaAB_solution(self, k, part):
		"""
		Function that returns the numerical power spectrum evaluated at k
		part = "Re" or "Im" to consider both initial conditions
		This function is mostly used for the bispectrum calculation
		"""
		Rtol, Atol = self.Rtol, self.Atol
		Nfield = self.Nfield
		Nspan = self.Nspan
		if part == "Re":
			Sigma0 = self.Sigma_AB_Re_init(k)
			S0 = np.reshape(Sigma0, 2*Nfield*2*Nfield)
			rtol, atol = Rtol[0], Atol[0]
		elif part == "Im":
			Sigma0 = self.Sigma_AB_Im_init(k)
			S0 = np.reshape(Sigma0, 2*Nfield*2*Nfield)
			rtol, atol = Rtol[1], Atol[1]
		N_boundary = [Nspan[0], Nspan[-1]]
		sol = solve_ivp(lambda N, y: self.dSigma_dN(y, N, k, part), N_boundary, S0, t_eval = Nspan, dense_output = False, method = "RK45", rtol = rtol, atol = atol)
		SigmaAB = sol.y
		SigmaAB = np.reshape(SigmaAB, (2*Nfield, 2*Nfield, len(Nspan)))

		return SigmaAB


	############################
	############################
	#Numerical solver for the spectra and the bispectrum simultaneously
	############################
	############################

	def Sigma_Re_init(self, k):
		"""
		Function that defines the real part of the initial Sigma matrix
		"""
		Nfield = self.Nfield
		s = self.initial.scale
		S = np.ones((Nfield, Nfield)) + (s-1)*np.eye(Nfield)
		SigmaAB_Re_init = np.zeros((2*Nfield, 2*Nfield))
		SigmaAB_Re_init[:Nfield, :Nfield] = self.initial.Delta_ab(k) * (1 + self.initial.H**2 * self.initial.a**2/k**2)
		SigmaAB_Re_init[Nfield:, Nfield:] = k**2/self.initial.a**2 * np.diag(1/np.diagonal(self.initial.Delta_ab(k))) *S*S
		SigmaAB_Re_init[Nfield:, :Nfield] = - self.initial.H * np.identity(Nfield) *S
		SigmaAB_Re_init[:Nfield, Nfield:] = - self.initial.H * np.identity(Nfield) *S

		#Speed of sound
		mdl = model(N = self.Nspan, Nfield = self.Nfield, interpolated = self.interpolated)
		cs = mdl.cs_f(self.Nspan[0])
		SigmaAB_Re_init[0, 0] = (1 + self.initial.H**2 * self.initial.a**2/k**2/cs**2)/cs
		SigmaAB_Re_init[Nfield, Nfield] *= cs
		SigmaAB_Re_init[0, Nfield] *= 1/cs
		SigmaAB_Re_init[Nfield, 0] *= 1/cs

		return 1/(2*k*self.initial.a**2) * SigmaAB_Re_init

	def Sigma_Im_init(self, k):
		"""
		Function that defines the imaginary part of the initial Sigma matrix
		"""
		Nfield = self.Nfield
		s = self.initial.scale
		S = np.ones((Nfield, Nfield)) + (s-1)*np.eye(Nfield)
		SigmaAB_Im_init = np.zeros((2*Nfield, 2*Nfield))
		SigmaAB_Im_init[Nfield:, :Nfield] = - np.identity(Nfield) *S
		SigmaAB_Im_init[:Nfield, Nfield:] = np.identity(Nfield) *S

		return 1/(2 * self.initial.a**3) * SigmaAB_Im_init

	def fffCalc(self, k1, k2, k3):
		"""
		Function that defines the field-field-field initial conditions
		"""
		Nfield = self.Nfield
		a = self.initial.a
		Hi = self.initial.H

		kt = k1 + k2 + k3
		K2 = k1*k2 + k1*k3 + k2*k3

		Af123 = self.Xfff*self.initial.A_abc_fast(k1, k2, k3)
		Af132 = self.Xfff*self.initial.A_abc_fast(k1, k3, k2)
		Af231 = self.Xfff*self.initial.A_abc_fast(k2, k3, k1)

		As123 = self.Xfff*self.initial.A_abc_slow(k1, k2, k3)
		As132 = self.Xfff*self.initial.A_abc_slow(k1, k3, k2)
		As231 = self.Xfff*self.initial.A_abc_slow(k2, k3, k1)

		B123 = self.Xffp*self.initial.B_abc(k1, k2, k3)
		B132 = self.Xffp*self.initial.B_abc(k1, k3, k2)
		B231 = self.Xffp*self.initial.B_abc(k2, k3, k1)

		C123 = self.Xppf*self.initial.C_abc(k1, k2, k3)
		C132 = self.Xppf*self.initial.C_abc(k1, k3, k2)
		C231 = self.Xppf*self.initial.C_abc(k2, k3, k1)

		D123 = self.Xppp*self.initial.D_abc(k1, k2, k3)
		D132 = self.Xppp*self.initial.D_abc(k1, k3, k2)
		D231 = self.Xppp*self.initial.D_abc(k2, k3, k1)


		fff = np.zeros((Nfield, Nfield, Nfield))

		
		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					fff[i, j, k] += 1/(a*a*a*a)/4/(k1*k2*k3)/kt*(-C123[i, j, k]*k1*k2 - C132[i, k, j]*k1*k3 - C231[j, k, i]*k2*k3
													+ a*a*As123[i, j, k] + a*a*As132[i, k, j] + a*a*As231[j, k, i]
													+ a*a*Hi*B123[i, j, k]*((k1+k2)*k3/k1/k2 - K2/k1/k2)
													+ a*a*Hi*B132[i, k, j]*((k1+k3)*k2/k1/k3 - K2/k1/k3)
													+ a*a*Hi*B231[j, k, i]*((k2+k3)*k1/k2/k3 - K2/k2/k3)
													+ D123[i, j, k]*Hi*(K2 - 2*k1*k2*k3/kt)
													+ D132[i, k, j]*Hi*(K2 - 2*k1*k2*k3/kt)
													+ D231[j, k, i]*Hi*(K2 - 2*k1*k2*k3/kt)
													)
					fff[i, j, k] += 1./(a*a*a*a)/4./(k1*k2*k3)/kt * (Af123[i, j, k] + Af132[i, k, j] + Af231[j, k, i])

		return self.Xfff*fff

	def pffCalc(self, k1, k2, k3):
		"""
		Function that defines the momentum-field-field initial conditions
		"""
		Nfield = self.Nfield
		a = self.initial.a
		Hi = self.initial.H

		kt = k1 + k2 + k3
		K2 = k1*k2 + k1*k3 + k2*k3
		K3 = k1*k1*k1 * k2*k2*k2 * k3*k3*k3

		Af123 = self.Xfff*self.initial.A_abc_fast(k1, k2, k3)
		Af132 = self.Xfff*self.initial.A_abc_fast(k1, k3, k2)
		Af231 = self.Xfff*self.initial.A_abc_fast(k2, k3, k1)

		As123 = self.Xfff*self.initial.A_abc_slow(k1, k2, k3)
		As132 = self.Xfff*self.initial.A_abc_slow(k1, k3, k2)
		As231 = self.Xfff*self.initial.A_abc_slow(k2, k3, k1)

		B123 = self.Xffp*self.initial.B_abc(k1, k2, k3)
		B132 = self.Xffp*self.initial.B_abc(k1, k3, k2)
		B231 = self.Xffp*self.initial.B_abc(k2, k3, k1)

		C123 = self.Xppf*self.initial.C_abc(k1, k2, k3)
		C132 = self.Xppf*self.initial.C_abc(k1, k3, k2)
		C231 = self.Xppf*self.initial.C_abc(k2, k3, k1)

		D123 = self.Xppp*self.initial.D_abc(k1, k2, k3)
		D132 = self.Xppp*self.initial.D_abc(k1, k3, k2)
		D231 = self.Xppp*self.initial.D_abc(k2, k3, k1)


		pff = np.zeros((Nfield, Nfield, Nfield))
		
		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					pff[i, j, k] += - 1./(a*a*a)/4./K3 * Hi * (-k1*k1*(k2+k3)/kt* k1*k2*k3) * (-C123[i, j, k]*k1*k2 - C132[i, k, j]*k1*k3 - C231[j, k, i]*k2*k3
		 																			+ a*a*As123[i, j, k] + a*a*As132[i, k, j] + a*a*As231[j, k, i]
		 																			-2*D123[i, j, k]*Hi*k1*k2*k3/kt - 2*D132[i, k, j]*Hi*k1*k2*k3/kt - 2*D231[j, k, i]*Hi*k1*k2*k3/kt
		 																			)
					pff[i, j, k] += - 1./(a*a*a)/4./K3 * Hi * (-k1*k1*(k2+k3)/kt* k1*k2*k3) * (Af123[i, j, k] + Af132[i, k, j] + Af231[j, k, i])
					pff[i, j, k] += - 1./(a*a*a)/4./K3 * Hi * (-k1*k1*k2*k3/kt) * (C123[i, j, k]*k1*k1*k2*k2*(1.+k3/kt)
																	+ C132[i, k, j]*k1*k1*k3*k3*(1.+k2/kt)
																	+ C231[j, k, i]*k3*k3*k2*k2*(1.+k1/kt)
																	- a*a*As123[i, j, k]*(K2 - k1*k2*k3/kt)
																	- a*a*As132[i, k, j]*(K2 - k1*k2*k3/kt)
																	- a*a*As231[j, k, i]*(K2 - k1*k2*k3/kt)
																	- D123[i, j, k]*(k1*k2*k3)**2/Hi/a/a
																	- D132[i, k, j]*(k1*k2*k3)**2/Hi/a/a
																	- D231[j, k, i]*(k1*k2*k3)**2/Hi/a/a
																	)
					pff[i, j, k] += - 1./(a*a*a)/4./K3 * Hi * (-k1*k1*k2*k3/kt) * (B123[i, j, k]/Hi*k1*k2*k3*k3
																	+ B132[i, k, j]/Hi*k1*k3*k2*k2
																	+ B231[j, k, i]/Hi*k2*k3*k1*k1
																	)
					pff[i, j, k] += - 1./(a*a*a)/4./K3 * Hi * (-k1*k1*k2*k3/kt) * (-Af123[i, j, k]*(K2 +k1*k2*k3/kt) - Af132[i, k, j]*(K2 +k1*k2*k3/kt) - Af231[j, k, i]*(K2 +k1*k2*k3/kt))

		return self.Xpff*pff


	def ppfCalc(self, k1, k2, k3):
		"""
		Function that defines the momentum-momentum-field initial conditions 
		"""
		Nfield = self.Nfield
		a = self.initial.a
		Hi = self.initial.H

		kt = k1 + k2 + k3
		K3 = k1*k1*k1 * k2*k2*k2 * k3*k3*k3

		Af123 = self.Xfff*self.initial.A_abc_fast(k1, k2, k3)
		Af132 = self.Xfff*self.initial.A_abc_fast(k1, k3, k2)
		Af231 = self.Xfff*self.initial.A_abc_fast(k2, k3, k1)

		As123 = self.Xfff*self.initial.A_abc_slow(k1, k2, k3)
		As132 = self.Xfff*self.initial.A_abc_slow(k1, k3, k2)
		As231 = self.Xfff*self.initial.A_abc_slow(k2, k3, k1)

		B123 = self.Xffp*self.initial.B_abc(k1, k2, k3)
		B132 = self.Xffp*self.initial.B_abc(k1, k3, k2)
		B231 = self.Xffp*self.initial.B_abc(k2, k3, k1)

		C123 = self.Xppf*self.initial.C_abc(k1, k2, k3)
		C132 = self.Xppf*self.initial.C_abc(k1, k3, k2)
		C231 = self.Xppf*self.initial.C_abc(k2, k3, k1)

		D123 = self.Xppp*self.initial.D_abc(k1, k2, k3)
		D132 = self.Xppp*self.initial.D_abc(k1, k3, k2)
		D231 = self.Xppp*self.initial.D_abc(k2, k3, k1)


		ppf = np.zeros((Nfield, Nfield, Nfield))
		
		
		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					ppf[i, j, k] += -1./(a*a*a*a)/4./K3 * (k1*k2*k3)*(k1*k2*k3)/kt*k1*k2*(-C123[i, j, k]*k1*k2 - C132[i, k, j]*k1*k3 - C231[j, k, i]*k2*k3
																	+ a*a*As123[i, j, k] + a*a*As132[i, k, j] + a*a*As231[j, k, i]
																	+ a*a*Hi*B123[i, j, k]*(k1+k2)*k3/k1/k2
																	+ a*a*Hi*B132[i, k, j]*(k1+k3)*k2/k1/k3
																	+ a*a*Hi*B231[j, k, i]*(k2+k3)*k1/k2/k3
																	+ a*a*Hi*B123[i, j, k]*k3*k3/k2/k2
																	+ a*a*Hi*B132[i, k, j]*k2*k2/k1/k1
																	+ a*a*Hi*B231[j, k, i]*k1*k1/k3/k3
																	- 2*D123[i, j, k]*Hi*k1*k2*k3/kt
																	- 2*D132[i, k, j]*Hi*k1*k2*k3/kt
																	- 2*D231[j, k, i]*Hi*k1*k2*k3/kt
																	)
					ppf[i, j, k] += - 1./(a*a*a*a)/4/K3 * (k1*k2*k3)*(k1*k2*k3)/kt*k1*k2 * (Af123[i, j, k] + Af132[i, k, j] + Af231[j, k, i])
		
		return self.Xppf*ppf

	def pppCalc(self, k1, k2, k3):
		"""
		Function that defines the momentum-momentum-momentum initial conditions 
		"""
		Nfield = self.Nfield
		a = self.initial.a
		Hi = self.initial.H

		kt = k1 + k2 + k3
		K2 = k1*k2 + k1*k3 + k2*k3
		K3 = k1*k1*k1 * k2*k2*k2 * k3*k3*k3

		Af123 = self.Xfff*self.initial.A_abc_fast(k1, k2, k3)
		Af132 = self.Xfff*self.initial.A_abc_fast(k1, k3, k2)
		Af231 = self.Xfff*self.initial.A_abc_fast(k2, k3, k1)

		As123 = self.Xfff*self.initial.A_abc_slow(k1, k2, k3)
		As132 = self.Xfff*self.initial.A_abc_slow(k1, k3, k2)
		As231 = self.Xfff*self.initial.A_abc_slow(k2, k3, k1)

		B123 = self.Xffp*self.initial.B_abc(k1, k2, k3)
		B132 = self.Xffp*self.initial.B_abc(k1, k3, k2)
		B231 = self.Xffp*self.initial.B_abc(k2, k3, k1)

		C123 = self.Xppf*self.initial.C_abc(k1, k2, k3)
		C132 = self.Xppf*self.initial.C_abc(k1, k3, k2)
		C231 = self.Xppf*self.initial.C_abc(k2, k3, k1)

		D123 = self.Xppp*self.initial.D_abc(k1, k2, k3)
		D132 = self.Xppp*self.initial.D_abc(k1, k3, k2)
		D231 = self.Xppp*self.initial.D_abc(k2, k3, k1)

		
		ppp = np.zeros((Nfield, Nfield, Nfield))
		

		
		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					ppp[i, j, k] += - 1./(a*a*a)/4./K3 * Hi * (k1*k1*k2*k2*k3*k3)/kt  * (C123[i, j, k]*k1*k1*k2*k2*(1.+k3/kt)
																+ C132[i, k, j]*k1*k1*k3*k3*(1.+k2/kt)
																+ C231[j, k, i]*k3*k3*k2*k2*(1.+k1/kt)
																- a*a*As123[i, j, k]*(K2 - k1*k2*k3/kt)
																- a*a*As132[i, k, j]*(K2 - k1*k2*k3/kt)
																- a*a*As231[j, k, i]*(K2 - k1*k2*k3/kt)
																- D123[i, j, k]*(k1*k2*k3)**2/Hi/a/a
																- D132[i, k, j]*(k1*k2*k3)**2/Hi/a/a
																- D231[j, k, i]*(k1*k2*k3)**2/Hi/a/a
																)
					ppp[i, j, k] += - 1./(a*a*a)/4./K3 * Hi * (k1*k1*k2*k2*k3*k3)/kt * (B123[i, j, k]/Hi*k1*k2*k3*k3
																+ B132[i, k, j]/Hi*k1*k3*k2*k2
																+ B231[j, k, i]/Hi*k2*k3*k1*k1
																)
					ppp[i, j, k] += - 1./(a*a*a)/4./K3 * Hi * (k1*k1*k2*k2*k3*k3)/kt * (-Af123[i, j, k]*(K2 +k1*k2*k3/kt) - Af132[i, k, j]*(K2 +k1*k2*k3/kt) - Af231[j, k, i]*(K2 +k1*k2*k3/kt))

		return self.Xppp*ppp

	def B_init(self, k1, k2, k3):
		"""
		Function that defines the real part of the initial B matrix
		"""
		Nfield = self.Nfield
		a = self.initial.a
		H = self.initial.H
		s = self.initial.scale
		S = np.ones((Nfield, Nfield, Nfield)) + (s-1)*np.eye(Nfield)

		B = np.zeros((2*Nfield, 2*Nfield, 2*Nfield))


		fff = self.fffCalc(k1, k2, k3)

		pff = self.pffCalc(k1, k2, k3)
		fpf = self.pffCalc(k2, k1, k3)
		ffp = self.pffCalc(k3, k1, k2)
      
		ppf = self.ppfCalc(k1, k2, k3)
		pfp = self.ppfCalc(k1, k3, k2)
		fpp = self.ppfCalc(k2, k3, k1)
        
		ppp = self.pppCalc(k1, k2, k3)


		for i in range(Nfield):
			for j in range(Nfield):
				for k in range(Nfield):
					B[i, j, k] = fff[i, j, k]
					B[Nfield+i, j, k] = pff[i, j, k]/a   *s
					B[i, Nfield+j, k] = fpf[j, i, k]/a   *s
					B[i, j, Nfield+k] = ffp[k, i, j]/a   *s
					B[i, Nfield+j, Nfield+k] = fpp[j, k, i]/a/a  *s*s
					B[Nfield+i, Nfield+j, k] = ppf[i, j, k]/a/a  *s*s
					B[Nfield+i, j,  Nfield+k] = pfp[i, k, j]/a/a   *s*s
					B[Nfield+i, Nfield+j,  Nfield+k] = ppp[i, j, k]/a/a/a  *s*s*s

		return B


	def f_init(self, k1, k2, k3):
		"""
		Function that defines the initial conditions for spectra/bispectra
		"""
		Nfield = self.Nfield
		return np.concatenate((np.reshape(self.Sigma_Re_init(k1), 2*Nfield*2*Nfield), np.reshape(self.Sigma_Re_init(k2), 2*Nfield*2*Nfield), np.reshape(self.Sigma_Re_init(k3), 2*Nfield*2*Nfield), 
						np.reshape(self.Sigma_Im_init(k1), 2*Nfield*2*Nfield), np.reshape(self.Sigma_Im_init(k2), 2*Nfield*2*Nfield), np.reshape(self.Sigma_Im_init(k3), 2*Nfield*2*Nfield),
						np.reshape(self.B_init(k1, k2, k3), 2*Nfield*2*Nfield*2*Nfield)))


	def df_dN(self, y, N, k1, k2, k3):
		"""
		Function that defines the differential equation for the spectrum/bispectrum in a flatten way
		y is a flat array of dimension (2*Nfield*2*Nfield * 2*Nfield*2*Nfield * 2*Nfield*2*Nfield  * 2*Nfield*2*Nfield * 2*Nfield*2*Nfield * 2*Nfield*2*Nfield  *  2Nfield*2Nfield*2Nfield )
		for (Sigma_Re_k1, Sigma_Re_k2, Sigma_Re_k3, Sigma_Im_k1, Sigma_Im_k2, Sigma_Im_k2, B_Re)
		"""
		#print(N)
		Nfield = self.Nfield

		n = 2*Nfield*2*Nfield
		Sigma_Re_k1 = np.reshape(y[:n], (2*Nfield, 2*Nfield))
		Sigma_Re_k2 = np.reshape(y[n:n*2], (2*Nfield, 2*Nfield))
		Sigma_Re_k3 = np.reshape(y[n*2:n*3], (2*Nfield, 2*Nfield))
		Sigma_Im_k1 = np.reshape(y[n*3:n*4], (2*Nfield, 2*Nfield))
		Sigma_Im_k2 = np.reshape(y[n*4:n*5], (2*Nfield, 2*Nfield))
		Sigma_Im_k3 = np.reshape(y[n*5:n*6], (2*Nfield, 2*Nfield))
		B = np.reshape(y[n*6:], (2*Nfield, 2*Nfield, 2*Nfield))
	
		mdl = model(N = N, Nfield = self.Nfield, interpolated = self.interpolated)
		uAB_k1 = mdl.u_AB(k1)
		uAB_k2 = mdl.u_AB(k2)
		uAB_k3 = mdl.u_AB(k3)
		uABC_k1k2k3 = mdl.u_ABC(k1, k2, k3)
		uABC_k2k1k3 = mdl.u_ABC(k2, k1, k3)
		uABC_k3k1k2 = mdl.u_ABC(k3, k1, k2)


		#Spectrum differential equation
		#k1 (Re)
		dSig1 = np.dot(uAB_k1, Sigma_Re_k1)
		dSig2 = np.einsum("bc, ac -> ab", uAB_k1, Sigma_Re_k1)
		dSig_dN_Re_k1 = dSig1 + dSig2
		#k2 (Re)
		dSig1 = np.dot(uAB_k2, Sigma_Re_k2)
		dSig2 = np.einsum("bc, ac -> ab", uAB_k2, Sigma_Re_k2)
		dSig_dN_Re_k2 = dSig1 + dSig2
		#k3 (Re)
		dSig1 = np.dot(uAB_k3, Sigma_Re_k3)
		dSig2 = np.einsum("bc, ac -> ab", uAB_k3, Sigma_Re_k3)
		dSig_dN_Re_k3 = dSig1 + dSig2

		#k1 (Im)
		dSig1 = np.dot(uAB_k1, Sigma_Im_k1)
		dSig2 = np.einsum("bc, ac -> ab", uAB_k1, Sigma_Im_k1)
		dSig_dN_Im_k1 = dSig1 + dSig2
		#k2 (Im)
		dSig1 = np.dot(uAB_k2, Sigma_Im_k2)
		dSig2 = np.einsum("bc, ac -> ab", uAB_k2, Sigma_Im_k2)
		dSig_dN_Im_k2 = dSig1 + dSig2
		#k3 (Im)
		dSig1 = np.dot(uAB_k3, Sigma_Im_k3)
		dSig2 = np.einsum("bc, ac -> ab", uAB_k3, Sigma_Im_k3)
		dSig_dN_Im_k3 = dSig1 + dSig2
		#Impose the imaginary part to be zero because of the commutation relation (except field and its conjugate momentum)
		for i in range(2*Nfield):
				for j in range(2*Nfield):
					if i != Nfield + j and j != Nfield + i:
						dSig_dN_Im_k1[i, j] = 0
						dSig_dN_Im_k2[i, j] = 0
						dSig_dN_Im_k3[i, j] = 0
		
		dSig_dN_Re_k1 = np.reshape(dSig_dN_Re_k1, (2*Nfield)**2)
		dSig_dN_Re_k2 = np.reshape(dSig_dN_Re_k2, (2*Nfield)**2)
		dSig_dN_Re_k3 = np.reshape(dSig_dN_Re_k3, (2*Nfield)**2)
		dSig_dN_Im_k1 = np.reshape(dSig_dN_Im_k1, (2*Nfield)**2)
		dSig_dN_Im_k2 = np.reshape(dSig_dN_Im_k2, (2*Nfield)**2)
		dSig_dN_Im_k3 = np.reshape(dSig_dN_Im_k3, (2*Nfield)**2)

		
		#Bispectrum differential equation
		dB1 = np.einsum("ad, dbc -> abc", uAB_k1, B)
		dB11 = np.einsum("ade, db, ec -> abc", uABC_k1k2k3, Sigma_Re_k2, Sigma_Re_k3)
		dB22 = np.einsum("ade, db, ec -> abc", uABC_k1k2k3, Sigma_Im_k2, Sigma_Im_k3)

		dB2 = np.einsum("bd, adc -> abc", uAB_k2, B)
		dB33 = np.einsum("bde, ad, ec -> abc", uABC_k2k1k3, Sigma_Re_k1, Sigma_Re_k3)
		dB44 = np.einsum("bde, ad, ec -> abc", uABC_k2k1k3, Sigma_Im_k1, Sigma_Im_k3)

		dB3 = np.einsum("cd, abd -> abc", uAB_k3, B)
		dB55 = np.einsum("cde, ad, be -> abc", uABC_k3k1k2, Sigma_Re_k1, Sigma_Re_k2)
		dB66 = np.einsum("cde, ad, be -> abc", uABC_k3k1k2, Sigma_Im_k1, Sigma_Im_k2)

		dB_dN = np.reshape(dB1 + dB2 + dB3 + dB11 - dB22 + dB33 - dB44 + dB55 - dB66, (2*Nfield)**3)

		#Concatenate all differential equations into a flatten array (right format for the solver)
		df_dN = np.concatenate((dSig_dN_Re_k1, dSig_dN_Re_k2, dSig_dN_Re_k3, dSig_dN_Im_k1, dSig_dN_Im_k2, dSig_dN_Im_k3, dB_dN))
		
		return df_dN

	def f_solution(self, k1, k2, k3):
		"""
		Solver function that solves for spectra/bispectra given k1, k2, k3 and a tensor of initial conditions
		"""
		rtol, atol = self.Rtol[2], self.Atol[2]
		Nfield = self.Nfield
		Nspan = self.Nspan

		f0 = self.f_init(k1, k2, k3)
		dimension = 2*Nfield*2*Nfield * 2*Nfield*2*Nfield * 2*Nfield*2*Nfield  * 2*Nfield*2*Nfield * 2*Nfield*2*Nfield * 2*Nfield*2*Nfield  *  2*Nfield*2*Nfield*2*Nfield
		N_boundary = [Nspan[0], Nspan[-1]]

		sol = solve_ivp(lambda N, y: self.df_dN(y, N, k1, k2, k3), N_boundary, f0, t_eval = Nspan, dense_output = True, method = "RK45", rtol = rtol, atol = atol)
		f = sol.y

		n = 2*Nfield*2*Nfield
		Nl = len(Nspan)
		Sigma_Re_k1 = np.reshape(f[:n], (2*Nfield, 2*Nfield, Nl))
		Sigma_Re_k2 = np.reshape(f[n:n*2], (2*Nfield, 2*Nfield, Nl))
		Sigma_Re_k3 = np.reshape(f[n*2:n*3], (2*Nfield, 2*Nfield, Nl))
		Sigma_Im_k1 = np.reshape(f[n*3:n*4], (2*Nfield, 2*Nfield, Nl))
		Sigma_Im_k2 = np.reshape(f[n*4:n*5], (2*Nfield, 2*Nfield, Nl))
		Sigma_Im_k3 = np.reshape(f[n*5:n*6], (2*Nfield, 2*Nfield, Nl))
		B           = np.reshape(f[n*6:], (2*Nfield, 2*Nfield, 2*Nfield, Nl))

		f = [Sigma_Re_k1, Sigma_Re_k2, Sigma_Re_k3, Sigma_Im_k1,Sigma_Im_k2, Sigma_Im_k3, B]

		return f


