import numpy as np

from background_inputs import background_inputs
from model import model
from solver import solver

#Function that turns on interactions adiabatically (numerical i\epsilon prescription) 
def adiabatic(N_load, DeltaN):
    return (np.tanh((N_load + DeltaN - 1)/0.1) + 1)/2


def squeezed(kappa):

	n_back = 50000 #Number of points for the background
	DeltaN = 5 #Number of efolds before horizon crossing

	#-----------
	N_load = np.linspace(-10, 20, n_back)
	H_load = np.ones(n_back) #Hubble scale set to unity

	#Quadratic theory
	cs_load   = 1*np.ones(n_back)
	m_load    = 1*np.ones(n_back)
	rho_load  = 10*np.ones(n_back) * adiabatic(N_load, DeltaN)

	#Cubic theory
	lambda1_load = 1*np.ones(n_back) * adiabatic(N_load, DeltaN)
	lambda2_load = 0*np.ones(n_back) * adiabatic(N_load, DeltaN)
	lambda3_load = 0*np.ones(n_back) * adiabatic(N_load, DeltaN)
	lambda4_load = 0*np.ones(n_back) * adiabatic(N_load, DeltaN)
	lambda5_load = 0*np.ones(n_back) * adiabatic(N_load, DeltaN)
	lambda6_load = 0*np.ones(n_back) * adiabatic(N_load, DeltaN)

	background = background_inputs(N_load, H_load, cs_load, m_load, rho_load, lambda1_load, lambda2_load, lambda3_load, lambda4_load, lambda5_load, lambda6_load)
	interpolated = background.output()

	#-----------

	n_pt = len(kappa)
	Shape = [] #Stores the shape function

	for i in range(n_pt):
		Nspan = np.linspace(-10, 20, 500)
		Rtol, Atol = [1e-5, 1e-5, 1e-5], [1e-180, 1e-180, 1e-180]

		mdl = model(N = Nspan, interpolated = interpolated)

		N_exit = 0
		kt = mdl.k_mode(N_exit)
		k1, k2, k3 = kt/kappa[i], kt/kappa[i], kt
		print("-----", "kappa = ", kappa[i], "-----")

		Ni, Nf = N_exit - DeltaN, 20 # sets initial and final efolds for transport equation integration
		N = np.linspace(Ni, Nf, 1000)
		s = solver(Nspan = N, interpolated = interpolated, Rtol = Rtol, Atol = Atol)

		f = s.f_solution(k1 = k1, k2 = k2, k3 = k3)

		S = 1/(2*np.pi)**4/((f[0][0, 0][-1]*k1**3/2/np.pi**2 + f[1][0, 0][-1]*k2**3/2/np.pi**2 + f[2][0, 0][-1]*k3**3/2/np.pi**2)/3)**2 * (k1*k2*k3)**2 * f[6][0, 0, 0]

		Shape.append(S[-1])
	return Shape



