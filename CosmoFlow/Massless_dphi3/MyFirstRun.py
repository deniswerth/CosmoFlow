import numpy as np # import Numpy package for vectorisation
import matplotlib.pyplot as plt # import matplotlib for visualisation

# Import CosmoFlow modules
from Parameters import parameters
from Theory import theory
from Solver import solver

# Define the numerical i\epsilon prescription
def adiabatic(N_load, DeltaN):
    return (np.tanh((N_load + DeltaN - 1)/0.1) + 1)/2

n = 10000 # Number of points for the parameter evaluation
N_load = np.linspace(-10, 20, n) # Time grid array in e-folds for the parameters
DeltaN = 4 # Number of e-folds before horizon crossing

# Theory 
g_load = 1 * np.ones(n) * adiabatic(N_load, DeltaN) # Cubic coupling constant
H_load = np.ones(n) # Hubble scale

# Load the parameters and define continuous functions
param = parameters(N_load, H_load, g_load) # Load the class parameters
interpolated = param.output() # Define list with continuous parameters

# Numerical parameters
Nspan = np.linspace(-10, 20, 500) # Time span in e-folds for the numerical integration
Nfield = 1 # Number of fields
Rtol, Atol = 1e-4, 1e-180 # Relative and absolute tolerance of the integrator
N_exit = 0 # Horizon exit for a mode
Ni, Nf = N_exit - DeltaN, 10 # Sets initial and final time for integration
N = np.linspace(Ni, Nf, 1000) # Define the time array for output correlators

# Initialise the integrator
theo = theory(N = Nspan, Nfield = Nfield, interpolated = interpolated)

# Kinematic configuration
k = theo.k_mode(N_exit) # Mode corresponding to N = 0 horizon exit
k1, k2, k3 = k, k, k # Kinematic configuration for 3-pt function (here equilateral)

# Solve flow equations
s = solver(Nspan = N, Nfield = Nfield, interpolated = interpolated, Rtol = Rtol, Atol = Atol)
f = s.f_solution(k1 = k1, k2 = k2, k3 = k3)

# Plot correlators
plt.semilogy(N, np.absolute(f[0][0, 0])) # field-field correlator
plt.semilogy(N, np.absolute(f[6][0, 0, 0])) # field-field-field correlator
plt.show()
