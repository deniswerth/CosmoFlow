import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#plt.rc('text', usetex = True)
#plt.rc('font', family = 'serif')



from background_inputs import background_inputs
from model import model
from solver import solver


"""
This .py is written to compute the bispectrum
in all triangle configurations in order to 
generate the full shape at a certain scale

Set at least DeltaN = 7 and n_point = 100
to have something decent
"""


n_point = 3  # number of points on the shape grid of size n_point * n_point

Shape = np.zeros((n_point, n_point))  # shape function to compute
x = np.linspace(0, 1, n_point)  # k1/k3
y = np.linspace(0.5, 1, n_point)  # k2/k3


n_back = 10000  # mumber of points for the background
N_load = np.linspace(-20, 20, n_back)  # 
H_load = np.ones(n_back)  # constant Hubble scale set to unity


# Quadratic theory  
cs_load   = 1*np.ones(n_back)  # speed of sound of pi
mu        = 5  # mu parameter controling the mass of the heavy field
m_load    = np.sqrt(mu**2 + 9/4)*np.ones(n_back)  # mass of the heavy field
rho_load  = 0.1*np.ones(n_back) * (np.tanh((N_load + 5)/0.1) + 1)/2  # quadratic mixing


# Cubic theory
lambda1_load = 0*np.ones(n_back)  # self-interaction \dot{\pi} (\partial_i \pi)^2
lambda2_load = 0*np.ones(n_back)  # self-interaction \dot{\pi}^3
mu_load      = 0*np.ones(n_back)  # self-interaction \sigma^3
alpha_load   = 0*np.ones(n_back)  # \dot{\pi} \sigma^2
kappa1_load  = 1*np.ones(n_back) * (np.tanh((N_load + 5)/0.1) + 1)/2  # (\partial_i \pi)^2 \sigma 
kappa2_load  = 0*np.ones(n_back)  # \dot{\pi}^2 \sigma


background = background_inputs(N_load, H_load, cs_load, m_load, rho_load, lambda1_load, lambda2_load, mu_load, alpha_load, kappa1_load, kappa2_load)
interpolated = background.output()


print("\n")
print("----------------")
print("cs = ", cs_load[0])
print("mu = ", mu)
print("rho/H = ", rho_load[-1])
print("----------------\n")



for i in range(n_point):
	for j in range(n_point):
		if y[j] >= x[i] and y[j] >= -x[i] + 1:

			print("i = ", i, ", j = ", j, "/", n_point)

			DeltaN = 1  # number of efolds before horizon crossing
			N_exit = 0  # fixing a reference for the efolds
			Nspan = np.linspace(-20, 20, 10000)
			Nfield = 2
			Rtol, Atol = [1e-4, 1e-4, 1e-4], [1e-180, 1e-180, 1e-180]
			mdl = model(N = Nspan, Nfield = Nfield, interpolated = interpolated)


			# triangle configuration
			kt = mdl.k_mode(N_exit)
			k3 = kt
			k1, k2 = x[i]*k3, y[j]*k3

			# avoid k_i == 0 because it is problematic for the code
			if k1 == 0:
				k1 = 0.01
			if k2 == 0:
				k2 = 0.01

			Ni, Nf = N_exit - DeltaN, 20 # sets initial and final efolds for transport equation integration
			N = np.linspace(Ni, Nf, 10000)

			s = solver(Nspan = N, Nfield = Nfield, interpolated = interpolated, Rtol = Rtol, Atol = Atol)
			f = s.f_solution(k1 = k1, k2 = k2, k3 = k3)
			S = 1/(2*np.pi)**4/(f[0][0, 0][-1]*k1**3/2/np.pi**2)**2 * (k1*k2*k3)**2 * f[6][0, 0, 0]
			Shape[i, j] = S[-1]
		else:
			Shape[i, j] = np.log(0)  # if the point is outside of valid triangle configuration, fill the shape array with -inf




# save the data in .npy format
np.save("k1k3.npy", x)
np.save("k2k3.npy", y)
np.save("Shape.npy", Shape)


# save the data in .txt format (might be better if you want to import it in Mathematica)
np.savetxt("k1k3.txt", x)
np.savetxt("k2k3.txt", y)
np.savetxt("Shape.txt", Shape)



# plot the data
fig = plt.figure()
ax = fig.add_subplot(111)

X, Y = np.meshgrid(x, y)

ax.set_xlim(0, 1)
ax.set_ylim(0.5, 1)

ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0.5, 0.75, 1])
ax.plot(x, x, color = "k", ls = "--")
ax.plot(x, -x+1, color = "k", ls = "--")

ax.set_xlabel("$k_1/k_3$", fontsize = 12)
ax.set_ylabel("$k_2/k_3$", fontsize = 12)

# plot the shape function normalised to unity in the equilateral configuration
contourf_ = ax.contourf(X, Y, np.transpose(Shape)/Shape[-1, -1], n = n_point, cmap = "coolwarm")
cbar = fig.colorbar(contourf_)
plt.show()

