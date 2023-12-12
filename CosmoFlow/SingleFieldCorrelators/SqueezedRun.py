import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

from squeezed import *

#-----------

n = 50
kappa = np.logspace(-1, 0, n)

#-----------

Shape = squeezed(kappa)

np.save("kappa.npy", kappa)
np.save("Shape", Shape)

plt.semilogx(kappa, Shape/kappa)
plt.show()
