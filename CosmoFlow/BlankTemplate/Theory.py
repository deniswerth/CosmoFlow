import numpy as np  # import Numpy package for vectorisation
from scipy.misc import derivative  # import derivative from Scipy package

class theory:
    """
    This class defines the theory, including:
    (i) time-dependent functions,
    (ii) tensors Delta, M, I, A, B, C, and D,
    (iii) tensors u^A_B and u^A_BC.

    All functions have been optimized to be computed once at each integration step.
    """

    def __init__(self, N, Nfield, interpolated):
        """
        Initialise the theory class with given parameters.

        :param N: Time grid array in e-folds.
        :type N: numpy.ndarray
        :param Nfield: Number of fields in the theory.
        :type Nfield: int
        :param interpolated: List of interpolated continuous functions.
        :type interpolated: list of scipy.interpolate.interpolate.interp1d
        """
        self.N = N
        self.Nfield = Nfield
        self.interpolated = interpolated

        # Evaluate the parameters at N
        self.H = self.H_f(N)
        self.parameter = self.parameter_f(N)

        # Additional functions and scale functions
        self.a = self.a_f(N)  # Scale factor
        self.dH = self.dH_f(N)  # Derivative of Hubble
        self.scale = self.scale_f(N)  # Rescale function to improve performance
        self.dscale = self.dscale_f(N)  # Derivative of rescale function

    def H_f(self, N):
        """
        Interpolated Hubble parameter as a function of N.

        :param N: Time grid array in e-folds.
        :type N: numpy.ndarray
        :return: Hubble parameter evaluated at N.
        :rtype: numpy.ndarray
        """
        return self.interpolated[0](N)

    def parameter_f(self, N):
        """
        Interpolated parameter as a function of N.

        :param N: Time grid array in e-folds.
        :type N: numpy.ndarray
        :return: Parameter evaluated at N.
        :rtype: numpy.ndarray
        """
        return self.interpolated[1](N)

    def a_f(self, N):
        """
        Scale factor as a function of the number of e-folds.

        :param N: Time grid array in e-folds.
        :type N: numpy.ndarray
        :return: Scale factor evaluated at N.
        :rtype: numpy.ndarray
        """
        return np.exp(N)

    def dH_f(self, N):
        """
        Derivative of the Hubble rate with respect to cosmic time.

        :param N: Time grid array in e-folds.
        :type N: numpy.ndarray
        :return: Derivative of the Hubble rate evaluated at N.
        :rtype: numpy.ndarray
        """
        dHdN = derivative(self.H_f, N, dx=1e-6)
        return self.H_f(N) * dHdN

    def k_mode(self, N_exit):
        """
        Return the mode k horizon crossing corresponding to N_exit.

        :param N_exit: Time of horizon crossing in e-folds.
        :type N_exit: numpy.ndarray
        :return: Mode k at horizon crossing.
        :rtype: numpy.ndarray
        """
        return self.a_f(N_exit) * self.H_f(N_exit)

    def scale_f(self, N):
        """
        Define the rescaled scale factor to improve performance.

        :param N: Time grid array in e-folds.
        :type N: numpy.ndarray
        :return: Rescaled scale factor evaluated at N.
        :rtype: numpy.ndarray
        """
        k = 1
        a = self.a_f(N)
        H = self.H_f(N)
        return a / (1. + a * H / k) / H

    def dscale_f(self, N):
        """
        Define the derivative of the rescale function.

        :param N: Time grid array in e-folds.
        :type N: numpy.ndarray
        :return: Derivative of the rescale function evaluated at N.
        :rtype: numpy.ndarray
        """
        k = 1
        a = self.a_f(N)
        H = self.H_f(N)
        Hd = self.dH_f(N)
        return -Hd / H / H * a / (1. + a * H / k) + a / (1. + a * H / k) - a * (a * H * H / k + a * Hd / k) / (1. + a * H / k) / (1. + a * H / k) / H

    def Delta_ab(self):
        """
        Define the quadratic theory tensor Delta_ab.

        :return: Identity matrix of size Nfield x Nfield.
        :rtype: numpy.ndarray
        """
        return np.eye(self.Nfield)

    def I_ab(self):
        """
        Define the quadratic theory tensor I_ab.

        :return: Zero matrix of size Nfield x Nfield.
        :rtype: numpy.ndarray
        """
        return np.zeros((self.Nfield, self.Nfield))

    def M_ab(self, k):
        """
        Define the quadratic theory tensor M_ab.

        :param k: Mode k.
        :type k: float
        :return: Identity matrix of size Nfield x Nfield.
        :rtype: numpy.ndarray
        """
        return np.eye(self.Nfield)

    def A_abc(self, k1, k2, k3):
        """
        Define the cubic theory tensor A_abc.

        :param k1: Mode k1.
        :type k1: float
        :param k2: Mode k2.
        :type k2: float
        :param k3: Mode k3.
        :type k3: float
        :return: Zero matrix of size Nfield x Nfield x Nfield.
        :rtype: numpy.ndarray
        """
        return np.zeros((self.Nfield, self.Nfield, self.Nfield))

    def A_abc_fast(self, k1, k2, k3):
        """
        Define the fast version of the cubic theory tensor A_abc for initial conditions.

        :param k1: Mode k1.
        :type k1: float
        :param k2: Mode k2.
        :type k2: float
        :param k3: Mode k3.
        :type k3: float
        :return: Zero matrix of size Nfield x Nfield x Nfield.
        :rtype: numpy.ndarray
        """
        return np.zeros((self.Nfield, self.Nfield, self.Nfield))

    def A_abc_slow(self, k1, k2, k3):
        """
        Define the slow version of the cubic theory tensor A_abc.

        :param k1: Mode k1.
        :type k1: float
        :param k2: Mode k2.
        :type k2: float
        :param k3: Mode k3.
        :type k3: float
        :return: Zero matrix of size Nfield x Nfield x Nfield.
        :rtype: numpy.ndarray
        """
        return np.zeros((self.Nfield, self.Nfield, self.Nfield))

    def B_abc(self, k1, k2, k3):
        """
        Define the cubic theory tensor B_abc.

        :param k1: Mode k1.
        :type k1: float
        :param k2: Mode k2.
        :type k2: float
        :param k3: Mode k3.
        :type k3: float
        :return: Zero matrix of size Nfield x Nfield x Nfield.
        :rtype: numpy.ndarray
        """
        return np.zeros((self.Nfield, self.Nfield, self.Nfield))

    def C_abc(self, k1, k2, k3):
        """
        Define the cubic theory tensor C_abc.

        :param k1: Mode k1.
        :type k1: float
        :param k2: Mode k2.
        :type k2: float
        :param k3: Mode k3.
        :type k3: float
        :return: Zero matrix of size Nfield x Nfield x Nfield.
        :rtype: numpy.ndarray
        """
        return np.zeros((self.Nfield, self.Nfield, self.Nfield))

    def D_abc(self, k1, k2, k3):
        """
        Define the cubic theory tensor D_abc.

        :param k1: Mode k1.
        :type k1: float
        :param k2: Mode k2.
        :type k2: float
        :param k3: Mode k3.
        :type k3: float
        :return: Zero matrix of size Nfield x Nfield x Nfield.
        :rtype: numpy.ndarray
        """
        return np.zeros((self.Nfield, self.Nfield, self.Nfield))

    def u_AB(self, k):
        """
        Define the u_AB tensor.

        :param k: Mode k.
        :type k: float
        :return: u_AB tensor of size 2*Nfield x 2*Nfield.
        :rtype: numpy.ndarray
        """
        Nfield = self.Nfield
        H = self.H
        s = self.scale
        ds = self.dscale
        S = np.ones((Nfield, Nfield)) + (s - 1) * np.eye(Nfield)
        uAB = np.zeros((2 * Nfield, 2 * Nfield))
        uAB[:Nfield, :Nfield] = -self.I_ab() / H
        uAB[:Nfield, Nfield:] = self.Delta_ab() / H / s
        uAB[Nfield:, :Nfield] = self.M_ab(k) / H * s
        uAB[Nfield:, Nfield:] = (self.I_ab()).T / H - 3 * self.H * np.eye(Nfield) / H + ds / s * np.eye(Nfield) / H
        return uAB

    def u_ABC(self, k1, k2, k3):
        """
        Define the u_ABC tensor.

        :param k1: Mode k1.
        :type k1: float
        :param k2: Mode k2.
        :type k2: float
        :param k3: Mode k3.
        :type k3: float
        :return: u_ABC tensor of size 2*Nfield x 2*Nfield x 2*Nfield.
        :rtype: numpy.ndarray
        """
        Nfield = self.Nfield
        s = self.scale
        S = np.ones((Nfield, Nfield, Nfield)) + (s - 1) * np.eye(Nfield)
        H = self.H
        uABC = np.zeros((2 * Nfield, 2 * Nfield, 2 * Nfield))

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
                    uABC[i, j, k] = -B231[j, k, i] / H
                    uABC[i, Nfield + j, k] = -C123[i, j, k] / H / s
                    uABC[i, j, Nfield + k] = -C132[i, k, j] / H / s
                    uABC[Nfield + i, Nfield + j, Nfield + k] = C321[k, j, i] / H / s
                    uABC[i, Nfield + j, Nfield + k] = 3. * D123[i, j, k] / H / s / s
                    uABC[Nfield + i, j, k] = 3. * A123[i, j, k] / H * s
                    uABC[Nfield + i, Nfield + j, k] = B132[i, k, j] / H
                    uABC[Nfield + i, j, Nfield + k] = B123[i, j, k] / H
        return uABC

