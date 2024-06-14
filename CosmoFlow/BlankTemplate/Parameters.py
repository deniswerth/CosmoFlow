from scipy.interpolate import interp1d  # import interp1d from Scipy package

class parameters:
    """
    The parameters class takes as inputs the parameters of the theory and creates interpolated continuous functions 
    to be passed to the classes theory and solver.
    """

    def __init__(self, N_load: np.ndarray, H_load: np.ndarray, parameter_load: np.ndarray):
        """
        Initialise the parameters class with the given arrays and create interpolated continuous functions.

        :param N_load: A one-dimensional array for the time grid array in e-folds.
        :type N_load: numpy.ndarray
        :param H_load: A one-dimensional array for the Hubble parameter as a function of e-folds evaluated on N_load.
        :type H_load: numpy.ndarray
        :param parameter_load: A one-dimensional array for any parameter as a function of e-folds evaluated on N_load.
        :type parameter_load: numpy.ndarray
        """

        # Importing pre-computed parameters of the theory
        self.N_load = N_load
        self.H_load = H_load
        self.parameter_load = parameter_load

        # Creating continuous functions out of the imported parameters
        self.H_f = interp1d(self.N_load, self.H_load, bounds_error=False, kind='cubic', fill_value="extrapolate")
        self.parameter_f = interp1d(self.N_load, self.parameter_load, bounds_error=False, kind='cubic', fill_value="extrapolate")

        # Creating interpolating list containing continuous functions
        self.interpolated = [self.H_f, self.parameter_f]

    def output(self):
        """
        Returns the interpolated continuous functions that can be evaluated at any N (time in e-folds).

        :return: A list containing the interpolated continuous functions for Hubble parameter and the given parameter.
        :rtype: list of scipy.interpolate.interpolate.interp1d
        """
        return self.interpolated






