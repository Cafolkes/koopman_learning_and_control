from .basis_functions import BasisFunctions
from sklearn import preprocessing
import numpy as np
import itertools as it

class CartPoleBasis(BasisFunctions):
    """Abstract class for basis functions that can be used to "lift" the state values.
    
    Override construct_basis
    """

    def __init__(self, n, Nlift):
        """
        Parameters
        ----------
        n : int
            number of basis functions
        Nlift : int
            Number of lifing functions
        """
        super(CartPoleBasis, self).__init__(n, Nlift)

    def construct_basis(self, poly_deg=2):
        poly_features = preprocessing.PolynomialFeatures(degree=poly_deg)
        poly_features.fit(np.zeros((1, self.n)))
        poly_func = lambda x: poly_features.transform(x)
        #poly_func = lambda x: np.concatenate((np.ones((x.shape[0],1)), x, np.square(x[:,3])), axis=1)
        sine_func = lambda x: np.concatenate((np.ones((x.shape[0],1)), np.sin(x[:,1:2]), np.cos(x[:,1:2])), axis=1)

        self.basis = lambda x: self.basis_product_(x, poly_func, sine_func)

    def basis_product_(self, x, basis_1, basis_2):
        basis_1_eval = basis_1(x)
        basis_2_eval = basis_2(x)

        return np.multiply(np.tile(basis_1_eval, (1,basis_2_eval.shape[1])), np.repeat(basis_2_eval, basis_1_eval.shape[1], axis=1))
