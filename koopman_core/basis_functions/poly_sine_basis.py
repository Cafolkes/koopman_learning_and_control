from .basis_functions import BasisFunctions
from sklearn import preprocessing
import numpy as np
import itertools as it

class PolySineBasis(BasisFunctions):
    """Abstract class for basis functions that can be used to "lift" the state values.
    
    Override construct_basis
    """

    def __init__(self, n, poly_deg=2, n_lift=None, cross_terms=True):
        """
        Parameters
        ----------
        n : int
            number of basis functions
        Nlift : int
            Number of lifing functions
        """
        self.poly_deg = poly_deg
        self.cross_terms = cross_terms
        super(PolySineBasis, self).__init__(n, n_lift)

    def construct_basis(self):
        poly_features = preprocessing.PolynomialFeatures(degree=self.poly_deg)
        poly_features.fit(np.zeros((1, self.n)))
        poly_func = lambda x: poly_features.transform(x)

        if self.cross_terms:
            sine_func = lambda x: np.concatenate((np.ones((x.shape[0],1)), np.sin(x[:,2:3]), np.cos(x[:,2:3])),axis=1)
            self.basis = lambda x: self.basis_product_(x, poly_func, sine_func)
            self.n_lift = poly_features.n_output_features_*sine_func(np.zeros((1,self.n))).shape[1]
        else:
            sine_func = lambda x: np.concatenate((np.sin(x[:, 2:3]), np.cos(x[:, 2:3])), axis=1)
            self.basis = lambda x: np.concatenate((poly_func(x), sine_func(x)),axis=1)
            self.n_lift = poly_features.n_output_features_ + sine_func(np.zeros((1, self.n))).shape[1]


    def basis_product_(self, x, basis_1, basis_2):
        basis_1_eval = basis_1(x)
        basis_2_eval = basis_2(x)

        return np.multiply(np.tile(basis_1_eval, (1,basis_2_eval.shape[1])), np.repeat(basis_2_eval, basis_1_eval.shape[1], axis=1))
