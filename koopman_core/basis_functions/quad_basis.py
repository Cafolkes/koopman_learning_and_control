from .basis_functions import BasisFunctions
from sklearn import preprocessing
import numpy as np
import itertools as it

class QuadBasis(BasisFunctions):
    """Abstract class for basis functions that can be used to "lift" the state values.
    
    Override construct_basis
    """

    def __init__(self, n, poly_deg=2, n_lift=None, cross_terms=False):
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
        super(QuadBasis, self).__init__(n, n_lift)

    def construct_basis(self):
        poly_features = preprocessing.PolynomialFeatures(degree=self.poly_deg)
        poly_features.fit(np.zeros((1, 6)))
        poly_func = lambda x: poly_features.transform(np.hstack((x[:,3:6], x[:,9:12])))
        sine_func = lambda x: np.concatenate((np.sin(x[:,3:6]),
                                              np.cos(x[:,3:6]),
                                              np.multiply(np.cos(x[:,5:6]), np.multiply(np.sin(x[:,4:5]), np.cos(x[:,3:4]))),
                                              np.multiply(np.sin(x[:, 5:6]), np.sin(x[:,3:4])),
                                              np.multiply(np.sin(x[:, 5:6]), np.multiply(np.sin(x[:, 4:5]), np.cos(x[:, 3:4]))),
                                              np.multiply(np.cos(x[:, 5:6]), np.sin(x[:, 3:4])),
                                              np.multiply(np.cos(x[:, 4:5]), np.cos(x[:, 3:4])),
                                              ), axis=1)

        if self.cross_terms:
            self.basis = lambda x: np.hstack((np.ones((x.shape[0],1)),
                                              x,
                                              self.basis_product_(x, poly_func, sine_func)))
            self.n_lift = 1 + self.n + poly_features.n_output_features_*sine_func(np.zeros((1,self.n))).shape[1]
        else:
            self.basis = lambda x: np.hstack((np.ones((x.shape[0], 1)),
                                              x,
                                              poly_func(x),
                                              sine_func(x)))
            self.n_lift = 1 + self.n + poly_features.n_output_features_ + sine_func(np.zeros((1, self.n))).shape[1]

    def basis_product_(self, x, basis_1, basis_2):
        basis_1_eval = basis_1(x)
        basis_2_eval = basis_2(x)

        return np.multiply(np.tile(basis_1_eval, (1,basis_2_eval.shape[1])), np.repeat(basis_2_eval, basis_1_eval.shape[1], axis=1))
