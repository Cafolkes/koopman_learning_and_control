from numpy import array, dot
from numpy.linalg import solve

from .. import arr_map

class GaussianProcessEstimator:
    def __init__(self, kernel, data):
        self.kernel = kernel
        self.data = data
        self.kernel_mat = arr_map(self.embedding, data)
        self.weights = None

    def embedding(self, x):
        return array([self.kernel.eval(x_i, x) for x_i in self.data])

    def fit(self, targets):
        self.weights = solve(self.kernel_mat, targets)
        return self

    def eval(self, x):
        return dot(self.weights, self.embedding(x))
