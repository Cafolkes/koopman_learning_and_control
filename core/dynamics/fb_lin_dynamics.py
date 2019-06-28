from numpy import argsort, cumsum, diag, identity, ones
from scipy.linalg import block_diag

from .affine_dynamics import AffineDynamics
from .linearizable_dynamics import LinearizableDynamics

class FBLinDynamics(AffineDynamics, LinearizableDynamics):
    """Abstract class for feedback linearizable affine dynamics.

    Representation must be block form, with each block corresponding to an
    output coordinate. If an output has relative degree gamma, then the
    corresponding block must express derivatives of order 0 through gamma - 1,
    in that order.

    If dynamics are specified in a different order, specify a permutation into
    block form.

    Override eval, drift, act.
    """

    def __init__(self, relative_degrees, perm=None):
        """Create an FBLinDynamics object.

        Inputs:
        Relative degrees of each output coordinate, relative_degrees: int list
        Indices of coordinates that make up each block, perm: numpy array
        """

        self.relative_degrees = relative_degrees
        self.relative_degree_idxs = cumsum(relative_degrees) - 1
        if perm is None:
            perm = arange(sum(relative_degrees))
        self.perm = perm
        self.inv_perm = argsort(perm)

    def select(self, arr):
        """Select coordinates of block order corresponding to highest-order derivatives.

        Inputs:
        Array, arr: numpy array

        Outputs:
        Array of selected coordinates: numpy array
        """

        return arr[self.relative_degree_idxs]

    def permute(self, arr):
        """Permute array into block order.

        Inputs:
        Array, arr: numpy array

        Outputs:
        Array permuted into block form: numpy array
        """

        return arr[self.perm]

    def inv_permute(self, arr):
        """Permute array out of block order.

        Inputs:
        Array in block form, arr: numpy array

        Outputs:
        Array out of block form: numpy array
        """

        return arr[self.inv_perm]

    def linear_system(self):
        F = block_diag(*[diag(ones(gamma - 1), 1) for gamma in self.relative_degrees])
        G = (identity(sum(self.relative_degrees))[self.relative_degree_idxs]).T

        F = (self.inv_permute((self.inv_permute(F)).T)).T
        G = self.inv_permute(G)

        return F, G