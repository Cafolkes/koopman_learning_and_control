from .basis_functions import BasisFunctions

class IdentityBF(BasisFunctions):
    """
    Implements Radial Basis Functions (RBD) as a basis function
    """

    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            number of basis functions
        Nlift : int
            Number of lifing functions
        """
        self.n = n
        self.n_lift = n

    def lift(self, q, q_d):
        """
        Call this function to get the variables in lifted space

        Parameters
        ----------
        q : numpy array
            State vector

        Returns
        -------
        basis applied to q
        """
        return q

    def construct_basis(self):
        pass

