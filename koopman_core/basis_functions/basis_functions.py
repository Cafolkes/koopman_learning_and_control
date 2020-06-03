

class BasisFunctions():
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
        self.n = n
        self.Nlift = Nlift
        self.basis = None

    def lift(self, q):
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

        return self.basis(q)

    def construct_basis(self, poly_deg=2):
        pass