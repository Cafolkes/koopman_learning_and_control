from sklearn.metrics.pairwise import rbf_kernel
from .basis_functions import BasisFunctions
from .utils import rbf
from numpy import array, atleast_2d, tile, diag, reshape

class Monomials(BasisFunctions):
    """
    Implements Radial Basis Functions (RBD) as a basis function
    """

    def __init__(self, n, Nlift):
        """__init__ [summary]
        
        Arguments:
            n {integer} -- Ns, number of states
        
        Keyword Arguments:

        """
        self.n = n
        self.Nlift = Nlift
        self.Lambda = None
        self.basis = None

    def lift(self, q, q_d):
        """lift Lift the state using the basis function
        
        Arguments:
            q {numpy array [Ns,Nt]} -- state
            q_d {numpy array [Nt,Nt]} -- desired state
        
        Returns:
            numpy array [Nz,Nt] -- lifted state
        """
        if q.ndim == 1:
            q = reshape(q,(q.shape[0],1))

        return atleast_2d([self.basis(q[:,ii].reshape(-1,1), q_d[:,ii].reshape(-1,1)).squeeze() for ii in range(q.shape[1])])

    def construct_basis(self):
        from itertools import combinations_with_replacement, permutations
        from numpy import unique, concatenate, lexsort, amax, prod, power

        p = array([ii for ii in range(self.Nlift)])
        combinations = array(list(combinations_with_replacement(p, self.n)))
        powers = array([list(permutations(c, self.n)) for c in combinations])  # Find all permutations of powers
        powers = unique(powers.reshape((powers.shape[0] * powers.shape[1], powers.shape[2])),
                        axis=0)  # Remove duplicates
        powers = concatenate((powers, amax(powers[:,:2], axis=1).reshape(-1,1)), axis=1)
        powers = powers[lexsort((powers[:,1], powers[:,0], powers[:,2])), :]
        powers = powers[3:3+self.Nlift,:2]

        self.basis = lambda q, q_d: prod(power(q, powers.T), axis=0)

