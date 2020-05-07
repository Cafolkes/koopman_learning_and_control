from .basis_functions import BasisFunctions
from .utils import rbf
from numpy import array, atleast_2d, tile, diag, reshape

class RBF(BasisFunctions):
    """
    Implements Radial Basis Functions (RBD) as a basis function
    """

    def __init__(self, rbf_centers, n, gamma=1., type='gaussian'):
        """__init__ [summary]
        
        Arguments:
            rbf_centers {numpy array [Ns,Nc]} -- points for the RBFs
            n {integer} -- Ns, number of states
        
        Keyword Arguments:
            gamma {float} -- gamma value for gaussian RBF (default: {1.})
            type {str} -- RBF Type (default: {'gaussian'})
        """
        self.n = n
        self.Nlift = rbf_centers.shape[1]
        self.rbf_centers = rbf_centers
        self.gamma = gamma
        self.type = type
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

        return atleast_2d(self.basis(q, q_d).squeeze())

    def construct_basis(self):
        if self.type == 'gaussian':
            self.basis = lambda q, q_t: rbf(q, self.rbf_centers, eps=self.gamma)
        else:
            raise Exception('RBF kernels other than Gaussian not implemented')

