from numpy import array, atleast_2d, concatenate, reshape, multiply

class BilinearBasis():
    """
    Implements Radial Basis Functions (RBD) as a basis function
    """

    def __init__(self, n, m, Nlift, core_basis, add_pure_ctrl=False):
        """__init__ [summary]
        
        Arguments:
            n {integer} -- Ns, number of states
        
        Keyword Arguments:

        """
        self.n = n
        self.m = m
        self.Nlift = Nlift
        self.core_basis = core_basis
        self.basis = None
        self.add_pure_ctrl = add_pure_ctrl

    def lift(self, x, u):
        """Lift the state using the basis function
        
        Arguments:
            x {numpy array [Ns,Nt]} -- state
            u {numpy array [Nt,Nt]} -- control action
        
        Returns:
            numpy array [Nz,Nt] -- lifted state
        """
        if x.ndim == 1:
            x = reshape(x,(x.shape[0],1))

        z = atleast_2d([self.basis(x[ii,:].reshape(1,-1), u[ii,:].reshape(1,-1)).squeeze() for ii in range(x.shape[0])])
        if self.add_pure_ctrl:
            z = concatenate((z, u), axis=1)

        return z

    def construct_basis(self):
        basis_lst = []
        basis_lst.append(lambda x, u: self.core_basis(x))
        for ii in range(self.m):
            basis_lst.append(lambda x, u: multiply(self.core_basis(x), u[:,ii]))

        self.basis = lambda x, u: array([basis_lst[ii](x, u) for ii in range(self.m+1)]).flatten()

