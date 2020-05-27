from .utils import differentiate_vec
from .edmd import Edmd
import numpy as np

class BilinearEdmd(Edmd):
    def __init__(self, n, m, basis, n_lift, n_traj, optimizer, standardizer=None, C=None):
        super(BilinearEdmd, self).__init__(n, m, basis, n_lift, n_traj, optimizer, standardizer=standardizer, C=C)
        self.B = []

    def fit(self, X, y):

        self.optimizer.fit(X, y)

        if self.standardizer is None:
            coefs = self.optimizer.coef_
        else:
            coefs = self.standardizer.transform(self.optimizer.coef_)

        self.A = coefs[:, :self.n_lift]
        for ii in range(self.m):
            self.B.append(coefs[:, self.n_lift * (ii + 1):self.n_lift * (ii + 2)])

        #TODO: Add possibility of learning C-matrix.

    def process(self, x, u, t):
        assert x.shape[2] == self.n

        self.construct_bilinear_basis_()
        z = super(BilinearEdmd, self).lift(x, u)
        z_dot = np.array([differentiate_vec(z[ii, :, :], t[ii, :-1]) for ii in range(self.n_traj)])
        z_bilinear = self.lift(x, u) #TODO: Wrong shape, look into lifting func

        order = 'F'
        n_data_pts = self.n_traj * (t[0,:].shape[0] - 1)
        z_bilinear_flat = z_bilinear.T.reshape(((self.m+1)*self.n_lift, n_data_pts), order=order)
        z_dot_flat = z_dot.T.reshape((self.n_lift, n_data_pts), order=order)

        if self.standardizer is None:
            return z_bilinear_flat.T, z_dot_flat.T
        else:
            self.standardizer.fit(z_bilinear_flat.T)
            return self.standardizer.transform(z_bilinear_flat.T), z_dot_flat.T

    def predict(self, x, u):
        return np.dot(self.C, np.dot(self.A, x) + np.dot(self.B, u))

    def lift(self, x, u):
        return np.atleast_2d([self.bilinear_basis(x[ii,:-1,:], u[ii,:,:]) for ii in range(self.n_traj)])

    def construct_bilinear_basis_(self):
        basis_lst = []
        basis_lst.append(lambda x, u: self.basis(x))
        for ii in range(self.m):
            basis_lst.append(lambda x, u: np.multiply(self.basis(x), u[:, ii].reshape(-1,1)))

        basis_stacked = lambda x, u: np.array([basis_lst[ii](x, u) for ii in range(self.m + 1)]).flatten()
        self.bilinear_basis = lambda x, u: np.array([basis_stacked(x[ii,:].reshape(1,-1), u[ii,:].reshape(1,-1)) for ii in range(x.shape[0])]) #TODO: Get rid of .flatten() (makes it too flat)