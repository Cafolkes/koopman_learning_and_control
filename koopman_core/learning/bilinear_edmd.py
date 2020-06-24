from .utils import differentiate_vec
from .edmd import Edmd
import numpy as np

class BilinearEdmd(Edmd):
    def __init__(self, n, m, basis, n_lift, n_traj, optimizer, cv=None, standardizer=None, C=None, first_obs_const=True):
        super(BilinearEdmd, self).__init__(n, m, basis, n_lift, n_traj, optimizer, cv=cv, standardizer=standardizer, C=C, first_obs_const=first_obs_const)
        self.B = []

    def fit(self, X, y, cv=False, override_kinematics=False):
        if override_kinematics:
            y = y[:, int(self.n / 2) + int(self.first_obs_const):]

        if cv:
            assert self.cv is not None, 'No cross validation method specified.'
            self.cv.fit(X,y)
            coefs = self.cv.coef_
        else:
            self.optimizer.fit(X, y)
            coefs = self.optimizer.coef_

        if self.standardizer is not None:
            coefs = self.standardizer.transform(coefs)

        if override_kinematics:
            const_dyn = np.zeros((int(self.first_obs_const), self.n_lift))
            kin_dyn = np.concatenate((np.zeros((int(self.n / 2), int(self.n / 2) + int(self.first_obs_const))),
                                      np.eye(int(self.n / 2)),
                                      np.zeros((int(self.n / 2), self.n_lift - self.n - int(self.first_obs_const)))),
                                     axis=1)
            self.A = np.concatenate((const_dyn, kin_dyn, coefs[:, :self.n_lift]), axis=0)

            for ii in range(self.m):
                self.B.append(np.concatenate((np.zeros((int(self.n/2)+int(self.first_obs_const), self.n_lift)),
                                     coefs[:, self.n_lift * (ii + 1):self.n_lift * (ii + 2)]), axis=0))
        else:
            self.A = coefs[:, :self.n_lift]
            for ii in range(self.m):
                self.B.append(coefs[:, self.n_lift * (ii + 1):self.n_lift * (ii + 2)])

        #TODO: Add possibility of learning C-matrix.

    def process(self, x, u, t, downsample_rate=1):
        assert x.shape[2] == self.n

        self.construct_bilinear_basis_()
        #z = super(BilinearEdmd, self).lift(x, u)
        z = np.array([super(BilinearEdmd, self).lift(x[ii, :-1, :], u[ii, :, :]) for ii in range(self.n_traj)])
        z_dot = np.array([differentiate_vec(z[ii, :, :], t[ii, :-1]) for ii in range(self.n_traj)])
        z_bilinear = self.lift(x, u) #TODO: Wrong shape, look into lifting func

        order = 'F'
        n_data_pts = self.n_traj * (t[0,:].shape[0] - 1)
        z_bilinear_flat = z_bilinear.T.reshape(((self.m+1)*self.n_lift, n_data_pts), order=order)
        z_dot_flat = z_dot.T.reshape((self.n_lift, n_data_pts), order=order)

        if self.standardizer is None:
            z_bilinear_flat, z_dot_flat = z_bilinear_flat.T, z_dot_flat.T
        else:
            self.standardizer.fit(z_bilinear_flat.T)
            z_bilinear_flat, z_dot_flat = self.standardizer.transform(z_bilinear_flat.T), z_dot_flat.T

        return z_bilinear_flat[::downsample_rate, :], z_dot_flat[::downsample_rate, :]

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