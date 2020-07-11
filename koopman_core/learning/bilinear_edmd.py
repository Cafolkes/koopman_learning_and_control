from .utils import differentiate_vec
from .edmd import Edmd
import numpy as np

class BilinearEdmd(Edmd):
    def __init__(self, n, m, basis, n_lift, n_traj, optimizer, cv=None, standardizer=None, C=None, first_obs_const=True):
        super(BilinearEdmd, self).__init__(n, m, basis, n_lift, n_traj, optimizer, cv=cv, standardizer=standardizer, C=C, first_obs_const=first_obs_const)
        self.B = []

        self.basis_reduced = None
        self.n_lift_reduced = None
        self.obs_in_use = None

    def fit(self, X, y, cv=False, override_kinematics=False, first_obs_const=True):

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
        z = np.array([super(BilinearEdmd, self).lift(x[ii, :-1, :], u[ii, :, :]) for ii in range(self.n_traj)])
        z_dot = np.array([differentiate_vec(z[ii, :, :], t[ii, :-1]) for ii in range(self.n_traj)])
        z_bilinear = np.array([self.lift(x[ii, :-1, :], u[ii, :, :]) for ii in range(self.n_traj)])

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
        pass

    def lift(self, x, u):
            return np.array([self.bilinear_basis(x[ii, :], u[ii, :]) for ii in range(x.shape[0])])

    def reduce_mdl(self):
        # Identify what basis functions are in use:
        in_use = np.unique(np.nonzero(self.C)[1]) # Identify observables used for state prediction
        n_obs_used = 0
        while n_obs_used < in_use.size:
            n_obs_used = in_use.size
            in_use = np.unique(np.nonzero(self.A[in_use,:])[1])
            for ii in range(self.m):
                in_use = np.unique(np.concatenate((in_use, np.nonzero(self.B[ii][in_use,:])[1])))

        self.A = self.A[in_use,:]
        self.A = self.A[:, in_use]
        for ii in range(self.m):
            self.B[ii] = self.B[ii][in_use, :]
            self.B[ii] = self.B[ii][:, in_use]
        self.C = self.C[:, in_use]
        self.basis_reduced = lambda x: self.basis(x)[:,in_use]
        self.n_lift_reduced = in_use.size
        self.obs_in_use = in_use

    def construct_bilinear_basis_(self):
        basis_lst = [lambda x, u: self.basis(x)]

        #TODO: Implement bilinearization for general number of inputs (below implementation not working)
        #for ii in range(self.m):
        #    basis_lst.append(lambda x, u: np.multiply(self.basis(x), u[ii]))
        basis_lst.append(lambda x, u: np.multiply(self.basis(x), u[0]))
        basis_lst.append(lambda x, u: np.multiply(self.basis(x), u[1]))

        self.bilinear_basis = lambda x, u: np.array([basis_lst[ii](x.reshape(1,-1),u) for ii in range(self.m+1)]).flatten()
