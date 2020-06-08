from .utils import differentiate_vec
from .edmd import Edmd
import numpy as np

class BilinearEdmd(Edmd):
    def __init__(self, n, m, basis, n_lift, n_traj, optimizer, cv=None, standardizer=None, C=None):
        super(BilinearEdmd, self).__init__(n, m, basis, n_lift, n_traj, optimizer, cv=cv, standardizer=standardizer, C=C)
        self.B = []

    def fit(self, X, y, cv=False, override_kinematics=False):
        if override_kinematics:
            y = y[:, int(self.n / 2):]

        if cv:
            assert self.cv is not None, 'No cross validation method specified.'
            self.cv.fit(X,y)
            mdl_coefs = self.cv.coef_
        else:
            self.optimizer.fit(X, y)
            mdl_coefs = self.optimizer.coef_

        if self.standardizer is None:
            coefs = mdl_coefs
        else:
            coefs = self.standardizer.transform(mdl_coefs)

        if override_kinematics:
            kin_dyn = np.concatenate((np.zeros((int(self.n/2),int(self.n/2))),
                                       np.eye(int(self.n/2)),
                                       np.zeros((int(self.n/2),self.n_lift-self.n))),axis=1)
            self.A = np.concatenate((kin_dyn, coefs[:, :self.n_lift]),axis=0)
            for ii in range(self.m):
                self.B.append(np.concatenate((np.zeros((int(self.n/2), self.n_lift)),
                                                       coefs[:, self.n_lift * (ii + 1):self.n_lift * (ii + 2)]), axis=0))

        else:
            self.A = coefs[:, :self.n_lift]
            for ii in range(self.m):
                self.B.append(coefs[:, self.n_lift * (ii + 1):self.n_lift * (ii + 2)])

        #TODO: Add possibility of learning C-matrix.

    def process(self, x, u, t):
        assert x.shape[2] == self.n

        z = np.array([super(BilinearEdmd, self).lift(x[ii, :-1, :], u[ii, :, :]) for ii in range(self.n_traj)])
        z_dot = np.array([differentiate_vec(z[ii, :, :], t[ii, :-1]) for ii in range(self.n_traj)])
        z_bilinear = self.lift(x, u)

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
        pass

    def lift(self, x, u):
        z = np.array([super(BilinearEdmd, self).lift(x[ii, :-1, :], u[ii, :, :]) for ii in range(self.n_traj)])
        z_bilinear = z.copy()
        for ii in range(self.m):
            z_bilinear = np.concatenate((z_bilinear, np.multiply(z,np.tile(u[:,:,ii:ii+1], (1,1,z.shape[2])))),axis=2)
        return z_bilinear