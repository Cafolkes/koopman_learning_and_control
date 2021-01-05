from .utils import differentiate_vec
import numpy as np

class Edmd_aut():
    '''
    Base class for edmd-type methods. Implements baseline edmd with the possible addition of l1 and/or l2 regularization.
    Overload fit for more specific methods.
    '''
    def __init__(self, n, basis, n_lift, n_traj, optimizer, cv=None, standardizer=None, C=None, first_obs_const=True,
                 continuous_mdl=True, dt=None):
        self.n = n
        self.n_lift = n_lift
        self.n_traj = n_traj
        self.A = None
        self.C = C

        self.basis = basis
        self.first_obs_const = first_obs_const
        self.optimizer = optimizer
        self.cv = cv
        self.standardizer = standardizer
        self.continuous_mdl = continuous_mdl
        self.dt = dt

    def fit(self, X, y, cv=False, override_kinematics=False):
        if not self.continuous_mdl:
            y = y - self.standardizer.inverse_transform(X)[:,:self.n_lift]

        if override_kinematics:
            y = y[:,int(self.n/2)+int(self.first_obs_const):]

        if cv:
            assert self.cv is not None, 'No cross validation method specified.'
            self.cv.fit(X,y)
            coefs = self.cv.coef_
        else:
            self.optimizer.fit(X, y)
            coefs = self.optimizer.coef_

        if self.standardizer is not None:
            coefs = self.standardizer.transform(coefs)

        self.construct_dyn_mat_(coefs, override_kinematics)

    def construct_dyn_mat_(self, coefs, override_kinematics):
        if override_kinematics:
            if self.continuous_mdl:
                self.construct_dyn_mat_continuous_(coefs)
            else:
                self.construct_dyn_mat_discrete_(coefs)
        else:
            self.A = coefs[:, :self.n_lift]

            if not self.continuous_mdl:
                fixed_inds = int(self.first_obs_const) + int(override_kinematics) * int(self.n / 2)
                self.A[fixed_inds:, fixed_inds:] += np.eye(self.n_lift - fixed_inds)


    def construct_dyn_mat_continuous_(self, coefs):
        const_dyn = np.zeros((int(self.first_obs_const), self.n_lift))
        kin_dyn = np.concatenate((np.zeros((int(self.n / 2), int(self.n / 2) + int(self.first_obs_const))),
                                  np.eye(int(self.n / 2)),
                                  np.zeros((int(self.n / 2), self.n_lift - self.n - int(self.first_obs_const)))),
                                 axis=1)
        self.A = np.concatenate((const_dyn, kin_dyn, coefs[:, :self.n_lift] + np.eye(self.n_lift)[int(self.n / 2) + int(
            self.first_obs_const):, :]), axis=0)

    def construct_dyn_mat_discrete_(self, coefs):
        const_dyn = np.hstack(
            (np.ones((int(self.first_obs_const), 1)), np.zeros((int(self.first_obs_const), self.n_lift - 1))))
        kin_dyn = np.concatenate((np.zeros((int(self.n / 2), int(self.first_obs_const))),
                                  np.eye(int(self.n / 2)),
                                  self.dt * np.eye(int(self.n / 2)),
                                  np.zeros((int(self.n / 2), self.n_lift - self.n - int(self.first_obs_const)))),
                                 axis=1)
        self.A = np.concatenate((const_dyn, kin_dyn, coefs[:, :self.n_lift] + np.eye(self.n_lift)[int(self.n / 2) + int(
            self.first_obs_const):, :]), axis=0)

    def process(self, x, t, downsample_rate=1):
        assert x.shape[0] == self.n_traj
        assert x.shape[2] == self.n

        z = np.array([self.lift(x[ii, :-1, :]) for ii in range(self.n_traj)])
        if self.continuous_mdl:
            z_prime = np.array([differentiate_vec(z[ii, :, :], t[ii,:-1]) for ii in range(self.n_traj)])
        else:
            z_prime = np.array([self.lift(x[ii, 1:, :]) for ii in range(self.n_traj)])

        order = 'F'
        n_data_pts = self.n_traj * (t[0,:].shape[0] - 1)
        z_flat = z.T.reshape((self.n_lift, n_data_pts), order=order)
        z_prime_flat = z_prime.T.reshape((self.n_lift, n_data_pts), order=order)

        if self.standardizer is None:
            z_flat, z_prime_flat = z_flat.T, z_prime_flat.T
        else:
            self.standardizer.fit(z_flat.T)
            z_flat, z_prime_flat = self.standardizer.transform(z_flat.T), z_prime_flat.T

        return z_flat[::downsample_rate,:], z_prime_flat[::downsample_rate,:]

    def predict(self, x):
        """predict compute the right hand side of z_dot
        
        Arguments:
            X {numpy array [Ns,Nt]} -- state        
            U {numpy array [Nu,Nt]} -- control input
        
        Returns:
            numpy array [Ns,Nt] -- Az+Bu in z_dot = Az+Bu
        """
        return np.dot(self.C, np.dot(self.A, x))

    def score(self, x, u):
        pass

    def lift(self, x):
        return self.basis(x)