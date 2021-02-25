from .utils import differentiate_vec
import numpy as np

class Edmd():
    '''
    Base class for edmd-type methods. Implements baseline edmd with the possible addition of l1 and/or l2 regularization.
    Overload fit for more specific methods.
    '''
    def __init__(self, n, m, basis, n_lift, n_traj, optimizer, cv=None, standardizer=None, C=None, first_obs_const=True,
                 continuous_mdl=True, dt=None):
        self.n = n
        self.n_lift = n_lift
        self.n_traj = n_traj
        self.m = m
        self.A = None
        self.B = None
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

        if override_kinematics:
            if self.continuous_mdl:
                const_dyn = np.zeros((int(self.first_obs_const), self.n_lift))
                kin_dyn = np.concatenate((np.zeros((int(self.n/2),int(self.n/2)+int(self.first_obs_const))),
                                           np.eye(int(self.n/2)),
                                           np.zeros((int(self.n/2),self.n_lift-self.n-int(self.first_obs_const)))),axis=1)
            else:
                const_dyn = np.hstack((np.ones((int(self.first_obs_const),1)), np.zeros((int(self.first_obs_const), self.n_lift-1))))
                kin_dyn = np.concatenate((np.zeros((int(self.n/2),int(self.first_obs_const))),
                                           np.eye(int(self.n/2)),
                                           self.dt*np.eye(int(self.n/2)),
                                           np.zeros((int(self.n/2),self.n_lift-self.n-int(self.first_obs_const)))),axis=1)
            self.A = np.concatenate((const_dyn, kin_dyn, coefs[:, :self.n_lift]+np.eye(self.n_lift)[int(self.n/2)+int(self.first_obs_const):,:]),axis=0)
            self.B = np.concatenate((np.zeros((int(self.n/2)+int(self.first_obs_const), self.m)),
                                     coefs[:,self.n_lift:]), axis=0)

        else:
            if self.continuous_mdl:
                self.A = coefs[:, :self.n_lift]
            else:
                self.A = coefs[:, :self.n_lift] + np.eye(self.n_lift)
            self.B = coefs[:, self.n_lift:]

        #TODO: Add possibility of learning C-matrix.

    def process(self, x, u, t, downsample_rate=1):
        assert x.shape[0] == self.n_traj
        assert x.shape[2] == self.n

        z = np.array([self.lift(x[ii, :-1, :], u[ii,:,:]) for ii in range(self.n_traj)])
        z_u = np.concatenate((z, u), axis=2)
        if self.continuous_mdl:
            z_prime = np.array([differentiate_vec(z[ii, :, :], t[ii,:-1]) for ii in range(self.n_traj)])
        else:
            z_prime = np.array([self.lift(x[ii, 1:, :], u[ii, :, :]) for ii in range(self.n_traj)])

        order = 'F'
        n_data_pts = self.n_traj * (t[0,:].shape[0] - 1)
        z_u_flat = z_u.T.reshape((self.n_lift + self.m, n_data_pts), order=order)
        z_prime_flat = z_prime.T.reshape((self.n_lift, n_data_pts), order=order)

        if self.standardizer is None:
            z_u_flat, z_prime_flat = z_u_flat.T, z_prime_flat.T
        else:
            self.standardizer.fit(z_u_flat.T)
            z_u_flat, z_prime_flat = self.standardizer.transform(z_u_flat.T), z_prime_flat.T

        return z_u_flat[::downsample_rate,:], z_prime_flat[::downsample_rate,:]

    def predict(self, x, u):
        """predict compute the right hand side of z_dot
        
        Arguments:
            X {numpy array [Ns,Nt]} -- state        
            U {numpy array [Nu,Nt]} -- control input
        
        Returns:
            numpy array [Ns,Nt] -- Az+Bu in z_dot = Az+Bu
        """
        return np.dot(self.C, np.dot(self.A, x) + np.dot(self.B, u))

    def reduce_mdl(self):
        # Identify what basis functions are in use:
        in_use = np.unique(np.nonzero(self.C)[1]) # Identify observables used for state prediction
        n_obs_used = 0
        while n_obs_used < in_use.size:
            n_obs_used = in_use.size
            in_use = np.unique(np.nonzero(self.A[in_use,:])[1])

        self.A = self.A[in_use,:]
        self.A = self.A[:, in_use]
        self.B = self.B[in_use, :]
        self.C = self.C[:, in_use]
        self.basis_reduced = lambda x: self.basis(x)[:,in_use]
        self.n_lift = in_use.size
        self.obs_in_use = in_use

    def score(self, x, u):
        pass

    def lift(self, x, u):
        return self.basis(x)
