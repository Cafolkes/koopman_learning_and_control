from .utils import differentiate_vec
import numpy as np

class Edmd():
    '''
    Base class for edmd-type methods. Implements baseline edmd with the possible addition of l1 and/or l2 regularization.
    Overload fit for more specific methods.
    '''
    def __init__(self, n, m, basis, n_lift, n_traj, optimizer, standardizer=None, C=None):
        self.n = n
        self.n_lift = n_lift
        self.n_traj = n_traj
        self.m = m
        self.A = None
        self.B = None
        self.C = C

        self.basis = basis
        self.optimizer = optimizer
        self.standardizer = standardizer

    def fit(self, X, y):

        self.optimizer.fit(X, y)

        if self.standardizer is None:
            coefs = self.optimizer.coef_
        else:
            coefs = self.standardizer.transform(self.optimizer.coef_)

        self.A = coefs[:, :self.n_lift]
        self.B = coefs[:, self.n_lift:]

        #TODO: Add possibility of learning C-matrix.

    def process(self, x, u, t):
        assert x.shape[2] == self.n

        z = self.lift(x)
        z_u = np.concatenate((z, u), axis=2)
        z_dot = np.array([differentiate_vec(z[ii, :, :], t[ii,:-1]) for ii in range(self.n_traj)])

        order = 'F'
        n_data_pts = self.n_traj * (t[0,:].shape[0] - 1)
        z_u_flat = z_u.T.reshape((self.n_lift + self.m, n_data_pts), order=order)
        z_dot_flat = z_dot.T.reshape((self.n_lift, n_data_pts), order=order)

        if self.standardizer is None:
            return z_u_flat.T, z_dot_flat.T
        else:
            self.standardizer.fit(z_u_flat.T)
            return self.standardizer.transform(z_u_flat.T), z_dot_flat.T

    def predict(self, x, u):
        """predict compute the right hand side of z_dot
        
        Arguments:
            X {numpy array [Ns,Nt]} -- state        
            U {numpy array [Nu,Nt]} -- control input
        
        Returns:
            numpy array [Ns,Nt] -- Az+Bu in z_dot = Az+Bu
        """
        return np.dot(self.C, np.dot(self.A, x) + np.dot(self.B, u))

    def lift(self, x):
        return np.array([self.basis(x[ii, :-1, :]) for ii in range(self.n_traj)])