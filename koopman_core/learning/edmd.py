from .utils import differentiate_vec
from sklearn import linear_model
from scipy.linalg import expm
from numpy import array, concatenate, zeros, dot, linalg, eye, ones, std, where, divide, multiply, tile, argwhere, diag, copy, ones_like
from .basis_functions import BasisFunctions

class Edmd():
    '''
    Base class for edmd-type methods. Implements baseline edmd with the possible addition of l1 and/or l2 regularization.
    Overload fit for more specific methods.
    '''
    def __init__(self, basis=BasisFunctions(0,0), system_dim=0, l1=0., l1_ratio=0.5, acceleration_bounds=None, override_C=True, add_ones=True, add_state=True):
        self.A = None
        self.B = None
        self.C = None
        self.basis = basis  # Basis class to use as lifting functions
        self.l1 = l1  # Strength of l1 regularization
        self.l1_ratio = l1_ratio  # Strength of l2 regularization
        self.n = system_dim
        self.n_lift = None
        self.m = None
        self.override_C = override_C
        self.acceleration_bounds = acceleration_bounds #(nx1)
        self.add_ones=add_ones
        self.add_state=add_state

        lift_dim = basis.Nlift
        if self.add_ones:
            lift_dim += 1
        if self.add_state:
            lift_dim += basis.n
        self.Z_std = ones((lift_dim, 1))

    def fit(self, X, X_d, Z, Z_dot, U=None, U_nom=None, X_dot=None):
        """
        Fit a EDMD object with the given basis function

        Sizes:
        - Ntraj: number of trajectories
        - N: number of timesteps
        - ns: number or original states
        - nu: number of control inputs

        Inputs:
        - X: state with all trajectories, numpy 3d array [NtrajxN, ns]
        - X_d: desired state with all trajectories, numpy 3d array [NtrajxN, ns]
        - Z: lifted state with all trajectories, numpy[NtrajxN, ns]
        - Z: derivative of lifted state with all trajectories, numpy[NtrajxN, ns]
        - U: control input, numpy 3d array [NtrajxN, nu]
        - U_nom: nominal control input, numpy 3d array [NtrajxN, nu]
        - t: time, numpy 2d array [Ntraj, N]
        """

        if self.l1 == 0.:
            # Construct EDMD matrices as described in M. Korda, I. Mezic, "Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control":
            W = concatenate((Z_dot, X), axis=0)
            if U is None and U_nom is None:
                V = Z
            else:
                V = concatenate((Z, U), axis=0)
            VVt = dot(V,V.transpose())
            WVt = dot(W,V.transpose())
            M = dot(WVt, linalg.pinv(VVt))
            self.A = M[:self.n_lift,:self.n_lift]
            if U is None and U_nom is None:
                self.B = None
            else:
                self.B = M[:self.n_lift,self.n_lift:]
            self.C = M[self.n_lift:,:self.n_lift]

            if self.override_C:
                self.C = zeros(self.C.shape)
                self.C[:self.n,:self.n] = eye(self.n)
                self.C = multiply(self.C, self.Z_std.transpose())

        else:
            # Construct EDMD matrices using Elastic Net L1 and L2 regularization
            if U is None and U_nom is None:
                input = Z.transpose()
            else:
                input = concatenate((Z.transpose(), U.transpose()), axis=1)
            output = Z_dot.transpose()

            CV = False
            if CV:
                reg_model = linear_model.MultiTaskElasticNetCV(alphas=None, copy_X=True, cv=5, eps=0.001, fit_intercept=True,
                                        l1_ratio=self.l1_ratio, max_iter=1e6, n_alphas=100, n_jobs=None,
                                        normalize=False, positive=False, precompute='auto', random_state=0,
                                        selection='random', tol=0.0001, verbose=0)
            else:
                reg_model = linear_model.ElasticNet(alpha=self.l1, l1_ratio=self.l1_ratio, fit_intercept=False, normalize=False, selection='random', max_iter=1e5)
            reg_model.fit(input,output)

            self.A = reg_model.coef_[:self.n_lift,:self.n_lift]
            if not (U is None and U_nom is None):
                self.B = reg_model.coef_[:self.n_lift, self.n_lift:]
            if self.override_C:
                self.C = zeros((self.n,self.n_lift))
                self.C[:self.n,:self.n] = eye(self.n)
                self.C = multiply(self.C, self.Z_std.transpose())
            else:
                input = Z.T
                output = X.T
                reg_model_C = linear_model.ElasticNet(alpha=self.l1, l1_ratio=self.l1_ratio, fit_intercept=False,
                                                    normalize=False, selection='random', max_iter=1e5)
                reg_model_C.fit(input, output)
                self.C = reg_model_C.coef_

    def process(self, X, X_d, U, U_nom, t):
        """process filter data
        
        Arguments:
            X {numpy array []} -- state
            X_d {numpy array []} -- desired state
            U {numpy array } -- command
            U_nom {numpy array } -- nominal command
            t {numpy array [Nt,]} -- time vector
        
        Returns:
            [type] --  state flat, desired state flat, lifted state flat, 
                       lifted state dot flat, control flat, nominal control flat, time flat
        """
        # Remove points where not every input is available:
        if U.shape[1] < X.shape[1]:
            X = X[:,:U.shape[1],:]
            X_d = X_d[:, :U.shape[1], :]
            t = t[:,:U.shape[1]]

        Ntraj = X.shape[0]  # Number of trajectories in dataset
        Z = array([self.lift(X[ii,:,:].transpose(), X[ii,:,:].transpose()) for ii in range(Ntraj)])  # Lift x
        #Z_old = copy(Z)  #TODO: Remove after debug
        # Vectorize Z- data
        n_data = Z.shape[0] * Z.shape[1]
        self.n_lift = Z.shape[2]
        self.m = U.shape[2]
        order = 'F'
        Z_vec = Z.transpose().reshape((self.n_lift, n_data), order=order)

        # Normalize data
        self.Z_std = std(Z_vec, axis=1)
        self.Z_std[argwhere(self.Z_std == 0.)] = 1.
        self.Z_std[:self.n] = 1.  # Do not rescale states. Note: Assumes state is added to beginning of observables
        self.Z_std = self.Z_std.reshape((self.Z_std.shape[0], 1))
        #self.Z_std = ones_like(self.Z_std)  #TODO: Remove after debug
        Z_norm = array([divide(Z[ii,:,:], self.Z_std.transpose()) for ii in range(Z.shape[0])])
        #Z_norm = Z  #TODO: Remove after debug

        Z_dot = array([differentiate_vec(Z_norm[ii,:,:],t[ii,:]) for ii in range(Ntraj)])  #Numerical differentiate lifted state

        #Vectorize remaining data
        X_flat, X_d_flat, Z_flat, Z_dot_flat, U_flat, U_nom_flat, t_flat = X.transpose().reshape((self.n,n_data),order=order), \
                                                                        X_d.transpose().reshape((self.n, n_data),order=order), \
                                                                        Z_norm.transpose().reshape((self.n_lift, n_data),order=order), \
                                                                        Z_dot.transpose().reshape((self.n_lift,n_data),order=order), \
                                                                        U.transpose().reshape((self.m,n_data),order=order), \
                                                                        U_nom.transpose().reshape((self.m, n_data),order=order), \
                                                                        t.transpose().reshape((1,n_data),order=order)

        return X_flat, X_d_flat, Z_flat, Z_dot_flat, U_flat, U_nom_flat, t_flat

    def lift(self, X, X_d):
        """lift Lift the state using the basis function
        
        Arguments:
            X {numpy array [Ns,Nt]} -- state
            X_d {numpy array [Nt,Nt]} -- desired state
        
        Returns:
            numpy array [Nz,Nt] -- lifted state
        """
        Z = self.basis.lift(X, X_d)
        if not X.shape[1] == Z.shape[1]:
            Z = Z.transpose()

        if self.add_ones:
            one_vec = ones((1, Z.shape[1]))
            Z = concatenate((one_vec, Z), axis=0)
        if self.add_state:
            Z = concatenate((X, Z), axis=0)

        output_norm = divide(Z, self.Z_std)
        return output_norm.transpose()

    def predict(self,X, U):
        """predict compute the right hand side of z dot
        
        Arguments:
            X {numpy array [Ns,Nt]} -- state        
            U {numpy array [Nu,Nt]} -- control input
        
        Returns:
            numpy array [Ns,Nt] -- Az+Bu in z_dot = Az+Bu
        """
        return dot(self.C, dot(self.A,X) + dot(self.B, U))

    def tune_fit(self, X, X_d, Z, Z_dot, U=None, U_nom=None, l1_ratio=array([1])):

        # Construct EDMD matrices using Elastic Net L1 and L2 regularization
        if U is None and U_nom is None:
            input = Z.transpose()
        else:
            input = concatenate((Z.transpose(), U.transpose()), axis=1)
        output = Z_dot.transpose()

        reg_model_cv = linear_model.MultiTaskElasticNetCV(l1_ratio=l1_ratio, fit_intercept=False, normalize=False, cv=5, n_jobs=-1, selection='random')
        reg_model_cv.fit(input, output)

        self.A = reg_model_cv.coef_[:self.n_lift, :self.n_lift]
        if not (U is None and U_nom is None):
            self.B = reg_model_cv.coef_[:self.n_lift, self.n_lift:]

        if self.override_C:
            self.C = zeros((self.n, self.n_lift))
            self.C[:self.n, :self.n] = eye(self.n)
            self.C = multiply(self.C,self.Z_std.transpose())

        else:
            raise Exception('Warning: Learning of C not implemented for regularized regression.')

        self.l1 = reg_model_cv.alpha_
        self.l1_ratio = reg_model_cv.l1_ratio_

        print('EDMD l1: ', self.l1, self.l1_ratio)

    def discretize(self,dt):
        '''
        Discretizes continous dynamics
        '''
        self.dt = dt
        self.Bd = self.B*dt
        self.Ad = expm(self.A*self.dt)
