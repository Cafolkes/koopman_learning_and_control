import cvxpy as cp
import numpy as np
import time

from koopman_core.learning import BilinearEdmd


class FlBilinearLearner(BilinearEdmd):
    def __init__(self, n, m, basis, n_lift, n_traj, optimizer, C_h, cv=None, standardizer=None, C=None, first_obs_const=True, alpha=1e-10):
        super(FlBilinearLearner, self).__init__(n, m, basis, n_lift, n_traj, optimizer, cv=cv, standardizer=standardizer, C=C, first_obs_const=first_obs_const)
        self.C_h = C_h
        self.alpha=alpha

    def fit(self, X, y, cv=False, override_kinematics=False, l1_reg=0.):
        # Perform bEDMD to learn A:
        super(FlBilinearLearner, self).fit(X, y, cv=cv, override_kinematics=override_kinematics)

        # Prepare data matrices for learning Bs:
        X_sdp, y_sdp = self._process_sdp(X, y)
        if override_kinematics:
            y_sdp = y_sdp[:, int(self.n / 2) + int(self.first_obs_const):]

        # Formulate SDP that learns Bs guaranteed to be feedback linearizable:
        coefs = self._fit_sdp_constrained_cvxpy(X_sdp, y_sdp, X[:,:self.n_lift], l1_reg=l1_reg)

        # Store the learned Bs:
        if self.standardizer is not None:
            coefs = self.standardizer.transform(np.concatenate((np.zeros((coefs.shape[0],self.n_lift)), coefs),axis=1))[:,self.n_lift:]

        self.B = []
        if override_kinematics:
            for ii in range(self.m):
                self.B.append(np.concatenate((np.zeros((int(self.n / 2) + int(self.first_obs_const), self.n_lift)),
                                              coefs[:, self.n_lift * ii:self.n_lift * (ii + 1)]), axis=0))
        else:
            for ii in range(self.m):
                self.B.append(coefs[:, self.n_lift * (ii):self.n_lift * (ii + 1)])

    def _process_sdp(self, X, y):
        X_sdp = X[:,self.n_lift:]
        y_sdp = y - X[:,:self.n_lift]@self.A.T  # y_sdp = y - Ax
        return X_sdp, y_sdp

    def _fit_sdp_constrained_cvxpy(self, X, y, X_constraint, l1_reg=1e-2):
        print('Solving SDP...')
        start_time = time.time()
        l = int(self.n/2) + int(self.first_obs_const)
        X_const_processed = [np.concatenate((np.concatenate((x,np.zeros_like(x)),axis=0).reshape(-1,1),
                                             np.concatenate((np.zeros_like(x),x),axis=0).reshape(-1,1)),axis=1)
                             for x in X_constraint]

        # Define cvx problem
        B = cp.Variable((self.m*self.n_lift, y.shape[1]))

        cost = cp.norm(y - X@B,'fro')
        cost += l1_reg*cp.norm(B, p=1)
        constraints = []
        for x in X_const_processed:
            constraints += [self.C_h[:,l:]@self.A[l:,l:]@B.T@x + (self.C_h[:,l:]@self.A[l:,l:]@B.T@x).T
                            - 2*self.alpha*np.eye(self.m) >> 0]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        mosek_params = {'MSK_IPAR_NUM_THREADS': 8}
        prob.solve(solver=cp.MOSEK, mosek_params=mosek_params)

        print("\nThe optimal value is", prob.value)
        print("The norm of the residual is ", cp.norm(X@B - y, p=2).value)
        print("Total elapsed time: ", time.time()-start_time, '\n')
        return B.value.T
