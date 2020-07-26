import cvxpy as cp
import numpy as np
import time

from koopman_core.learning import BilinearEdmd


class FlBilinearLearner(BilinearEdmd):
    def __init__(self, n, m, basis, n_lift, n_traj, optimizer, C_h, cv=None, standardizer=None, C=None, first_obs_const=True, alpha=1e-16):
        super(FlBilinearLearner, self).__init__(n, m, basis, n_lift, n_traj, optimizer, cv=cv, standardizer=standardizer, C=C, first_obs_const=first_obs_const)
        self.C_h = C_h
        self.alpha=alpha

    def fit(self, X, y, cv=False, override_kinematics=False, l1_reg=0., equilibrium_pts=None):
        # Perform bEDMD to learn A:
        super(FlBilinearLearner, self).fit(X, y, cv=False, override_kinematics=override_kinematics)

        # Formulate SDP that learns Bs guaranteed to be feedback linearizable:
        if cv:
            assert self.cv is not None, 'No cross validation method specified.'
            X_sdp, y_sdp = self._process_sdp(X, y)
            if override_kinematics:
                y_sdp = y_sdp[:, int(self.n / 2) + int(self.first_obs_const):]

            self.cv.fit(X_sdp, y_sdp)
            l1_reg = self.cv.alpha_
            print('Bilinear tuned value: ', l1_reg)

        coefs = self._em_fit_sdp_constrained_cvxpy(X, y, self.A, l1_reg=l1_reg, equilibrium_pts=equilibrium_pts)

        # Store the learned Bs:
        if self.standardizer is not None:
            coefs = self.standardizer.transform(coefs)

        if override_kinematics:
            const_dyn = np.zeros((int(self.first_obs_const), self.n_lift))
            kin_dyn = np.concatenate((np.zeros((int(self.n / 2), int(self.n / 2) + int(self.first_obs_const))),
                                      np.eye(int(self.n / 2)),
                                      np.zeros((int(self.n / 2), self.n_lift - self.n - int(self.first_obs_const)))),
                                     axis=1)
            self.A = np.concatenate((const_dyn, kin_dyn, coefs[:, :self.n_lift]), axis=0)

            self.B = []
            for ii in range(self.m):
                self.B.append(np.concatenate((np.zeros((int(self.n/2)+int(self.first_obs_const), self.n_lift)),
                                     coefs[:, self.n_lift * (ii + 1):self.n_lift * (ii + 2)]), axis=0))
        else:
            pass

    def _process_sdp(self, X, y):
        X_sdp = X[:,self.n_lift:]
        y_sdp = y - X[:,:self.n_lift]@self.A.T  # y_sdp = y - Ax
        return X_sdp, y_sdp

    def _em_fit_sdp_constrained_cvxpy(self, X, y, A_init, n_iter=30, l1_reg=1e-2, equilibrium_pts=None):
        l = int(self.n/2) + int(self.first_obs_const)

        X_const_processed = [np.concatenate((np.concatenate((x,np.zeros_like(x)),axis=0).reshape(-1,1),
                                             np.concatenate((np.zeros_like(x),x),axis=0).reshape(-1,1)),axis=1)
                             for x in X[:, :self.n_lift]]

        # Define cvx problem to learn A-matrix:
        A_A = cp.Variable((A_init[l:, :].shape))
        B_A = cp.Parameter((y.shape[1] - l, self.m * self.n_lift))

        # Objective:
        cost_A = cp.norm(y[:, l:].T - (A_A@X[:, :self.n_lift].T + B_A@X[:, self.n_lift:].T), 'fro')
        for ii in range(y.shape[1]-l):
            cost_A += l1_reg * cp.norm(A_A[ii,:], p=1)

        # Positive singular values constraint:
        constraints_A = []
        for x in X_const_processed:
            constraints_A += [(self.C_h @ cp.vstack([self.A[:l,:], A_A]))[:,l:] @ B_A @ x + ((self.C_h @ cp.vstack([self.A[:l,:], A_A]))[:,l:] @ B_A @ x).T
                              - 2 * self.alpha * np.eye(self.m) >> 0]

        prob_A = cp.Problem(cp.Minimize(cost_A), constraints_A)

        # Define cvx problem to learn B-matrix:
        A_B = cp.Parameter((A_init[l:,:].shape))
        B_B = cp.Variable((y.shape[1] - l, self.m * self.n_lift))

        cost_B = cp.norm(y[:, l:].T - (A_B@X[:, :self.n_lift].T + B_B@X[:, self.n_lift:].T), 'fro')
        for ii in range(y.shape[1]-l):
            cost_B += l1_reg * cp.norm(B_B[ii, :], p=1)
        constraints_B = []
        for x in X_const_processed:
            constraints_B += [(self.C_h @ cp.vstack([self.A[:l,:], A_B]))[:,l:]@B_B@x + ((self.C_h @ cp.vstack([self.A[:l,:], A_B]))[:,l:]@B_B@x).T
                            - 2*self.alpha*np.eye(self.m) >> 0]
        prob_B = cp.Problem(cp.Minimize(cost_B), constraints_B)

        start_time = time.time()
        A_B.value = A_init[l:,:]
        prob_B.solve(solver=cp.MOSEK)

        norm_train = [cp.norm(y[:, l:].T - (A_B@X[:, :self.n_lift].T + B_B@X[:, self.n_lift:].T), p='fro').value]
        print('Iteration  0: Norm of the residual = ', "{:.2f}".format(norm_train[-1]) , ' comp. time = ', "{:.2f}".format(time.time()-start_time))

        for ii in range(n_iter):
            start_time = time.time()
            B_A.value = B_B.value
            prob_A.solve(solver=cp.MOSEK)

            A_B.value = A_A.value
            prob_B.solve(solver=cp.MOSEK)

            norm_train.append(cp.norm(y[:, l:].T - (A_B@X[:, :self.n_lift].T + B_B@X[:, self.n_lift:].T), p='fro').value)
            improvement = 1 - norm_train[-1]/norm_train[-2]
            print('Iteration ', ii + 1, ': Norm of the residual = ', "{:.2f}".format(norm_train[-1]), ' comp. time = ',
                  "{:.2f}".format(time.time() - start_time), ', iteration improvement: ', "{:.2f}".format(improvement*100),'%')
            if improvement < 5e-3:
                break

        return np.concatenate((A_B.value, B_B.value), axis=1)
 