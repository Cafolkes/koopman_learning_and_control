from koopman_core.learning import BilinearEdmd
import cvxpy as cp
import numpy as np
import osqp
from scipy import sparse

class FlBilinearLearner(BilinearEdmd):
    def __init__(self, n, m, basis, n_lift, n_traj, optimizer, C_h, cv=None, standardizer=None, C=None, first_obs_const=True):
        super(FlBilinearLearner, self).__init__(n, m, basis, n_lift, n_traj, optimizer, cv=cv, standardizer=standardizer, C=C, first_obs_const=first_obs_const)
        self.C_h = C_h

    def fit(self, X, y, cv=False, override_kinematics=False, l1_reg=0.):
        if override_kinematics:
            y = y[:, int(self.n / 2) + int(self.first_obs_const):]

        # Perform bEDMD to learn A:
        super(FlBilinearLearner, self).fit(X, y, cv=cv, override_kinematics=override_kinematics)

        # Prepare data matrices for learning Bs:
        X_sdp, y_sdp = self._process_sdp(X, y)

        # Formulate SDP that learns Bs guaranteed to be feedback linearizable:
        #coefs = self._fit_sdp_cvxpy(X_sdp, y_sdp, l1_reg=l1_reg)
        coefs = self._fit_sdp_osqp(X_sdp, y_sdp, l1_reg=l1_reg)

        # Store the learned Bs:
        if self.standardizer is not None:
            coefs = self.standardizer.transform(np.concatenate((np.zeros((self.n_lift,self.n_lift)), coefs),axis=1))[:,self.n_lift:]

        self.B = []
        if override_kinematics:
            for ii in range(self.m):
                self.B.append(np.concatenate((np.zeros((int(self.n / 2) + int(self.first_obs_const), self.n_lift)),
                                              coefs[:, self.n_lift * (ii + 1):self.n_lift * (ii + 2)]), axis=0))
        else:
            for ii in range(self.m):
                self.B.append(coefs[:, self.n_lift * (ii + 1):self.n_lift * (ii + 2)])

    def fit_test(self, X, y, cv=False, override_kinematics=False, l1_reg=0.):
        if override_kinematics:
            y = y[:, int(self.n / 2) + int(self.first_obs_const):]

        # Formulate SDP that learns Bs guaranteed to be feedback linearizable:
        #coefs = self._fit_sdp_cvxpy(X_sdp, y_sdp, l1_reg=l1_reg)
        coefs = self._fit_sdp_osqp(X, y, l1_reg=l1_reg)

        if override_kinematics:
            const_dyn = np.zeros((int(self.first_obs_const), self.n_lift))
            kin_dyn = np.concatenate((np.zeros((int(self.n / 2), int(self.n / 2) + int(self.first_obs_const))),
                                      np.eye(int(self.n / 2)),
                                      np.zeros((int(self.n / 2), self.n_lift - self.n - int(self.first_obs_const)))),
                                     axis=1)
            self.A = np.concatenate((const_dyn, kin_dyn, coefs[:, :self.n_lift]), axis=0)

            for ii in range(self.m):
                self.B.append(np.concatenate((np.zeros((int(self.n / 2) + int(self.first_obs_const), self.n_lift)),
                                              coefs[:, self.n_lift * (ii + 1):self.n_lift * (ii + 2)]), axis=0))
        else:
            self.A = coefs[:, :self.n_lift]
            for ii in range(self.m):
                self.B.append(coefs[:, self.n_lift * (ii + 1):self.n_lift * (ii + 2)])

    def _process_sdp(self, X, y):
        X_sdp = X[:,self.n_lift:]
        y_sdp = y - X[:,:self.n_lift]@self.A.T  # y_sdp = y - Az
        return X_sdp, y_sdp

    def _fit_sdp_cvxpy(self, X, y, l1_reg=1e-2):
        #print(cp.installed_solvers())
        b_var = cp.Variable((self.m*self.n_lift, self.n_lift))
        cost = 0.
        #for ii in range(self.n_lift):
        #    cost += cp.sum_squares(X@b_var[ii,:] - y[:,ii])
        cost = cp.sum_squares(y - X@b_var)
        cost += l1_reg*cp.norm(b_var,p=1)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve(solver=cp.OSQP)

        print("\nThe optimal value is", prob.value)
        print("The norm of the residual is ", cp.norm(X@b_var - y, p=2).value)
        print("The setup time was ", prob.solver_stats.setup_time)
        print("The computation time was ", prob.solver_stats.solve_time)
        return b_var.value.T

    def _fit_sdp_osqp(self, X, y, l1_reg=1e-2):

        # OSQP data
        m = X.shape[0]
        n = X.shape[1]

        coefs, obj_val, comp_time = [], [], []

        # Auxiliary data
        In = sparse.eye(n)
        Im = sparse.eye(m)
        On = sparse.csc_matrix((n, n))
        Onm = sparse.csc_matrix((n, m))
        #A = sparse.hstack([X, -sparse.eye(m)])

        P = sparse.block_diag([On, sparse.eye(m), On], format='csc')
        q = np.hstack([np.zeros(n+m), l1_reg*np.ones(n)])
        A = sparse.vstack([sparse.hstack([X, -Im, Onm.T]),
                           sparse.hstack([In, Onm, -In]),
                           sparse.hstack([In, Onm, In])], format='csc')
        for y_feat in y.T:
            #P = sparse.block_diag([sparse.csc_matrix((n, n)), sparse.eye(m)], format='csc')
            #q = np.zeros(n + m)
            #b = sparse.csc_matrix(y_feat)
            #b = y_feat
            #A = sparse.vstack([
            #    sparse.hstack([Ad, -sparse.eye(m)]),
            #    sparse.hstack([sparse.eye(n), sparse.csc_matrix((n, m))])], format='csc')
            #l = y_feat
            #u = y_feat

            # OSQP data
            l = np.hstack([y_feat, -np.inf * np.ones(n), np.zeros(n)])
            u = np.hstack([y_feat, np.zeros(n), np.inf * np.ones(n)])

            # Create an OSQP object
            prob = osqp.OSQP()

            # Setup workspace
            prob.setup(P, q, A, l, u)

            # Solve problem
            res = prob.solve()

            # Store solver values
            #coefs.append(res.x[:self.n_lift*self.m]) #TODO: Reinsert after debug
            coefs.append(res.x[:self.n_lift * (self.m+1)])
            obj_val.append(res.info.obj_val)
            comp_time.append(res.info.solve_time)

        print('Total comp time: ', sum(comp_time))

        return np.array(coefs)