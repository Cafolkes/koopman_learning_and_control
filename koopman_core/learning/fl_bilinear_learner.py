import cvxpy as cp
import numpy as np
import time

from koopman_core.learning import BilinearEdmd


class FlBilinearLearner(BilinearEdmd):
    def __init__(self, n, m, basis, n_lift, n_traj, optimizer, C_h, cv=None, standardizer=None, C=None, first_obs_const=True, alpha=1e-6):
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

    def _cv_sdp_constrained_cvxpy(self, X, y, X_constraint, l1_reg=1e-2):
        print('Solving SDP...')
        l = int(self.n/2) + int(self.first_obs_const)
        X_const_processed = [np.concatenate((np.concatenate((x,np.zeros_like(x)),axis=0).reshape(-1,1),
                                             np.concatenate((np.zeros_like(x),x),axis=0).reshape(-1,1)),axis=1)
                             for x in X_constraint]

        # Define cvx problem
        B = cp.Variable((self.m*self.n_lift, y.shape[1]))
        gamma = cp.Parameter(nonneg=True)

        import sklearn as sk
        import matplotlib.pyplot as plt

        train_loss, test_loss = [], []
        k_fold = sk.model_selection.KFold(n_splits=3, shuffle=True)
        for train_index, test_index in k_fold.split(X):
            X_train, y_train, X_const_train = X[train_index,:], y[train_index,:], [X_const_processed[ii] for ii in train_index]
            X_test, y_test = X[test_index, :], y[test_index, :]

            cost = cp.norm(y_train - X_train @ B, 'fro')
            cost += gamma * cp.norm(B, p=1)
            constraints = []
            for x in X_const_train:
                constraints += [self.C_h[:,l:]@self.A[l:,l:]@B.T@x + (self.C_h[:,l:]@self.A[l:,l:]@B.T@x).T
                                - 2*self.alpha*np.eye(self.m) >> 0]
            prob = cp.Problem(cp.Minimize(cost), constraints)
            mosek_params = {'MSK_IPAR_NUM_THREADS': 8}

            train_loss.append([])
            test_loss.append([])
            for g_val in self.cv_values:
                gamma.value = g_val
                start_time = time.time()
                prob.solve(solver=cp.MOSEK, mosek_params=mosek_params)
                train_loss[-1].append(cp.norm(X_train@B - y_train, p=2).value/train_index.size)
                test_loss[-1].append(cp.norm(X_test @ B - y_test, p=2).value/test_index.size)
                print("Solve time gamma=",g_val,": ", time.time() - start_time, '\n')

        train_loss, test_loss = np.array(train_loss), np.array(test_loss)
        train_loss_mean, test_loss_mean = np.mean(train_loss,axis=0), np.mean(test_loss, axis=0)
        plot_cv = True
        if plot_cv:
            plt.figure()
            plt.plot(self.cv_values, train_loss_mean, label='Train loss')
            plt.plot(self.cv_values, test_loss_mean, label='Test loss')
            plt.xlabel('$\\gamma$ (regularization strength)')
            plt.ylabel('Loss (root MSE)')
            plt.title('Cross validation (3 folds) mean training and test loss')
            plt.legend()

        return self.cv_values[np.argmin(test_loss_mean)]

    def _em_fit_sdp_constrained_cvxpy(self, X, y, A_init, n_iter=10, l1_reg=1e-2, equilibrium_pts=None):
        print('Solving SDP...')
        l = int(self.n/2) + int(self.first_obs_const)
        X_const_processed = [np.concatenate((np.concatenate((x,np.zeros_like(x)),axis=0).reshape(-1,1),
                                             np.concatenate((np.zeros_like(x),x),axis=0).reshape(-1,1)),axis=1)
                             for x in X[:, :self.n_lift]]

        # Define cvx problem to learn A-matrix:
        A_A = cp.Variable((A_init[l:, :].shape))
        B_A = cp.Parameter((self.m * self.n_lift, y.shape[1] - l))

        # Objective:
        cost_A = cp.norm(y[:, l:] - (X[:, :self.n_lift] @ A_A.T + X[:, self.n_lift:] @ B_A), 'fro')
        cost_A += l1_reg * cp.norm(A_A, p=1)

        # Positive singular values constraint:
        constraints_A = []
        for x in X_const_processed:
            constraints_A += [(self.C_h @ cp.vstack([self.A[:l,:], A_A]))[:,l:] @ B_A.T @ x + ((self.C_h @ cp.vstack([self.A[:l,:], A_A]))[:,l:] @ B_A.T @ x).T
                              - 2 * self.alpha * np.eye(self.m) >> 0]

        # Equilibrium point constraint:
        for x in equilibrium_pts:
            constraints_A += [cp.vstack([self.A[:l,:], A_A])@x == np.zeros_like(x)]

        prob_A = cp.Problem(cp.Minimize(cost_A), constraints_A)

        # Define cvx problem to learn B-matrix:
        A_B = cp.Parameter((A_init[l:,:].shape))
        #A_B = cp.Variable((A_init[l:, :].shape)) #TODO: Remove
        B_B = cp.Variable((self.m*self.n_lift, y.shape[1]-l))

        cost_B = cp.norm(y[:,l:] - (X[:,:self.n_lift]@A_B.T + X[:,self.n_lift:]@B_B),'fro')
        cost_B += l1_reg*cp.norm(B_B, p=1)
        #cost_B += l1_reg * cp.norm(cp.hstack([A_B, B_B.T]), p=1)
        constraints_B = []
        for x in X_const_processed:
            constraints_B += [(self.C_h @ cp.vstack([self.A[:l,:], A_B]))[:,l:]@B_B.T@x + ((self.C_h @ cp.vstack([self.A[:l,:], A_B]))[:,l:]@B_B.T@x).T
                            - 2*self.alpha*np.eye(self.m) >> 0]
        prob_B = cp.Problem(cp.Minimize(cost_B), constraints_B)
        mosek_params = {'MSK_IPAR_NUM_THREADS': 8}

        start_time = time.time()
        A_B.value = A_init[l:,:]
        prob_B.solve(solver=cp.MOSEK, mosek_params=mosek_params)

        norm_train = [cp.norm(y[:,l:] - (X[:,:self.n_lift]@A_B.T + X[:,self.n_lift:]@B_B), p=2).value]
        print("The norm of the residual is ", norm_train[-1])
        print("Total elapsed time: ", time.time()-start_time, '\n')

        for ii in range(n_iter):
            start_time = time.time()
            B_A.value = B_B.value
            prob_A.solve(solver=cp.MOSEK, mosek_params=mosek_params)

            A_B.value = A_A.value
            prob_B.solve(solver=cp.MOSEK, mosek_params=mosek_params)

            norm_train.append(cp.norm(y[:, l:] - (X[:, :self.n_lift] @ A_B.T + X[:, self.n_lift:] @ B_B), p=2).value)
            print("\nThe norm of the residual is ", norm_train[-1])
            print("Total elapsed time: ", time.time() - start_time)

            improvement = 1 - norm_train[-1]/norm_train[-2]
            print('Iteration improvement: ', improvement, ', total improvement: ', 1 - norm_train[-1]/norm_train[0])
            if improvement < 1e-3:
                break

        return np.concatenate((A_B.value, B_B.value.T), axis=1)
