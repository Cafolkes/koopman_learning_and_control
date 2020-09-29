import cvxpy as cvx
import osqp
import numpy as np
from scipy import sparse
import scipy as sp
from koopman_core.learning.utils import differentiate_vec

from core.controllers.controller import Controller

class BilinearMpcControllerCVX(Controller):

    def __init__(self, n , m, k, n_lift, n_pred, fl_dynamics, bilinear_dynamics, C_x, C_h, xmin, xmax, umin, umax, Q, Qn, R, xr, tr, const_offset=0):

        super(BilinearMpcControllerCVX, self).__init__(fl_dynamics)
        self.n = n
        self.m = m
        self.k = k
        self.n_lift = n_lift
        self.n_pred = n_pred
        self.fl_dynamics = fl_dynamics
        self.bilinear_dynamics = bilinear_dynamics
        self.C_x = C_x
        self.C_h = C_h
        self.xmin = xmin
        self.xmax = xmax
        self.umin = umin
        self.umax = umax
        self.Q = Q
        self.Qn = Qn
        self.R = R
        self.xr = xr
        self.tr = tr
        self.const_offset = const_offset

        #MPC variables:
        self.n_mpc, self.m_mpc = self.fl_dynamics.B.shape
        self.mpc_prob = None
        self.u_prev = np.zeros(self.m)
        self.nu_prev = np.zeros(self.n_lift)  # TODO: Remove when cvxpy mpc is removed
        self.dt = self.tr[1] - self.tr[0]

        self.l = None
        self.u = None
        self.CFGumin = None
        self.CFGumax = None
        self.comp_time = []

        self.calculate_lifted_trajectories_()
        self.construct_controller_()

    def calculate_lifted_trajectories_(self):
        self.zd = np.array([self.bilinear_dynamics.phi_fun(x.reshape(1,-1)).squeeze() for x in self.xr.T]).T
        self.zd_dot = differentiate_vec(self.zd.T, self.tr).T
        self.zd_ddot = differentiate_vec(self.zd_dot.T, self.tr).T
        self.eta_zd = np.zeros((2 * self.n_lift))
        '''
        import matplotlib.pyplot as plt
        i = 3
        plt.figure()
        plt.plot(self.tr, self.zd_dot[:,i])
        plt.plot(self.tr, self.zd[:,i+3])
        plt.plot(self.tr, self.zd_ddot[:, i])
        plt.plot(self.tr, self.zd_dot[:, i + 3])
        plt.show()
        '''

    def update_mpc_(self, x, t):
        # Update desired trajectory:
        t_index = int(np.ceil((t - self.tr[0]) / self.dt))
        if t_index <= self.tr.size-self.n_pred:
            self.zd_cvx.value = self.zd[:,t_index:t_index+self.n_pred]
            #zd_dot_mpc_hor = self.zd_dot[t_index:t_index+self.n_pred,:]
            self.zd_ddot_cvx.value = self.zd_ddot[:,t_index:t_index+self.n_pred]
        else:
            n_tiles = t_index+self.n_pred-self.tr.size
            self.zd_cvx.value = np.concatenate((self.zd[:,t_index:], np.tile(self.zd[:,-1].reshape(-1,1),(1,n_tiles))), axis=1)
            #zd_dot_mpc_hor = np.concatenate((self.zd_dot[t_index:, :], np.tile(self.zd_dot[-1,:],(n_tiles,1))),axis=0)
            self.zd_ddot_cvx.value = np.concatenate((self.zd_ddot[:,t_index:], np.tile(self.zd[:,-1].reshape(-1,1),(1,n_tiles))), axis=1)

        # Update initial condition:
        zd, zd_dot, zd_ddot = self.zd[:,t_index], self.zd_dot[:,t_index], self.zd_ddot[:,t_index]
        z = self.bilinear_dynamics.phi_fun(x.reshape((1, -1))).squeeze()
        z_dot = self.bilinear_dynamics.eval_dot(z, self.u_prev, t)
        self.eta_z0_cvx.value = np.concatenate((z - zd, z_dot - zd_dot), axis=0)
        print(self.eta_z0_cvx.value[2:4])

        return z, zd, zd_dot, zd_ddot

    def construct_controller_old(self):
        # Precompute static matrices:
        try:
            F = self.bilinear_dynamics.F
            G = self.bilinear_dynamics.G
        except:
            F = self.bilinear_dynamics.A
            G = self.bilinear_dynamics.B
        G_umax = sp.sparse.csc_matrix(np.sum(np.array([G[ii] * self.umax[ii] for ii in range(self.m)]), axis=0))
        G_umin = sp.sparse.csc_matrix(np.sum(np.array([G[ii] * self.umin[ii] for ii in range(self.m)]), axis=0))

        # Construct cvx problem:
        nu = cvx.Variable((self.n_lift, self.n_pred))
        eta_z_cvx = cvx.Variable((int(2*self.n_lift), self.n_pred+1))
        self.eta_z0_cvx = cvx.Parameter(int(2*self.n_lift))
        self.eta_zd_cvx = cvx.Parameter(int(2*self.n_lift))
        self.zd_cvx = cvx.Parameter((self.n_lift, self.n_pred))
        self.zd_ddot_cvx = cvx.Parameter((self.n_lift, self.n_pred))

        self.eta_zd_cvx.value = self.eta_zd
        objective = 0
        constraints = [eta_z_cvx[:,0] == self.eta_z0_cvx]
        for k in range(self.n_pred):
            objective += cvx.quad_form(eta_z_cvx[:,k+1], self.Q) + cvx.quad_form(nu[:,k], self.R)

            constraints += [eta_z_cvx[:,k+1] == self.fl_dynamics.A @ eta_z_cvx[:,k] + self.fl_dynamics.B @ nu[:,k]]
            constraints += [self.xmin <= self.C_x@(eta_z_cvx[:self.n_lift,k] + self.eta_zd_cvx[:self.n_lift]),
                            self.C_x@(eta_z_cvx[:self.n_lift,k] + self.eta_zd_cvx[:self.n_lift]) <= self.xmax]
            constraints += [self.C_h@F@G_umin@(eta_z_cvx[:self.n_lift,k] + self.zd_cvx[:,k]) + self.C_h@(-self.zd_ddot_cvx[:,k] + F@F@self.zd_cvx[:,k]) <= self.C_h@nu[:,k],
                            self.C_h@F@G_umax@(eta_z_cvx[:self.n_lift,k] + self.zd_cvx[:,k]) + self.C_h@(-self.zd_ddot_cvx[:,k] + F@F@self.zd_cvx[:,k]) >= self.C_h@nu[:,k]]

        objective += cvx.quad_form(eta_z_cvx[:,self.n_pred], self.Qn)
        #objective += cvx.quad_form(self.C_h @ (nu[:,0] - self.nu_prev), 5e-1*np.eye(self.k))
        self.mpc_prob = cvx.Problem(cvx.Minimize(objective), constraints)

    def construct_controller_(self):
        # Precompute static matrices:
        Q = sp.sparse.csc_matrix(self.Q)
        Qn = sp.sparse.csc_matrix(self.Qn)
        R = sp.sparse.csc_matrix(self.R)
        try:
            F = sp.sparse.csc_matrix(self.bilinear_dynamics.F)
            G = self.bilinear_dynamics.G
        except:
            F = sp.sparse.csc_matrix(self.bilinear_dynamics.A)
            G = self.bilinear_dynamics.B
        G_umax = sp.sparse.csc_matrix(np.sum(np.array([G[ii] * self.umax[ii] for ii in range(self.m)]), axis=0))
        G_umin = sp.sparse.csc_matrix(np.sum(np.array([G[ii] * self.umin[ii] for ii in range(self.m)]), axis=0))
        A = sp.sparse.csc_matrix(self.fl_dynamics.A)
        B = sp.sparse.csc_matrix(self.fl_dynamics.B)
        C_h = sp.sparse.csc_matrix(self.C_h)
        C_x = sp.sparse.csc_matrix(self.C_x)

        # Construct cvx problem:
        nu = cvx.Variable((self.n_lift, self.n_pred))
        eta_z_cvx = cvx.Variable((int(2*self.n_lift), self.n_pred+1))
        self.eta_z0_cvx = cvx.Parameter(int(2*self.n_lift))
        self.eta_zd_cvx = cvx.Parameter(int(2*self.n_lift))
        self.zd_cvx = cvx.Parameter((self.n_lift, self.n_pred))
        self.zd_ddot_cvx = cvx.Parameter((self.n_lift, self.n_pred))

        self.eta_zd_cvx.value = self.eta_zd
        objective = 0
        constraints = [eta_z_cvx[:,0] == self.eta_z0_cvx]
        for k in range(self.n_pred):
            objective += cvx.quad_form(eta_z_cvx[:,k+1], Q) + cvx.quad_form(nu[:,k], R)

            constraints += [eta_z_cvx[:,k+1] == A @ eta_z_cvx[:,k] + B @ nu[:,k]]
            constraints += [self.xmin <= C_x@(eta_z_cvx[:self.n_lift,k] + self.eta_zd_cvx[:self.n_lift]),
                            C_x@(eta_z_cvx[:self.n_lift,k] + self.eta_zd_cvx[:self.n_lift]) <= self.xmax]
            constraints += [C_h@F@G_umin@(eta_z_cvx[:self.n_lift,k] + self.zd_cvx[:,k]) + C_h@(-self.zd_ddot_cvx[:,k] + F@F@self.zd_cvx[:,k]) <= C_h@nu[:,k],
                            C_h@F@G_umax@(eta_z_cvx[:self.n_lift,k] + self.zd_cvx[:,k]) + C_h@(-self.zd_ddot_cvx[:,k] + F@F@self.zd_cvx[:,k]) >= C_h@nu[:,k]]

        objective += cvx.quad_form(eta_z_cvx[:,self.n_pred], Qn)
        #objective += cvx.quad_form(self.C_h @ (nu[:,0] - self.nu_prev), 5e-1*np.eye(self.k))
        self.mpc_prob = cvx.Problem(cvx.Minimize(objective), constraints)


    def eval(self, x, t):
        # Solve auxillary model predictive controller:
        z, zd, zd_dot, zd_ddot = self.update_mpc_(x, t)
        self.mpc_prob.solve(solver=cvx.OSQP, verbose=False, warm_start=True) #, warm_start=True)
        assert self.mpc_prob.status == 'optimal', 'MPC not solved to optimality'
        nu = self.mpc_prob.variables()[1].value[:,0]
        self.comp_time.append(self.mpc_prob.solver_stats.solve_time)
        print('MPC solved. Runtime: ', self.comp_time[-1])

        # Calculate feedback linearization matrices:
        F = self.bilinear_dynamics.F
        act = self.bilinear_dynamics.act(z, t)
        C = self.C_h

        u = np.linalg.solve(C@F@act, C@(zd_ddot - F@F@zd + nu)) + self.const_offset
        self.u_prev = u
        self.nu_prev = nu

        return u

