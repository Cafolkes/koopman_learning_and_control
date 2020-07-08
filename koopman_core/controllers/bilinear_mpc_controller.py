import cvxpy as cvx
import osqp
import numpy as np
from scipy import sparse
import scipy as sp
from koopman_core.learning.utils import differentiate_vec

from core.controllers.controller import Controller

class BilinearMpcController(Controller):

    def __init__(self, n , m, k, n_lift, n_pred, fl_dynamics, bilinear_dynamics, C_x, C_h, xmin, xmax, umin, umax, Q, Qn, R, xr, tr, const_offset=0):

        super(BilinearMpcController, self).__init__(fl_dynamics)
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
        self.zd = np.array([self.bilinear_dynamics.phi_fun(x.reshape(1,-1)).squeeze() for x in self.xr.T])
        self.zd_dot = differentiate_vec(self.zd, self.tr)
        self.zd_ddot = differentiate_vec(self.zd_dot, self.tr)
        self.eta_z_d = np.zeros((2 * self.n_lift))
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

    def construct_controller_(self):
        #Dummy parameters:
        zd_mpc_hor = np.zeros((self.n_pred, self.n_lift))
        zd_dot_mpc_hor = np.zeros((self.n_pred, self.n_lift))
        zd_ddot_mpc_hor = np.zeros((self.n_pred, self.n_lift))

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(self.n_pred), self.Q), self.Qn,
                               sparse.kron(sparse.eye(self.n_pred), self.R)], format='csc')
        # - linear objective
        q = np.hstack([np.kron(np.ones(self.n_pred), -self.Q.dot(self.eta_z_d)), -self.Qn.dot(self.eta_z_d),
                       np.zeros(self.n_pred * self.m_mpc)])
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(self.n_pred + 1), -sparse.eye(self.n_mpc)) + sparse.kron(sparse.eye(self.n_pred + 1, k=-1), self.fl_dynamics.A)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.n_pred)), sparse.eye(self.n_pred)]), self.fl_dynamics.B)
        Aeq = sparse.hstack([Ax, Bu])

        self.eta_z_0  = np.zeros((2 * self.n_lift))
        leq = np.hstack([-self.eta_z_0 , np.zeros(self.n_pred * self.n_mpc)])
        ueq = leq

        # - input and state constraints
        C_x_ext = np.concatenate((self.C_x, np.zeros_like(self.C_x)), axis=1)
        Aineq_x = np.zeros((self.n * (self.n_pred + 1), self.n_mpc * (self.n_pred + 1) + self.n_pred * self.m_mpc))
        Aineq_x[:, :self.n_mpc * (self.n_pred + 1)] = np.kron(np.eye(self.n_pred + 1), C_x_ext)

        self.CFGumin = self.C_h @ self.bilinear_dynamics.F @ np.sum(
            np.array([self.bilinear_dynamics.G[ii] * self.umin[ii] for ii in range(self.m)]), axis=0)
        self.CFGumax = self.C_h @ self.bilinear_dynamics.F @ np.sum(
            np.array([self.bilinear_dynamics.G[ii] * self.umax[ii] for ii in range(self.m)]), axis=0)
        CFGumin_ext = np.concatenate((self.CFGumin, np.zeros_like(self.CFGumin)), axis=1)
        CFGumax_ext = np.concatenate((self.CFGumax, np.zeros_like(self.CFGumax)), axis=1)
        Aineq_nu = np.zeros((2 * self.m * (self.n_pred), self.n_mpc * (self.n_pred + 1) + self.n_pred * self.m_mpc))
        Aineq_nu[:self.m * self.n_pred, :self.n_mpc * self.n_pred] = np.kron(np.eye(self.n_pred), -CFGumin_ext)
        Aineq_nu[:self.m * self.n_pred, self.n_mpc * (self.n_pred + 1):] = np.kron(np.eye(self.n_pred), self.C_h)
        Aineq_nu[self.m * self.n_pred:, :self.n_mpc * self.n_pred] = np.kron(np.eye(self.n_pred), -CFGumax_ext)
        Aineq_nu[self.m * self.n_pred:, self.n_mpc * (self.n_pred + 1):] = np.kron(np.eye(self.n_pred), self.C_h)

        lineq_x = np.kron(np.ones(self.n_pred + 1), self.xmin)
        uineq_x = np.kron(np.ones(self.n_pred + 1), self.xmax)

        lineq_nu = np.array([self.C_h @ (-zd_ddot_mpc_hor[k,:] + self.bilinear_dynamics.F @ self.bilinear_dynamics.F @ zd_mpc_hor[k,:])
                             + self.CFGumin@zd_mpc_hor[k,:] for k in range(self.n_pred)]).flatten(order='C')
        uineq_nu = np.array([self.C_h @ (-zd_ddot_mpc_hor[k,:] + self.bilinear_dynamics.F @ self.bilinear_dynamics.F @ zd_mpc_hor[k,:])
                             + self.CFGumax @ zd_mpc_hor[k,:] for k in range(self.n_pred)]).flatten(order='C')

        Aineq = sparse.vstack([Aineq_x, Aineq_nu], format='csc')
        lineq = np.hstack([lineq_x, lineq_nu, -np.inf*np.ones_like(lineq_nu)])
        uineq = np.hstack([uineq_x, np.inf*np.ones_like(uineq_nu), uineq_nu])

        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        self.l = np.hstack([leq, lineq])
        self.u = np.hstack([ueq, uineq])
        self.n_eq = leq.size
        self.n_ineq = lineq.size
        # Create an OSQP object
        self.mpc_prob = osqp.OSQP()

        # Setup workspace
        self.mpc_prob.setup(P, q, A, self.l, self.u, warm_start=True, linsys_solver='mkl pardiso',
                            eps_abs=1e-1, eps_rel=1e-1, verbose=False)

    def update_mpc_(self, x, t):
        # Update desired trajectory:
        t_index = int(np.ceil((t - self.tr[0]) / self.dt))
        if t_index <= self.tr.size-self.n_pred:
            zd_mpc_hor = self.zd[t_index:t_index+self.n_pred,:]
            zd_dot_mpc_hor = self.zd_dot[t_index:t_index+self.n_pred,:]
            zd_ddot_mpc_hor = self.zd_ddot[t_index:t_index+self.n_pred,:]
        else:
            n_tiles = t_index+self.n_pred-self.tr.size
            zd_mpc_hor = np.concatenate((self.zd[t_index:, :], np.tile(self.zd[-1,:],(n_tiles,1))),axis=0)
            zd_dot_mpc_hor = np.concatenate((self.zd_dot[t_index:, :], np.tile(self.zd_dot[-1,:],(n_tiles,1))),axis=0)
            zd_ddot_mpc_hor = np.concatenate((self.zd_ddot[t_index:, :], np.tile(self.zd_ddot[-1,:],(n_tiles,1))),axis=0)

        # Update actuation constraints:
        lineq_nu = np.array([self.C_h @ (
                    -zd_ddot_mpc_hor[k, :] + self.bilinear_dynamics.F @ self.bilinear_dynamics.F @ zd_mpc_hor[k, :])
                             + self.CFGumin @ zd_mpc_hor[k, :] for k in range(self.n_pred)]).flatten(order='C')
        uineq_nu = np.array([self.C_h @ (
                    -zd_ddot_mpc_hor[k, :] + self.bilinear_dynamics.F @ self.bilinear_dynamics.F @ zd_mpc_hor[k, :])
                             + self.CFGumax @ zd_mpc_hor[k, :] for k in range(self.n_pred)]).flatten(order='C')

        self.l[self.n_eq + self.n*(self.n_pred+1):self.n_eq + self.n*(self.n_pred+1)+self.m*self.n_pred] = lineq_nu
        self.u[self.n_eq + self.n * (self.n_pred + 1) + self.m * self.n_pred:] = uineq_nu

        # Update initial condition:
        zd, zd_dot, zd_ddot = zd_mpc_hor[0,:], zd_dot_mpc_hor[0,:], zd_ddot_mpc_hor[0,:]
        z = self.bilinear_dynamics.phi_fun(x.reshape((1, -1))).squeeze()
        z_dot = self.bilinear_dynamics.eval_dot(z, self.u_prev, t)
        self.eta_z_0 = np.concatenate((z - zd, z_dot - zd_dot), axis=0)
        print(self.eta_z_0[2:4])

        self.l[:self.n_mpc] = -self.eta_z_0
        self.u[:self.n_mpc] = -self.eta_z_0

        self.mpc_prob.update(l=self.l, u=self.u)

        return z, zd, zd_dot, zd_ddot

    def eval(self, x, t):
        # Solve auxillary model predictive controller:
        z, zd, zd_dot, zd_ddot = self.update_mpc_(x, t)
        res = self.mpc_prob.solve()
        assert res.info.status == 'solved', 'MPC not solved to optimality'
        nu = res.x[-self.n_pred*self.m_mpc:-(self.n_pred-1)*self.m_mpc]
        self.comp_time.append(res.info.run_time)

        # Calculate feedback linearization matrices:
        F = self.bilinear_dynamics.F
        act = self.bilinear_dynamics.act(z, t)
        C = self.C_h

        u = np.linalg.solve(C@F@act, C@(zd_ddot - F@F@zd + nu)) + self.const_offset
        self.u_prev = u
        self.nu_prev = nu

        return u






    def construct_controller_old(self):
        # Precompute static matrices:
        try:
            F = self.bilinear_dynamics.F
            G = self.bilinear_dynamics.G
        except:
            F = self.bilinear_dynamics.A
            G = self.bilinear_dynamics.B
        G_umax = np.sum(np.array([G[ii] * self.umax[ii] for ii in range(self.m)]), axis=0)
        G_umin = np.sum(np.array([G[ii] * self.umin[ii] for ii in range(self.m)]), axis=0)

        # Construct cvx problem:
        nu = cvx.Variable((self.n_lift, self.n_pred))
        eta_z = cvx.Variable((int(2*self.n_lift), self.n_pred+1))
        self.eta_z_init = cvx.Parameter(int(2*self.n_lift))
        self.eta_z_d = cvx.Parameter(int(2*self.n_lift))
        self.zd = cvx.Parameter(self.n_lift)
        self.zd_dot = cvx.Parameter(self.n_lift)
        self.zd_ddot = cvx.Parameter(self.n_lift)

        objective = 0
        constraints = [eta_z[:,0] == self.eta_z_init]
        for k in range(self.n_pred):
            objective += cvx.quad_form(eta_z[:,k+1], self.Q) + cvx.quad_form(nu[:,k], self.R)

            constraints += [eta_z[:,k+1] == self.fl_dynamics.A @ eta_z[:,k] + self.fl_dynamics.B @ nu[:,k]]
            constraints += [self.xmin <= self.C_x@(eta_z[:self.n_lift,k] + self.eta_z_d[:self.n_lift]), #TODO: Reinsert
                            self.C_x@(eta_z[:self.n_lift,k] + self.eta_z_d[:self.n_lift]) <= self.xmax]
            constraints += [self.C_h@F@G_umin@(eta_z[:self.n_lift,k] + self.zd) + self.C_h@(-self.zd_ddot + F@F@self.zd) <= self.C_h@nu[:,k],
                            self.C_h@F@G_umax@(eta_z[:self.n_lift,k] + self.zd) + self.C_h@(-self.zd_ddot + F@F@self.zd) >= self.C_h@nu[:,k]]

        objective += cvx.quad_form(eta_z[:,self.n_pred], self.Q_n)
        objective += cvx.quad_form(self.C_h @ (nu[:,0] - self.nu_prev), 5e-1*np.eye(self.k))
        self.mpc_prob = cvx.Problem(cvx.Minimize(objective), constraints)

    def eval_old(self, x, t):
        # TODO: Add support for update of reference trajectory (time-varying)
        zd = self.bilinear_dynamics.phi_fun(self.set_pt.reshape((1,-1))).squeeze()  #TODO
        zd_dot = np.zeros(self.n_lift)  #TODO
        zd_ddot = np.zeros(self.n_lift)  #TODO
        z = self.bilinear_dynamics.phi_fun(x.reshape((1,-1))).squeeze()
        z_dot = self.bilinear_dynamics.eval_dot(z, self.u_prev, t)

        # Update all reference parameters:
        self.eta_z_0.value = np.concatenate((z-zd, z_dot-zd_dot), axis=0)
        self.eta_z_d.value = np.concatenate((zd, zd_dot), axis=0)
        self.zd.value = zd
        self.zd_dot.value = zd_dot
        self.zd_ddot.value = zd_ddot

        # Solve auxillary model predictive controller:
        res = self.mpc_prob.solve()
        assert res.info.status == 'optimal', 'MPC not solved to optimality'
        nu = self.mpc_prob.variables()[1].value[:,0]

        # Calculate feedback linearization matrices:
        F = self.bilinear_dynamics.F
        act = self.bilinear_dynamics.act(z, t)
        C = self.C_h

        u = np.linalg.solve(C@F@act, C@(zd_ddot - F@F@zd + nu)) + self.const_offset
        self.u_prev = u
        self.nu_prev = nu

        return u


