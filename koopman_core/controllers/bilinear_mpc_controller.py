import cvxpy as cvx
import numpy as np

from core.controllers.controller import Controller

class BilinearMpcController(Controller):

    def __init__(self, n, m, k, n_lift, n_pred, fl_dynamics, bilinear_dynamics, xmin, xmax, umin, umax, Q, Q_n, R, set_pt):

        super(BilinearMpcController, self).__init__(fl_dynamics)
        self.n = n
        self.m = m
        self.k = k
        self.n_lift = n_lift
        self.n_pred = n_pred
        self.fl_dynamics = fl_dynamics
        self.bilinear_dynamics = bilinear_dynamics
        self.xmin = xmin
        self.xmax = xmax
        self.umin = umin
        self.umax = umax
        self.Q = Q
        self.Q_n = Q_n
        self.R = R
        self.set_pt = set_pt

        self.mpc_prob = None
        self.eta_init = None
        self.eta_d = None
        self.zd = None
        self.zd_dot = None
        self.zd_ddot = None
        self.u_prev = np.zeros(self.m)

    def construct_controller(self):
        # Precompute static matrices:
        F = self.bilinear_dynamics.F
        G = self.bilinear_dynamics.G
        C = self.bilinear_dynamics.Cx
        C_stacked = np.zeros((int(2*self.k), int(2*self.n_lift)))
        C_stacked[:self.k, :self.n_lift] = C
        C_stacked[self.k:, self.n_lift:] = C
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
            objective += cvx.quad_form(eta_z[:,k], self.Q) + cvx.quad_form(nu[:,k], self.R)
            constraints += [eta_z[:,k+1] == self.fl_dynamics.A * eta_z[:,k] + self.fl_dynamics.B * nu[:,k]]
            constraints += [self.xmin <= C_stacked@(eta_z[:,k] + self.eta_z_d),
                            C_stacked@(eta_z[:,k] + self.eta_z_d) <= self.xmax]
            constraints += [C@F@G_umin@(eta_z[:self.n_lift,k] + self.zd) + C@(-self.zd_ddot + F@F@self.zd) <= C@nu[:,k],
                            C@F@G_umax@(eta_z[:self.n_lift,k] + self.zd) + C@(-self.zd_ddot + F@F@self.zd) >= C@nu[:,k]]

        objective += cvx.quad_form(eta_z[:,self.n_pred], self.Q_n)
        self.mpc_prob = cvx.Problem(cvx.Minimize(objective), constraints)

    def eval(self, x, t):
        # TODO: Add support for update of reference trajectory (time-varying)
        zd = self.bilinear_dynamics.phi_fun(self.set_pt)  #TODO
        zd_dot = np.zeros(self.n_lift)  #TODO
        zd_ddot = np.zeros(self.n_lift)  #TODO
        z = self.bilinear_dynamics.phi_fun(x)
        z_dot = self.bilinear_dynamics.eval_dot(z, self.u_prev, t)

        # Update all reference parameters:
        self.eta_z_init.value = np.concatenate((z-zd, z_dot-zd_dot), axis=0)
        self.eta_z_d.value = np.concatenate((zd, zd_dot), axis=0)
        self.zd.value = zd
        self.zd_dot.value = zd_dot
        self.zd_ddot.value = zd_ddot

        # Solve auxillary model predictive controller:
        self.mpc_prob.solve(solver=cvx.OSQP, warm_start=True)
        assert self.mpc_prob.status == 'optimal', 'MPC not solved to optimality'
        nu = self.mpc_prob.variables()[1].value[:,0]

        # Calculate feedback linearization matrices:
        F = self.bilinear_dynamics.F
        act = self.bilinear_dynamics.act(z, t)
        C = self.bilinear_dynamics.Cx

        u = np.linalg.solve(C@F@act, C@(zd_ddot - F@F@zd + nu))
        self.u_prev = u

        return u

