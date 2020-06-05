import cvxpy as cvx
import time

from core.controllers.controller import Controller

class LinearMpcController(Controller):

    def __init__(self, n, m, n_lift, n_pred, linear_dynamics, xmin, xmax, umin, umax, Q, Q_n, R, set_pt):

        super(LinearMpcController, self).__init__(linear_dynamics)
        self.n = n
        self.m = m
        self.n_lift = n_lift
        self.n_pred = n_pred
        self.linear_dynamics = linear_dynamics
        self.xmin = xmin
        self.xmax = xmax
        self.umin = umin
        self.umax = umax
        self.Q = Q
        self.Q_n = Q_n
        self.R = R
        self.set_pt = set_pt

        self.mpc_prob = None
        self.x_init = None
        self.comp_time = []

    def construct_controller(self):
        u = cvx.Variable((self.m, self.n_pred))
        x = cvx.Variable((self.n, self.n_pred+1))
        self.x_init = cvx.Parameter(self.n)
        objective = 0
        constraints = [x[:,0] == self.x_init]

        for k in range(self.n_pred):
            objective += cvx.quad_form(x[:,k] - self.set_pt, self.Q) + cvx.quad_form(u[:,k], self.R)
            constraints += [x[:,k+1] == self.linear_dynamics.A * x[:,k] + self.linear_dynamics.B * u[:,k]]
            constraints += [self.xmin <= x[:,k], x[:,k] <= self.xmax]
            constraints += [self.umin <= u[:,k], u[:,k] <= self.umax]

        objective += cvx.quad_form(x[:,self.n_pred] - self.set_pt, self.Q_n)
        self.mpc_prob = cvx.Problem(cvx.Minimize(objective), constraints)

    def eval(self, x, t):
        # TODO: Add support for update of reference trajectory

        self.x_init.value = x
        time_eval0 = time.time()
        self.mpc_prob.solve(solver=cvx.OSQP, warm_start=True)
        self.comp_time.append(time.time()-time_eval0)
        assert self.mpc_prob.status == 'optimal', 'MPC not solved to optimality'
        return self.mpc_prob.variables()[1].value[:,0]

