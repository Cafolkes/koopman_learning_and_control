import os
import sys
import time

import numpy as np
from hpipm_python import *
from hpipm_python.common import *
from scipy import sparse

from core.controllers.controller import Controller
from koopman_core.dynamics import BilinearLiftedDynamics

class NonlinearMPCControllerHPIPM(Controller):
    """
    Class for nonlinear MPC with control-affine dynamics.

    Quadratic programs are solved using HPIPM.
    """

    def __init__(self, dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, const_offset=None,
                 terminal_constraint=False, add_slack=False, q_slack=1e4):
        """
        Initialize the nonlinear mpc class.
        :param dynamics: (AffindeDynamics) dynamics object describing system dynamics
        :param N: (int) Prediction horizon in number of timesteps
        :param dt: (float) Time interval between time steps
        :param umin: (np.array) Actuation lower bounds
        :param umax: (np.array) Actuation upper bounds
        :param xmin: (np.array) State lower bounds
        :param xmax: (np.array) State upper bounds
        :param Q: (sparse.csc_matrix) State deviation penalty matrix
        :param R: (sparse.csc_matrix) Actuation penalty matrix
        :param QN: (sparse.csc_matrix) Terminal state deviation penalty matrix
        :param xr: (np.array) Desired state, setpoint
        :param const_offset: (np.array) Constant offset of the control inputs
        :param terminal_constraint: (boolean) Constrain terminal state to be xr
        :param add_slack: (boolean) Add slack variables to state constraints
        :param q_slack: (float) Penalty value of slack terms q||s||^2, where s are the slack variables
        """

        Controller.__init__(self, dynamics)
        self.dynamics_object = dynamics
        self.nx = self.dynamics_object.n
        self.nu = self.dynamics_object.m
        self.dt = dt
        if type(self.dynamics_object) == BilinearLiftedDynamics:
            self.C = self.dynamics_object.C
        else:
            self.C = np.eye(self.nx)
            self.dynamics_object.lift = lambda x, t: x

        self.Q = Q
        self.QN = QN
        self.R = R
        self.N = N

        if xmin.ndim == 1 and xmax.ndim == 1:
            self.xmin = xmin.reshape(1, -1)
            self.xmax = xmax.reshape(1, -1)
        else:
            self.xmin = xmin
            self.xmax = xmax

        if umin.ndim == 1 and umax.ndim == 1:
            self.umin = umin.reshape(1, -1)
            self.umax = umax.reshape(1, -1)
        else:
            self.umin = umin
            self.umax = umax

        if const_offset is None:
            self.const_offset = np.zeros(self.nu)
        else:
            self.const_offset = const_offset

        assert xr.ndim == 1, 'Desired trajectory not supported'
        self.xr = xr
        self.ns = xr.shape[0]
        self.terminal_constraint = terminal_constraint

        self.add_slack = add_slack
        self.Q_slack = q_slack * sparse.eye(self.ns * (self.N))

        self.setup_time = []
        self.qp_time = []
        self.comp_time = []
        self.x_iter = []
        self.u_iter = []

    def construct_controller(self, z_init, u_init):
        """
        Construct NMPC controller.
        :param z_init: (np.array) Initial guess of z-solution
        :param u_init: (np.array) Initial guess of u-solution
        :return:
        """
        z0 = z_init[0, :]
        self.z_init = z_init
        self.u_init = u_init
        self.x_init = self.z_init @ self.C.T

        dim = hpipm_ocp_qp_dim(self.N)
        dim.set('nx', self.ns, 0, self.N)
        dim.set('nu', self.nu, 0, self.N-1)
        dim.set('nbx', self.ns, 0, self.N)  # number of state bounds
        dim.set('nbu', self.nu, 0, self.N-1)

        # Prepare problem matrices:
        A_lst, B_lst, r_lst = self.update_linearization_()  # TODO: Consider storing in self
        self.construct_linear_objective_()
        self.construct_constraint_vecs_(z0, None)
        S = np.zeros((self.nu, self.ns))
        Ju = np.eye(self.nu)
        Jx = self.C

        # TODO: prepare all data matrices..
        self.qp = hpipm_ocp_qp(dim)

        # Set static problem data:
        self.qp.set('Q', self.Q, 0, self.N-1) #TODO: Verify that method works ->
        self.qp.set('Q', self.QN, self.N)
        self.qp.set('S', S, 0, self.N-1)
        self.qp.set('R', self.R, 0, self.N-1)
        self.qp.set('Ju', Ju, 0, self.N-1)
        self.qp.set('Jx', Jx, 0, self.N)
        # TODO: Implement slack variable penalty

        # Set dynamic problem data:
        self.update_qp_problem_(A_lst, B_lst, r_lst)
        self.qp_sol = hpipm_ocp_qp_sol(dim)

        # set up solver arg
        # TODO: Put parameters in param_dict...
        #mode = 'speed_abs'
        mode = 'speed'
        #mode = 'balance'
        #mode = 'robust'
        # create and set default arg based on mode
        arg = hpipm_ocp_qp_solver_arg(dim, mode)

        # create and set default arg based on mode
        #arg.set('mu0', 1e4)
        #arg.set('iter_max', 30)
        #arg.set('tol_stat', 1e-4)
        #arg.set('tol_eq', 1e-5)
        #arg.set('tol_ineq', 1e-5)
        #arg.set('tol_comp', 1e-5)
        #arg.set('reg_prim', 1e-12)
        #arg.set('warm_start', 1)

        # set up solver
        self.solver = hpipm_ocp_qp_solver(dim, arg)

    def update_qp_problem_(self, A_lst, B_lst, r_lst):
        self.qp.set('lbx', self._hpipm_dz0, 0)  # Initial state
        self.qp.set('ubx', self._hpipm_dz0, 0)  # Initial state
        self.qp.set('q', self._hpipm_qN, self.N)

        for ii in range(self.N):
            self.qp.set('A', A_lst[ii], ii)
            self.qp.set('B', B_lst[ii], ii)
            self.qp.set('b', r_lst[ii], ii)

            self.qp.set('q', self._hpipm_q[ii], ii)
            self.qp.set('r', self._hpipm_r[ii], ii)

            self.qp.set('lbu', self._hpipm_lbu[ii], ii)
            self.qp.set('ubu', self._hpipm_ubu[ii], ii)

            if ii > 0:
                self.qp.set('lbx', self._hpipm_lbx[ii], ii)
                self.qp.set('ubx', self._hpipm_ubx[ii], ii)

        self.qp.set('lbx', self._hpipm_lbx[-1], self.N)
        self.qp.set('ubx', self._hpipm_ubx[-1], self.N)

    def update_solver_settings(self, warm_start=True, check_termination=25, max_iter=4000, polish=True,
                               linsys_solver='qdldl'):
        """
        Update the OSQP solver settings (see OSQP documentation for detailed description of each setting)
        :param warm_start: (boolean) Warm start the solver with solution from previous timestep
        :param check_termination: (int) Frequency of checking wheter the solution has converged (number of iterations)
        :param max_iter: (int) Maximum iterations allowed by the solver
        :param polish: (boolean) Execute polish step at the end of solve
        :param linsys_solver: (string) Which linear system solver to use as part of OSQP algorithm
        :return:
        """
        pass
        #self.prob.update_settings(warm_start=warm_start, check_termination=check_termination, max_iter=max_iter,
        #                          polish=polish, linsys_solver=linsys_solver)

    def solve_to_convergence(self, z, t, z_init_0, u_init_0, eps=1e-3, max_iter=1):
        """
        Run SQP-algorithm to convergence
        :param z: (np.array) Initial value of z
        :param t: (float) Initial value of t (for time-dependent dynamics)
        :param z_init_0: (np.array) Initial guess of z-solution
        :param u_init_0: (np.array) Initial guess of u-solution
        :param eps: (float) Stop criterion, normed difference of the control input sequence
        :param max_iter: (int) Maximum SQP-iterations to run
        :return:
        """
        iter = 0
        self.cur_z = z_init_0
        self.cur_u = u_init_0
        u_prev = np.zeros_like(u_init_0)

        while (iter == 0 or np.linalg.norm(u_prev - self.cur_u) / np.linalg.norm(u_prev) > eps) and iter < max_iter:
            t0 = time.time()
            u_prev = self.cur_u.copy()
            self.z_init = self.cur_z.copy()
            self.x_init = self.z_init@self.C.T
            self.u_init = self.cur_u.copy()

            # Solve MPC Instance
            self.construct_linear_objective_()
            self.construct_constraint_vecs_(z, None)
            A_lst, B_lst, r_lst = self.update_linearization_()
            self.update_qp_problem_(A_lst, B_lst, r_lst)

            self.solve_mpc_()

            alpha = 1
            self.cur_z = self.z_init + alpha * self.dz
            self.cur_u = self.u_init + alpha * self.du

            iter += 1
            self.comp_time.append(time.time() - t0)
            self.x_iter.append(self.cur_z.copy().T)
            self.u_iter.append(self.cur_u.copy().T)

    def eval(self, x, t):
        """
        Run single iteration of SQP-algorithm to get control signal in closed-loop control
        :param x: (np.array) Current state
        :param t: (float) Current time (for time-dependent dynamics)
        :return: u: (np.array) Current control input
        """
        t0 = time.time()
        z = self.dynamics_object.lift(x.reshape((1, -1)), None).squeeze()
        self.update_initial_guess_()
        self.construct_linear_objective_()
        self.construct_constraint_vecs_(z, t)
        A_lst, B_lst, r_lst = self.update_linearization_()
        self.update_qp_problem_(A_lst, B_lst, r_lst)
        setup_time = time.time() - t0
        self.solve_mpc_()
        self.cur_z = self.z_init + self.dz
        self.cur_u = self.u_init + self.du

        self.comp_time.append(time.time() - t0)
        self.setup_time.append(setup_time)
        self.qp_time.append(self.comp_time[-1] - self.setup_time[-1])

        return self.cur_u[0, :]

    def construct_linear_objective_(self):
        """
        Construct MPC objective function
        :return:
        """
        self._hpipm_q = (self.C.T @ self.Q @ (self.C @ self.z_init[:-1, :].T - self.xr.reshape(-1, 1))).T#.flatten(order='F')
        self._hpipm_qN = self.C.T @ self.QN @ (self.C @ self.z_init[-1, :] - self.xr)
        self._hpipm_r = (self.R @ self.u_init.T).T#.flatten(order='F')  # TODO: Unflatten

    def construct_constraint_vecs_(self, z, t):
        """
        Construct MPC constraint vectors (lower and upper bounds)
        :param z: (np.array) Current state
        :param t: (float) Current time (for time-dependent dynamics)
        :return:
        """

        # Input constraints:
        self._hpipm_lbu = self.umin - self.u_init
        self._hpipm_ubu = self.umax - self.u_init

        # State constraints:
        self._hpipm_dz0 = z - self.z_init[0]
        self._hpipm_lbx = self.xmin - self.x_init
        self._hpipm_ubx = self.xmax - self.x_init

        if self.terminal_constraint:
            pass  #TODO: Implement terminal constraints
            #lineq_x[-self.ns:] = self.xr - self.x_init[:, -1]
            #uineq_x[-self.ns:] = lineq_x[-self.ns:]

    def solve_mpc_(self):
        """
        Solve the MPC sub-problem
        :return:
        """
        self.solver.solve(self.qp, self.qp_sol)
        self.dz = np.array(self.qp_sol.get('x', 0, self.N)).squeeze()
        self.du = np.array(self.qp_sol.get('u', 0, self.N-1)).squeeze()

    def update_initial_guess_(self):
        """
        Update the intial guess of the solution (z_init, u_init)
        :return:
        """
        z_last = self.cur_z[-1, :]
        u_new = self.cur_u[-1, :]
        z_new = self.dynamics_object.eval_dot(z_last, u_new, None)

        self.z_init[:-1, :] = self.cur_z[1:, :]
        self.z_init[-1, :] = z_new

        self.u_init[:-1, :] = self.cur_u[1:, :]
        self.u_init[-1, :] = u_new
        #self.u_init_flat[:-self.nu] = self.u_init_flat[self.nu:]
        #self.u_init_flat[-self.nu:] = u_new

        #self.x_init = self.C @ self.z_init.T
        self.x_init = self.z_init @ self.C.T
        #self.x_init_flat = self.x_init.flatten(order='F')

        # Warm start of OSQP:
        # du_new = self.du_flat[-self.nu:]
        # dz_last = self.dz_flat[-self.nx:]
        # dz_new = self.dynamics_object.eval_dot(dz_last, du_new, None)
        # self.warm_start[:self.nx*self.N] = self.dz_flat[self.nx:]
        # self.warm_start[self.nx*self.N:self.nx*(self.N+1)] = dz_new
        # self.warm_start[self.nx*(self.N+1):-self.nu] = self.du_flat[self.nu:]
        # self.warm_start[-self.nu:] = du_new

    def update_linearization_(self):
        """
        Update the linearization of the dyanmics around the initial guess
        :return: A_lst: (list(np.array)) List of dynamics matrices, A, for each timestep in the prediction horizon
                 B_lst: (list(np.array)) List of dynamics matrices, B, for each timestep in the prediction horizon
        """
        print(self.z_init.shape, self.u_init.shape)
        lin_mdl_list = [self.dynamics_object.get_linearization(z, z_next, u, None) for z, z_next, u in
                        zip(self.z_init[:-1, :], self.z_init[1:, :], self.u_init)]
        A_lst = [lin_mdl[0] for lin_mdl in lin_mdl_list]
        B_lst = [lin_mdl[1] for lin_mdl in lin_mdl_list]
        r_lst = [lin_mdl[2] for lin_mdl in lin_mdl_list]
        # TODO: Optimize, avoid 4x loop through all steps..

        return A_lst, B_lst, r_lst

    def get_state_prediction(self):
        """
        Get the state prediction from the MPC problem
        :return: Z (np.array) current state prediction
        """
        return self.cur_z

    def get_control_prediction(self):
        """
        Get the control prediction from the MPC problem
        :return: U (np.array) current control prediction
        """
        return self.cur_u