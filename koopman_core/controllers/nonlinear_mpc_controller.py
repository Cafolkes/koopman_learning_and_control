import time

import numpy as np
import osqp
from scipy import sparse

from core.controllers.controller import Controller
from koopman_core.dynamics import BilinearLiftedDynamics


class NonlinearMPCController(Controller):
    """
    Class for nonlinear MPC with control-affine dynamics.

    Quadratic programs are solved using OSQP.
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
        self.xmin = xmin
        self.xmax = xmax
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
        self.x_init = self.C @ z_init.T
        self.u_init_flat = self.u_init.flatten()
        self.x_init_flat = self.x_init.flatten(order='F')
        # self.warm_start = np.zeros(self.nx*(self.N+1) + self.nu*self.N)

        A_lst = [np.ones((self.nx, self.nx)) for _ in range(self.N)]
        B_lst = [np.ones((self.nx, self.nu)) for _ in range(self.N)]
        r_lst = [np.ones(self.nx) for _ in range(self.N)]
        self.r_vec = np.array(r_lst).flatten()

        self.construct_objective_()
        self.construct_constraint_vecs_(z0, None)
        self.construct_constraint_matrix_(A_lst, B_lst)
        self.construct_constraint_matrix_data_(A_lst, B_lst)

        # Create an OSQP object and setup workspace
        self.prob = osqp.OSQP()
        self.prob.setup(self._osqp_P, self._osqp_q, self._osqp_A, self._osqp_l, self._osqp_u,
                        warm_start=True, verbose=False, polish=True, check_termination=25)

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
        self.prob.update_settings(warm_start=warm_start, check_termination=check_termination, max_iter=max_iter,
                                  polish=polish, linsys_solver=linsys_solver)

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
            self.x_init = (self.C @ self.z_init.T)
            self.u_init = self.cur_u.copy()

            # Update equality constraint matrices:
            A_lst, B_lst = self.update_linearization_()

            # Solve MPC Instance
            self.update_objective_()
            self.construct_constraint_vecs_(z, None)
            self.update_constraint_matrix_data_(A_lst, B_lst)

            self.solve_mpc_()
            dz = self.dz_flat.reshape(self.N + 1, self.nx)
            du = self.du_flat.reshape(self.N, self.nu)

            alpha = 1
            self.cur_z = self.z_init + alpha * dz
            self.cur_u = self.u_init + alpha * du
            self.u_init_flat = self.u_init_flat + alpha * self.du_flat

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
        self.update_objective_()
        A_lst, B_lst = self.update_linearization_()
        self.update_constraint_matrix_data_(A_lst, B_lst)
        self.update_constraint_vecs_(z, t)

        self.solve_mpc_()
        self.cur_z = self.z_init + self.dz_flat.reshape(self.N + 1, self.nx)
        self.cur_u = self.u_init + self.du_flat.reshape(self.N, self.nu)
        self.u_init_flat = self.u_init_flat + self.du_flat
        self.comp_time.append(time.time() - t0)

        return self.cur_u[0, :]

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

    def construct_objective_(self):
        """
        Construct MPC objective function
        :return:
        """
        # Quadratic objective:

        if not self.add_slack:
            self._osqp_P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.C.T @ self.Q @ self.C),
                                              self.C.T @ self.QN @ self.C,
                                              sparse.kron(sparse.eye(self.N), self.R)], format='csc')

        else:
            self._osqp_P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.C.T @ self.Q @ self.C),
                                              self.C.T @ self.QN @ self.C,
                                              sparse.kron(sparse.eye(self.N), self.R),
                                              self.Q_slack], format='csc')

        # Linear objective:
        if not self.add_slack:
            self._osqp_q = np.hstack(
                [(self.C.T @ self.Q @ (self.C @ self.z_init[:-1, :].T - self.xr.reshape(-1, 1))).flatten(order='F'),
                 self.C.T @ self.QN @ (self.C @ self.z_init[-1, :] - self.xr),
                 (self.R @ self.u_init.T).flatten(order='F')])

        else:
            self._osqp_q = np.hstack(
                [(self.C.T @ self.Q @ (self.C @ self.z_init[:-1, :].T - self.xr.reshape(-1, 1))).flatten(order='F'),
                 self.C.T @ self.QN @ (self.C @ self.z_init[-1, :] - self.xr),
                 (self.R @ self.u_init.T).flatten(order='F'),
                 np.zeros(self.ns * (self.N))])

    def update_objective_(self):
        """
        Construct MPC objective function
        :return:
        """
        # TODO: Change with direct memory allocation:
        self._osqp_q[:self.nx * (self.N + 1) + self.nu * self.N] = np.hstack(
            [(self.C.T @ self.Q @ (self.x_init[:, :-1] - self.xr.reshape(-1, 1))).flatten(order='F'),
             self.C.T @ self.QN @ (self.x_init[:, -1] - self.xr),
             (self.R @ self.u_init.T).flatten(order='F')])

    def construct_constraint_matrix_(self, A_lst, B_lst):
        """
        Construct MPC constraint matrix
        :param A_lst: (list(np.array)) List of dynamics matrices, A, for each timestep in the prediction horizon
        :param B_lst: (list(np.array)) List of dynamics matrices, B, for each timestep in the prediction horizon
        :return:
        """
        # Linear dynamics constraints:
        A_dyn = sparse.vstack((sparse.csc_matrix((self.nx, (self.N + 1) * self.nx)),
                               sparse.hstack(
                                   (sparse.block_diag(A_lst), sparse.csc_matrix((self.N * self.nx, self.nx))))))
        Ax = -sparse.eye((self.N + 1) * self.nx) + A_dyn
        Bu = sparse.vstack((sparse.csc_matrix((self.nx, self.N * self.nu)),
                            sparse.block_diag(B_lst)))

        if not self.add_slack:
            # Input constraints:
            Aineq_u = sparse.hstack(
                [sparse.csc_matrix((self.N * self.nu, (self.N + 1) * self.nx)), sparse.eye(self.N * self.nu)])

            # State constraints:
            Aineq_x = sparse.hstack([sparse.kron(sparse.eye(self.N + 1), self.C),
                                     sparse.csc_matrix(((self.N + 1) * self.ns, self.N * self.nu))])

            Aeq = sparse.hstack([Ax, Bu])
        else:
            # Input constraints:
            Aineq_u = sparse.hstack(
                [sparse.csc_matrix((self.N * self.nu, (self.N + 1) * self.nx)),
                 sparse.eye(self.N * self.nu),
                 sparse.csc_matrix((self.nu * self.N, self.ns * self.N))])

            # State constraints:
            Aineq_x = sparse.hstack([sparse.kron(sparse.eye(self.N + 1), self.C),
                                     sparse.csc_matrix(((self.N + 1) * self.ns, self.N * self.nu)),
                                     sparse.vstack([sparse.eye(self.ns * self.N),
                                                    sparse.csc_matrix((self.ns, self.ns * self.N))])])

            Aeq = sparse.hstack([Ax, Bu, sparse.csc_matrix((self.nx * (self.N + 1), self.ns * (self.N)))])

        self._osqp_A = sparse.vstack([Aeq, Aineq_u, Aineq_x], format='csc')

    def construct_constraint_matrix_data_(self, A_lst, B_lst):
        """
        Manually build csc_matrix.data array
        :param A_lst: (list(np.array)) List of dynamics matrices, A, for each timestep in the prediction horizon
        :param B_lst: (list(np.array)) List of dynamics matrices, B, for each timestep in the prediction horizon
        :return:
        """
        C_data = [np.atleast_1d(self.C[np.nonzero(self.C[:, i]), i].squeeze()).tolist() for i in range(self.nx)]

        # State variables:
        data = []
        A_inds = []
        start_ind_A = 1
        for t in range(self.N):
            for i in range(self.nx):
                data.append(np.hstack((-np.ones(1), A_lst[t][:, i], np.array(C_data[i]))))
                A_inds.append(np.arange(start_ind_A, start_ind_A + self.nx))
                start_ind_A += self.nx + 1 + len(C_data[i])

        for i in range(self.nx):
            data.append(np.hstack((-np.ones(1), np.array(C_data[i]))))

        # Input variables:
        B_inds = []
        start_ind_B = start_ind_A + self.nx + np.nonzero(self.C)[0].size - 1
        for t in range(self.N):
            for i in range(self.nu):
                data.append(np.hstack((B_lst[t][:, i], np.ones(1))))
                B_inds.append(np.arange(start_ind_B, start_ind_B + self.nx))
                start_ind_B += self.nx + 1

        # Slack variables:
        for t in range(self.N):
            for i in range(self.ns):
                data.append(np.ones(1))

        flat_data = []
        for arr in data:
            for d in arr:
                flat_data.append(d)

        self._osqp_A_data = np.array(flat_data)
        self._osqp_A_data_A_inds = np.array(A_inds).flatten().tolist()
        self._osqp_A_data_B_inds = np.array(B_inds).flatten().tolist()

    def update_constraint_matrix_data_(self, A_lst, B_lst):
        """
        Manually update csc_matrix.data array
        :param A_lst: (list(np.array)) List of dynamics matrices, A, for each timestep in the prediction horizon
        :param B_lst: (list(np.array)) List of dynamics matrices, B, for each timestep in the prediction horizon
        :return:
        """
        self._osqp_A_data[self._osqp_A_data_A_inds] = np.hstack(A_lst).flatten(order='F')
        self._osqp_A_data[self._osqp_A_data_B_inds] = np.hstack(B_lst).flatten(order='F')

    def construct_constraint_vecs_(self, z, t):
        """
        Construct MPC constraint vectors (lower and upper bounds)
        :param z: (np.array) Current state
        :param t: (float) Current time (for time-dependent dynamics)
        :return:
        """
        self.n_opt_x = self.nx * (self.N + 1)
        self.n_opt_x_u = self.nx * (self.N + 1) + self.nu * self.N

        dz0 = z - self.z_init[0, :]
        leq = np.hstack([-dz0, -self.r_vec])
        ueq = leq

        # Input constraints:
        u_init_flat = self.u_init.flatten()
        self.umin_tiled = np.tile(self.umin, self.N)
        self.umax_tiled = np.tile(self.umax, self.N)
        lineq_u = self.umin_tiled - u_init_flat
        uineq_u = self.umax_tiled - u_init_flat

        # State constraints:
        x_init_flat = self.x_init.flatten(order='F')
        self.xmin_tiled = np.tile(self.xmin, self.N + 1)
        self.xmax_tiled = np.tile(self.xmax, self.N + 1)
        lineq_x = self.xmin_tiled - x_init_flat
        uineq_x = self.xmax_tiled - x_init_flat

        if self.terminal_constraint:
            lineq_x[-self.ns:] = self.xr - self.x_init[:, -1]
            uineq_x[-self.ns:] = lineq_x[-self.ns:]

        self._osqp_l = np.hstack([leq, lineq_u, lineq_x])
        self._osqp_u = np.hstack([ueq, uineq_u, uineq_x])

    def update_constraint_vecs_(self, z, t):
        """
        Update MPC constraint vectors (lower and upper bounds)
        :param z: (np.array) Current state
        :param t: (float) Current time (for time-dependent dynamics)
        :return:
        """
        # Equality constraints:
        self._osqp_l[:self.nx] = -(z - self.z_init[0, :])
        self._osqp_l[self.nx:self.nx * (self.N + 1)] = -self.r_vec

        self._osqp_u[:self.nx * (self.N + 1)] = self._osqp_l[:self.nx * (self.N + 1)]

        # Input constraints:
        self._osqp_l[self.n_opt_x:self.n_opt_x_u] = self.umin_tiled - self.u_init_flat
        self._osqp_u[self.n_opt_x:self.n_opt_x_u] = self.umax_tiled - self.u_init_flat

        # State constraints:
        self._osqp_l[self.n_opt_x_u:] = self.xmin_tiled - self.x_init_flat
        self._osqp_u[self.n_opt_x_u:] = self.xmax_tiled - self.x_init_flat

        if self.terminal_constraint:
            self._osqp_l[-self.ns:] = self.xr - self.x_init_flat[-self.ns:]
            self._osqp_u[-self.ns:] = self._osqp_l[-self.ns:]

    def solve_mpc_(self):
        """
        Solve the MPC sub-problem
        :return:
        """
        self.prob.update(q=self._osqp_q, Ax=self._osqp_A_data, l=self._osqp_l, u=self._osqp_u)
        self.res = self.prob.solve()
        self.dz_flat = self.res.x[:(self.N + 1) * self.nx]
        self.du_flat = self.res.x[(self.N + 1) * self.nx:(self.N + 1) * self.nx + self.nu * self.N]

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
        self.u_init_flat[:-self.nu] = self.u_init_flat[self.nu:]
        self.u_init_flat[-self.nu:] = u_new

        self.x_init = self.C @ self.z_init.T
        self.x_init_flat = self.x_init.flatten(order='F')

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
        lin_mdl_list = [self.dynamics_object.get_linearization(z, z_next, u, None) for z, z_next, u in
                        zip(self.z_init[:-1, :], self.z_init[1:, :], self.u_init)]
        A_lst = [lin_mdl[0] for lin_mdl in lin_mdl_list]
        B_lst = [lin_mdl[1] for lin_mdl in lin_mdl_list]
        self.r_vec = np.array([lin_mdl[2] for lin_mdl in lin_mdl_list]).flatten()

        return A_lst, B_lst
