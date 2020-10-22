import numpy as np
from scipy import sparse
import osqp
from core.controllers.controller import Controller
import time


class BilinearMPCController(Controller):
    """
    Class for linear MPC with lifted linear dynamics.

    Quadratic programs are solved using OSQP.
    """
    def __init__(self, lifted_bilinear_dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, const_offset=None):
        """__init__ Create an MPC controller
        
        Arguments:
            lifted_bilinear_dynamics {LinearLiftedDynamics} -- Lifted linear continuous-time dynamics
            N {integer} -- MPC prediction horizon, number of timesteps
            dt {float} -- time step in seconds
            umin {numpy array [Nu,]} -- minimum control bound
            umax {numpy array [Nu,]} -- maximum control bound
            xmin {numpy array [Ns,]} -- minimum state bound
            xmax {numpy array [Ns,]} -- maximum state bound
            Q {numpy array [Ns,Ns]} -- state cost matrix
            R {numpy array [Nu,Nu]} -- control cost matrix
            QN {numpy array [Ns,]} -- final state cost
            xr {numpy array [Ns,]} -- reference trajectory
        """

        Controller.__init__(self, lifted_bilinear_dynamics)
        self.dynamics_object = lifted_bilinear_dynamics
        self.nx = self.dynamics_object.n
        self.nu = self.dynamics_object.m
        self.dt = dt
        if self.dynamics_object.continuous:
            pass
        else:
            self.A = self.dynamics_object.A
            self.B = self.dynamics_object.B
            assert self.dt == self.dynamics_object.dt
        self.C = lifted_bilinear_dynamics.C

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

        assert xr.ndim==1, 'Desired trajectory not supported'
        self.xr = xr
        self.ns = xr.shape[0]

        self.comp_time = []
        self.iter_sol = []

    def construct_controller(self, z_init, u_init):
        z0 = z_init[0,:]
        #A_lst = [self.linearize_dynamics(z, z_next, u, None)[0] for z, z_next, u in zip(z_init[:-1, :], z_init[1:, :], u_init)]
        #B_lst = [self.linearize_dynamics(z, z_next, u, None)[1] for z, z_next, u in zip(z_init[:-1, :], z_init[1:, :], u_init)]
        #r_lst = [self.linearize_dynamics(z, z_next, u, None)[2] for z, z_next, u in zip(z_init[:-1, :], z_init[1:, :], u_init)]
        A_lst = [np.ones((self.nx,self.nx)) for _ in range(self.N)]
        B_lst = [np.ones((self.nx, self.nu)) for _ in range(self.N)]
        r_lst = [np.ones(self.nx) for _ in range(self.N)]

        self.construct_objective_(z_init, u_init)
        self.construct_constraint_vecs_(z0, None, z_init, u_init, r_lst)
        self.construct_constraint_matrix_(A_lst, B_lst)
        self.construct_constraint_matrix_data_(A_lst, B_lst)

        # Create an OSQP object and setup workspace
        self.prob = osqp.OSQP()
        self.prob.setup(self._osqp_P, self._osqp_q, self._osqp_A, self._osqp_l, self._osqp_u,
                        warm_start=True, verbose=False, polish=True, max_iter=1000)

    def linearize_dynamics(self, z0, z1, u0, t):
        A_lin = self.A + np.sum(np.array([b*u for b,u in zip(self.B, u0)]),axis=0)
        B_lin = np.array([b @ z0 for b in self.B]).T

        if z1 is None:
            z1 = A_lin@z0 + B_lin@u0

        f_d = self.dynamics_object.drift(z0, t) + np.dot(self.dynamics_object.act(z0,t),u0)
        r_lin = f_d - z1

        return A_lin, B_lin, r_lin

    def solve_to_convergence(self, z, t, z_init_0, u_init_0, eps=1e-3, max_iter=1):
        iter = 0
        self.cur_z = z_init_0
        self.cur_u = u_init_0
        u_prev = np.zeros_like(u_init_0)

        while (iter == 0 or np.linalg.norm(u_prev-self.cur_u) > eps) and iter < max_iter:
            u_prev = self.cur_u.copy()
            z_init = self.cur_z.copy()
            u_init = self.cur_u.copy()

            # Update equality constraint matrices:
            A_lst = [self.linearize_dynamics(z, z_next, u, None)[0] for z, z_next, u in zip(z_init[:-1,:], z_init[1:,:], u_init)]
            B_lst = [self.linearize_dynamics(z, z_next, u, None)[1] for z, z_next, u in zip(z_init[:-1,:], z_init[1:,:], u_init)]
            r_lst = [self.linearize_dynamics(z, z_next, u, None)[2] for z, z_next, u in zip(z_init[:-1,:], z_init[1:,:], u_init)]

            # Solve MPC Instance

            t0_c = time.time()
            self.update_objective_(z_init, u_init)
            self.construct_constraint_vecs_(z, None, z_init, u_init, r_lst)
            self.update_constraint_matrix_data_(A_lst, B_lst)
            tf_c = time.time() - t0_c

            t0_s = time.time()
            dz, du = self.solve_mpc_(z, t, z_init, u_init, A_lst, B_lst, r_lst)
            tf_s = time.time()-t0_s
            self.cur_z = z_init + dz.T
            self.cur_u = u_init + du.T

            iter += 1
            print('OSQP ', iter,': ', np.linalg.norm(u_prev-self.cur_u), 'Construct t: ', tf_c, 'Solve t: ', tf_s)

    def solve_mpc_(self, z, t, z_init, u_init, A_lst, B_lst, r_lst):
        self.prob.update(q=self._osqp_q, Ax=self._osqp_A_data, l=self._osqp_l, u=self._osqp_u)
        self.res = self.prob.solve()
        dz = self.res.x[:(self.N+1)*self.nx].reshape(self.nx,self.N+1, order='F')
        du = self.res.x[(self.N+1)*self.nx:].reshape(self.nu,self.N, order='F')

        return dz, du

    def construct_objective_(self, z_init, u_init):
        # Quadratic objective:
        self._osqp_P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.C.T @ self.Q @ self.C),
                                          self.C.T @ self.QN @ self.C,
                                          sparse.kron(sparse.eye(self.N), self.R)], format='csc')

        # Linear objective:
        self._osqp_q = np.hstack(
            [(self.C.T @ self.Q@(self.C@z_init[:-1,:].T-self.xr.reshape(-1,1))).flatten(order='F'),
             self.C.T @ self.QN@(self.C@z_init[-1,:] - self.xr),
             (self.R@u_init.T).flatten(order='F')])

    def construct_constraint_matrix_(self, A_lst, B_lst):
        # Linear dynamics constraints:
        A_dyn = sparse.vstack((sparse.csc_matrix((self.nx,(self.N+1)*self.nx)),
                               sparse.hstack((sparse.block_diag(A_lst), sparse.csc_matrix((self.N*self.nx,self.nx))))))
        Ax = -sparse.eye((self.N+1)*self.nx) + A_dyn
        Bu = sparse.vstack((sparse.csc_matrix((self.nx,self.N*self.nu)),
                            sparse.block_diag(B_lst)))
        Aeq = sparse.hstack([Ax, Bu])

        # Input constraints:
        Aineq_u = sparse.hstack([sparse.csc_matrix((self.N*self.nu,(self.N+1)*self.nx)), sparse.eye(self.N*self.nu)])

        # State constraints:
        Aineq_x = sparse.hstack([sparse.kron(sparse.eye(self.N+1),self.C), sparse.csc_matrix(((self.N+1)*self.ns, self.N*self.nu))])

        self._osqp_A = sparse.vstack([Aeq, Aineq_u, Aineq_x], format='csc')

    def construct_constraint_vecs_(self, z, t, z_init, u_init, r_lst):
        dz0 = z - z_init[0, :]
        r_vec = np.array(r_lst).flatten()
        leq = np.hstack([-dz0, -r_vec])
        ueq = leq

        # Input constraints:
        lineq_u = np.tile(self.umin, self.N) - u_init.flatten()
        uineq_u = np.tile(self.umax, self.N) - u_init.flatten()

        # State constraints:
        lineq_x = np.tile(self.xmin, self.N + 1) - (self.C @ z_init.T).flatten(order='F')
        uineq_x = np.tile(self.xmax, self.N + 1) - (self.C @ z_init.T).flatten(order='F')

        self._osqp_l = np.hstack([leq, lineq_u, lineq_x])
        self._osqp_u = np.hstack([ueq, uineq_u, uineq_x])

    def update_objective_(self, z_init, u_init):
        self._osqp_q = np.hstack(
            [(self.C.T @ self.Q @ (self.C @ z_init[:-1, :].T - self.xr.reshape(-1, 1))).flatten(order='F'),
             self.C.T @ self.QN @ (self.C @ z_init[-1, :] - self.xr),
             (self.R @ u_init.T).flatten(order='F')])

    def construct_constraint_matrix_data_(self, A_lst, B_lst):
        '''Manually build csc_matrix.data array'''
        C_data = [np.atleast_1d(self.C[np.nonzero(self.C[:, i]), i].squeeze()).tolist() for i in range(self.nx)]

        # State variables:
        data = []
        A_inds = []
        start_ind_A = 1
        for t in range(self.N):
            for i in range(self.nx):
                data.append(np.hstack((-np.ones(1), A_lst[t][:,i], np.array(C_data[i]))))
                A_inds.append(np.arange(start_ind_A, start_ind_A+self.nx))
                start_ind_A += self.nx + 1 + len(C_data[i])

        for i in range(self.nx):
            data.append(np.hstack((-np.ones(1), np.array(C_data[i]))))

        # Input variables:
        B_inds = []
        start_ind_B = start_ind_A + self.nx + np.nonzero(self.C)[0].size - 1
        for t in range(self.N):
            for i in range(self.nu):
                data.append(np.hstack((B_lst[t][:,i], np.ones(1))))
                B_inds.append(np.arange(start_ind_B, start_ind_B + self.nx))
                start_ind_B += self.nx + 1

        flat_data = []
        for arr in data:
            for d in arr:
                flat_data.append(d)

        self._osqp_A_data = np.array(flat_data)
        self._osqp_A_data_A_inds = np.array(A_inds).flatten().tolist()
        self._osqp_A_data_B_inds = np.array(B_inds).flatten().tolist()

    def update_constraint_matrix_data_(self, A_lst, B_lst):
        self._osqp_A_data[self._osqp_A_data_A_inds] = np.hstack(A_lst).flatten(order='F')
        self._osqp_A_data[self._osqp_A_data_B_inds] = np.hstack(B_lst).flatten(order='F')

    def eval(self, x, t):
        """eval Function to evaluate controller
        
        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time
        
        Returns:
            control action -- numpy array [Nu,]
        """
        pass

    def get_state_prediction(self):
        return self.cur_z

    def get_control_prediction(self):
        return self.cur_u
