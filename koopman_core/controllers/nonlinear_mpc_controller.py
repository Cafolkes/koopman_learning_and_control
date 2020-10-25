import numpy as np
from scipy import sparse
import osqp
from core.controllers.controller import Controller
import time
from koopman_core.dynamics import BilinearLiftedDynamics


class NonlinearMPCController(Controller):
    """
    Class for nonlinear MPC with control-affine dynamics.

    Quadratic programs are solved using OSQP.
    """
    def __init__(self, dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, const_offset=None, terminal_constraint=False):
        """__init__ Create an MPC controller
        
        Arguments:
            dynamics {AffineDynamics} -- Control-affine discrete-time dynamics
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

        assert xr.ndim==1, 'Desired trajectory not supported'
        self.xr = xr
        self.ns = xr.shape[0]
        self.terminal_constraint = terminal_constraint

        self.comp_time = []
        self.x_iter = []
        self.u_iter = []

    def construct_controller(self, z_init, u_init):
        z0 = z_init[0,:]
        A_lst = [np.ones((self.nx,self.nx)) for _ in range(self.N)]
        B_lst = [np.ones((self.nx, self.nu)) for _ in range(self.N)]
        r_lst = [np.ones(self.nx) for _ in range(self.N)]
        self.r_vec = np.array(r_lst).flatten()

        self.construct_objective_(z_init, u_init)
        self.construct_constraint_vecs_(z0, None, z_init, u_init)
        self.construct_constraint_matrix_(A_lst, B_lst)
        self.construct_constraint_matrix_data_(A_lst, B_lst)

        # Create an OSQP object and setup workspace
        self.prob = osqp.OSQP()
        self.prob.setup(self._osqp_P, self._osqp_q, self._osqp_A, self._osqp_l, self._osqp_u,
                        warm_start=True, verbose=False, polish=True)

    def solve_to_convergence(self, z, t, z_init_0, u_init_0, eps=1e-2, max_iter=1):
        iter = 0
        self.cur_z = z_init_0
        self.cur_u = u_init_0
        u_prev = np.zeros_like(u_init_0)

        while (iter == 0 or np.linalg.norm(u_prev-self.cur_u) > eps) and iter < max_iter:
            t0 = time.time()
            u_prev = self.cur_u.copy()
            z_init = self.cur_z.copy()
            u_init = self.cur_u.copy()

            # Update equality constraint matrices:
            A_lst = [self.dynamics_object.get_linearization(z, z_next, u, None)[0] for z, z_next, u in zip(z_init[:-1,:], z_init[1:,:], u_init)]
            B_lst = [self.dynamics_object.get_linearization(z, z_next, u, None)[1] for z, z_next, u in zip(z_init[:-1,:], z_init[1:,:], u_init)]
            r_lst = [self.dynamics_object.get_linearization(z, z_next, u, None)[2] for z, z_next, u in zip(z_init[:-1,:], z_init[1:,:], u_init)]
            self.r_vec = np.array(r_lst).flatten()

            # Solve MPC Instance
            self.update_objective_(z_init, u_init)
            self.construct_constraint_vecs_(z, None, z_init, u_init)
            self.update_constraint_matrix_data_(A_lst, B_lst)

            dz, du = self.solve_mpc_()

            #alpha = min((iter+1)/5,1)
            alpha = 1
            self.cur_z = z_init + alpha*dz.T
            self.cur_u = u_init + alpha*du.T

            iter += 1
            self.comp_time.append(time.time()-t0)
            self.x_iter.append(self.cur_z.copy().T)
            self.u_iter.append(self.cur_u.copy().T)

    def solve_mpc_(self):
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

    def construct_constraint_vecs_(self, z, t, z_init, u_init):
        # TODO (runtime optimization): Utilize the fact that z_init and u_init are shifted such that not every element of the vectors must be reconstructed at every timestep
        dz0 = z - z_init[0, :]
        leq = np.hstack([-dz0, -self.r_vec])
        ueq = leq

        # Input constraints:
        lineq_u = np.tile(self.umin, self.N) - u_init.flatten()
        uineq_u = np.tile(self.umax, self.N) - u_init.flatten()

        # State constraints:
        lineq_x = np.tile(self.xmin, self.N + 1) - (self.C @ z_init.T).flatten(order='F')
        uineq_x = np.tile(self.xmax, self.N + 1) - (self.C @ z_init.T).flatten(order='F')

        if self.terminal_constraint:
            lineq_x[-self.ns:] = self.xr - self.C@z_init[-1,:]
            uineq_x[-self.ns:] = lineq_x[-self.ns:]

        self._osqp_l = np.hstack([leq, lineq_u, lineq_x])
        self._osqp_u = np.hstack([ueq, uineq_u, uineq_x])

    def update_objective_(self, z_init, u_init):
        # TODO (runtime optimzation): Utilize the fact that z_init and u_init are shifted such that not every element of the vectors must be reconstructed at every timestep
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
        t0 = time.time()
        z = self.dynamics_object.lift(x.reshape((1, -1)), None).squeeze()
        x_init, u_init = self.update_initial_guess_()
        self.update_objective_(x_init, u_init)
        self.update_linearization_(x_init[-2,:], x_init[-1,:], u_init[-1,:])
        self.construct_constraint_vecs_(z, t, x_init, u_init)

        dz, du = self.solve_mpc_()
        self.cur_z = x_init + dz.T
        self.cur_u = u_init + du.T
        self.comp_time.append(time.time()-t0)

        return self.cur_u[0,:]

    def update_initial_guess_(self):
        x_last = self.cur_z[-1,:]
        u_last = self.cur_u[-1,:]
        x_new = self.dynamics_object.eval_dot(x_last, u_last, None)
        u_new = u_last

        x_init = np.vstack((self.cur_z[1:,:], x_new))
        u_init = np.vstack((self.cur_u[1:,:], u_new))

        return x_init, u_init

    def update_linearization_(self, x, x_next, u):
        # TODO (Runtime optimization): Rewrite get_linearization to output flat dynamics matrices (avoid numpy.flatten step)
        A_new, B_new, r_new = self.dynamics_object.get_linearization(x, x_next, u, None)

        self._osqp_A_data[self._osqp_A_data_A_inds[:-self.nx**2]] = self._osqp_A_data[self._osqp_A_data_A_inds[self.nx**2:]]
        self._osqp_A_data[self._osqp_A_data_A_inds[-self.nx**2:]] = A_new.flatten(order='F')

        self._osqp_A_data[self._osqp_A_data_B_inds[:-self.nx*self.nu]] = self._osqp_A_data[self._osqp_A_data_B_inds[self.nx*self.nu:]]
        self._osqp_A_data[self._osqp_A_data_B_inds[-self.nx * self.nu:]] = B_new.flatten(order='F')

        self.r_vec[:-self.nx] = self.r_vec[self.nx:]
        self.r_vec[-self.nx:] = r_new

    def get_state_prediction(self):
        return self.cur_z

    def get_control_prediction(self):
        return self.cur_u
