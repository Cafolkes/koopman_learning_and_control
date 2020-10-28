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
    def __init__(self, dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr,
                 const_offset=None, terminal_constraint=False, obstacle=None):
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
        self.obstacle = obstacle

        self.comp_time = []
        self.x_iter = []
        self.u_iter = []

    def construct_controller(self, z_init, u_init):
        z0 = z_init[0,:]
        A_lst = [np.ones((self.nx,self.nx)) for _ in range(self.N)]
        B_lst = [np.ones((self.nx, self.nu)) for _ in range(self.N)]
        r_lst = [np.ones(self.nx) for _ in range(self.N)]
        self.r_vec = np.array(r_lst).flatten()

        if self.obstacle is not None:
            D_lst = [np.ones((1, self.nx)) for _ in range(self.N)]
            h_lst = [np.ones(1) for _ in range(self.N)]
            self.h_vec = np.array(h_lst).flatten()
            self.construct_constraint_matrix_(A_lst, B_lst, D_lst=D_lst)
            self.construct_constraint_matrix_data_(A_lst, B_lst, D_lst=D_lst)
        else:
            self.construct_constraint_matrix_(A_lst, B_lst)
            self.construct_constraint_matrix_data_(A_lst, B_lst)

        self.construct_objective_(z_init, u_init)
        self.construct_constraint_vecs_(z0, None, z_init, u_init)

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
            x_init = (self.C @ z_init.T)
            u_init = self.cur_u.copy()

            # Update equality constraint matrices:
            A_lst, B_lst = self.update_dynamics_linearization_(z_init, u_init)
            if self.obstacle is None:
                self.update_constraint_matrix_data_(A_lst, B_lst)
            else:
                D_lst = self.update_obstacle_linearization_(z_init, u_init)
                self.update_constraint_matrix_data_(A_lst, B_lst, D_lst=D_lst)

            self.update_objective_(x_init, u_init)
            self.construct_constraint_vecs_(z, None, z_init, u_init)

            # Solve MPC Instance
            dz, du = self.solve_mpc_(z_init.flatten(), u_init.flatten())

            #alpha = min((iter+1)/5,1)
            alpha = 1
            self.cur_z = z_init + alpha*dz.T
            self.cur_u = u_init + alpha*du.T

            iter += 1
            self.comp_time.append(time.time()-t0)
            self.x_iter.append(self.cur_z.copy().T)
            self.u_iter.append(self.cur_u.copy().T)

    def solve_mpc_(self, z_init, u_init):
        self.prob.update(q=self._osqp_q, Ax=self._osqp_A_data, l=self._osqp_l, u=self._osqp_u)
        # TODO (comp time optimzation): Store flat z_init, u_init and update directly
        self.prob.warm_start(x=np.hstack((z_init, u_init)))
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

    def construct_constraint_matrix_(self, A_lst, B_lst, D_lst=None):
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

        if self.obstacle is None:
            self._osqp_A = sparse.vstack([Aeq, Aineq_u, Aineq_x], format='csc')
        else:
            # Obstacle constraints:
            Aineq_obs = sparse.hstack([sparse.block_diag(D_lst), sparse.csc_matrix((self.N, self.nx+self.N*self.nu))])
            self._osqp_A = sparse.vstack([Aeq, Aineq_u, Aineq_x, Aineq_obs], format='csc')

    def construct_constraint_vecs_(self, z, t, z_init, u_init):
        self.n_opt_z = self.nx * (self.N + 1)
        self.n_opt_z_u = self.n_opt_z + self.nu*self.N
        self.n_opt_z_u_x = self.n_opt_z_u + self.ns * (self.N + 1)

        dz0 = z - z_init[0, :]
        leq = np.hstack([-dz0, -self.r_vec])
        ueq = leq

        # Input constraints:
        u_init_flat = u_init.flatten()
        self.umin_tiled = np.tile(self.umin, self.N)
        self.umax_tiled = np.tile(self.umax, self.N)
        lineq_u = self.umin_tiled - u_init_flat
        uineq_u = self.umax_tiled - u_init_flat

        # State constraints:
        x_init_flat = (self.C @ z_init.T).flatten(order='F')
        self.xmin_tiled = np.tile(self.xmin, self.N+1)
        self.xmax_tiled = np.tile(self.xmax, self.N+1)
        lineq_x = self.xmin_tiled - x_init_flat
        uineq_x = self.xmax_tiled - x_init_flat

        if self.terminal_constraint:
            lineq_x[-self.ns:] = self.xr - self.C@z_init[-1,:]
            uineq_x[-self.ns:] = lineq_x[-self.ns:]

        if self.obstacle is None:
            self._osqp_l = np.hstack([leq, lineq_u, lineq_x])
            self._osqp_u = np.hstack([ueq, uineq_u, uineq_x])
        else:
            # Obstacle constraints:
            lineq_obs = -self.h_vec
            uineq_obs = np.inf*np.ones(self.N)
            self._osqp_l = np.hstack([leq, lineq_u, lineq_x, lineq_obs])
            self._osqp_u = np.hstack([ueq, uineq_u, uineq_x, uineq_obs])

    def update_constraint_vecs_(self, z, t, x_init_flat, z_init, u_init_flat):
        # Equality constraints:
        self._osqp_l[:self.nx] = -(z - z_init[0, :])
        self._osqp_l[self.nx:self.nx*(self.N+1)] = -self.r_vec

        self._osqp_u[:self.nx*(self.N+1)] = self._osqp_l[:self.nx*(self.N+1)]

        # Input constraints:
        self._osqp_l[self.n_opt_z:self.n_opt_z_u] = self.umin_tiled - u_init_flat
        self._osqp_u[self.n_opt_z:self.n_opt_z_u] = self.umax_tiled - u_init_flat

        # State constraints:
        self._osqp_l[self.n_opt_z_u:self.n_opt_z_u_x] = self.xmin_tiled - x_init_flat
        self._osqp_u[self.n_opt_z_u:self.n_opt_z_u_x] = self.xmax_tiled - x_init_flat

        if self.terminal_constraint:
            self._osqp_l[self.n_opt_z_u_x-self.ns:self.n_opt_z_u_x] = self.xr - x_init_flat[-self.ns:]
            self._osqp_u[self.n_opt_z_u_x-self.ns:self.n_opt_z_u_x] = self._osqp_l[self.n_opt_z_u_x-self.ns:self.n_opt_z_u_x]

        if self.obstacle is not None:
            self._osqp_l[self.n_opt_z_u_x:] = -self.h_vec

    def update_objective_(self, x_init, u_init):
        self._osqp_q = np.hstack(
            [(self.C.T @ self.Q @ (x_init[:,:-1] - self.xr.reshape(-1, 1))).flatten(order='F'),
             self.C.T @ self.QN @ (x_init[:,-1] - self.xr),
             (self.R @ u_init.T).flatten(order='F')])

    def construct_constraint_matrix_data_(self, A_lst, B_lst, D_lst=None):
        '''Manually build csc_matrix.data array'''
        C_data = [np.atleast_1d(self.C[np.nonzero(self.C[:, i]), i].squeeze()).tolist() for i in range(self.nx)]

        # State variables:
        data = []
        A_inds = []
        D_inds = []
        start_ind_A = 1
        for t in range(self.N):
            for i in range(self.nx):
                if self.obstacle is None:
                    data.append(np.hstack((-np.ones(1), A_lst[t][:,i], np.array(C_data[i]))))
                    A_inds.append(np.arange(start_ind_A, start_ind_A+self.nx))
                    start_ind_A += self.nx + 1 + len(C_data[i])
                else:
                    data.append(np.hstack((-np.ones(1), A_lst[t][:, i], np.array(C_data[i]), D_lst[t][:, i])))
                    A_inds.append(np.arange(start_ind_A, start_ind_A + self.nx))  #TODO: Currently only implemented for single obstacle constraint
                    D_inds.append(start_ind_A + self.nx + len(C_data[i]))
                    start_ind_A += self.nx + 1 + len(C_data[i]) + 1
        for i in range(self.nx):
            data.append(np.hstack((-np.ones(1), np.array(C_data[i]))))
            # TODO: Add terminal obstacle constraint.

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
        if self.obstacle is not None:
            self._osqp_A_data_D_inds = np.array(D_inds).flatten().tolist()

    def update_constraint_matrix_data_(self, A_lst, B_lst, D_lst=None):
        self._osqp_A_data[self._osqp_A_data_A_inds] = np.hstack(A_lst).flatten(order='F')
        self._osqp_A_data[self._osqp_A_data_B_inds] = np.hstack(B_lst).flatten(order='F')
        if self.obstacle is not None:
            self._osqp_A_data[self._osqp_A_data_D_inds] = np.hstack(D_lst).flatten(order='F')

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
        z_init, u_init = self.update_initial_guess_()
        x_init = (self.C @ z_init.T)
        z_init_flat = z_init.flatten()
        u_init_flat = u_init.flatten()
        x_init_flat = x_init.flatten(order='F')


        self.update_objective_(x_init, u_init)
        A_lst, B_lst = self.update_dynamics_linearization_(z_init, u_init)

        if self.obstacle is None:
            self.update_constraint_matrix_data_(A_lst, B_lst)
        else:
            D_lst = self.update_obstacle_linearization_(z_init, u_init)
            self.update_constraint_matrix_data_(A_lst, B_lst, D_lst=D_lst)

        self.update_constraint_vecs_(z, t, x_init_flat, z_init, u_init_flat)

        dz, du = self.solve_mpc_(z_init_flat, u_init_flat)
        self.cur_z = z_init + dz.T
        self.cur_u = u_init + du.T
        self.comp_time.append(time.time()-t0)

        return self.cur_u[0,:]

    def update_initial_guess_(self):
        x_last = self.cur_z[-1,:]
        u_last = self.cur_u[-1,:]
        x_new = self.dynamics_object.eval_dot(x_last, u_last, None)
        u_new = u_last

        # TODO: (Comp time optimization) Store x_init between time steps to avoid new memory allocation and index directly
        x_init = np.vstack((self.cur_z[1:,:], x_new))
        u_init = np.vstack((self.cur_u[1:,:], u_new))

        return x_init, u_init

    def update_dynamics_linearization_(self, z_init, u_init):
        lin_mdl_list = [self.dynamics_object.get_linearization(z,z_next,u,None) for z,z_next,u in zip(z_init[:-1,:], z_init[1:,:], u_init)]
        A_lst = [lin_mdl[0] for lin_mdl in lin_mdl_list]
        B_lst = [lin_mdl[1] for lin_mdl in lin_mdl_list]
        self.r_vec = np.array([lin_mdl[2] for lin_mdl in lin_mdl_list]).flatten()

        return A_lst, B_lst

    def update_obstacle_linearization_(self, z_init, u_init):
        lin_mdl_list = [self.obstacle(z) for z in z_init[:-1,:]]
        D_lst = [lin_mdl[0] for lin_mdl in lin_mdl_list]
        self.h_vec = np.array([lin_mdl[1] for lin_mdl in lin_mdl_list]).flatten()

        return D_lst

    def get_state_prediction(self):
        return self.cur_z

    def get_control_prediction(self):
        return self.cur_u
