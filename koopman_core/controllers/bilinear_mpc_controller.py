from koopman_core.controllers import NonlinearMPCController
import numpy as np
import time
from scipy import sparse

class BilinearMPCController(NonlinearMPCController):
    """
    Class for bilinear MPC.
    Quadratic programs are solved using OSQP.
    """

    def __init__(self, dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, const_offset=None,
                 terminal_constraint=False):

        NonlinearMPCController.__init__(self, dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr,
                                        const_offset=const_offset,
                                        terminal_constraint=terminal_constraint)

        self.A_flat = self.dynamics_object.A.flatten(order='F')
        self.B_flat = np.array([b.flatten(order='F') for b in self.dynamics_object.B])
        self.B_arr = np.vstack(self.dynamics_object.B).T

    def update_constraint_matrix_data_(self, A_lst, B_lst):
        self._osqp_A_data[self._osqp_A_data_A_inds] = A_lst
        self._osqp_A_data[self._osqp_A_data_B_inds] = B_lst

    def update_dynamics_linearization_(self, z_init, u_init):
        A_lst = u_init@self.B_flat + self.A_flat
        A_lst_flat = A_lst.flatten()
        B_lst_flat = (z_init[:-1,:]@self.B_arr).flatten()

        #TODO: (Comp time optimzation) Try to further improve calculation of r_vec
        A_reshaped = A_lst.reshape(self.nx*self.N,self.nx)
        self.r_vec[:] = (np.array([z_init[i,:]@A_reshaped[i*self.nx:(i+1)*self.nx,:]
                                  for i in range(self.N)]) - z_init[1:,:]).flatten()

        return A_lst_flat, B_lst_flat

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

        if self.dynamics_object.C_h is None:
            self._osqp_A = sparse.vstack([Aeq, Aineq_u, Aineq_x], format='csc')
        else:
            # Obstacle constraints:
            Aineq_obs = sparse.hstack([sparse.kron(sparse.eye(self.N+1), self.dynamics_object.C_h),
                                     sparse.csc_matrix(((self.N+1)*self.dynamics_object.C_h.shape[0], self.N*self.nu))])
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

        if self.dynamics_object.C_h is None:
            self._osqp_l = np.hstack([leq, lineq_u, lineq_x])
            self._osqp_u = np.hstack([ueq, uineq_u, uineq_x])
        else:
            # Obstacle constraints:
            lineq_obs = np.zeros(self.N+1)
            uineq_obs = np.inf*np.ones(self.N+1)
            self._osqp_l = np.hstack([leq, lineq_u, lineq_x, lineq_obs])
            self._osqp_u = np.hstack([ueq, uineq_u, uineq_x, uineq_obs])

    def construct_constraint_matrix_data_(self, A_lst, B_lst, D_lst=None):
            '''Manually build csc_matrix.data array'''
            C_data = [np.atleast_1d(self.C[np.nonzero(self.C[:, i]), i].squeeze()).tolist() for i in range(self.nx)]
            if self.dynamics_object.C_h is not None:
                C_h_data = [np.atleast_1d(self.dynamics_object.C_h[np.nonzero(self.dynamics_object.C_h[:, i]), i].squeeze()).tolist() for i in range(self.nx)]

            # State variables:
            data = []
            A_inds = []
            start_ind_A = 1
            for t in range(self.N):
                for i in range(self.nx):
                    if self.dynamics_object.C_h is None:
                        data.append(np.hstack((-np.ones(1), A_lst[t][:, i], np.array(C_data[i]))))
                        A_inds.append(np.arange(start_ind_A, start_ind_A + self.nx))
                        start_ind_A += self.nx + 1 + len(C_data[i])
                    else:
                        data.append(np.hstack((-np.ones(1), A_lst[t][:, i], np.array(C_data[i]), np.array(C_h_data[i]))))
                        A_inds.append(np.arange(start_ind_A, start_ind_A + self.nx))
                        start_ind_A += self.nx + 1 + len(C_data[i]) + len(C_h_data[i])
            for i in range(self.nx):
                if self.dynamics_object.C_h is None:
                    data.append(np.hstack((-np.ones(1), np.array(C_data[i]))))
                else:
                    data.append(np.hstack((-np.ones(1), np.array(C_data[i]), np.array(C_h_data[i]))))

            # Input variables:
            B_inds = []
            if self.dynamics_object.C_h is None:
                start_ind_B = start_ind_A + self.nx + np.nonzero(self.C)[0].size - 1
            else:
                start_ind_B = start_ind_A + self.nx + np.nonzero(self.C)[0].size + np.nonzero(self.dynamics_object.C_h)[0].size - 1
            for t in range(self.N):
                for i in range(self.nu):
                    data.append(np.hstack((B_lst[t][:, i], np.ones(1))))
                    B_inds.append(np.arange(start_ind_B, start_ind_B + self.nx))
                    start_ind_B += self.nx + 1

            flat_data = []
            for arr in data:
                for d in arr:
                    flat_data.append(d)

            self._osqp_A_data = np.array(flat_data)
            self._osqp_A_data_A_inds = np.array(A_inds).flatten().tolist()
            self._osqp_A_data_B_inds = np.array(B_inds).flatten().tolist()
