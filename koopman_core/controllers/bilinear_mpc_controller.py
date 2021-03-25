import numpy as np
import os
import importlib

from koopman_core.controllers import NonlinearMPCController


class BilinearMPCController(NonlinearMPCController):
    """
    Class for bilinear MPC.
    Quadratic programs are solved using OSQP.
    """

    def __init__(self, dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, solver_settings, const_offset=None,
                 terminal_constraint=False, add_slack=False, q_slack=1e3):

        super(BilinearMPCController, self).__init__(dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, solver_settings,
                                        const_offset=const_offset,
                                        terminal_constraint=terminal_constraint,
                                        add_slack=add_slack,
                                        q_slack=q_slack)

        self.embed_pkg_str = 'knmpc_' + str(self.nx) + '_' + str(self.nu) + '_' + str(self.N)
        self.A_flat = self.dynamics_object.A.flatten(order='F')
        self.B_flat = np.array([b.flatten(order='F') for b in self.dynamics_object.B])
        self.B_arr = np.vstack(self.dynamics_object.B).T

    def update_constraint_matrix_data_(self, A_lst, B_lst):
        self._osqp_A_data[self._osqp_A_data_A_inds] = A_lst
        self._osqp_A_data[self._osqp_A_data_B_inds] = B_lst

    def update_linearization_(self):
        A_lst = self.u_init @ self.B_flat + self.A_flat
        A_lst_flat = A_lst.flatten()
        B_lst_flat = (self.z_init[:-1, :] @ self.B_arr).flatten()

        # TODO: (Comp time optimzation) Try to further improve calculation of r_vec
        A_reshaped = A_lst.reshape(self.nx * self.N, self.nx)
        self.r_vec[:] = (np.array([self.z_init[i, :] @ A_reshaped[i * self.nx:(i + 1) * self.nx, :]
                                   for i in range(self.N)]) - self.z_init[1:, :]).flatten()

        return A_lst_flat, B_lst_flat