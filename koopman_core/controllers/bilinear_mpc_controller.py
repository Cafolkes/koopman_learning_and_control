import numpy as np
import os
import importlib

from ..controllers import NonlinearMPCController


class BilinearMPCController(NonlinearMPCController):
    """
    Class for bilinear MPC.
    Quadratic programs are solved using OSQP.
    """

    def __init__(self, dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, solver_settings,
                 terminal_constraint=False, add_slack=False, q_slack=1e-4, standardizer_x=None, standardizer_u=None):

        const_offset = None
        if standardizer_u is not None:
            umin = standardizer_u.transform(umin.reshape(1,-1)).squeeze()
            umax = standardizer_u.transform(umax.reshape(1,-1)).squeeze()
            if standardizer_u.with_mean:
                const_offset = standardizer_u.mean_
        if standardizer_x is not None:
            xmin = standardizer_x.transform(xmin.reshape(1,-1)).squeeze()
            xmax = standardizer_x.transform(xmax.reshape(1,-1)).squeeze()
            xr = standardizer_x.transform(xr.reshape(1,-1)).squeeze()

        super(BilinearMPCController, self).__init__(dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, solver_settings,
                                        const_offset=const_offset,
                                        terminal_constraint=terminal_constraint,
                                        add_slack=add_slack,
                                        q_slack=q_slack)

        self.embed_pkg_str = 'knmpc_' + str(self.nx) + '_' + str(self.nu) + '_' + str(self.N)
        self.A_flat = self.dynamics_object.A.flatten(order='F')
        self.B_flat = np.array([b.flatten(order='F') for b in self.dynamics_object.B])
        self.B_arr = np.vstack(self.dynamics_object.B).T
        self.standardizer_x = standardizer_x
        self.standardizer_u = standardizer_u

    def update_constraint_matrix_data_(self, A_lst, B_lst):
        self._osqp_A_data[self._osqp_A_data_A_inds] = A_lst
        self._osqp_A_data[self._osqp_A_data_B_inds] = B_lst

    def update_linearization_(self):
        A_lst_flat = (self.u_init @ self.B_flat + self.A_flat).flatten()
        B_lst_flat = (self.z_init[:-1, :] @ self.B_arr).flatten()

        self.r_vec[:] = (np.array([self.dynamics.eval_dot(z, u, None)
                                   for z, u in zip(self.z_init[:-1,:], self.u_init)]) - self.z_init[1:, :]).flatten()

        return A_lst_flat, B_lst_flat

    def eval(self, x, t):
        u = super(BilinearMPCController, self).eval(x, t)

        if self.standardizer_u is not None:
            return self.standardizer_u.inverse_transform(u.reshape(1,-1)).squeeze()
        else:
            return u

    def get_control_prediction(self):
        if self.standardizer_u is not None:
            return self.standardizer_u.inverse_transform(self.cur_u)
        else:
            return self.cur_u
