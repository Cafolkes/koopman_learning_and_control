import numpy as np

from koopman_core.controllers import NMPCTrajControllerNb
from numba import njit

@njit(fastmath=True, cache=True)
def update_linearization_AB(A_flat, B_flat, B_arr, z_init, u_init):
    A_stacked = (B_flat @ u_init + A_flat).T.flatten()
    B_stacked = (B_arr @ z_init[:, :-1]).T.flatten()
    return A_stacked, B_stacked

class BilinearMPCTrajControllerNb(NMPCTrajControllerNb):
    """
    Class for bilinear MPC.
    Quadratic programs are solved using OSQP.
    """

    def __init__(self, dynamics, N, dt, umin, umax, xmin, xmax, C_x, C_obj, Q, R, QN, R0, xr, solver_settings,
                 terminal_constraint=False, add_slack=False, q_slack=1e-4):

        super(BilinearMPCTrajControllerNb, self).__init__(dynamics, N, dt, umin, umax, xmin, xmax, C_x, C_obj, Q, R, QN,
                                                          R0, xr, solver_settings, const_offset = None,
                                                          terminal_constraint = terminal_constraint,
                                                          add_slack = add_slack, q_slack = q_slack)

        self.embed_pkg_str = 'knmpc_' + str(self.nx) + '_' + str(self.nu) + '_' + str(self.N)
        self.A_flat = self.dynamics_object.A.flatten(order='F').reshape(-1,1)
        #self.B_flat = np.array([b.flatten(order='F') for b in self.dynamics_object.B])
        self.B_flat = np.array([b.flatten(order='F') for b in self.dynamics_object.B]).T
        self.B_arr = np.vstack(self.dynamics_object.B)

    def update_linearization_(self):
        self.A_stacked, self.B_stacked = update_linearization_AB(self.A_flat, self.B_flat, self.B_arr, self.z_init,
                                                                 self.u_init)
        self.r_vec[:] = (np.array([self.dynamics.eval_dot(z, u, None)
                                   for z, u in zip(self.z_init[:,:-1].T, self.u_init.T)]) - self.z_init[:, 1:].T).flatten()

    def eval(self, x, t):
        u = super(BilinearMPCTrajControllerNb, self).eval(x, t)

        if self.dynamics_object.standardizer_u is not None:
            return self.dynamics_object.standardizer_u.inverse_transform(u.reshape(1,-1)).squeeze()
        else:
            return u

    def get_state_prediction(self):
        """
        Get the state prediction from the MPC problem
        :return: Z (np.array) current state prediction
        """
        if self.dynamics_object.standardizer_x is None:
            return (self.dynamics_object.C@self.cur_z.T).T
        else:
            return self.dynamics_object.standardizer_x.inverse_transform(self.dynamics_object.C@self.cur_z.T).T

    def get_control_prediction(self):
        if self.dynamics_object.standardizer_u is None:
            return self.cur_u
        else:
            return self.dynamics_object.standardizer_u.inverse_transform(self.cur_u)

