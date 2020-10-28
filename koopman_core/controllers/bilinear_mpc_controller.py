from koopman_core.controllers import NonlinearMPCController
import numpy as np

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

    def update_linearization_(self, z_init, u_init):
        A_lst = u_init@self.B_flat + self.A_flat
        A_lst_flat = A_lst.flatten()
        B_lst_flat = (z_init[:-1,:]@self.B_arr).flatten()

        #TODO: (Comp time optimzation) Try to further improve calculation of r_vec
        self.r_vec[:] = np.array([a.reshape((self.nx,self.nx), order='F')@z - z_next for
                          a,z, z_next in zip(A_lst, z_init[:-1,:], z_init[1:,:])]).flatten()

        return A_lst_flat, B_lst_flat


