import numpy as np
from scipy import sparse as sparse
from scipy.signal import cont2discrete
from scipy.linalg import solve
import osqp
import cvxpy as cp
from core.controllers.controller import Controller


class BilinearMPCControllerCvx(Controller):
    """
    Class for controllers MPC.

    MPCs are solved using osqp.
    """

    def __init__(self, bilinear_dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr,
                 const_offset=None):
        """__init__ Create an MPC controller

        Arguments:
            linear_dynamics {dynamical sytem} -- it contains the A and B matrices in continous time
            N {integer} -- number of timesteps
            dt {float} -- time step in seconds
            umin {numpy array [Nu,]} -- minimum control bound
            umax {numpy array [Nu,]} -- maximum control bound
            xmin {numpy array [Ns,]} -- minimum state bound
            xmax {numpy array [Ns,]} -- maximum state bound
            Q {numpy array [Ns,Ns]} -- state cost matrix
            R {numpy array [Nu,Nu]} -- control cost matrix
            QN {numpy array [Ns,]} -- final state cost
            xr {numpy array [Ns,]} -- reference trajectory

        Keyword Arguments:
            plotMPC {bool} -- flag to plot results (default: {False})
            plotMPC_filename {str} -- plotting filename (default: {""})
            lifting {bool} -- flag to use state lifting (default: {False})
            edmd_object {edmd object} -- lifting object. It contains projection matrix and lifting function (default: {Edmd()})
        """
        Controller.__init__(self, bilinear_dynamics)

        # Load arguments
        Ac, Bc = bilinear_dynamics.A, bilinear_dynamics.B
        for ii,b in enumerate(Bc):
            if ii == 0:
                B_mpc = b
            else:
                B_mpc = np.concatenate((B_mpc, b), axis=1)
        self.nx, self.nu = Ac.shape[0], len(Bc)
        self.ns = xr.shape[0]
        self.dt = dt
        lin_model_d = cont2discrete((Ac, B_mpc, np.eye(self.nx), np.zeros((self.nx*self.nu, 1))), dt)
        self.Ad = lin_model_d[0]
        self.Bd = lin_model_d[1]
        self.C = bilinear_dynamics.C
        self.q_d = xr
        self.umin=umin
        self.umax=umax
        self.xmin=xmin
        self.xmax=xmax
        self.N = N
        if self.q_d.ndim == 2:
            self.Nqd = self.q_d.shape[1]

        #self.Q = self.C.T@Q@self.C
        #self.QN = self.C.T@QN@self.C
        self.Q = Q
        self.QN = QN

        self.R = np.zeros((self.nx*self.nu,self.nx*self.nu))
        for i in range(self.nu):
            for j in range(self.nu):
                self.R[self.nx*i,self.nx*j] = R[i,j]

        if const_offset is None:
            self.const_offset = np.zeros(self.nu)
        else:
            self.const_offset = const_offset

    def gen_trajectory(self, x0, t):
        z0 = self.dynamics.basis(x0.reshape((1, -1))).squeeze()
        z = cp.Variable((self.nx, self.N + 1))
        w = cp.Variable((self.nx*self.nu, self.N))

        cost = cp.quad_form(self.C@z[:,self.N]-self.q_d,self.QN)
        constr = []

        for t in range(self.N):
            cost += cp.quad_form(w[:,t],self.R)
            constr += [z[:, t + 1] == self.Ad @ z[:, t] + self.Bd @ w[:, t]]

        # sums problem objectives and concatenates constraints.
        constr += [z[:, 0] == z0]
        self.problem = cp.Problem(cp.Minimize(cost), constr)
        self.problem.solve(solver=cp.ECOS)

        z_traj = z.value
        x_traj = self.C@z_traj
        w_traj = w.value
        u_traj = []
        for ii in range(self.N):
            if all(z_traj[:,ii] == 0):
                u_traj.append(np.zeros(self.nu))
            else:
                z_inv = np.linalg.pinv(z_traj[:,ii].reshape(-1, 1))
                u_traj.append(
                    np.array([np.dot(z_inv, w_traj[i * self.nx:(i + 1) * self.nx, ii]) for i in range(self.nu)]).squeeze())

        return x_traj, np.array(u_traj).T

    def eval(self, x, t):
        """eval Function to evaluate controller

        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time

        Returns:
            control action -- numpy array [Nu,]
        """
        pass

    def eval_mpc_(self, x, t):
        pass

    def parse_result(self):
        pass

    def get_control_prediction(self):
        pass



