import numpy as np
import scipy.sparse as sparse
from scipy.signal import cont2discrete
import osqp
from core.controllers.controller import Controller


class BilinearMPCController(Controller):
    """
    Class for controllers MPC.

    MPCs are solved using osqp.

    Use lifting=True to solve MPC in the lifted space
    """

    def __init__(self, bilinear_dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr,
                 lifting=False, const_offset=None):
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
        nx, nu = Ac.shape[0], len(Bc)
        self.dt = dt
        lin_model_d = cont2discrete((Ac, B_mpc, np.eye(nx), np.zeros((nx*nu, 1))), dt)
        self._osqp_Ad = sparse.csc_matrix(lin_model_d[0])
        self._osqp_Bd = sparse.csc_matrix(lin_model_d[1])
        self.C = bilinear_dynamics.C
        self.q_d = xr

        self.ns = xr.shape[0]

        self.Q = Q
        self.R = np.zeros((nx*nu,nx*nu))
        for i in range(nu):
            for j in range(nu):
                self.R[nx*i,nx*j] = R[i,j]

        self.lifting = lifting

        self.nu = nu
        self.nx = nx
        if const_offset is None:
            self.const_offset = np.zeros(nu)
        else:
            self.const_offset = const_offset

        self.comp_time = []

        # Total desired path
        if self.q_d.ndim == 2:
            self.Nqd = self.q_d.shape[1]
            xr = self.q_d[:, :N + 1]

        # Prediction horizon
        self.N = N
        x0 = np.zeros(nx)

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),w(0),...,w(N-1))

        # Load EDMD objects
        offset_transform = np.zeros(nu*nx)
        for ii in range(nu):
            offset_transform[ii*nx] = self.const_offset[ii]

        # - quadratic objective
        CQC = sparse.csc_matrix(np.transpose(self.C).dot(Q.dot(self.C)))
        CQNC = sparse.csc_matrix(np.transpose(self.C).dot(QN.dot(self.C)))
        P = sparse.block_diag([sparse.kron(sparse.eye(N), CQC), CQNC,
                               sparse.kron(sparse.eye(N), self.R)]).tocsc()

        # - linear objective
        QCT = np.transpose(Q.dot(self.C))
        QNCT = np.transpose(QN.dot(self.C))
        if (xr.ndim == 1):
            q = np.hstack(
                [np.kron(np.ones(N), -QCT.dot(xr)), -QNCT.dot(xr), np.tile(2 * self.R.dot(offset_transform), (N))])
        elif (xr.ndim == 2):
            q = np.hstack(
                [np.reshape(-QCT.dot(xr), ((N + 1) * nx,), order='F'), np.tile(2 * self.R.dot(offset_transform), (N))])

        # - input and state constraints
        Aineq = sparse.block_diag([self.C for _ in range(N + 1)])
        Aineq = sparse.hstack([Aineq, np.zeros((Aineq.shape[0], N*nu*nx))])
        # Actuation lower bound:
        for ii in range(N):
            for jj in range(nu):
                Aineq_temp = np.hstack(
                    (np.zeros((nx, ii*nx)), -np.diag(np.repeat(umin[jj], nx)), np.zeros((nx, (N-ii)*nx)),
                     np.zeros((nx, ii*nx*nu+jj*nx)), np.eye(nx), np.zeros((nx, (N-1-ii)*nx*nu+(nu-jj-1)*nx))))
                Aineq = sparse.vstack([Aineq, Aineq_temp])
        # Actuation upper bound:
        for ii in range(N):
            for jj in range(nu):
                Aineq_temp = np.hstack(
                    (np.zeros((nx, ii * nx)), -np.diag(np.repeat(umax[jj], nx)), np.zeros((nx, (N - ii) * nx)),
                     np.zeros((nx, ii * nx * nu + jj * nx)), np.eye(nx),
                     np.zeros((nx, (N - 1 - ii) * nx * nu + (nu - jj - 1) * nx))))
                Aineq = sparse.vstack([Aineq, Aineq_temp])

        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(N + 1, k=-1), self._osqp_Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), self._osqp_Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N * nx)])

        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.zeros(N*nu*nx), -np.inf*np.ones(N*nu*nx)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.inf*np.ones(N*nu*nx), np.zeros(N*nu*nx)])

        ueq = leq
        self._osqp_q = q

        A = sparse.vstack([Aeq, Aineq]).tocsc()
        self._osqp_l = np.hstack([leq, lineq])
        self._osqp_u = np.hstack([ueq, uineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P, q, A, self._osqp_l, self._osqp_u, warm_start=True, verbose=False)

    def eval(self, x, t):
        """eval Function to evaluate controller

        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time

        Returns:
            control action -- numpy array [Nu,]
        """

        pass

    def eval_mpc(self, x, t):
        N = self.N
        nu = self.nu
        nx = self.nx

        tindex = int(t / self.dt)

        ## Update inequalities
        if self.q_d.ndim == 2:

            # Update the local reference trajectory
            if (tindex + N) < self.Nqd:  # if we haven't reach the end of q_d yet
                xr = self.q_d[:, tindex:tindex + N + 1]
            else:  # we fill xr with copies of the last q_d
                xr = np.hstack(
                    [self.q_d[:, tindex:], np.transpose(np.tile(self.q_d[:, -1], (N + 1 - self.Nqd + tindex, 1)))])

            # Construct the new _osqp_q objects
            if (self.lifting):
                QCT = np.transpose(self.Q.dot(self.C))
                self._osqp_q = np.hstack([np.reshape(-QCT.dot(xr), ((N + 1) * nx,), order='F'), np.zeros(N * nu)])
            else:
                self._osqp_q = np.hstack([np.reshape(-self.Q.dot(xr), ((N + 1) * nx,), order='F'), np.zeros(N * nu)])

        if self.q_d.ndim == 1:
            # Update the local reference trajectory
            xr = np.transpose(np.tile(self.q_d, N + 1))

        # Lift the current state if necessary
        if (self.lifting):
            x = self.dynamics.lift(x.reshape((1, -1)), None).squeeze()

        self._osqp_l[:self.nx] = -x
        self._osqp_u[:self.nx] = -x

        self.prob.update(q=self._osqp_q, l=self._osqp_l, u=self._osqp_u)

        ## Solve MPC Instance
        self._osqp_result = self.prob.solve()
        self.comp_time.append(self._osqp_result.info.run_time)

        return self._osqp_result.x[-N * nu:-(N - 1) * nu]

    def parse_result(self):
        return np.transpose(np.reshape(self._osqp_result.x[:(self.N + 1) * self.nx], (self.N + 1, self.nx)))

    def get_control_prediction(self):
        return np.transpose(np.reshape(self._osqp_result.x[-self.N * self.nu:], (self.N, self.nu)))



