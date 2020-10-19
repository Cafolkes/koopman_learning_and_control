import numpy as np
from scipy import sparse as sparse
from scipy.signal import cont2discrete
from scipy.linalg import solve
import osqp
from core.controllers.controller import Controller


class BilinearMPCController(Controller):
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
        self._osqp_Ad = sparse.csc_matrix(lin_model_d[0])
        self._osqp_Bd = sparse.csc_matrix(lin_model_d[1])
        self.C = bilinear_dynamics.C
        self.q_d = xr
        self.umin=umin
        self.umax=umax
        self.xmin=xmin
        self.xmax=xmax
        self.N = N
        if self.q_d.ndim == 2:
            self.Nqd = self.q_d.shape[1]

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

        self.comp_time = []

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),w(0),...,w(N-1), S+(0),...,S+(N-1),S-(1),...,S-(N-1))
        P, q = self.build_objective_()
        Aeq, leq, ueq = self.build_eq_constraints_()
        Aineq, lineq, uineq = self.build_ineq_constraints_()

        A = sparse.vstack([Aeq, Aineq]).tocsc()
        self._osqp_q = q
        self._osqp_l = np.hstack([leq, lineq])
        self._osqp_u = np.hstack([ueq, uineq])

        # Create an OSQP object and setup workspace
        self.prob = osqp.OSQP()
        self.prob.setup(P, q, A, self._osqp_l, self._osqp_u, warm_start=True, verbose=False)

    def build_objective_(self):
        nx_osqp = (self.N+1)*self.nx
        nu_osqp = self.N*self.nu*self.nx
        nslack_osqp = 2*self.N*self.nu*self.nx
        slack_cost = 5e-4

        # Load EDMD objects
        offset_transform = np.zeros(self.nu * self.nx)
        for ii in range(self.nu):
            offset_transform[ii * self.nx] = self.const_offset[ii]

        # - quadratic objective
        CQC = sparse.csc_matrix(np.transpose(self.C).dot(self.Q.dot(self.C)))
        CQNC = sparse.csc_matrix(np.transpose(self.C).dot(self.QN.dot(self.C)))
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), CQC), CQNC,
                               sparse.kron(sparse.eye(self.N), self.R),
                               np.zeros((nslack_osqp,nslack_osqp))]).tocsc()

        # - linear objective
        self.QCT = np.transpose(self.Q.dot(self.C))
        self.QNCT = np.transpose(self.QN.dot(self.C))
        if (self.q_d.ndim == 1):
            xr = self.q_d
            q = np.hstack(
                [np.kron(np.ones(self.N), -self.QCT.dot(xr)), -self.QNCT.dot(xr),
                 np.tile(2 * self.R.dot(offset_transform), (self.N)),
                 slack_cost * np.ones(nslack_osqp)])
        elif (self.q_d.ndim == 2):
            xr = self.q_d[:, :self.N + 1]
            q = np.hstack(
                [np.reshape(-self.QCT.dot(xr), ((self.N + 1) * self.nx,), order='F'),
                 np.tile(2 * self.R.dot(offset_transform), (self.N)),
                 slack_cost * np.ones(nslack_osqp)])

        return P, q

    def build_eq_constraints_(self):
        nx_osqp = (self.N + 1) * self.nx
        nu_osqp = self.N * self.nu * self.nx
        nslack_osqp = 2 * self.N * self.nu * self.nx
        x0 = np.zeros(self.nx)

        Ax = sparse.kron(sparse.eye(self.N + 1), -sparse.eye(self.nx)) + sparse.kron(sparse.eye(self.N + 1, k=-1), self._osqp_Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), self._osqp_Bd)
        Aeq = sparse.hstack([Ax, Bu, np.zeros((nx_osqp,nslack_osqp))])
        leq = np.hstack([-x0, np.zeros(self.N * self.nx)])
        ueq = leq

        return Aeq, leq, ueq

    def build_ineq_constraints_(self):
        nx_osqp = (self.N + 1) * self.nx
        nu_osqp = self.N * self.nu * self.nx
        nslack_osqp = 2 * self.N * self.nu * self.nx

        # State constraints:
        Ax = sparse.block_diag([self.C for _ in range(self.N + 1)])
        Aineq = sparse.hstack([Ax, np.zeros((Ax.shape[0], nu_osqp+nslack_osqp))])
        lineq = np.kron(np.ones(self.N + 1), self.xmin)
        uineq = np.kron(np.ones(self.N + 1), self.xmax)

        # Actuation constraints:
        for ii in range(self.N):
            for jj in range(self.nu):
                # - lower bound, u_min-terms:
                Ax = np.hstack((np.zeros((self.nx, ii * self.nx)), -np.diag(np.repeat(self.umin[jj], self.nx)),
                                np.zeros((self.nx, (self.N - ii) * self.nx))))
                Aw = np.hstack((np.zeros((self.nx, ii * self.nx * self.nu + jj * self.nx)), np.eye(self.nx),
                        np.zeros((self.nx, (self.N - 1 - ii) * self.nx * self.nu + (self.nu - jj - 1) * self.nx))))
                As_pos = Aw
                As_neg = As_pos
                Aineq = sparse.vstack([Aineq, np.hstack((Ax, Aw, As_pos, As_neg))])
                lineq = np.hstack((lineq, np.zeros(self.nx)))
                uineq = np.hstack((uineq, np.inf*np.ones(self.nx)))

                # - lower bound, u_max-terms:
                Ax = np.hstack((np.zeros((self.nx, ii * self.nx)), -np.diag(np.repeat(self.umax[jj], self.nx)),
                                np.zeros((self.nx, (self.N - ii) * self.nx))))
                Aw = np.hstack((np.zeros((self.nx, ii * self.nx * self.nu + jj * self.nx)), np.eye(self.nx),
                                np.zeros((self.nx, (self.N - 1 - ii) * self.nx * self.nu + (self.nu - jj - 1) * self.nx))))
                As_pos = Aw
                As_neg = As_pos
                Aineq = sparse.vstack([Aineq, np.hstack((Ax, Aw, As_pos, As_neg))])
                lineq = np.hstack((lineq, np.zeros(self.nx)))
                uineq = np.hstack((uineq, np.inf * np.ones(self.nx)))

                # - upper bound, u_min-terms:
                Ax = np.hstack((np.zeros((self.nx, ii * self.nx)), -np.diag(np.repeat(self.umin[jj], self.nx)),
                                np.zeros((self.nx, (self.N - ii) * self.nx))))
                Aw = np.hstack((np.zeros((self.nx, ii * self.nx * self.nu + jj * self.nx)), np.eye(self.nx),
                                np.zeros(
                                    (self.nx, (self.N - 1 - ii) * self.nx * self.nu + (self.nu - jj - 1) * self.nx))))
                As_pos = np.hstack((np.zeros((self.nx, ii * self.nx * self.nu + jj * self.nx)), -np.eye(self.nx),
                        np.zeros((self.nx, (self.N - 1 - ii) * self.nx * self.nu + (self.nu - jj - 1) * self.nx))))
                As_neg = As_pos
                Aineq = sparse.vstack([Aineq, np.hstack((Ax, Aw, As_pos, As_neg))])
                lineq = np.hstack((lineq, -np.inf * np.ones(self.nx)))
                uineq = np.hstack((uineq, np.zeros(self.nx)))

                # - upper bound, u_max-terms:
                Ax = np.hstack((np.zeros((self.nx, ii * self.nx)), -np.diag(np.repeat(self.umax[jj], self.nx)),
                                np.zeros((self.nx, (self.N - ii) * self.nx))))
                Aw = np.hstack((np.zeros((self.nx, ii * self.nx * self.nu + jj * self.nx)), np.eye(self.nx),
                                np.zeros(
                                    (self.nx, (self.N - 1 - ii) * self.nx * self.nu + (self.nu - jj - 1) * self.nx))))
                As_pos = np.hstack((np.zeros((self.nx, ii * self.nx * self.nu + jj * self.nx)), -np.eye(self.nx),
                                    np.zeros((self.nx, (self.N - 1 - ii) * self.nx * self.nu + (self.nu - jj - 1) * self.nx))))
                As_neg = As_pos
                Aineq = sparse.vstack([Aineq, np.hstack((Ax, Aw, As_pos, As_neg))])
                lineq = np.hstack((lineq, -np.inf * np.ones(self.nx)))
                uineq = np.hstack((uineq, np.zeros(self.nx)))
        
        # Slack variables lower bound:
        for ii in range(self.N):
            for jj in range(self.nu):
                # S_pos - umin*z + umax*z >= 0:
                Ax = np.hstack((np.zeros((self.nx, ii * self.nx)), -np.diag(np.repeat(self.umin[jj]-self.umax[jj], self.nx)),
                                np.zeros((self.nx, (self.N - ii) * self.nx))))
                Aw = np.zeros((self.nx,nu_osqp))
                As_pos = np.hstack((np.zeros((self.nx, ii * self.nx * self.nu + jj * self.nx)), np.eye(self.nx),
                                np.zeros((self.nx, (self.N - 1 - ii) * self.nx * self.nu + (self.nu - jj - 1) * self.nx))))
                As_neg = np.zeros((self.nx,nu_osqp))
                Aineq = sparse.vstack([Aineq, np.hstack((Ax, Aw, As_pos, As_neg))])
                lineq = np.hstack((lineq, np.zeros(self.nx)))
                uineq = np.hstack((uineq, np.inf * np.ones(self.nx)))

                # S_neg - umax*z + umin*z >= 0:
                Ax = np.hstack(
                    (np.zeros((self.nx, ii * self.nx)), -np.diag(np.repeat(self.umax[jj] - self.umin[jj], self.nx)),
                     np.zeros((self.nx, (self.N - ii) * self.nx))))
                Aw = np.zeros((self.nx, nu_osqp))
                As_pos = np.zeros((self.nx, nu_osqp))
                As_neg = np.hstack((np.zeros((self.nx, ii * self.nx * self.nu + jj * self.nx)), np.eye(self.nx),
                                    np.zeros((self.nx,(self.N - 1 - ii) * self.nx * self.nu + (self.nu - jj - 1) * self.nx))))
                Aineq = sparse.vstack([Aineq, np.hstack((Ax, Aw, As_pos, As_neg))])
                lineq = np.hstack((lineq, np.zeros(self.nx)))
                uineq = np.hstack((uineq, np.inf * np.ones(self.nx)))

        # Slack variables >= 0:
        Ax = np.zeros((nslack_osqp,nx_osqp))
        Aw = np.zeros((nslack_osqp,nu_osqp))
        As = np.eye(nslack_osqp)
        Aineq = sparse.vstack([Aineq, np.hstack((Ax, Aw, As))])
        lineq = np.hstack((lineq, np.zeros(nslack_osqp)))
        uineq = np.hstack((uineq, np.inf * np.ones(nslack_osqp)))

        return Aineq, lineq, uineq

    def eval(self, x, t):
        """eval Function to evaluate controller

        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time

        Returns:
            control action -- numpy array [Nu,]
        """
        z, w = self.eval_mpc_(x,t)
        if all(z == 0):
            return np.zeros(self.nu)
        else:
            z_inv = np.linalg.pinv(z.reshape(-1,1))
            return np.array([np.dot(z_inv,w[i*self.nx:(i+1)*self.nx]) for i in range(self.nu)]).squeeze()

    def eval_mpc_(self, x, t):
        nx_osqp = (self.N + 1) * self.nx
        tindex = int(np.ceil(t / self.dt))

        # Update the local reference trajectory
        if self.q_d.ndim == 2:
            if (tindex + self.N) < self.Nqd:  # if we haven't reach the end of q_d yet
                xr = self.q_d[:, tindex:tindex + self.N + 1]
            else:  # we fill xr with copies of the last q_d
                xr = np.hstack(
                    [self.q_d[:, tindex:], np.transpose(np.tile(self.q_d[:, -1], (self.N + 1 - self.Nqd + tindex, 1)))])

            # Construct the new _osqp_q objects
            self._osqp_q[:(self.N+1)*self.nx] = np.reshape(-self.QCT.dot(xr), ((self.N+1)*self.nx,), order='F')

        # Lift the current state:
        z = self.dynamics.basis(x.reshape((1, -1))).squeeze()
        self._osqp_l[:self.nx] = -z
        self._osqp_u[:self.nx] = -z

        # Solve MPC Instance
        self.prob.update(q=self._osqp_q, l=self._osqp_l, u=self._osqp_u)
        self._osqp_result = self.prob.solve()
        self.comp_time.append(self._osqp_result.info.run_time)

        return z, self._osqp_result.x[nx_osqp:nx_osqp+self.nx*self.nu]

    def parse_result(self):
        return np.transpose(np.reshape(self._osqp_result.x[:(self.N + 1) * self.nx], (self.N + 1, self.nx)))

    def get_control_prediction(self):
        nx_osqp = (self.N + 1) * self.nx
        nu_osqp = self.N*self.nx*self.nu

        z = self._osqp_result.x[:nx_osqp]
        w = self._osqp_result.x[nx_osqp:nx_osqp+nu_osqp]
        u = []
        for ii in range(self.N):
            z_tmp = z[ii*self.nx:(ii+1)*self.nx]
            w_tmp = w[ii*self.nx*self.nu:(ii+1)*self.nx*self.nu]
            if all(z_tmp == 0):
                u.append(np.zeros(self.nu))
            else:
                z_inv = np.linalg.pinv(z_tmp.reshape(-1,1))
                u.append(np.array([np.dot(z_inv,w_tmp[i*self.nx:(i+1)*self.nx]) for i in range(self.nu)]).squeeze())

        return np.array(u).T



