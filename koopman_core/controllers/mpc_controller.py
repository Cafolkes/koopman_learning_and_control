import numpy as np
import scipy.sparse as sparse
from scipy.signal import cont2discrete
import osqp
from ...core.controllers.controller import Controller

class MPCController(Controller):
    """
    Class for linear MPC with lifted linear dynamics.

    Quadratic programs are solved using OSQP.
    """
    def __init__(self, lifted_linear_dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, q_d, const_offset=None, terminal_constraint=False, add_slack=False):
        """__init__ Create an MPC controller
        
        Arguments:
            lifted_linear_dynamics {LinearLiftedDynamics} -- Lifted linear continuous-time dynamics
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

        Controller.__init__(self, lifted_linear_dynamics)
        self.dynamics_object = lifted_linear_dynamics
        self.dt = dt
        if lifted_linear_dynamics.continuous_mdl:
            Ac = lifted_linear_dynamics.A
            Bc = lifted_linear_dynamics.B
            [self.nx, self.nu] = Bc.shape
            lin_model_d = cont2discrete((Ac,Bc,np.eye(self.nx),np.zeros((self.nu,1))),dt)
            self._osqp_Ad = sparse.csc_matrix(lin_model_d[0])
            self._osqp_Bd = sparse.csc_matrix(lin_model_d[1])
        else:
            self._osqp_Ad = lifted_linear_dynamics.A
            self._osqp_Bd = lifted_linear_dynamics.B
            [self.nx, self.nu] = self._osqp_Bd.shape
        self.C = lifted_linear_dynamics.C

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

        # Total desired path
        self.q_d = q_d
        self.ns = q_d.shape[0]
        if self.q_d.ndim==2:
            # Add copies of the final state in the desired trajectory to enable prediction beyond trajectory horizon:
            self.q_d = np.hstack([self.q_d, np.transpose(np.tile(self.q_d[:, -1], (self.N + 1, 1)))])
        self.terminal_constraint = terminal_constraint
        self.add_slack = add_slack

        # Initialize OSQP MPC Problem:
        self.build_objective_()
        self.build_constraints_()
        self.prob = osqp.OSQP()
        self.prob.setup(self._osqp_P, self._osqp_q, self._osqp_A, self._osqp_l, self._osqp_u, warm_start=True, verbose=False)
        self._osqp_result = None
        self.comp_time = []

    def build_objective_(self):
        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        CQC = sparse.csc_matrix(np.transpose(self.C).dot(self.Q.dot(self.C)))
        CQNC = sparse.csc_matrix(np.transpose(self.C).dot(self.QN.dot(self.C)))
        Q_slack = 1e3*sparse.eye(self.ns*(self.N+1))

        if not self.add_slack:
            self._osqp_P = sparse.block_diag([sparse.kron(sparse.eye(self.N), CQC), CQNC,
                                sparse.kron(sparse.eye(self.N), self.R)]).tocsc()
        else:
            self._osqp_P = sparse.block_diag([sparse.kron(sparse.eye(self.N), CQC), CQNC,
                                              sparse.kron(sparse.eye(self.N), self.R),
                                              Q_slack]).tocsc()

        # - linear objective
        if self.q_d.ndim==2:
            xr = self.q_d[:,:self.N+1]
        else:
            xr = self.q_d
        QCT = np.transpose(self.Q.dot(self.C))
        QNCT = np.transpose(self.QN.dot(self.C))
        if (xr.ndim==1):
            if not self.add_slack:
                self._osqp_q = np.hstack([np.kron(np.ones(self.N), -QCT.dot(xr)), -QNCT.dot(xr), np.tile(2*self.R.dot(self.const_offset),(self.N))])
            else:
                self._osqp_q = np.hstack([np.kron(np.ones(self.N), -QCT.dot(xr)), -QNCT.dot(xr),
                                          np.tile(2 * self.R.dot(self.const_offset), (self.N)),
                                          np.zeros(self.ns*(self.N+1))])
        elif (xr.ndim==2):
            if not self.add_slack:
                self._osqp_q = np.hstack([np.reshape(-QCT.dot(xr),((self.N+1)*self.nx,),order='F'), np.tile(2*self.R.dot(self.const_offset),(self.N))])
            else:
                self._osqp_q = np.hstack([np.reshape(-QCT.dot(xr), ((self.N + 1) * self.nx,), order='F'),
                                          np.tile(2 * self.R.dot(self.const_offset), (self.N)),
                                          np.zeros(self.ns*(self.N+1))])

    def build_constraints_(self):
        # - linear dynamics
        x0 = np.zeros(self.nx)

        # - input and state constraints
        Ax = sparse.kron(sparse.eye(self.N + 1), -sparse.eye(self.nx)) + sparse.kron(sparse.eye(self.N + 1, k=-1), self._osqp_Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), self._osqp_Bd)

        if not self.add_slack:
            Aineq = sparse.block_diag([self.C for _ in range(self.N+1)]+[np.eye(self.N*self.nu)])
            Aeq = sparse.hstack([Ax, Bu])
        else:
            Aineq = sparse.hstack([sparse.block_diag([self.C for _ in range(self.N + 1)] + [np.eye(self.N * self.nu)]),
                                   sparse.vstack([sparse.eye(self.ns * (self.N + 1)),
                                                  sparse.csc_matrix((self.nu * self.N, self.ns * (self.N + 1)))])])
            Aeq = sparse.hstack([Ax, Bu, sparse.csc_matrix((self.nx*(self.N+1), self.ns*(self.N+1)))])

        leq = np.hstack([-x0, np.zeros(self.N*self.nx)])
        ueq = leq

        lineq = np.hstack([np.kron(np.ones(self.N+1), self.xmin), np.kron(np.ones(self.N), self.umin)])
        uineq = np.hstack([np.kron(np.ones(self.N+1), self.xmax), np.kron(np.ones(self.N), self.umax)])

        if self.terminal_constraint:
            lineq[-self.ns-self.N*self.nu:-self.N*self.nu] = self.q_d
            uineq[-self.ns-self.N*self.nu:-self.N*self.nu] = self.q_d

        self._osqp_A = sparse.vstack([Aeq, Aineq]).tocsc()
        self._osqp_l = np.hstack([leq, lineq])
        self._osqp_u = np.hstack([ueq, uineq])

    def eval(self, x, t):
        """eval Function to evaluate controller
        
        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time
        
        Returns:
            control action -- numpy array [Nu,]
        """

        ## Update inequalities
        x = self.dynamics_object.lift(x.reshape((1, -1)), None).squeeze()
        self._osqp_l[:self.nx] = -x
        self._osqp_u[:self.nx] = -x

        if self.q_d.ndim==2:
            # Update the local reference trajectory
            tindex = int(t / self.dt)
            xr = self.q_d[:,tindex:tindex+self.N+1]

            # Construct the new _osqp_q objects
            QCT = np.transpose(self.Q.dot(self.C))
            #self._osqp_q = np.hstack([np.reshape(-QCT.dot(xr),((self.N+1)*self.nx,),order='F'), np.zeros(self.N*self.nu)])
            self._osqp_q[:self.nx*(self.N+1)] = np.reshape(-QCT.dot(xr), ((self.N + 1) * self.nx,), order='F')

            self.prob.update(q=self._osqp_q, l=self._osqp_l, u=self._osqp_u)
        else:
            self.prob.update(l=self._osqp_l, u=self._osqp_u)

        ## Solve MPC Instance
        self._osqp_result = self.prob.solve()
        self.comp_time.append(self._osqp_result.info.run_time)

        return self._osqp_result.x[self.nx*(self.N+1):self.nx*(self.N+1)+self.nu]

    def parse_result(self):
        return np.transpose(np.reshape(self._osqp_result.x[:(self.N+1)*self.nx], (self.N+1,self.nx)))

    def get_control_prediction(self):
        return np.transpose(np.reshape(self._osqp_result.x[self.nx*(self.N+1):self.nx*(self.N+1)+self.nu*self.N], (self.N,self.nu)))