import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.signal import cont2discrete
import osqp
from core.controllers.controller import Controller


class BilinearMPCController(Controller):
    """
    Class for linear MPC with lifted linear dynamics.

    Quadratic programs are solved using OSQP.
    """
    def __init__(self, lifted_bilinear_dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, q_d, const_offset=None):
        """__init__ Create an MPC controller
        
        Arguments:
            lifted_bilinear_dynamics {LinearLiftedDynamics} -- Lifted linear continuous-time dynamics
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

        Controller.__init__(self, lifted_bilinear_dynamics)
        self.dynamics_object = lifted_bilinear_dynamics
        self.nx = self.dynamics_object.n
        self.nu = self.dynamics_object.m
        self.dt = dt
        if self.dynamics_object.continuous:
            pass
        else:
            self.A = self.dynamics_object.A
            self.B = self.dynamics_object.B
            assert self.dt == self.dynamics_object.dt
        self.C = lifted_bilinear_dynamics.C

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

        # Initialize OSQP MPC Problem:
        self._osqp_result = None
        self.comp_time = []
        self.iter_sol = []

    def initialize_controller(self, x0, u0):
        z0 = self.dynamics_object.lift(x0.reshape((1, -1)), None).squeeze()
        A, B, r = self.linearize_dynamics(z0, None, u0, None)
        self._osqp_Ad = sparse.csc_matrix(A)
        self._osqp_Bd = sparse.csc_matrix(B)
        self._osqp_r = r

        self.build_objective_()
        self.build_constraints_()
        self.prob = osqp.OSQP()
        self.prob.setup(self._osqp_P, self._osqp_q, self._osqp_A, self._osqp_l, self._osqp_u, warm_start=True, verbose=False)

    def linearize_dynamics(self, z0, z1, u0, t):
        A_lin = self.A + np.sum(np.array([b*u for b,u in zip(self.B, u0)]),axis=0)
        B_lin = np.array([b @ z0 for b in self.B]).T

        if z1 is None:
            z1 = A_lin@z0 + B_lin@u0

        f_d = self.dynamics_object.drift(z0, t) + np.dot(self.dynamics_object.act(z0,t),u0)
        r_lin = f_d - z1

        return A_lin, B_lin, r_lin

    def build_objective_(self):
        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        CQC  = sparse.csc_matrix(np.transpose(self.C).dot(self.Q.dot(self.C)))
        CQNC = sparse.csc_matrix(np.transpose(self.C).dot(self.QN.dot(self.C)))
        self._osqp_P = sparse.block_diag([sparse.kron(sparse.eye(self.N), CQC), CQNC,
                            sparse.kron(sparse.eye(self.N), self.R)]).tocsc()

        # - linear objective
        if self.q_d.ndim==2:
            xr = self.q_d[:,:self.N+1]
        else:
            xr = self.q_d
        QCT = np.transpose(self.Q.dot(self.C))
        QNCT = np.transpose(self.QN.dot(self.C))
        if (xr.ndim==1):
            self._osqp_q = np.hstack([np.kron(np.ones(self.N), -QCT.dot(xr)), -QNCT.dot(xr), np.tile(self.R.dot(self.const_offset),(self.N))])
        elif (xr.ndim==2):
            self._osqp_q = np.hstack([np.reshape(-QCT.dot(xr),((self.N+1)*self.nx,),order='F'), np.tile(self.R.dot(self.const_offset),(self.N))])

    def build_constraints_(self):
        # - input and state constraints
        Aineq = sparse.block_diag([self.C for i in range(self.N+1)]+[np.eye(self.N*self.nu)])

        # - linear dynamics
        x0 = np.zeros(self.nx)

        Ax = sparse.kron(sparse.eye(self.N+1),-sparse.eye(self.nx)) + sparse.kron(sparse.eye(self.N+1, k=-1), self._osqp_Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), self._osqp_Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, -np.tile(self._osqp_r,self.N)])

        lineq = np.hstack([np.kron(np.ones(self.N+1), self.xmin), np.kron(np.ones(self.N), self.umin)])
        uineq = np.hstack([np.kron(np.ones(self.N+1), self.xmax), np.kron(np.ones(self.N), self.umax)])

        ueq = leq

        #self._osqp_A = sparse.vstack([Aeq, Aineq]).tocsc()
        self._osqp_A = sparse.csc_matrix(Aeq)
        #self._osqp_l = np.hstack([leq, lineq])
        #self._osqp_u = np.hstack([ueq, uineq])
        self._osqp_l = leq
        self._osqp_u = ueq

    def solve_to_convergence(self, x, t, eps=1e-3, x_init=None, u_init=None, max_iter=1):
        self.eval(x, t)

        CQC = sparse.csc_matrix(np.transpose(self.C).dot(self.Q.dot(self.C)))
        CQNC = sparse.csc_matrix(np.transpose(self.C).dot(self.QN.dot(self.C)))
        QCT = np.transpose(self.Q.dot(self.C))
        CTQ = self.C.T@self.Q
        CTQN = self.C.T@self.QN
        QNCT = np.transpose(self.QN.dot(self.C))
        if self.q_d.ndim == 2:
            xr = self.q_d[:, :self.N+1]
            q_0 = np.hstack([np.reshape(-CTQ.dot(xr[:,:-1]), ((self.N) * self.nx,), order='F'),
                                      np.reshape(-0.5*(CTQN.dot(xr[:,-1])),self.nx),
                                      np.tile(self.R.dot(self.const_offset), (self.N))]) # TODO: Test that final state penalty is correct
        else:
            xr = self.q_d
            q_0 = np.hstack([np.tile(-CTQ.dot(xr), self.N),
                                      -CTQN.dot(xr),
                                      np.tile(self.R.dot(self.const_offset), (self.N))])

        sol_init = self._osqp_result.x
        iter = 0
        if x_init is None and u_init is None:
            self.cur_sol = sol_init.copy()
            x_init = self.parse_result().T
            u_init = self.get_control_prediction().T
        else:
            self.cur_sol = np.hstack((x_init.flatten(), u_init.flatten()))

        while (iter==0 or np.linalg.norm(sol_init-self.cur_sol) > eps) and iter < max_iter:
            sol_init = self.cur_sol.copy()
            if iter > 0:
                x_init = self.parse_result().T
                u_init = self.get_control_prediction().T

            # Update objective:
            q_diff = np.hstack([np.reshape(CQC.dot(x_init.T[:, :-1]), ((self.N) * self.nx,), order='F'),
                             CQNC.dot(x_init.T[:, -1]),
                             np.reshape(self.R.dot(u_init.T), (self.N*self.nu), order='F')])

            # Update equality constraint matrices:
            A_lst = [self.linearize_dynamics(z, z_next, u, None)[0] for z, z_next, u in zip(x_init[:-1,:], x_init[1:,:], u_init)]
            B_lst = [self.linearize_dynamics(z, z_next, u, None)[1] for z, z_next, u in zip(x_init[:-1,:], x_init[1:,:], u_init)]
            r_lst = [self.linearize_dynamics(z, z_next, u, None)[2] for z, z_next, u in zip(x_init[:-1,:], x_init[1:,:], u_init)]

            Ax = np.kron(np.eye(self.N + 1), -np.eye(self.nx))
            Ax[self.nx:,:-self.nx] += sp.linalg.block_diag(*A_lst)
            Bu = np.zeros(((self.N+1)*self.nx,self.N*self.nu))
            Bu[self.nx:,:] += sp.linalg.block_diag(*B_lst)
            Aeq = sparse.hstack([sparse.csc_matrix(Ax), sparse.csc_matrix(Bu)])

            # Update equality constraint vectors:
            x = np.zeros(self.nx)
            r_vec = np.array(r_lst).flatten()
            leq = np.hstack((-x, -r_vec))
            ueq = leq

            # Update inequality constraint vectors:
            #lineq = np.hstack([np.tile(self.xmin, self.N+1)-(self.C@x_init.T).flatten(order='F'), np.tile(self.umin,self.N)-u_init.flatten()])
            #uineq = np.hstack([np.tile(self.xmax, self.N+1)-(self.C@x_init.T).flatten(order='F'), np.tile(self.umax,self.N)-u_init.flatten()])

            # Add updates to MPC problem:
            self._osqp_q = q_0 + q_diff
            self._osqp_A[:(self.N+1)*self.nx,:] = Aeq  # Linearization along initial guess, state evolution
            #self._osqp_l = np.hstack([leq, lineq])
            #self._osqp_u = np.hstack([ueq, uineq])
            self._osqp_l = leq
            self._osqp_u = ueq
            self.prob.update(q=self._osqp_q, Ax=self._osqp_A.data, l=self._osqp_l, u=self._osqp_u)

            # Solve MPC Instance
            #self.prob.warm_start(x=self.cur_sol)
            self._osqp_result = self.prob.solve()
            self.comp_time.append(self._osqp_result.info.run_time)
            self.cur_sol = sol_init + self._osqp_result.x
            iter += 1
            self.iter_sol.append(self.cur_sol.copy())
            print(iter,': ', np.linalg.norm(sol_init-self.cur_sol))

    def eval(self, x, t):
        """eval Function to evaluate controller
        
        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time
        
        Returns:
            control action -- numpy array [Nu,]
        """

        ## Update inequalities
        self.update_constraints_(x, t)

        ## Solve MPC Instance
        self._osqp_result = self.prob.solve()
        self.comp_time.append(self._osqp_result.info.run_time)

        return self._osqp_result.x[-self.N*self.nu:-(self.N-1)*self.nu]

    def update_constraints_(self, x, t):
        x = self.dynamics_object.lift(x.reshape((1, -1)), None).squeeze()
        self._osqp_l[:self.nx] = -x
        self._osqp_u[:self.nx] = -x

        if self.q_d.ndim == 2:
            # Update the local reference trajectory
            tindex = int(t / self.dt)
            xr = self.q_d[:, tindex:tindex + self.N + 1]

            # Construct the new _osqp_q objects
            QCT = np.transpose(self.Q.dot(self.C))
            self._osqp_q = np.hstack(
                [np.reshape(-QCT.dot(xr), ((self.N + 1) * self.nx,), order='F'), np.zeros(self.N * self.nu)])

            # TODO: Update dynamics linearization ...

            self.prob.update(q=self._osqp_q, l=self._osqp_l, u=self._osqp_u)
        else:
            self.prob.update(l=self._osqp_l, u=self._osqp_u)

    def parse_result(self):
        return np.transpose(np.reshape(self.cur_sol[:(self.N+1)*self.nx], (self.N+1,self.nx)))

    def get_control_prediction(self):
        return np.transpose(np.reshape(self.cur_sol[-self.N*self.nu:], (self.N,self.nu)))