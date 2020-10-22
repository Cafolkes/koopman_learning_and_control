import numpy as np
import cvxpy as cp
from core.controllers.controller import Controller


class BilinearMPCControllerCVX(Controller):
    """
    Class for linear MPC with lifted linear dynamics.

    Quadratic programs are solved using OSQP.
    """
    def __init__(self, lifted_bilinear_dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, const_offset=None):
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
        assert xr.ndim == 1, 'Desired trajectory not supported'
        self.xr = xr
        self.ns = xr.shape[0]
        
        self.comp_time = []
        self.iter_sol = []

    def linearize_dynamics(self, z0, z1, u0, t):
        A_lin = self.A + np.sum(np.array([b*u for b,u in zip(self.B, u0)]),axis=0)
        B_lin = np.array([b @ z0 for b in self.B]).T

        if z1 is None:
            z1 = A_lin@z0 + B_lin@u0

        f_d = self.dynamics_object.drift(z0, t) + np.dot(self.dynamics_object.act(z0,t),u0)
        r_lin = f_d - z1

        return A_lin, B_lin, r_lin

    def solve_to_convergence(self, z, t, z_init, u_init, eps=1e-3, max_iter=1):
        iter = 0
        self.cur_z = z_init
        self.cur_u = u_init
        u_prev = np.zeros_like(u_init)

        while (iter==0 or np.linalg.norm(u_prev-self.cur_u) > eps) and iter < max_iter:
            u_prev = self.cur_u.copy()
            z_init = self.cur_z.copy()
            u_init = self.cur_u.copy()

            # Update equality constraint matrices:
            A_lst = [self.linearize_dynamics(z, z_next, u, None)[0] for z, z_next, u in zip(z_init[:-1,:], z_init[1:,:], u_init)]
            B_lst = [self.linearize_dynamics(z, z_next, u, None)[1] for z, z_next, u in zip(z_init[:-1,:], z_init[1:,:], u_init)]
            r_lst = [self.linearize_dynamics(z, z_next, u, None)[2] for z, z_next, u in zip(z_init[:-1,:], z_init[1:,:], u_init)]

            # Solve MPC Instance
            dz, du = self.solve_mpc_(z, t, z_init, u_init, A_lst, B_lst, r_lst)
            self.cur_z = z_init + dz.T
            self.cur_u = u_init + du.T

            iter += 1
            print(iter,': ', np.linalg.norm(u_prev-self.cur_u))

    def solve_mpc_(self, z, t, z_init, u_init, A_lst, B_lst, r_lst):
        dz0 = z - z_init[0,:]
        dz = cp.Variable((self.nx, self.N + 1))
        du = cp.Variable((self.nu, self.N))

        cost = cp.quad_form(self.C@(z_init[-1,:] + dz[:,self.N])-self.xr,self.QN)
        constr = []

        for t in range(self.N):
            cost += cp.quad_form(u_init[t,:] + du[:,t],self.R)
            constr += [dz[:, t+1] == A_lst[t] @ dz[:,t] + B_lst[t]@du[:,t] + r_lst[t],
                       du[:,t] <= self.umax - u_init[t,:],
                       du[:,t] >= self.umin - u_init[t,:],
                       self.C@dz[:, t] <= self.xmax - self.C@z_init[t, :],
                       self.C@dz[:, t] >= self.xmin - self.C@z_init[t, :]
                       ]

        # sums problem objectives and concatenates constraints.
        constr += [dz[:,0] == dz0]
        #constr += [self.C@(z_init[-1,:] + dz[:,-1]) == xr]
        self.problem = cp.Problem(cp.Minimize(cost), constr)
        res = self.problem.solve(solver=cp.OSQP)
        return dz.value, du.value

    def eval(self, x, t):
        """eval Function to evaluate controller
        
        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time
        
        Returns:
            control action -- numpy array [Nu,]
        """
        pass

    def update_constraints_(self, x, t):
        pass

    def get_state_prediction(self):
        return self.cur_z

    def get_control_prediction(self):
        return self.cur_u