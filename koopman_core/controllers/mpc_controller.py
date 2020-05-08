import numpy as np
import scipy.sparse as sparse
import osqp
import matplotlib.pyplot as plt

from core.controllers.controller import Controller
from koopman_core.learning.edmd import Edmd


class MPCController(Controller):
    """
    Class for controllers MPC.

    MPC are solved using osqp.

    Use lifting=True to solve MPC in the lifted space
    """
    def __init__(self, linear_dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, plotMPC=False, plotMPC_filename="",lifting=False, edmd_object=Edmd()):
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

        Controller.__init__(self, linear_dynamics)

        # Load arguments
        Ac, Bc = linear_dynamics.linear_system()
        [nx, nu] = Bc.shape
        self.dt = dt
        self._osqp_Ad = sparse.eye(nx)+Ac*self.dt
        self._osqp_Bd = Bc*self.dt
        self.plotMPC = plotMPC
        self.plotMPC_filename = plotMPC_filename
        self.q_d = xr
        
        self.ns = xr.shape[0]

        self.Q = Q
        self.lifting = lifting

        self.nu = nu
        self.nx = nx

        # Total desired path
        if self.q_d.ndim==2:
            self.Nqd = self.q_d.shape[1]
            xr = self.q_d[:,:N+1]

        # Prediction horizon
        self.N = N
        x0 = np.zeros(nx)

        
        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        if (self.lifting):
            # Load eDMD objects
            self.C = edmd_object.C
            self.edmd_object = edmd_object

            # - quadratic objective
            CQC  = sparse.csc_matrix(np.transpose(edmd_object.C).dot(Q.dot(edmd_object.C)))
            CQNC = sparse.csc_matrix(np.transpose(edmd_object.C).dot(QN.dot(edmd_object.C)))
            P = sparse.block_diag([sparse.kron(sparse.eye(N), CQC), CQNC,
                                sparse.kron(sparse.eye(N), R)]).tocsc()

            # - linear objective
            QCT = np.transpose(Q.dot(edmd_object.C))
            QNCT = np.transpose(QN.dot(edmd_object.C))
            if (xr.ndim==1):
                q = np.hstack([np.kron(np.ones(N), -QCT.dot(xr)), -QNCT.dot(xr), np.zeros(N*nu)])
            elif (xr.ndim==2):
                q = np.hstack([np.reshape(-QCT.dot(xr),((N+1)*nx,),order='F'), np.zeros(N*nu)])

            # - input and state constraints
            Aineq = sparse.block_diag([edmd_object.C for i in range(N+1)]+[np.eye(N*nu)])


        else:
            # - quadratic objective
            P =  sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                                sparse.kron(sparse.eye(N), R)]).tocsc()
            # - linear objective
            if (xr.ndim==1):
                q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr), np.zeros(N*nu)])
            elif (xr.ndim==2):
                q = np.hstack([np.reshape(-Q.dot(xr),((N+1)*nx,),order='F'), np.zeros(N*nu)])  #TODO: Check if reshape is reshaping in the expected order

            # - input and state constraints
            Aineq = sparse.eye((N+1)*nx + N*nu)




        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), self._osqp_Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), self._osqp_Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])

        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])

        ueq = leq
        self._osqp_q = q

        A = sparse.vstack([Aeq, Aineq]).tocsc()
        self._osqp_l = np.hstack([leq, lineq])
        self._osqp_u = np.hstack([ueq, uineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P, q, A, self._osqp_l, self._osqp_u, warm_start=True, verbose=False)

        if self.plotMPC:
            # Figure to plot MPC thoughts
            self.fig, self.axs = plt.subplots(self.ns+self.nu)
            ylabels = ['$x$', '$\\theta$', '$\\dot{x}$', '$\\dot{\\theta}$']
            for ii in range(self.ns):
                self.axs[ii].set(xlabel='Time(s)',ylabel=ylabels[ii])
                self.axs[ii].grid()
            for ii in range(self.ns,self.ns+self.nu):
                self.axs[ii].set(xlabel='Time(s)',ylabel='u')
                self.axs[ii].grid()




    def eval(self, x, t):
        """eval Function to evaluate controller
        
        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time
        
        Returns:
            control action -- numpy array [Nu,]
        """

        N = self.N
        nu = self.nu
        nx = self.nx

        tindex = int(t/self.dt)
        #print("iteration {}".format(tindex))
        
        ## Update inequalities
        if self.q_d.ndim==2: 
            
            # Update the local reference trajectory
            if (tindex+N) < self.Nqd: # if we haven't reach the end of q_d yet
                xr = self.q_d[:,tindex:tindex+N+1]
            else: # we fill xr with copies of the last q_d
                xr = np.hstack( [self.q_d[:,tindex:],np.transpose(np.tile(self.q_d[:,-1],(N+1-self.Nqd+tindex,1)))])

            # Construct the new _osqp_q objects
            if (self.lifting):
                QCT = np.transpose(self.Q.dot(self.C))                        
                self._osqp_q = np.hstack([np.reshape(-QCT.dot(xr),((N+1)*nx,),order='F'), np.zeros(N*nu)])                    
            else:
                self._osqp_q = np.hstack([np.reshape(-self.Q.dot(xr),((N+1)*nx,),order='F'), np.zeros(N*nu)])

        if self.q_d.ndim==1:
            # Update the local reference trajectory
            xr = np.transpose(np.tile(self.q_d,N+1))

        # Lift the current state if necessary
        if (self.lifting): 
            x = np.transpose(self.edmd_object.lift(x.reshape((x.shape[0],1)),xr[:,0].reshape((xr.shape[0],1))))[:,0]
        
        self._osqp_l[:self.nx] = -x
        self._osqp_u[:self.nx] = -x

        self.prob.update(q=self._osqp_q, l=self._osqp_l, u=self._osqp_u)

        ## Solve MPC Instance
        self._osqp_result = self.prob.solve()

        # Check solver status
        #if self._osqp_result.info.status != 'solved':
        #    raise ValueError('OSQP did not solve the problem!')

        if self.plotMPC:
            self.plot_MPC(t, xr, tindex)
        return  self._osqp_result.x[-N*nu:-(N-1)*nu]

    def parse_result(self):
        return  np.transpose(np.reshape( self._osqp_result.x[:(self.N+1)*self.nx], (self.N+1,self.nx)))

    def get_control_prediction(self):
        return np.transpose(np.reshape( self._osqp_result.x[-self.N*self.nu:], (self.N,self.nu)))

    def plot_MPC(self, current_time, xr, tindex):
        """plot mpc
        
       
        - current_time (float): time now
        - xr (2darray [N,ns]): local reference trajectory
        - tindex (int): index of the current time
        """


        # Unpack OSQP results
        nu = self.nu
        nx = self.nx
        N = self.N

        osqp_sim_state = np.transpose(np.reshape( self._osqp_result.x[:(N+1)*nx], (N+1,nx)))
        osqp_sim_forces = np.transpose(np.reshape( self._osqp_result.x[-N*nu:], (N,nu)))

        if self.lifting:
            osqp_sim_state = np.dot(self.C,osqp_sim_state)

        # Plot
        pos = current_time/(self.Nqd*self.dt) # position along the trajectory
        time = np.linspace(current_time,current_time+N*self.dt,num=N+1)
        timeu = np.linspace(current_time,current_time+N*self.dt,num=N)

        
        for ii in range(self.ns):
            if (tindex==0):
                self.axs[ii].plot(time,osqp_sim_state[ii,:],color=[0,1-pos,pos],label='x_0')
            elif (tindex==self.Nqd-2):
                self.axs[ii].plot(time,osqp_sim_state[ii,:],color=[0,1-pos,pos],label='x_f')
            else:
                self.axs[ii].plot(time,osqp_sim_state[ii,:],color=[0,1-pos,pos])
        for ii in range(self.nu):
            self.axs[ii+self.ns].plot(timeu,osqp_sim_forces[ii,:],color=[0,1-pos,pos])
            
    def finish_plot(self, x, u, u_pd, time_vector, filename):
        """
        Call this function to plot extra lines.

        - x: state, numpy 2darray [Nqd,n] 
        - u, input from this controller [Nqd-1,n] 
        - u_pd, input from a PD controller [Nqd-1,n] 
        - time_vector, 1d array [Nqd
        - filename, string
        """
        self.fig.suptitle(filename[:-4], fontsize=16)
        for ii in range(self.ns):
            self.axs[ii].plot(time_vector, self.q_d[ii,:], linewidth=2, label='$x_d$', color=[1,0,0])
            self.axs[ii].plot(time_vector, x[ii,:], linewidth=2, label='$x$', color=[0,0,0])
            self.axs[ii].legend(fontsize=10, loc='best')
        for ii in range(self.nu):
            self.axs[ii+self.ns].plot(time_vector[:-1],u[ii,:],label='$u$',color=[0,0,0])
            self.axs[ii+self.ns].plot(time_vector[:-1],u_pd[ii,:],label='$u_{PD}$',color=[0,1,1])
            self.axs[ii+self.ns].legend(fontsize=10, loc='best')
        self.fig.savefig(filename)
        #plt.close(self.fig)



 