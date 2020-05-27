

from numpy import zeros
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import osqp
import matplotlib.pyplot as plt

from core.controllers.controller import Controller
from koopman_core.learning.edmd import Edmd


def block_diag(M,n):
  """bd creates a sparse block diagonal matrix by repeating M n times
  
  Args:
      M (2d numpy array): matrix to be repeated
      n (float): number of times to repeat
  """
  return sparse.block_diag([M for i in range(n)])


class MPCControllerFast(Controller):
    """
    Class for controllers MPC.

    MPC are solved using osqp.

    Use lifting=True to solve MPC in the lifted space
    """
    def __init__(self, linear_dynamics, N, dt, umin, umax, xmin, xmax, 
                 Q, R, QN, xr, plotMPC=False, plotMPC_filename="",
                 lifting=False, edmd_object=None,name="noname", soft=False, D=None):
        """Create an MPC Controller object.

        Sizes:
        - N: number of timesteps for predictions
        - Nqd: number of timesteps of the desired trajectory
        - nx: number of states (original or lifted)
        - ns: number or original states
        - nu: number of control inputs

        Inputs:
        - initilized linear_dynamics, LinearSystemDynamics object. It takes Ac and Bc from it.
        - number of timesteps, N, integer
        - time step, dt, float
        - minimum control,  umin, numpy 1d array [nu,]
        - maximum control,  umax, numpy 1d array [nu,]
        - minimum state,    xmin, numpy 1d array [ns,]
        - maximum state,    xmax, numpy 1d array [ns,]
        - state cost matrix    Q, sparse numpy 2d array [ns,ns]. In practice it is always diagonal. 
        - control cost matrix, R, sparse numpy 2d array [nu,nu]. In practice it is always diagonal. 
        - final state cost matrix,  QN, sparse numpy 2d array [ns,ns]. In practice it is always diagonal. 
        - reference state trajectory, xr, numpy 2d array [ns,Nqd] OR numpy 1d array [ns,]
        (Optional)
        - flag to plot MPC thoughts, plotMPC=False, boolean
        - filename to save the previosu plot, plotMPC_filename="", string
        - flag to use or not lifting, lifting=False, boolean
        - object to store the eDMD data, edmd_object=Edmd(). It has been initialized. s
        """

        Controller.__init__(self, linear_dynamics)

        # Load arguments
        Ac, Bc = linear_dynamics.linear_system()
        [nx, nu] = Bc.shape
        ns = xr.shape[0]

        self.dt = dt
        self.plotMPC = plotMPC
        self.plotMPC_filename = plotMPC_filename
        self.verbose = False
        self.q_d = xr


        self.Q = Q
        self.R = R
        self.lifting = lifting
        self.soft = soft


        self.nu = nu
        self.nx = nx
        self.ns = ns
        
        #Discretize dynamics:
        self.dt = dt
        if lifting:
            self.C = edmd_object.C
            self.edmd_object = edmd_object
        else:
            self.C = sparse.eye(ns)
        lin_model_d = sp.signal.cont2discrete((Ac,Bc,self.C,zeros((ns,1))),dt)
        Ad = sparse.csc_matrix(lin_model_d[0]) #TODO: If bad behavior, delete this
        Bd = sparse.csc_matrix(lin_model_d[1]) #TODO: If bad behavior, delete this

        # Total desired path
        if self.q_d.ndim==2:
            self.Nqd = self.q_d.shape[1]
            xr = self.q_d[:,:N]

        # Prediction horizon
        self.N = N
        x0 = np.zeros(nx)
        self.run_time = np.zeros([0,])
        self.xr = self.q_d[:, :N]
               

        Rbd = sparse.kron(sparse.eye(N), R)
        Qbd = sparse.kron(sparse.eye(N), Q)
        Bbd = block_diag(Bd,nu).tocoo()


        # Check Xmin and Xmax
        if  xmin.shape[0]==ns and xmin.ndim==1: # it is a single vector we tile it
            x_min_flat = np.kron(np.ones(N), xmin)
            x_max_flat = np.kron(np.ones(N), xmax)
        elif xmin.shape[0]==ns*N: # if it is a long vector it is ok
            x_min_flat = xmin
            x_max_flat = xmax
        elif xmin.shape[0] == ns and xmin.shape[1] == N: # if it is a block we flatten it
            x_min_flat = np.reshape(xmin,(N*ns,),order='F')
            x_max_flat = np.reshape(xmax,(N*ns,),order='F')
        else:
            raise ValueError('xmin has wrong dimensions. xmin shape={}'.format(xmin.shape))
        self.x_min_flat = x_min_flat 
        self.x_max_flat = x_max_flat


        # Check Umin and Umax
        if  umin.shape[0]==nu and umin.ndim==1:
            u_min_flat = np.kron(np.ones(N), umin)
            u_max_flat = np.kron(np.ones(N), umax)
        elif umin.shape[0]==nu*N:
            u_min_flat = umin
            u_max_flat = umax
        elif umin.shape[0] == nu and umin.shape[1] == N: 
            u_min_flat = np.reshape(umin,(N*nu,),order='F')
            u_max_flat = np.reshape(umax,(N*nu,),order='F')
        else:
            raise ValueError('umin has wrong dimensions. Umin shape={}'.format(umin.shape))
        self.u_min_flat = u_min_flat 
        self.u_max_flat = u_max_flat 

        #! GET a & b
        # Write B:
        diag_AkB = Bd
        data_list = Bbd.data
        row_list = Bbd.row
        col_list = Bbd.col
        B = sparse.coo_matrix
        for i in range(N):
            if i<N-1:
                AkB_bd_temp = block_diag(diag_AkB,N-i)
            else:
                AkB_bd_temp = diag_AkB.tocoo()
            data_list = np.hstack([data_list,AkB_bd_temp.data])
            row_list  = np.hstack([row_list,AkB_bd_temp.row+np.full((AkB_bd_temp.row.shape[0],),nx*i)])
            col_list  = np.hstack([col_list,AkB_bd_temp.col])

            diag_AkB = Ad.dot(diag_AkB)            

        B = sparse.coo_matrix((data_list, (row_list, col_list)), shape=(N*nx, N*nu))

        a = Ad.copy()
        Ak = Ad.copy()
        for i in range(N-1):
            Ak = Ak.dot(Ad)
            a = sparse.vstack([a,Ak])    

        
        self.a = a
        self.B = B

        # check_ab = True
        # if check_ab:
        #     x0  = np.linspace(-5,40,nx)
        #     x00 = np.linspace(-5,40,nx)
        #     # Store data Init
        #     nsim = N
        #     xst = np.zeros((nx,nsim))
        #     ust = np.zeros((nu,nsim))

        #     # Simulate in closed loop

        #     for i in range(nsim):
        #         # Fake pd controller
        #         ctrl = np.zeros(nu,) #np.random.rand(nu,)
        #         x0 = Ad.dot(x0) + Bd.dot(ctrl)

        #         # Store Data
        #         xst[:,i] = x0
        #         ust[:,i] = ctrl

        #     x_dense = np.reshape(a @ x00 + B @ (ust.flatten('F')),(N,nx)).T

        #     plt.figure()
        #     plt.subplot(2,1,1)
        #     for i in range(nx):
        #         plt.plot(range(nsim),xst[i,:],'d',label="sim "+str(i))
        #         plt.plot(range(nsim),x_dense[i,:],'d',label="ax+bu "+str(i))
        #     plt.xlabel('Time(s)')
        #     plt.grid()
        #     plt.legend()

        #     plt.subplot(2,1,2)
        #     for i in range(nu):
        #         plt.plot(range(nsim),ust[i,:],label=str(i))
        #     plt.xlabel('Time(s)')
        #     plt.grid()
        #     plt.legend()
        #     plt.savefig("AB_check_for_"+name+".png",bbox_inches='tight')
        #     plt.close()


        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        if (self.lifting):
            # Compute Block Diagonal elements
            self.Cbd = sparse.kron(sparse.eye(N), self.C)
            CQCbd  = self.Cbd.T @ Qbd @ self.Cbd
            self.CtQ = self.C.T @ Q
            Cbd = self.Cbd
            
            P = Rbd + B.T @ CQCbd @ B            
            self.BTQbda =  B.T @ CQCbd @ a            
            Aineq_x = Cbd @ B

            xrQB  = B.T @ np.reshape(self.CtQ.dot(xr),(N*nx,),order='F')
            l = np.hstack([x_min_flat - Cbd @ a @ x0, u_min_flat])
            u = np.hstack([x_max_flat - Cbd @ a @ x0, u_max_flat])

        else:
            # - quadratic objective
            P = Rbd + B.T @ Qbd @ B
            self.BTQbda =  B.T @ Qbd @ a
            xrQB  = B.T @ np.reshape(Q.dot(xr),(N*nx,),order='F')
            Aineq_x = B
        
            l = np.hstack([x_min_flat - a @ x0, u_min_flat])
            u = np.hstack([x_max_flat - a @ x0, u_max_flat])

        x0aQb = self.BTQbda @ x0
        self.Cbda = Cbd @ a
        self.BQxr  = self.B.T @ np.reshape(self.CtQ.dot(self.xr),(N*nx,),order='F')
        q = x0aQb - xrQB 
        Aineq_u = sparse.eye(N*nu)
        A = sparse.vstack([Aineq_x, Aineq_u]).tocsc()

        if soft:
            Pdelta = sparse.kron(sparse.eye(N), D)
            P = sparse.block_diag([P,Pdelta])
            qdelta = np.zeros(N*ns)
            q = np.hstack([q,qdelta])
            Adelta = sparse.csc_matrix(np.vstack([np.eye(N*ns),np.zeros((N*nu,N*ns))]))
            A = sparse.hstack([A, Adelta])


        # plot_matrices = True
        # if plot_matrices:
        #     #! Visualize Matrices
        #     fig = plt.figure()

        #     fig.suptitle("QP Matrices to solve MP in dense form. N={}, nx={}, nu={}".format(N,nx,nu),fontsize=20)
        #     plt.subplot(2,4,1,xlabel="Ns*(N+1)", ylabel="Ns*(N+1)")
        #     plt.imshow(a.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
        #     plt.title("a in $x=ax_0+bu$")
        #     plt.subplot(2,4,2,xlabel="Ns*(N+1)", ylabel="Nu*N")
        #     plt.imshow(B.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
        #     plt.title("b in $x=ax_0+bu$")
        #     plt.subplot(2,4,3,xlabel="ns*(N+1) + ns*(N+1) + nu*N", ylabel="Ns*(N+1)+Nu*N")
        #     plt.imshow(A.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
        #     plt.title("A total in $l\\leq Ax \\geq u$")
        #     plt.subplot(2,4,4)
        #     plt.imshow(P.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
        #     plt.title("P in $J=u^TPu+q^Tu$")
        #     plt.subplot(2,4,5)
        #     plt.imshow(Qbd.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
        #     plt.title("Qbd")


        #     #! Visualize Vectors
        #     plt.subplot(2,4,6)
        #     plt.plot(l)
        #     plt.title('l in  $l\\leq Ax \\geq u$')
        #     plt.grid()
        #     plt.subplot(2,4,7)
        #     plt.plot(u)
        #     plt.title("l in  $l\\leq Ax \\geq u$")
        #     plt.grid()
        #     plt.subplot(2,4,8)
        #     plt.plot(q)
        #     plt.title("q in $J=u^TPu+q^Tu$")
        #     plt.grid()
        #     plt.tight_layout()
        #     plt.savefig("MPC_matrices_for_"+name+".png",bbox_inches='tight')
        #     plt.close()
            #plt.show()

        self.osqp_size = P.shape[0]
        # Create an OSQP object
        self.prob = osqp.OSQP()
        # Setup workspace
        self.prob.setup(P=P.tocsc(), q=q, A=A, l=l, u=u, warm_start=True, verbose=self.verbose)

        if self.plotMPC:
            # Figure to plot MPC thoughts
            self.fig, self.axs = plt.subplots(self.ns+self.nu)
            if nx==4:
                ylabels = ['$x$', '$\\theta$', '$\\dot{x}$', '$\\dot{\\theta}$']
            else:
                ylabels = [str(i) for i in range(nx)]

            for ii in range(self.ns):
                self.axs[ii].set(xlabel='Time(s)',ylabel=ylabels[ii])
                self.axs[ii].grid()
            for ii in range(self.ns,self.ns+self.nu):
                self.axs[ii].set(xlabel='Time(s)',ylabel='u')
                self.axs[ii].grid()


    def eval(self, x, t):
        '''
        Args:
        - x, numpy 1d array [ns,]
        - time, t, float
        '''
        N = self.N
        nu = self.nu
        nx = self.nx
        x_original = np.copy(x)

        # Construct the new _osqp_q objects
        if (self.lifting):
            x = np.transpose(self.edmd_object.lift(x.reshape((x.shape[0],1)),self.xr[:,0].reshape((self.xr.shape[0],1))))[:,0]

            l = np.hstack([self.x_min_flat - self.Cbda @ x, self.u_min_flat])
            u = np.hstack([self.x_max_flat - self.Cbda @ x, self.u_max_flat])

        else:
            l = np.hstack([self.x_min_flat - self.a @ x, self.u_min_flat])
            u = np.hstack([self.x_max_flat - self.a @ x, self.u_max_flat])

        # Update initial state 
        q = self.BTQbda @ x  - self.BQxr

        if self.soft:
            q = np.hstack([q,np.zeros(N*self.ns)])   

        self.prob.update(q=q,l=l,u=u)

        ## Solve MPC Instance
        self._osqp_result = self.prob.solve()
        u_out = self._osqp_result.x[:nu]

        # Check solver status
        if self._osqp_result.info.status != 'solved':
            #raise ValueError('OSQP did not solve the problem!')
            u_out = np.zeros(nu)
            print('WARNING: Augmenting MPC not solved at x = ', x_original)

        return u_out

    def parse_result(self,x,u):
        return  np.transpose(np.reshape( self.a @ x + self.B @ u, (self.N+1,self.nx)))

    def get_control_prediction(self):
        return np.transpose(np.reshape( self._osqp_result.x[-self.N*self.nu:], (self.N,self.nu)))

    def plot_MPC(self, current_time, x0, xr, tindex):
        """plot mpc
        
       
        - current_time (float): time now
        - xr (2darray [N,ns]): local reference trajectory
        - tindex (int): index of the current time
        """
        #* Unpack OSQP results
        nu = self.nu
        nx = self.nx
        N = self.N

        u_flat = self._osqp_result.x
        osqp_sim_state =  np.reshape(self.a @ x0 + self.B @ u_flat,(N,nx)).T
        osqp_sim_forces = np.reshape(u_flat,(N,nu)).T

        if self.lifting:
            osqp_sim_state = np.dot(self.C,osqp_sim_state)

        pos = current_time/(self.Nqd*self.dt) # position along the trajectory
        time = np.linspace(current_time,current_time+N*self.dt,num=N)

        
        for ii in range(self.ns):
            if (tindex==0):
                self.axs[ii].plot(time,osqp_sim_state[ii,:],color=[0,1-pos,pos],label='x_0')
            elif (tindex==self.Nqd-2):
                self.axs[ii].plot(time,osqp_sim_state[ii,:],color=[0,1-pos,pos],label='x_f')
            else:
                self.axs[ii].plot(time,osqp_sim_state[ii,:],color=[0,1-pos,pos])
        for ii in range(self.nu):
            self.axs[ii+self.ns].plot(time,osqp_sim_forces[ii,:],color=[0,1-pos,pos])

    def update(self, xmin=None, xmax=None, umax=None, umin= None, Q=None):
        
        N, ns, nu = [self.N, self.ns, self.nu]
        if xmin is not None and xmax is not None:
            # Check Xmin and Xmax
            if  xmin.shape[0]==ns and xmin.ndim==1: # it is a single vector we tile it
                x_min_flat = np.kron(np.ones(N), xmin)
                x_max_flat = np.kron(np.ones(N), xmax)
            elif xmin.shape[0]==ns*N and xmin.ndim==1: # if it is a long vector it is ok
                x_min_flat = xmin
                x_max_flat = xmax
            elif xmin.shape[0] == ns and xmin.shape[1] == N: # if it is a block we flatten it
                x_min_flat = np.reshape(xmin,(N*ns,),order='F')
                x_max_flat = np.reshape(xmax,(N*ns,),order='F')
            else:
                raise ValueError('xmin has wrong dimensions. xmin shape={}'.format(xmin.shape))
            self.x_min_flat = x_min_flat 
            self.x_max_flat = x_max_flat

        if umin is not None and umax is not None: #TODO check it works 
        # Check Umin and Umax
            if  umin.shape[0]==nu and umin.ndim==1:
                u_min_flat = np.kron(np.ones(N), umin)
                u_max_flat = np.kron(np.ones(N), umax)
            elif umin.shape[0]==nu*N and umin.ndim==1:
                u_min_flat = umin
                u_max_flat = umax
            elif umin.shape[0] == nu and umin.shape[1] == N: 
                u_min_flat = np.reshape(umin,(N*nu,),order='F')
                u_max_flat = np.reshape(umax,(N*nu,),order='F')
            else:
                raise ValueError('umin has wrong dimensions. Umin shape={}'.format(umin.shape))
            self.u_min_flat = u_min_flat 
            self.u_max_flat = u_max_flat 

        if Q is not None:
            raise ValueError('Q changes is not implemented') #TODO implemented Q change

            """             a, B = [self.a, self.B]
            Qbd = sparse.kron(sparse.eye(N), Q)

            P = Rbd + B.T @ Qbd @ B
            self.BTQbda =  B.T @ Qbd @ a
            self.prob.update(P=P,l=l,u=u) """

        
        
            
    def finish_plot(self, x, u, u_pd, time_vector, filename):
        """
        Call this function to plot extra lines.

        - x: state, numpy 2darray [Nqd,n] 
        - u, input from this controller [Nqd-1,n] 
        - u_pd, input from a PD controller [Nqd-1,n] 
        - time_vector, 1d array [Nqd
        - filename, string
        """
        u = u.squeeze()
        u_pd = u_pd.squeeze()
        
        self.fig.suptitle(filename[:-4], fontsize=16)
        for ii in range(self.ns):
            self.axs[ii].plot(time_vector, self.q_d[ii,:], linewidth=2, label='$x_d$', color=[1,0,0])
            self.axs[ii].plot(time_vector, x[ii,:], linewidth=2, label='$x$', color=[0,0,0])
            self.axs[ii].legend(fontsize=10, loc='best')
        self.fig.savefig(filename)
        plt.close()



 