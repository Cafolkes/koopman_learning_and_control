from numpy import zeros
import time

import numpy as np
import scipy.sparse as sparse
from scipy.signal import cont2discrete
import osqp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from core.controllers.controller import Controller


def block_diag(M,n):
  """bd creates a sparse block diagonal matrix by repeating M n times
  
  Args:
      M (2d numpy array): matrix to be repeated
      n (float): number of times to repeat
  """
  return sparse.block_diag([M for i in range(n)])


class MPCControllerDense(Controller):
    """
    Class for controllers MPC.

    MPC are solved using osqp.

    Use lifting=True to solve MPC in the lifted space
    """
    def __init__(self, linear_dynamics, N, dt, umin, umax, xmin, xmax, 
                Q, R, QN, xr, plotMPC=False, plotMPC_filename="",lifting=False, edmd_object=None, name="noname", soft=False, D=None):
        """__init__ [summary]
        
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
            name {str} -- name for all saved files (default: {"noname"})
            soft {bool} -- flag to enable soft constraints (default: {False})
            D {[type]} -- cost matrix for the soft variables (default: {None})
        """

        Controller.__init__(self, linear_dynamics)

        # Load arguments
        Ac, Bc = linear_dynamics.linear_system()
        [nx, nu] = Bc.shape
        ns = xr.shape[0]

        #Discretize dynamics:
        self.dt = dt
        if lifting:
            self.C = edmd_object.C
            self.edmd_object = edmd_object
        else:
            self.C = sparse.eye(ns)

        lin_model_d = cont2discrete((Ac,Bc,self.C,zeros((ns,1))),dt)
        Ad = sparse.csc_matrix(lin_model_d[0]) #TODO: If bad behavior, delete this
        Bd = sparse.csc_matrix(lin_model_d[1]) #TODO: If bad behavior, delete this
        self.plotMPC = plotMPC
        self.plotMPC_filename = plotMPC_filename
        self.q_d = xr


        self.Q = Q
        self.R = R
        self.lifting = lifting

        self.nu = nu
        self.nx = nx
        self.ns = ns
        self.soft = soft
        self.comp_time = []

        # Total desired path
        if self.q_d.ndim==2:
            self.Nqd = self.q_d.shape[1]
            xr = self.q_d[:,:N]

        # Prediction horizon
        self.N = N
        x0 = np.zeros(nx)
        self.run_time = np.zeros([0,])
               

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

        check_ab = False
        if check_ab:
            x0  = np.linspace(-5,40,nx)
            x00 = np.linspace(-5,40,nx)
            # Store data Init
            nsim = N
            xst = np.zeros((nx,nsim))
            ust = np.zeros((nu,nsim))

            # Simulate in closed loop

            for i in range(nsim):
                # Fake pd controller
                ctrl = np.zeros(nu,) #np.random.rand(nu,)
                x0 = Ad.dot(x0) + Bd.dot(ctrl)

                # Store Data
                xst[:,i] = x0
                ust[:,i] = ctrl

            x_dense = np.reshape(a @ x00 + B @ (ust.flatten('F')),(N,nx)).T

            plt.figure()
            plt.subplot(2,1,1)
            for i in range(nx):
                plt.plot(range(nsim),xst[i,:],'d',label="sim "+str(i))
                plt.plot(range(nsim),x_dense[i,:],'d',label="ax+bu "+str(i))
            plt.xlabel('Time(s)')
            plt.grid()
            plt.legend()

            plt.subplot(2,1,2)
            for i in range(nu):
                plt.plot(range(nsim),ust[i,:],label=str(i))
            plt.xlabel('Time(s)')
            plt.grid()
            plt.legend()
            plt.savefig("AB_check_for_"+name+".pdf",bbox_inches='tight',format='pdf', dpi=2400)
            plt.close()


        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
 

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

        x0aQb = self.BTQbda @ x0
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

        plot_matrices = False
        if plot_matrices:
            #! Visualize Matrices
            fig = plt.figure()

            fig.suptitle("QP Matrices to solve MP in dense form. N={}, nx={}, nu={}".format(N,nx,nu),fontsize=20)
            plt.subplot(2,4,1,xlabel="Ns*(N+1)", ylabel="Ns*(N+1)")
            plt.imshow(a.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
            plt.title("a in $x=ax_0+bu$")
            plt.subplot(2,4,2,xlabel="Ns*(N+1)", ylabel="Nu*N")
            plt.imshow(B.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
            plt.title("b in $x=ax_0+bu$")
            plt.subplot(2,4,3,xlabel="ns*(N+1) + ns*(N+1) + nu*N", ylabel="Ns*(N+1)+Nu*N")
            plt.imshow(A.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
            plt.title("A total in $l\\leq Ax \\geq u$")
            plt.subplot(2,4,4)
            plt.imshow(P.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
            plt.title("P in $J=u^TPu+q^Tu$")
            plt.subplot(2,4,5)
            plt.imshow(Qbd.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
            plt.title("Qbd")


            #! Visualize Vectors
            plt.subplot(2,4,6)
            plt.plot(l)
            plt.title('l in  $l\\leq Ax \\geq u$')
            plt.grid()
            plt.subplot(2,4,7)
            plt.plot(u)
            plt.title("l in  $l\\leq Ax \\geq u$")
            plt.grid()
            plt.subplot(2,4,8)
            plt.plot(q)
            plt.title("q in $J=u^TPu+q^Tu$")
            plt.grid()
            plt.tight_layout()
            plt.savefig("MPC_matrices_for_"+name+".pdf",bbox_inches='tight',format='pdf', dpi=2400)
            plt.close()
            #plt.show()

        # Create an OSQP object
        self.prob = osqp.OSQP()
        # Setup workspace

        self.prob.setup(P=P.tocsc(), q=q, A=A, l=l, u=u, warm_start=True, verbose=False)


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
        time_eval0 = time.time()
        N = self.N
        nu = self.nu
        nx = self.nx
        ns = self.ns



        tindex = int(np.ceil(t/self.dt))  #TODO: Remove ceil and add back +1 if bad performance
            
        #print("Eval at t={:.2f}, x={}".format(t,x))
        # Update the local reference trajectory
        if (tindex+N) < self.Nqd: # if we haven't reach the end of q_d yet
            xr = self.q_d[:,tindex:tindex+N]
        else: # we fill xr with copies of the last q_d
            xr = np.hstack([self.q_d[:,tindex:],np.transpose(np.tile(self.q_d[:,-1],(N-self.Nqd+tindex,1)))])

        # Construct the new _osqp_q objects
        if (self.lifting):
            x = self.edmd_object.lift(x.reshape((1,-1)), np.zeros((1,self.nu))).squeeze()
            BQxr  = self.B.T @ np.reshape(self.CtQ.dot(xr),(N*nx,),order='F')
            l = np.hstack([self.x_min_flat - self.Cbd @ self.a @ x, self.u_min_flat])
            u = np.hstack([self.x_max_flat - self.Cbd @ self.a @ x, self.u_max_flat])

        else:
            BQxr  = self.B.T @ np.reshape(self.Q.dot(xr),(N*nx,),order='F')
            l = np.hstack([self.x_min_flat - self.a @ x, self.u_min_flat])
            u = np.hstack([self.x_max_flat - self.a @ x, self.u_max_flat])

        # Update initial state
        BQax0 = self.BTQbda @ x
        q = BQax0  - BQxr

        if self.soft:
            q = np.hstack([q,np.zeros(N*ns)])        

        self.prob.update(q=q,l=l,u=u)

        #print('Time Setup {:.2f}ms'.format(1000*(time.time()-time_eval0)))
        time_eval0 = time.time() 
        ## Solve MPC Instance
        self._osqp_result = self.prob.solve()
        self.comp_time.append(time.time()-time_eval0)
        #print('Time Solve {:.2f}ms'.format(1000*(time.time()-time_eval0)))
        time_eval0 = time.time() 

        # Check solver status
        if self._osqp_result.info.status != 'solved':
            print('ERROR: MPC DENSE coudl not be solved at t ={}, x = {}'.format(t, x))
            raise ValueError('OSQP did not solve the problem!')

        if self.plotMPC:
            self.plot_MPC(t, x, xr, tindex)

        self.run_time = np.append(self.run_time,self._osqp_result.info.run_time)

        return  self._osqp_result.x[:nu]

    def parse_result(self,x,u):
        """parse_result obtain state from MPC optimization
        
        Arguments:
            x {numpy array [Ns,]} -- initial state
            u {numpy array [Nu*N]} -- control action
        
        Returns:
            numpy array [Ns,N] -- state in the MPC optimization
        """
        return  np.transpose(np.reshape( self.a @ x + self.B @ u, (self.N+1,self.nx)))

    def get_control_prediction(self):
        """get_control_prediction parse control command from MPC optimization
        
        Returns:
            numpy array [N,Nu] -- control command along MPC optimization
        """
        return np.transpose(np.reshape( self._osqp_result.x[-self.N*self.nu:], (self.N,self.nu)))

    def plot_MPC(self, current_time, x0, xr, tindex):
        """plot_MPC Plot MPC thoughts
        
        Arguments:
            current_time {float} -- current time
            x0 {numpy array [Ns,]} -- current state
            xr {numpy array [Ns,N]} -- reference state
            tindex {float} -- time index along reference trajectory
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
        """update Change QP parameters
        
        Keyword Arguments:
            xmin {numpy array [Ns,]} -- minimum state bound (default: {None})
            xmax {numpy array [Ns,]} -- maximum state bound (default: {None})
            umax {numpy array [Nu,]} -- maximum command bound (default: {None})
            umin {numpy array [Nu,]} -- minimum command bound (default: {None})
            Q {numpy array [Ns,Ns]} -- state cost matrix (default: {None})
        
        """
        
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
        self.fig.savefig(filename,format='pdf', dpi=2400)
        plt.close()



 