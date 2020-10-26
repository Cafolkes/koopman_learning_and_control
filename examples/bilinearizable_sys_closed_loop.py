#!/usr/bin/env python
# coding: utf-8

# $\newcommand{\mb}[1]{\mathbf{ #1 }}$
# $\newcommand{\bs}[1]{\boldsymbol{ #1 }}$
# $\newcommand{\bb}[1]{\mathbb{ #1 }}$
# 
# $\newcommand{\R}{\bb{R}}$
# 
# $\newcommand{\ip}[2]{\left\langle #1, #2 \right\rangle}$
# $\newcommand{\norm}[1]{\left\Vert #1 \right\Vert}$
# 
# $\newcommand{\der}[2]{\frac{\mathrm{d} #1 }{\mathrm{d} #2 }}$
# $\newcommand{\derp}[2]{\frac{\partial #1 }{\partial #2 }}$

# In[1]:


import numpy as np
import sys
import sympy as sym
sys.path.append('../../../')


# # Define experiment parameters

# In[2]:


from core.dynamics import RoboticDynamics

class FiniteDimKoopSys(RoboticDynamics):
    def __init__(self, lambd, mu, c):
        RoboticDynamics.__init__(self, 2, 2)
        self.params = lambd, mu, c
    
    def D(self, q):
        return np.array([[1, 0],[0, (q[0]+1)**(-1)]])
    
    def C(self, q, q_dot):
        labmd, mu, c = self.params
        return -np.array([[lambd, 0], [(q[0]+1)**(-1)*(2*lambd - mu)*c*q_dot[0], (q[0]+1)**(-1)*mu]])
    
    def G(self, q):
        return np.array([0, 0])
    
    def B(self, q):
        return np.array([[1, 0], [0, 1]])

class FiniteDimKoopSysDiscrete(FiniteDimKoopSys):
    def __init__(self, lambd, mu, c, dt):
        FiniteDimKoopSys.__init__(self, lambd, mu, c)
        self.dt = dt

    def eval_dot(self, x, u, t):
        return x + self.dt*self.drift(x, t) + self.dt*np.dot(self.act(x, t),u)

    def get_linearization(self, x0, x1, u0, t):
        lambd, mu, c = self.params
        A_lin = np.eye(self.n) + self.dt*np.array([[0, 0, 1, 0],
                                            [0, 0, 0, 1],
                                            [0, 0, lambd, 0],
                                            [u0[1], 0, 2*(2*lambd-mu)*c*x0[2], mu]])

        B_lin = self.dt*np.array([[0, 0],
                          [0, 0],
                          [1, 0],
                          [0, x0[0]+1]])

        if x1 is None:
            x1 = A_lin@x0 + B_lin@u0

        f_d = self.eval_dot(x0,u0,t)
        r_lin = f_d - x1

        return A_lin, B_lin, r_lin

n, m = 4, 2
lambd, mu, c = .3, .2, -.5
finite_dim_koop_sys = FiniteDimKoopSys(lambd, mu, c)


# In[3]:


from core.dynamics import LinearSystemDynamics

A_lin = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, lambd, 0],
                  [0, 0, 0, mu]])
B_lin = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
dt = 1e-2
linearized_sys = LinearSystemDynamics(A_lin, B_lin)

# MPC parameters:
umax = np.array([10., 10.])
#umax = np.array([np.inf, np.inf])
umin = -umax
xmax = np.array([3., 6., 3., 3.])
#xmax = np.array([np.inf, np.inf, np.inf, np.inf])
xmin = -xmax
q, r = 1e3, 1
Q_mpc = q*np.eye(n)
R_mpc = r*np.eye(m)
traj_length = 200

# Design trajectory:
x0 = np.array([0., 0., 0., 0.])
set_pt = np.array([2., 5., 0., 0.])


# # Define nonlinear MPC for trajectory generation

# #### Define model

# In[4]:


from casadi import *
import do_mpc

model = do_mpc.model.Model('continuous')
x = model.set_variable('_x', 'x', (2,1))
dx = model.set_variable('_x', 'dx', (2,1))
u = model.set_variable('_u', 'force', (2,1))
ddx = model.set_variable('_z', 'ddx', (2,1))

model.set_rhs('x', dx)
model.set_rhs('dx', ddx)

eul_lagrange = vertcat(
    ddx[0] - (lambd*dx[0] + u[0]),
    ddx[1] - (mu*dx[1] + (2*lambd-mu)*c*dx[0]**2 + (x[0]+1)*u[1])
)
model.set_alg('euler_lagrange', eul_lagrange)
quad_cost = q*((x[0]-set_pt[0])**2
              +(x[1]-set_pt[1])**2
              +(dx[0]-set_pt[2])**2
              +(dx[1]-set_pt[3])**2)
model.set_expression(expr_name='cost', expr=quad_cost)
model.setup()


# #### Define controller

# In[5]:


mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': traj_length,
    'n_robust': 0,
    'open_loop': 0,
    't_step': dt,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
}
mpc.set_param(**setup_mpc)
mpc.set_objective(lterm=SX.zeros(1,1), mterm=model.aux['cost'])
mpc.set_rterm(force=np.array([r, r]))

mpc.bounds['lower', '_x', 'x'] = xmin[:2]
mpc.bounds['lower', '_x', 'dx'] = xmin[2:]
mpc.bounds['upper', '_x', 'x'] = xmax[:2]
mpc.bounds['upper', '_x', 'dx'] = xmax[2:]

mpc.bounds['lower','_u','force'] = umin
mpc.bounds['upper', '_u', 'force'] = umax

mpc.setup()


# #### Set up simulation and evaluate open loop prediction

# In[6]:


mpc.x0 = x0
mpc.set_initial_guess()
mpc.make_step(x0)

xr_cl = np.vstack((mpc.data.prediction(('_x','x')).squeeze(), mpc.data.prediction(('_x','dx')).squeeze()))
ur_cl = mpc.data.prediction(('_u','force')).squeeze()


# # Construct Koopman bilinear form of the system

# As a result of the careful construction of this system, there exists a Koopman canonical transform, $\mb z = T(\mb x)$ that exactly transforms the control-affine dynamics into a bilinear system. Consider the transformation:

# \begin{equation}
#     T(\mb q, \mb{\dot{q}}) = \begin{bmatrix} 
#     \phi_1(\mb q, \mb{\dot{q}})\\
#     \phi_2(\mb q, \mb{\dot{q}})\\
#     \phi_3(\mb q, \mb{\dot{q}})\\
#     \phi_4(\mb q, \mb{\dot{q}})\\
#     \phi_5(\mb q, \mb{\dot{q}})\\
#     \phi_6(\mb q, \mb{\dot{q}})\\
#     \end{bmatrix}
#     = \begin{bmatrix} 
#     q_1 - \frac{1}{\lambda}\dot{q}_1\\
#     q_2 - \frac{1}{\mu} \dot{q}_2 + \frac{(2 \lambda - \mu)c}{2\lambda \mu} \dot{q}_1^2\\
#     \dot{q}_1\\
#     \dot{q}_2 - c \dot{q}_1^2\\
#     \dot{q}_1^2\\
#     1
#     \end{bmatrix},
# \end{equation}

# where $\phi_1, \phi_2, \phi_3, \phi_4, \phi_5, \phi_6$ are eigenfunctions of the Koopman operator associated with the drift 
# vector field $\mb f_0$. The matrix with the eigenvalue associated with the i-th eigenfunction on the i-th diagonal 
# element is $F=\text{diag}(0, 0, \lambda, \mu, 2 \lambda, 0)$. Then, to reformulate the dynamics we have: 

# \begin{equation} 
# L_{\mb f_1} T(\mb q, \mb{\dot{q}}) = \begin{bmatrix} -\frac{1}{\lambda}\\ \frac{(2\lambda - \mu)c}{\lambda \mu}\dot{q}_1\\ 1 \\ -2c\dot{q}_1 \\ 2\dot{q_1} \\ 0 \end{bmatrix}, \qquad 
# L_{\mb f_2} T(\mb q, \mb{\dot{q}}) = \begin{bmatrix} 0\\ -\frac{1}{\mu}(q_1 + 1)\\0 \\ q_1 + 1 \\ 0 \\ 0 \end{bmatrix} 
# \end{equation}

# and the dynamics can be equivalently transformed to a bilinear form $\mb{\dot{z}} = F \mb z + G_1\mb z u_1 + G_2\mb z u_2$ with 

# \begin{equation}
# F = \begin{bmatrix} 
# 0 & 0 & 0 & 0 & 0 & 0\\
# 0 & 0 & 0 & 0 & 0 & 0\\
# 0 & 0 & \lambda & 0 & 0 & 0\\
# 0 & 0 & 0 & \mu & 0 & 0\\
# 0 & 0 & 0 & 0 & 2 \lambda & 0\\
# 0 & 0 & 0 & 0 & 0 & 0\\  
# \end{bmatrix}, \qquad 
# G_1 = \begin{bmatrix} 
# 0 & 0 & 0 & 0 & 0 & -\frac{1}{\lambda}\\
# 0 & 0 & \frac{(2\lambda - \mu)c}{\lambda \mu} & 0 & 0 & 0\\
# 0 & 0 & 0 & 0 & 0 & 1\\
# 0 & 0 & -2c & 0 & 0 & 0\\
# 0 & 0 & 2 & 0 & 0 & 0\\
# 0 & 0 & 0 & 0 & 0 & 0\\  
# \end{bmatrix}
# , \qquad 
# G_2 = \begin{bmatrix} 
# 0 & 0 &  0 & 0 & 0 & 0\\
# -\frac{1}{\mu} & 0 &  -\frac{1}{\lambda \mu} & 0 & 0 & -\frac{1}{\mu}\\
# 0 & 0 & 0 & 0 & 0 & 0\\
# 1 & 0 & \frac{1}{\lambda} & 0 & 0 & 1\\
# 0 & 0 & 0 & 0 & 0 & 0\\
# 0 & 0 & 0 & 0 & 0 & 0\\  
# \end{bmatrix}
# \end{equation}

# In[7]:


from koopman_core.dynamics import BilinearLiftedDynamics

F = np.array([[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, lambd, 0, 0],
              [0, 0, 0, 0, mu, 0],
              [0, 0, 0, 0, 0, 2*lambd]])
G_1 = np.array([[0, 0, 0, 0, 0, 0],
                [-1/lambd, 0, 0, 0, 0, 0],
                [0, 0, 0, (2*lambd - mu)*c/(lambd*mu), 0, 0],
                [1, 0, 0, 0, 0, 0], 
                [0, 0, 0, -2*c, 0, 0], 
                [0, 0, 0, 2, 0, 0]])
G_2 = np.array([[0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-1/mu, -1/mu, 0, -1/(lambd*mu), 0, 0],
                [0, 0, 0, 0, 0, 0], 
                [1, 1, 0, 1/lambd, 0, 0], 
                [0, 0, 0, 0, 0, 0]])
G = [G_1, G_2]
phi_fun = lambda x: np.array([1,
                            x[0,0] - 1/lambd*x[0,2],
                            x[0,1] - 1/mu*x[0,3] + (2*lambd - mu)*c/(2*lambd*mu)*x[0,2]**2,
                            x[0,2],
                            x[0,3] - c*x[0,2]**2,
                            x[0,2]**2])
phi_fun_grad = lambda x: np.array([[0., 0., 0., 0.]
                                   [1., 0., -1/lambd, 0.],
                                   [0., 1., (2*lambd-mu)*c/(lambd*mu)*x[0,2], -1/mu],
                                   [0., 0., 1., 0.],
                                   [0., 0., -2*c*x[0,2], 1.],
                                   [0., 0., 2*x[0,2], 0.]])

C_q = np.array([[0, 1, 0, 1/lambd, 0, 0],
                [0, 0, 1, 0, 1/mu, c/mu - ((2*lambd - mu)*c)/(2*lambd*mu)]])
C_qdot = np.array([[0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, c]])
C = np.concatenate((C_q, C_qdot), axis=0)

n_koop = 6

xtest = np.array([10, 2, 3, 4])

koop_bilinear_sys = BilinearLiftedDynamics(n_koop, m, F, G, C, phi_fun)


# In[8]:


# # Design model predictive controllers based on the bilinear model

# In[9]:


#from koopman_core.controllers import BilinearMPCController

#controller_koop = BilinearMPCController(koop_bilinear_sys, traj_length, dt, umin, umax, xmin, xmax, np.zeros_like(Q_mpc), R_mpc, Q_mpc, set_pt)


# In[10]:


#controller_koop.eval(x0, 0.)
#xr_koop = koop_bilinear_sys.C@controller_koop.parse_result()
#ur_koop = controller_koop.get_control_prediction()


# #### QP-MPC based on linear transformation of the bilinear dynamics

# In[11]:


from koopman_core.controllers import MPCController
from koopman_core.dynamics import LinearLiftedDynamics

z0 = phi_fun(x0.reshape((1,-1)))
A_0 = koop_bilinear_sys.A
B_0 = np.array([np.dot(b,z0) for b in koop_bilinear_sys.B]).T

init_sys = LinearLiftedDynamics(A_0, B_0, C, koop_bilinear_sys.basis)

controller_koop = MPCController(init_sys, traj_length, dt, umin, umax, xmin, xmax, np.zeros_like(Q_mpc), R_mpc, Q_mpc, set_pt)
controller_koop.eval(x0,0.)
xr_koop = koop_bilinear_sys.C@controller_koop.parse_result()
ur_koop = controller_koop.get_control_prediction()

from koopman_core.controllers import BilinearMPCController, BilinearMPCControllerCVX, NonlinearMPCController
A_d = np.eye(n_koop) + koop_bilinear_sys.A*dt
B_d = [b*dt for b in koop_bilinear_sys.B]
kbf_d = BilinearLiftedDynamics(n_koop, m, A_d, B_d, C, koop_bilinear_sys.basis, continuous=False, dt=dt)
controller_kbf = NonlinearMPCController(kbf_d, traj_length, dt, umin, umax, xmin, xmax, np.zeros_like(Q_mpc), R_mpc, Q_mpc, set_pt, terminal_constraint=True)
controller_kbf_cvx = BilinearMPCControllerCVX(kbf_d, traj_length, dt, umin, umax, xmin, xmax, np.zeros_like(Q_mpc), R_mpc, Q_mpc, set_pt)

z_init = np.tile(z0.reshape((-1,1)),(1,traj_length+1)).T
u_init = np.zeros((m,traj_length)).T
#zr_cl = np.array([koop_bilinear_sys.lift(x.reshape(1,-1),None) for x in xr_cl.T])
#z_init = zr_cl
#u_init = ur_cl.T

controller_kbf.construct_controller(z_init, u_init)
controller_kbf.solve_to_convergence(z0, 0., z_init, u_init, max_iter=100)
xr_kbf = koop_bilinear_sys.C@controller_kbf.get_state_prediction().T
ur_kbf = controller_kbf.get_control_prediction().T

controller_kbf_cvx.solve_to_convergence(z0, 0., z_init, u_init, max_iter=1)
xr_kbf_cvx = koop_bilinear_sys.C@controller_kbf_cvx.get_state_prediction().T
ur_kbf_cvx = controller_kbf_cvx.get_control_prediction().T

finite_dim_koop_sys_d = FiniteDimKoopSysDiscrete(lambd, mu, c, dt)
controller_nmpc = NonlinearMPCController(finite_dim_koop_sys_d, traj_length, dt, umin, umax, xmin, xmax, np.zeros_like(Q_mpc), R_mpc, Q_mpc, set_pt)
x_init = np.tile(x0.reshape((-1,1)),(1,traj_length+1)).T
controller_nmpc.construct_controller(x_init, u_init)
controller_nmpc.solve_to_convergence(x0, 0., x_init, u_init, max_iter=100)
xr_nmpc = controller_nmpc.get_state_prediction().T
ur_nmpc = controller_nmpc.get_control_prediction().T

print('Number of SQP iterations Koopman NMPC: ', len(controller_kbf.x_iter))
print('Number of SQP iterations NMPC: ', len(controller_nmpc.x_iter))

# In[12]:
import matplotlib.pyplot as plt
t_eval = np.arange(0,traj_length+1)*dt
plot_inds = [0, 2, 1, 3, 0, 1]
labels = ['$x_1$ (m)', '$x_2$ (m)', '$\\dot{x}_1$ (m/s)','$\\dot{x}_2$ (m/s)','$F_1$ (N)','$F_2$ (N)']

plt.figure(figsize=(15,10))
plt.suptitle('Trajectory designed with nonlinear MPC\n(blue lines-designed trajectory, black dotted lines-state/actuation bounds')
for ii in range(6):
    ind = plot_inds[ii]
    if ii < 4:
        plt.subplot(3,2,ii+1)
        plt.plot(t_eval, xr_nmpc[ind,:], label='Nonlinear MPC')
        plt.plot(t_eval, xr_koop[ind,:], label='Linear MPC')
        plt.plot(t_eval, xr_kbf[ind, :], label='Bilinear MPC')
        plt.plot(t_eval, xr_kbf_cvx[ind, :], label='Bilinear MPC cvx')
        plt.plot([0, 2.], [xmax[ind], xmax[ind]], ':k')
        plt.plot([0, 2.], [xmin[ind], xmin[ind]], ':k')
        plt.scatter(t_eval[0], x0[ind], color='g')
        plt.scatter(t_eval[-1], set_pt[ind], color='r')
        plt.ylabel(labels[ind])
        plt.grid()
        if ii==1:
            plt.legend(loc='upper right', ncol=4)
    else:
        plt.subplot(3,2,ii+1)
        plt.plot(t_eval[:-1],ur_nmpc[ind,:])
        plt.plot(t_eval[:-1],ur_koop[ind,:])
        plt.plot(t_eval[:-1], ur_kbf[ind, :])
        plt.plot(t_eval[:-1], ur_kbf_cvx[ind, :])
        plt.plot([0, 2.], [umax[ind], umax[ind]], ':k')
        plt.plot([0, 2.], [umin[ind], umin[ind]], ':k')
        plt.ylabel(labels[ii])
        plt.xlabel('Time (sec)')
        plt.grid()
            
plt.show()


# In[14]:


traj_cost_ref = (xr_cl[:,-1]-set_pt).T@Q_mpc@(xr_cl[:,-1]-set_pt) + np.sum(np.diag(ur_cl.T@R_mpc@ur_cl))
traj_cost_koop = (xr_kbf[:,-1]-set_pt).T@Q_mpc@(xr_kbf[:,-1]-set_pt) + np.sum(np.diag(ur_kbf.T@R_mpc@ur_kbf))
print(xr_kbf[:,-1])
print('Closed loop quadratic cost:')
print('   Ref:   ', "{:.3f}".format(traj_cost_ref),
      '\n   bEDMD: ', "{:.3f}".format(traj_cost_koop),
      )
print('\n   Relative increased cost Ref -> koop:  ', "{:.2f}".format(100*((traj_cost_koop)/(traj_cost_ref)-1)), ' %'
      )


# #### Simulate system open loop with both control trajectories

# In[15]:


from koopman_core.controllers import OpenLoopController
ol_controller = OpenLoopController(finite_dim_koop_sys, ur_nmpc.T, t_eval[:-1])
xs_nmpc, us_nmpc = finite_dim_koop_sys.simulate(x0, ol_controller, t_eval)
xs_nmpc, us_nmpc = xs_nmpc.T, us_nmpc.T

ol_controller_koop = OpenLoopController(finite_dim_koop_sys, ur_koop.T, t_eval[:-1])
xs_koop, us_koop = finite_dim_koop_sys.simulate(x0, ol_controller_koop, t_eval)
xs_koop, us_koop = xs_koop.T, us_koop.T

ol_controller_kbf= OpenLoopController(finite_dim_koop_sys, ur_kbf.T, t_eval[:-1])
xs_kbf, us_kbf = finite_dim_koop_sys.simulate(x0, ol_controller_kbf, t_eval)
xs_kbf, us_kbf = xs_kbf.T, us_kbf.T

# In[16]:


plt.figure(figsize=(15,10))
plt.suptitle('Open loop simulation of control trajectories\n(blue lines-designed trajectory, black dotted lines-state/actuation bounds')
for ii in range(6):
    ind = plot_inds[ii]
    if ii < 4:
        plt.subplot(3,2,ii+1)
        plt.plot(t_eval, xs_nmpc[ind,:], label='Nonlinear MPC')
        plt.plot(t_eval, xs_koop[ind,:], label='Linear MPC')
        plt.plot(t_eval, xs_kbf[ind, :], label='Bilinear MPC')
        plt.plot([0, 2.], [xmax[ind], xmax[ind]], ':k')
        plt.plot([0, 2.], [xmin[ind], xmin[ind]], ':k')
        plt.scatter(t_eval[0], x0[ind], color='g')
        plt.scatter(t_eval[-1], set_pt[ind], color='r')
        plt.ylabel(labels[ind])
        plt.grid()
        if ii==1:
            plt.legend(loc='upper right', ncol=2)
    else:
        plt.subplot(3,2,ii+1)
        plt.plot(t_eval[:-1],us_nmpc[ind,:])
        plt.plot(t_eval[:-1],us_koop[ind,:])
        plt.plot(t_eval[:-1], us_kbf[ind, :])
        plt.plot([0, 2.], [umax[ind], umax[ind]], ':k')
        plt.plot([0, 2.], [umin[ind], umin[ind]], ':k')
        plt.ylabel(labels[ii])
        plt.xlabel('Time (sec)')
        plt.grid()
            
plt.show()

xs_knmpc_cl, us_knmpc_cl = finite_dim_koop_sys.simulate(x0, controller_kbf, t_eval)
xs_knmpc_cl, us_knmpc_cl = xs_knmpc_cl.T, us_knmpc_cl.T


# In[ ]:




