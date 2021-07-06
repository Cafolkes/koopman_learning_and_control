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
# 
# # Finite Dimensional Koopman Bilinear System

# Consider a nonlinear dynamical system that allows an exact finite dimensional Koopman canonical transform such that the control-affine dynamics can be transformed to a bilinear system. Consider the dynamical system 

# \begin{equation}
# \mb{\dot{x}}=\mb{f}_0(\mb x) + \mb f_1 ( \mb x) u_1 + \mb f_2(\mb x) u_2, 
# \end{equation}

# where we for this example choose $\mb f_0, \mb f_1$ as follows:

# \begin{equation}
# \mb f_0(\mb x) = \begin{bmatrix} x_3 \\ x_4 \\ \lambda x_3 \\ \mu x_4 + (2 \lambda - \mu) c x_3^2 \end{bmatrix}, \qquad
# \mb f_1(\mb x) = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}, \qquad
# \mb f_2(\mb x) = \begin{bmatrix} 0 \\ 0 \\ 0 \\ x_1+1 \end{bmatrix}, 
# \end{equation}   

# and $\lambda, \mu, c \in \mathbb{R}$ are scalar parameters of the system. Setting $ \mb x = [q_1 \, q_2 \, \dot{q_1} \, \dot{q_2}]^T$, 
# these equations of motion can be expressed as robotic dynamics of the form $\mb{D}(\mb{q})\ddot{\mb{q}} + \mb{C}(\mb{q}, \dot{\mb{q}})\dot{\mb{q}} + \mb{G}(\mb{q}) = \mb{B}\mb{u}$, where $\mb D$ is the inertia matrix, $\mb C$ is the matrix of Coriolis terms, $\mb G$ is the matrix of gravitational terms, and $\mb B$ is the static actuation matrix. Rewriting $\mb f_0, \mb f_1, \mb f_2$ in terms of $\mb D, \mb C, \mb G,$ and $\mb B$ yield
# 
# 

# \begin{equation}
# \mb D(\mb q) = \begin{bmatrix} 1 & 0\\ 0 &  \frac{1}{q_1+1} \end{bmatrix}, 
# \qquad \mb C(\mb q, \mb{\dot{q}}) = -\begin{bmatrix} \lambda & 0 \\ \frac{1}{q_1 + 1}(2 \lambda - \mu) c \dot{q}_1 & \frac{1}{q_1 +1} \mu \end{bmatrix}, \qquad 
# \mb G(\mb q) = \begin{bmatrix} 0 \\ 0 \end{bmatrix} 
# \qquad \mb B = \begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix},
# \end{equation}

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
#     1\\
#     q_1 - \frac{1}{\lambda}\dot{q}_1\\
#     q_2 - \frac{1}{\mu} \dot{q}_2 + \frac{(2 \lambda - \mu)c}{2\lambda \mu} \dot{q}_1^2\\
#     \dot{q}_1\\
#     \dot{q}_2 - c \dot{q}_1^2\\
#     \dot{q}_1^2\\
#     \end{bmatrix},
# \end{equation}

# where $\phi_1, \phi_2, \phi_3, \phi_4, \phi_5, \phi_6$ are eigenfunctions of the Koopman operator associated with the drift 
# vector field $\mb f_0$. The matrix with the eigenvalue associated with the i-th eigenfunction on the i-th diagonal 
# element is $F=\text{diag}(0, 0, \lambda, \mu, 2 \lambda, 0)$. Then, to reformulate the dynamics we have: 

# \begin{equation} 
# L_{\mb f_1} T(\mb q, \mb{\dot{q}}) = \begin{bmatrix} 0\\ -\frac{1}{\lambda}\\ \frac{(2\lambda - \mu)c}{\lambda \mu}\dot{q}_1\\ 1 \\ -2c\dot{q}_1 \\ 2\dot{q_1} \end{bmatrix}, \qquad 
# L_{\mb f_2} T(\mb q, \mb{\dot{q}}) = \begin{bmatrix} 0 \\ 0\\ -\frac{1}{\mu}(q_1 + 1)\\0 \\ q_1 + 1 \\ 0 \end{bmatrix} 
# \end{equation}

# and the dynamics can be equivalently transformed to a bilinear form $\mb{\dot{z}} = F \mb z + G_1\mb z u_1 + G_2\mb z u_2$ with 

# \begin{equation}
# F = \begin{bmatrix} 
# 0 &0 & 0 & 0 & 0 & 0\\  
# 0 & 0 & 0 & 0 & 0 & 0\\
# 0 &0 & 0 & 0 & 0 & 0\\
# 0 &0 & 0 & \lambda & 0 & 0\\
# 0 &0 & 0 & 0 & \mu & 0 \\
# 0 &0 & 0 & 0 & 0 & 2 \lambda\\
# \end{bmatrix}, \qquad 
# G_1 = \begin{bmatrix} 
# 0 & 0 & 0 & 0 & 0 & 0\\
# -\frac{1}{\lambda}& 0 & 0 & 0 & 0 & 0\\
# 0 & 0 & 0 & \frac{(2\lambda - \mu)c}{\lambda \mu} & 0 & 0\\
# 1 & 0 & 0 & 0 & 0 & 0\\
# 0 & 0 & 0 & -2c & 0 & 0\\
# 0 & 0 & 0 & 2 & 0 & 0\\  
# \end{bmatrix}
# , \qquad 
# G_2 = \begin{bmatrix} 
# 0 & 0 & 0 & 0 & 0 & 0\\
# 0 & 0 &  0 & 0 & 0 & 0\\
#  -\frac{1}{\mu} & -\frac{1}{\mu} & 0 &  -\frac{1}{\lambda \mu} & 0 & 0\\
# 0 & 0 & 0 & 0 & 0 & 0\\
# 1 & 1 & 0 & \frac{1}{\lambda} & 0 & 0\\
# 0 & 0 & 0 & 0 & 0 & 0\\  
# \end{bmatrix}
# \end{equation}

# In[1]:


import numpy as np
import sys
sys.path.append('../../')


# # Define experiment parameters

# In[2]:


from core.dynamics import RoboticDynamics, ConfigurationDynamics

class KoopPdOutput(ConfigurationDynamics):
    def __init__(self, dynamics, xd, n, m):
        ConfigurationDynamics.__init__(self, dynamics, 1)
        self.xd = xd
        self.n = n
        self.m = m

    def proportional(self, x, t):
        q = x[:int(n/2)]
        q_d = self.xd[:int(n/2)]

        return  q - q_d

    def derivative(self, x, t):
        q_dot = x[int(n/2):]
        q_dot_d = self.xd[int(n/2):]

        return q_dot - q_dot_d

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

n, m = 4, 2
lambd, mu, c = .3, .2, -.5
sys_name = 'bilinearizable_sys'
system = FiniteDimKoopSys(lambd, mu, c)


# In[3]:


from koopman_core.dynamics import LinearLiftedDynamics

A_lin = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, lambd, 0],
                  [0, 0, 0, mu]])
B_lin = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
dt = 1e-2
linearized_sys = LinearLiftedDynamics(A_lin, B_lin, np.eye(n), lambda x: x)


# # Collect data for learning 

# In[4]:


import scipy as sc
import os

q_dc, r_dc = 5e2, 1                                                 # State and actuation penalty values, data collection
Q_dc = q_dc * np.identity(n)                                        # State penalty matrix, data collection
R_dc = r_dc*np.identity(m)                                          # Actuation penalty matrix, data collection
P_dc = sc.linalg.solve_continuous_are(A_lin, B_lin, Q_dc, R_dc)     # Algebraic Ricatti equation solution, data collection
K_dc = np.linalg.inv(R_dc)@B_lin.T@P_dc                             # LQR feedback gain matrix, data collection
K_dc_p = K_dc[:,:int(n/2)]                                          # Proportional control gains, data collection
K_dc_d = K_dc[:,int(n/2):]                                          # Derivative control gains, data collection

# Data collection parameters:
collect_data = False
dt = 1.0e-2                                                         # Time step length
traj_length_dc = 2.                                                 # Trajectory length, data collection
n_pred_dc = int(traj_length_dc/dt)                                  # Number of time steps, data collection
t_eval = dt * np.arange(n_pred_dc + 1)                              # Simulation time points
n_traj_train = 100                                                      # Number of trajectories to execute, data collection
n_traj_val = int(0.2*n_traj_train)
noise_var = 5.                                                      # Exploration noise to perturb controller, data collection

xmax = np.array([1., 1., 1., 1.])                                   # State constraints, trajectory generation
xmin = -xmax
umax = np.array([10., 10.])                                         # Actuation constraint, trajectory generation
umin = -umax
x0_max = xmax/2                                                     # Initial value limits
sub_sample_rate = 1                                                 # Rate to subsample data for training
model_fname = 'examples/'                                           # Path to save learned models
n_cols = 10                                                         # Number of columns in training data plot
directory = os.path.abspath("")                                     # Path to save learned models


# In[5]:


from koopman_core.util import run_experiment
import dill

if collect_data:
    xs_train, us_train, t_train = run_experiment(system, n, n_traj_train, n_pred_dc, t_eval, x0_max, plot_experiment_data=True,
                                      m=m, K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var)
    xs_val, us_val, t_val = run_experiment(system, n, n_traj_val, n_pred_dc, t_eval, x0_max,
                                      m=m, K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var)

    data_list = [xs_train, us_train, t_train, n_traj_train, xs_val, us_val, t_val, n_traj_val]
    outfile = open(directory + '/data/' + sys_name + '_data.pickle', 'wb')
    dill.dump(data_list, outfile)
    outfile.close()
else:
    infile = open(directory + '/data/' + sys_name + '_data.pickle', 'rb')
    xs_train, us_train, t_train, n_traj_train, xs_val, us_val, t_val, n_traj_val = dill.load(infile)
    infile.close()


# # Learn Koopman-based models of the dynamics

# ### Learn bilinear EDMD model

# In[6]:


#Bilinear EDMD parameters:                                           
alpha_bedmd = 2.4e-5  # Regularization strength (LASSO) bEDMD
tune_mdl_bedmd = False


# In[7]:


from sklearn import preprocessing, linear_model
from koopman_core.learning import BilinearEdmd
from koopman_core.dynamics import BilinearLiftedDynamics

bedmd_features = preprocessing.PolynomialFeatures(2)
bedmd_features.fit(np.zeros((1,n)))
n_lift_bedmd = bedmd_features.transform((np.zeros((1,n)))).shape[1]

C_bedmd = np.zeros((n,n_lift_bedmd))
C_bedmd[:,1:n+1] = np.eye(n)

basis_bedmd = lambda x: bedmd_features.transform(x)
optimizer_bedmd = linear_model.MultiTaskLasso(alpha=alpha_bedmd, fit_intercept=False, selection='random')
cv_bedmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
#standardizer_bedmd = preprocessing.StandardScaler(with_mean=False)
standardizer_bedmd = None

model_bedmd = BilinearEdmd(n, m, basis_bedmd, n_lift_bedmd, n_traj_train, optimizer_bedmd, cv=cv_bedmd, 
                           standardizer=standardizer_bedmd, C=C_bedmd, continuous_mdl=False, dt=dt)
X_bedmd, y_bedmd = model_bedmd.process(xs_train, us_train, np.tile(t_train,(n_traj_train,1)), downsample_rate=sub_sample_rate)
model_bedmd.fit(X_bedmd, y_bedmd, cv=tune_mdl_bedmd, override_kinematics=True)
sys_bedmd = BilinearLiftedDynamics(model_bedmd.n_lift, m, model_bedmd.A, model_bedmd.B, model_bedmd.C, 
                                   model_bedmd.basis, continuous_mdl=False, dt=dt)
if tune_mdl_bedmd:
    print('$\\alpha$ bilinear EDMD: ', model_bedmd.cv.alpha_)


# ### Learn Koopman DNN model

# In[8]:


import dill, os, torch

load_tuned_params = False

if load_tuned_params:
    infile = open(os.path.abspath('') + '/data/analytic_koop_sys_best_params.pickle', 'rb')
    best_config, val_loss, test_loss, open_loop_mse, open_loop_std = dill.load(infile)
    infile.close()

else:
    net_params = {}
    net_params['state_dim'] = n
    net_params['ctrl_dim'] = m
    net_params['encoder_hidden_width'] = 100
    net_params['encoder_hidden_depth'] = 1
    net_params['encoder_output_dim'] = 1
    net_params['optimizer'] = 'adam'
    net_params['activation_type'] = 'relu'
    net_params['lr'] = 2e-3
    net_params['epochs'] = 100
    net_params['batch_size'] = 128
    net_params['lin_loss_penalty'] = 0.5
    net_params['l2_reg'] = 0
    net_params['l1_reg'] = 0
    net_params['first_obs_const'] = True
    net_params['override_kinematics'] = False # TODO: Fix override kin... 
    net_params['dt'] = dt

print(net_params)


# In[9]:


from koopman_core.learning import KoopDnn, KoopmanNetCtrl
from koopman_core.util import fit_standardizer

standardizer_x_kdnn = fit_standardizer(xs_train, preprocessing.StandardScaler())
standardizer_u_kdnn = fit_standardizer(us_train, preprocessing.StandardScaler())
n_tot = n + net_params['encoder_output_dim'] + int(net_params['first_obs_const'])

net = KoopmanNetCtrl(net_params, standardizer_x=standardizer_x_kdnn, standardizer_u=standardizer_u_kdnn)
model_koop_dnn = KoopDnn(net)
model_koop_dnn.set_datasets(xs_train, t_train, u_train=us_train, x_val=xs_val, u_val=us_val, t_val=t_val)
model_koop_dnn.model_pipeline(net_params)
model_koop_dnn.construct_koopman_model()
sys_koop_dnn = BilinearLiftedDynamics(n_tot, m, model_koop_dnn.A, model_koop_dnn.B, model_koop_dnn.C, 
                                      model_koop_dnn.basis_encode, continuous_mdl=False, dt=dt, 
                                      standardizer_x=standardizer_x_kdnn, standardizer_u=standardizer_u_kdnn)


# In[10]:


sys_koop_dnn.A


# # Evaluate open-loop prediction performance

# In[11]:


# Prediction performance evaluation parameters:
folder_plots = 'figures/'                                          # Path to save plots
n_traj_ol = 50                                                     # Number of trajectories to execute, open loop


# In[12]:


from koopman_core.util import evaluate_ol_pred
from tabulate import tabulate
import random as rand

xs_ol, us_ol, t_ol = run_experiment(system, n, n_traj_ol, n_pred_dc, t_eval, x0_max,
                                      m=m, K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var)

mdl_lst = [sys_koop_dnn, sys_bedmd]
mdl_names = ['Koop DNN', 'bEDMD']
error, mse, std = [], [], []

for sys in mdl_lst:
    err_tmp, mse_tmp, std_tmp = evaluate_ol_pred(sys, xs_ol, t_eval, us=us_ol)
    error.append(err_tmp)
    mse.append(mse_tmp)
    std.append(std_tmp)
    
print('\nOpen loop performance statistics:')
table_data = []
for name, mse_mdl, std_mdl in zip(mdl_names, mse, std):
    table_data.append([name, "{:.5f}".format(mse_mdl), "{:.5f}".format(std_mdl)])

print(tabulate(table_data, 
               headers=['Mean squared error', 'Standard deviation']))


# In[13]:


import matplotlib.pyplot as plt
import matplotlib

figwidth = 12
lw = 2
fs = 14
y_lim_gain = 1.2
row = 2
col = n/row

#Plot open loop results:
plt.figure(figsize=(figwidth,4))
axs = [plt.subplot(row,col,jj+1) for jj in range(n)]

for ii, err in enumerate(error):
    err_mean = np.mean(err, axis=0)
    err_std = np.std(err, axis=0)
    
    for jj in range(n):
        axs[jj].plot(t_eval[1:], err_mean[:,jj], label=mdl_names[ii])
        axs[jj].fill_between(t_eval[1:], err_mean[:,jj]-err_std[:,jj], err_mean[:,jj]+err_std[:,jj], alpha=0.1)

for jj in range(n):
    axs[jj].grid()
    axs[jj].set_xlabel('Time (sec)', fontsize=fs)
    axs[jj].set_ylabel('$x_'+ str(jj+1) + '$', fontsize=fs)

plt.legend(frameon=False, fontsize=fs)
stitle=plt.suptitle('Open loop prediction performance of learned models', fontsize=fs+2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.savefig(folder_plots + 'koop_sys_prediction.pdf', format='pdf', dpi=2400, bbox_extra_artists=(stitle,), bbox_inches="tight")
plt.show()


# In[14]:


print(standardizer_u_kdnn.mean_)
print(standardizer_u_kdnn.scale_)
print(standardizer_x_kdnn.mean_)
print(standardizer_x_kdnn.scale_)


# In[ ]:




