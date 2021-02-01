#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
import os
import dill

import numpy as np
import scipy as sc
import random as rand
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt

from core.controllers import ConstantController

from koopman_core.dynamics import LinearLiftedDynamics, BilinearLiftedDynamics
from koopman_core.learning import Edmd_aut, KoopDnn


# ## Autonomous system with analytic finite dimensional Koopman operator

# Consider the continuous-time dynamics
# 
# \begin{equation}
#     x = \begin{bmatrix} x_1\\x_2\\\dot{x}_1\\ \dot{x}_2 \end{bmatrix}, \qquad
#     \begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \\ \dot{x}_3 \\ \dot{x}_4 \end{bmatrix}
#     = \begin{bmatrix}
#     x_3 \\ 
#     x_4 \\
#     \mu x_3\\
#     -\lambda x_3^2 + \lambda x_4
#     \end{bmatrix}
# \end{equation}
# by carefully choosing observables, the drift vector field of the dynamics can be reformulated as an equilvalent linear system (global linearization). Define the observables
# 
# \begin{equation}
# \begin{bmatrix}
# y_1 \\ y_2 \\ y_3 \\ y_4 \\ y_5
# \end{bmatrix}
# = \begin{bmatrix}
# x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_3^2
# \end{bmatrix}
# \end{equation}
# 
# then the system can be rewritten as
# 
# \begin{equation}
# \begin{bmatrix}
# \dot{y}_1 \\ \dot{y}_2 \\ \dot{y}_3 \\ \dot{y}_4 \\ \dot{y}_5 \end{bmatrix} = 
# \begin{bmatrix}
# 0 & 0 & 1 & 0 & 0\\
# 0 & 0 & 0 & 1 & 0\\
# 0 & 0 & \mu & 0 & 0\\
# 0 & 0 & 0 & \lambda & - \lambda \\
# 0 & 0 & 0 & 0 & \mu \end{bmatrix}
# \begin{bmatrix}
# y_1 \\ y_2 \\ y_3 \\ y_4 \\ y_5 \end{bmatrix}
# \end{equation}
# 
# 
# 

# In[2]:


from core.dynamics import SystemDynamics

class FiniteKoopSys(SystemDynamics):
    def __init__(self, mu, lambd):
        SystemDynamics.__init__(self, 4, 0)
        self.params = mu, lambd
    
    def drift(self, x, t):
        mu, lambd = self.params

        return np.array([x[2], x[3], mu*x[2], -lambd*x[2]**2 + lambd*x[3]])

    def eval_dot(self, x, u, t):
        return self.drift(x, t)
    
n, m = 4, 0
mu, lambd = -0.3, -0.6
system = FiniteKoopSys(mu, lambd)
sys_name = 'analytic_koop_sys'


# ### Collect data for learning

# To collect data, a nominal controller is designed with LQR on the dynamics's linearization around hover. However, any
# controller can be used and the method does not require the knowledge of model's linearization. In addition, a
# exploratory white noise is added to the controller to ensure that the data is sufficiently excited. Note that the system
# is underactuated and that trajectory optimization is necessary to control the position of the vehicle. We use a
# simplified trajectory generator based on a model predictive controller for the linearized dynamics. More careful design
# of the desired trajectory may be necessary for more demanding applications and this is readily compatible with our method.
# 
# 4 data sets are collected to evaluate the impact of signal and process noise:
# - Nominal data set (no signal or process noise)
# - Data set with signal noise
# - Data set with process noise
# - Data set with signal and process noise

# In[3]:


# Data collection parameters:
collect_data = True
dt = 1.0e-2                                                         # Time step length
traj_length_dc = 8.                                                 # Trajectory length, data collection
n_pred_dc = int(traj_length_dc/dt)                                  # Number of time steps, data collection
t_eval = dt * np.arange(n_pred_dc + 1)                              # Simulation time points
n_traj_train = 250                                                  # Number of trajectories to execute, data collection
n_traj_val = int(0.2*n_traj_train)

xmax = np.array([1, 1, 1, 1])                                             # State constraints, trajectory generation
xmin = -xmax
x0_max = xmax                                                       # Initial value limits
sub_sample_rate = 1                                                 # Rate to subsample data for training
n_cols = 10                                                         # Number of columns in training data plot
folder_plots = 'figures/'                                           # Path to save plots
directory = os.path.abspath("")                                     # Path to save learned models

from koopman_core.util import run_experiment
if collect_data:
    xs_train, t_train = run_experiment(system, n, n_traj_train, n_pred_dc, t_eval, x0_max, plot_experiment_data=True)
    xs_val, t_val = run_experiment(system, n, n_traj_val, n_pred_dc, t_eval, x0_max, plot_experiment_data=True)

    data_list = [xs_train, t_train, n_traj_train, xs_val, t_val, n_traj_val]
    outfile = open(directory + '/data/' + sys_name + '_data.pickle', 'wb')
    dill.dump(data_list, outfile)
    outfile.close()
else:
    infile = open(directory + '/data/' + sys_name + '_data.pickle', 'rb')
    xs_train, t_train, n_traj_train, xs_val, t_val, n_traj_val, t_eval_test, n_traj_test = dill.load(infile)
    infile.close()

# ### Learn a lifted linear model with Koopman DNN

# In[5]:


net_params = {}
net_params['state_dim'] = n
net_params['encoder_hidden_width'] = 20
net_params['encoder_hidden_depth'] = 2
net_params['encoder_output_dim'] = 4
net_params['optimizer'] = 'adam'
net_params['lr'] = 1e-4
net_params['epochs'] = 100
net_params['batch_size'] = 16
net_params['lin_loss_penalty'] = 5e-1/net_params['encoder_output_dim']
net_params['l2_reg'] = 1e-5
net_params['l1_reg'] = 1e-5
net_params['n_multistep'] = 10
net_params['first_obs_const'] = True
net_params['override_kinematics'] = True
net_params['dt'] = dt


# In[7]:

from koopman_core.learning import KoopmanNetAut

standardizer_kdnn = preprocessing.StandardScaler()

net = KoopmanNetAut(net_params, standardizer=standardizer_kdnn)
model_koop_dnn = KoopDnn(net)
model_koop_dnn.set_datasets(xs_train, t_train, x_val=xs_val, t_val=t_val)
model_koop_dnn.model_pipeline(net_params, early_stop=True)
model_koop_dnn.construct_koopman_model()
sys_koop_dnn = LinearLiftedDynamics(model_koop_dnn.A, None, model_koop_dnn.C, model_koop_dnn.basis_encode,
                                    continuous_mdl=False, dt=dt, standardizer=standardizer_kdnn)



# ### Evaluate open loop prediction performance

# We now evaluate the open loop prediction performance of the implemented methods.
# This is done by generating a new data set in the same way as the training set, predicting the evolution of the system
# with the control sequence of each trajectory executed in the data set with each of the models, and finally comparing
# the mean and standard deviation of the error between the true and predicted evolution over the trajectories. All the models are evaluated on 2 test data sets. One nominal data set (no signal or process noise) and a data set with process noise. No test data with signal noise is used, as we would need to fix the signal noise sequence to do a fair comparison in open loop prediction, hence resulting in the same comparison as the 2 data sets used.

# In[12]:
# Prediction performance evaluation parameters:
n_traj_ol = n_traj_train                                                     # Number of trajectories to execute, open loop


# In[13]:


from tabulate import tabulate

t_eval = dt * np.arange(4./dt + 1)
xs_ol = np.empty((n_traj_ol, t_eval.shape[0], n))    
xs_koop_dnn_ol = np.empty((n_traj_ol, t_eval.shape[0]-1, n))
xs_dmd_ol = np.empty((n_traj_ol, t_eval.shape[0]-1, n))
xs_edmd_ol = np.empty((n_traj_ol, t_eval.shape[0]-1, n))
ctrl = ConstantController(system, 0.)
    
for ii in range(n_traj_ol):
    x0 = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
    xs_ol[ii,:,:], _ = system.simulate(x0, ctrl, t_eval)

    z_0_koop_dnn = sys_koop_dnn.basis(np.atleast_2d(x0)).squeeze()
    zs_koop_dnn_tmp, _ = sys_koop_dnn.simulate(z_0_koop_dnn, ctrl, t_eval[:-1])
    xs_koop_dnn_ol[ii,:,:] = sys_koop_dnn.standardizer.inverse_transform(np.dot(sys_koop_dnn.C, zs_koop_dnn_tmp.T).T)

data = [xs_koop_dnn_ol]
mdl_names = ['Koopman DNN']
error, mse, std = [], [], []

for ii, d in enumerate(data):
    err_tmp = xs_ol[:,:-1,:]-d
    error.append(err_tmp)
    mse.append(np.mean(np.square(error[ii])))
    std.append(np.std(error[ii]))
    
print('\nOpen loop performance statistics:')

table_data = []
for name, mse_mdl, std_mdl in zip(mdl_names, mse, std):
    table_data.append([name, "{:.5f}".format(mse_mdl), "{:.5f}".format(std_mdl)])

print(tabulate(table_data, 
               headers=['Mean squared error', 'Standard deviation']))


# In[14]:


import matplotlib.pyplot as plt
import matplotlib

figwidth = 12
lw = 2
fs = 14
y_lim_gain = 1.2
row = 2
col = 2

#Plot open loop results:
plt.figure(figsize=(figwidth,8))
axs = [plt.subplot(row,col,jj+1) for jj in range(n)]

for ii, err in enumerate(error):
    err_mean = np.mean(err, axis=0)
    err_std = np.std(err, axis=0)
    
    for jj in range(n):
        axs[jj].plot(t_eval[1:], err_mean[:,jj], label=mdl_names[ii])
        axs[jj].fill_between(t_eval[1:], err_mean[:,jj]-err_std[:,jj], err_mean[:,jj]+err_std[:,jj], alpha=0.1)

for jj in range(n):
    axs[jj].set_xlabel('Time (sec)', fontsize=fs)
    axs[jj].set_ylabel('$x_'+ str(jj+1) + '$', fontsize=fs)

plt.legend(frameon=False, fontsize=fs)
stitle=plt.suptitle('Open loop prediction performance of learned models', fontsize=fs+2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.savefig(folder_plots + 'koop_sys_prediction.pdf', format='pdf', dpi=2400, bbox_extra_artists=(stitle,), bbox_inches="tight")
plt.show()
