#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../../')
import os
import numpy as np
import torch
import scipy as sc
import dill
from core.dynamics import RoboticDynamics
from koopman_core.util import run_experiment, evaluate_ol_pred
from koopman_core.dynamics import BilinearLiftedDynamics
from koopman_core.learning import KoopDnn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import matplotlib.pyplot as plt

class FiniteDimKoopSys(RoboticDynamics):
    def __init__(self, lambd, mu, c):
        RoboticDynamics.__init__(self, 2, 2)
        self.params = lambd, mu, c

    def D(self, q):
        return np.array([[1, 0], [0, (q[0] + 1) ** (-1)]])

    def C(self, q, q_dot):
        labmd, mu, c = self.params
        return -np.array([[lambd, 0], [(q[0] + 1) ** (-1) * (2 * lambd - mu) * c * q_dot[0], (q[0] + 1) ** (-1) * mu]])

    def G(self, q):
        return np.array([0, 0])

    def B(self, q):
        return np.array([[1, 0], [0, 1]])

# Define system and system linearization:
sys_name = 'bilinearizable_sys'
n, m = 4, 2
lambd, mu, c = .3, .2, -.5
system = FiniteDimKoopSys(lambd, mu, c)
A_lin = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, lambd, 0],
                  [0, 0, 0, mu]])
B_lin = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])

# Define LQR controller for data collection:
q_dc, r_dc = 5e2, 1                                                 # State and actuation penalty values, data collection
Q_dc = q_dc * np.identity(n)                                        # State penalty matrix, data collection
R_dc = r_dc*np.identity(m)                                          # Actuation penalty matrix, data collection
P_dc = sc.linalg.solve_continuous_are(A_lin, B_lin, Q_dc, R_dc)     # Algebraic Ricatti equation solution, data collection
K_dc = np.linalg.inv(R_dc)@B_lin.T@P_dc                             # LQR feedback gain matrix, data collection
K_dc_p = K_dc[:,:int(n/2)]                                          # Proportional control gains, data collection
K_dc_d = K_dc[:,int(n/2):]                                          # Derivative control gains, data collection

# Data collection parameters:
collect_data = True
test_frac = 0.2
val_frac = 0.2
dt = 1.0e-2                                                         # Time step length
traj_length_dc = 2.                                                 # Trajectory length, data collection
n_pred_dc = int(traj_length_dc/dt)                                  # Number of time steps, data collection
t_eval = dt * np.arange(n_pred_dc + 1)                              # Simulation time points
n_traj_train = 250                                                      # Number of trajectories to execute, data collection
n_traj_test = 100                                                      # Number of trajectories to execute, data collection
noise_var = 5.                                                      # Exploration noise to perturb controller, data collection
x0_max = np.array([1., 1., 1., 1.])                                  # Initial value limits
directory = os.path.abspath("working_files/bkeedmd/")                                                  # Path to save learned models

# Model configuration parameters:
net_params = {}
net_params['state_dim'] = n
net_params['ctrl_dim'] = m
net_params['first_obs_const'] = True
net_params['override_kinematics'] = True
net_params['dt'] = dt
net_params['data_dir'] = directory + '/data'
net_params['n_multistep'] = 1

# DNN architecture parameters:
net_params['encoder_hidden_dim'] = [20, 20, 20]
net_params['encoder_output_dim'] = 10
net_params['epochs'] = 500
net_params['optimizer'] = 'adam'

# DNN tunable parameters:
net_params['lr'] = tune.loguniform(1e-5, 1e-1)
net_params['l2_reg'] = tune.uniform(1e-6, 1e0)
net_params['l1_reg'] = tune.uniform(1e-6, 1e0)
net_params['batch_size'] = tune.choice([8, 16, 32])
net_params['lin_loss_penalty'] = tune.uniform(1e-6, 1e0)

# Hyperparameter tuning parameters:
num_samples = -1
time_budget_s = 2*60*60                                                      # Time budget for tuning process for each n_multistep value
n_multistep_lst = [1, 3, 5, 10, 30, 50]
if torch.cuda.is_available():
    resources_cpu = 12
    resources_gpu = 1
else:
    resources_cpu = 8
    resources_gpu = 0


# Collect/load datasets:
if collect_data:
    xs_train, us_train, t_eval_train = run_experiment(system, n, m, K_dc_p, K_dc_d, n_traj_train, n_pred_dc, t_eval,
                                                      x0_max, noise_var)
    xs_test, us_test, t_eval_test = run_experiment(system, n, m, K_dc_p, K_dc_d, n_traj_test, n_pred_dc, t_eval,
                                                      x0_max, noise_var)

    data_list = [xs_train, us_train, t_eval_train, n_traj_train, xs_test, us_test, t_eval_test, n_traj_test]
    outfile = open(directory + '/data/' + sys_name + '_data.pickle', 'wb')
    dill.dump(data_list, outfile)
    outfile.close()
else:
    infile = open(directory + '/data/' + sys_name + '_data.pickle', 'rb')
    xs_train, us_train, t_eval_train, n_traj_train, xs_test, us_test, t_eval_test, n_traj_test = dill.load(infile)
    infile.close()

# Define Koopman DNN model:
model_kdnn = KoopDnn(net_params)
model_kdnn.set_datasets(xs_train, us_train, t_eval_train)

# Set up hyperparameter tuning:
trainable = lambda config: model_kdnn.model_pipeline(config)
tune.register_trainable('trainable_pipeline', trainable)

best_trial_lst = []
for n_multistep in n_multistep_lst:
    scheduler = ASHAScheduler(
        max_t=net_params['epochs'],
        grace_period=1,
        reduction_factor=2)

    net_params['n_multistep'] = n_multistep
    result = tune.run(
        'trainable_pipeline',
        config=net_params,
        metric='loss',
        mode='min',
        checkpoint_at_end=True,
        num_samples=num_samples,
        time_budget_s=time_budget_s,
        scheduler=scheduler,
        resources_per_trial={'cpu': resources_cpu, 'gpu': resources_gpu}
    )

    best_trial_lst.append(result.get_best_trial("loss", "min", "last"))

# Analyze the results:
val_loss = []
test_loss = []
open_loop_mse = []
open_loop_std = []

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

for best_trial in best_trial_lst:
    # Extract validation loss:
    val_loss.append(best_trial.last_result["loss"])

    # Calculate test loss:
    best_model = KoopDnn(best_trial.config)
    best_model.to(device)
    checkpoint_path = os.path.join(best_trial.checkpoint.value, 'checkpoint')
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_model.koopman_net.load_state_dict(model_state)
    test_loss.append(best_model.test_loss(xs_test, us_test, t_eval_test))

    # Calculate open loop mse and std:
    n_tot = net_params['state_dim'] + net_params['encoder_output_dim'] + int(net_params['first_obs_const'])
    best_model.construct_koopman_model()
    sys_kdnn = BilinearLiftedDynamics(n_tot, m, best_model.A, best_model.B, best_model.C,
                                          best_model.basis_encode, continuous_mdl=False, dt=dt)
    mse, std = evaluate_ol_pred(sys_kdnn, xs_test, us_test, t_eval_test)
    open_loop_mse.append(mse)
    open_loop_std.append(std)

plt.figure()
plt.plot(n_multistep_lst, val_loss, label='validation loss')
plt.plot(n_multistep_lst, test_loss, label='test loss')
plt.plot(n_multistep_lst, val_loss, label='open loop mse')
plt.plot(n_multistep_lst, val_loss, label='open loop std')
plt.xlabel('# of multistep prediction steps')
plt.ylabel('Loss')
plt.title('Best tuned model performance VS multistep horizon')
plt.legend()
plt.savefig(directory + '/figures/' + 'tuning_summary_' + sys_name + '.pdf')

outfile = open(directory + '/data/' + sys_name + '_best_params.pickle', 'wb')
data_list_tuning = [best_trial_lst, val_loss, test_loss, open_loop_mse, open_loop_std]
dill.dump(data_list_tuning, outfile)
outfile.close()