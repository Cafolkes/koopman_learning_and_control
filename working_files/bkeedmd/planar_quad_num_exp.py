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
from koopman_core.util import evaluate_ol_pred
from koopman_core.dynamics import BilinearLiftedDynamics
from koopman_core.learning import KoopDnn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import matplotlib.pyplot as plt

# TODO: Insert needed dynamics...

# Define system and system linearization:
sys_name = 'bilinearizable_sys'
n, m = 4, 2
lambd, mu, c = .3, .2, -.5
system = PlanarQuadrotor(lambd, mu, c)
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

# Experiment parameters:
n_traj_exp = [100]#[10, 20, 50, 100, 200]
n_lift_exp = [5]#[1, 4, 8, 10, 15, 20]
n_repeat = 10

# Data collection parameters:
collect_data = True
test_frac = 0.2
val_frac = 0.2
dt = 1.0e-2                                                         # Time step length
traj_length_dc = 2.                                                 # Trajectory length, data collection
n_pred_dc = int(traj_length_dc/dt)                                  # Number of time steps, data collection
t_eval = dt * np.arange(n_pred_dc + 1)                              # Simulation time points
n_traj_train = 250                                                  # Number of trajectories to execute, data collection
n_traj_test = 100                                                   # Number of trajectories to execute, data collection
noise_var = 5.                                                      # Exploration noise to perturb controller, data collection
x0_max = np.array([1., 1., 1., 1.])                                 # Initial value limits
directory = os.path.abspath("working_files/bkeedmd/")               # Path to save learned models

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
net_params['encoder_hidden_width'] = 100
net_params['encoder_hidden_depth'] = 1
net_params['encoder_output_dim'] = 10
net_params['epochs'] = 200
net_params['optimizer'] = 'adam'

# DNN tunable parameters:
net_params['lr'] = tune.loguniform(1e-5, 1e-2)
net_params['l2_reg'] = tune.loguniform(1e-6, 1e-1)
net_params['l1_reg'] = tune.loguniform(1e-6, 1e-1)
net_params['batch_size'] = tune.choice([16, 32, 64, 128])
net_params['lin_loss_penalty'] = tune.uniform(0, 1)

# Hyperparameter tuning parameters:
tune_mdl_kdnn = True
num_samples = -1
time_budget_s = 60*60                                      # Time budget for tuning process for each n_multistep value
if torch.cuda.is_available():
    resources_cpu = 2
    resources_gpu = 0.2
else:
    resources_cpu = 1
    resources_gpu = 0

# Collect/load for datasets:
if collect_data:
    # TODO: Collect xs_tune, us_tune, t_tune
    xs_train, us_train, t_eval_train = run_experiment(system, n, n_traj_train, n_pred_dc, t_eval,
                                                      x0_max, m=m, K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var)
    xs_test, us_test, t_eval_test = run_experiment(system, n, n_traj_test, n_pred_dc, t_eval,
                                                      x0_max, m=m, K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var)

    data_list = [xs_train, us_train, t_eval_train, n_traj_train, xs_test, us_test, t_eval_test, n_traj_test]
    outfile = open(directory + '/data/' + sys_name + '_data.pickle', 'wb')
    dill.dump(data_list, outfile)
    outfile.close()
else:
    infile = open(directory + '/data/' + sys_name + '_data.pickle', 'rb')
    xs_train, us_train, t_eval_train, n_traj_train, xs_test, us_test, t_eval_test, n_traj_test = dill.load(infile)
    infile.close()

best_config_lst = [], []
for n_lift in n_lift_exp:

    if tune_mdl_kdnn:
        # Define Koopman DNN model:
        model_kdnn = KoopDnn(net_params)
        model_kdnn.set_datasets(xs_train, us_train, t_eval_train)
        trainable = lambda config: model_kdnn.model_pipeline(config, print_epoch=False, tune_run=True)
        tune.register_trainable('trainable_pipeline', trainable)

        net_params['encoder_output_dim'] = n_lift
        scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='loss',
            mode='min',
            max_t=net_params['epochs'],
            grace_period=30,
        )
        result = tune.run(
            'trainable_pipeline',
            config=net_params,
            checkpoint_at_end=True,
            num_samples=num_samples,
            time_budget_s=time_budget_s,
            scheduler=scheduler,
            resources_per_trial={'cpu': resources_cpu, 'gpu': resources_gpu},
            verbose=1
        )
        config = result.get_best_config("loss", "min")
        best_config = {}
        best_config['n_lift'] = n_lift
        best_config['config'] = config
        best_config_lst.append(best_config)

    else:
        config = None  # TODO: Load pretuned configuration

    for n_data in n_traj_exp:
        n_tot = n + int(config['first_obs_const']) + n_lift
        model_kdnn = KoopDnn(config)
        model_kdnn.set_datasets(xs_train[:n_data, :, :], us_train[:n_data, :, :], t_eval_train)
        model_kdnn.model_pipeline(net_params, early_stop=True)
        model_kdnn.construct_koopman_model()
        sys_koop_dnn = BilinearLiftedDynamics(n_tot, m, model_kdnn.A, model_kdnn.B, model_kdnn.C,
                                              model_kdnn.basis_encode,
                                              continuous_mdl=False, dt=dt)

        _, mse_tmp, _ = evaluate_ol_pred(sys_koop_dnn, xs_test, t_eval, us=us_test - hover_thrust)

        res_kdnn = {}
        res_kdnn['n_lift'] = n_lift
        res_kdnn['n_data'] = n_data
        res_kdnn['mse'] = mse_tmp
        res_kdnn_lst.append(res)

outfile = open(directory + '/data/' + sys_name + '_best_params.pickle', 'wb')
data_list_tuning = [best_config_lst, val_loss, test_loss, open_loop_mse, open_loop_std]
dill.dump(data_list_tuning, outfile)
outfile.close()

def run_experiment(system, n, n_traj_train, n_pred_dc, t_eval, x0_max, m=m, K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var):

    return xs_train, us_train, t_eval_train
