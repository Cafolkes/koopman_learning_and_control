#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../../')
import os
import numpy as np
import torch
import dill
from core.dynamics import SystemDynamics
from koopman_core.util import run_experiment, evaluate_ol_pred
from koopman_core.dynamics import LinearLiftedDynamics
from koopman_core.learning import KoopDnn, KoopmanNetAut
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB, ASHAScheduler
import matplotlib.pyplot as plt

class FiniteKoopSys(SystemDynamics):
    def __init__(self, mu, lambd):
        SystemDynamics.__init__(self, 4, 0)
        self.params = mu, lambd

    def drift(self, x, t):
        mu, lambd = self.params

        return np.array([x[2], x[3], mu * x[2], -lambd * x[2] ** 2 + lambd * x[3]])

    def eval_dot(self, x, u, t):
        return self.drift(x, t)

# Define system and system linearization:
sys_name = 'analytic_koop_sys'
n, m = 4, 0
mu, lambd = -0.3, -0.6
system = FiniteKoopSys(mu, lambd)

# Data collection parameters:
collect_data = True
test_frac = 0.2
val_frac = 0.2
dt = 1.0e-2                                                         # Time step length
traj_length_dc = 8.                                                 # Trajectory length, data collection
n_pred_dc = int(traj_length_dc/dt)                                  # Number of time steps, data collection
t_eval = dt * np.arange(n_pred_dc + 1)                              # Simulation time points
n_traj_train = 250                                                  # Number of trajectories to execute, data collection
n_traj_test = 100                                                   # Number of trajectories to execute, data collection
x0_max = np.array([1., 1., 1., 1.])                                         # Initial value limits
directory = os.path.abspath("working_files/bkeedmd/")               # Path to save learned models

# Model configuration parameters:
net_params = {}
net_params['state_dim'] = n
net_params['ctrl_dim'] = m
net_params['first_obs_const'] = False
net_params['override_kinematics'] = False
net_params['dt'] = dt
net_params['data_dir'] = directory + '/data'
net_params['n_multistep'] = 1

# DNN architecture parameters:
net_params['epochs'] = 200
net_params['optimizer'] = 'adam'

# DNN tunable parameters:
net_params['encoder_hidden_width'] = tune.choice([20, 50, 100, 200])
net_params['encoder_hidden_depth'] = tune.choice([1, 2, 3, 4, 10])
net_params['encoder_output_dim'] = tune.choice([1, 5, 10, 20])
net_params['lr'] = tune.loguniform(1e-5, 1e-2)
net_params['l2_reg'] = tune.loguniform(1e-6, 1e-1)
net_params['l1_reg'] = tune.loguniform(1e-6, 1e-1)
net_params['batch_size'] = tune.choice([16, 32, 64, 128])
net_params['lin_loss_penalty'] = tune.uniform(0, 1)

# Hyperparameter tuning parameters:
num_samples = -1
time_budget_s = 5*60*60                                      # Time budget for tuning process for each n_multistep value
n_multistep_lst = [1, 10]
if torch.cuda.is_available():
    resources_cpu = 2
    resources_gpu = 0.2
else:
    resources_cpu = 1
    resources_gpu = 0

# Collect/load datasets:
if collect_data:
    xs_train, t_eval_train = run_experiment(system, n, n_traj_train, n_pred_dc, t_eval, x0_max)
    xs_test, t_eval_test = run_experiment(system, n, n_traj_test, n_pred_dc, t_eval, x0_max)

    data_list = [xs_train, t_eval_train, n_traj_train, xs_test, t_eval_test, n_traj_test]
    outfile = open(directory + '/data/' + sys_name + '_data.pickle', 'wb')
    dill.dump(data_list, outfile)
    outfile.close()
else:
    infile = open(directory + '/data/' + sys_name + '_data.pickle', 'rb')
    xs_train, t_eval_train, n_traj_train, xs_test, t_eval_test, n_traj_test = dill.load(infile)
    infile.close()

# Define Koopman DNN model:
net = KoopmanNetAut(net_params)
model_kdnn = KoopDnn(net)
model_kdnn.set_datasets(xs_train, t_eval_train)

# Set up hyperparameter tuning:
trainable = lambda config: model_kdnn.model_pipeline(config, print_epoch=False, tune_run=True)
tune.register_trainable('trainable_pipeline', trainable)

best_trial_lst, best_config_lst = [], []
for n_multistep in n_multistep_lst:
    net_params['n_multistep'] = n_multistep

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='loss',
        mode='min',
        max_t=net_params['epochs'],
        grace_period=10,
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
    # algo = TuneBOHB(metric='loss', mode='min')
    # bohb = HyperBandForBOHB(
    #    metric='loss',
    #    mode='min',
    #    max_t=net_params['epochs'],
    #    time_attr='training_iteration'
    # )
    # result = tune.run(
    #    'trainable_pipeline',
    #    config=net_params,
    #    checkpoint_at_end=True,
    #    num_samples=num_samples,
    #    time_budget_s=time_budget_s,
    #    scheduler=bohb,
    #    search_alg=algo,
    #    resources_per_trial={'cpu': resources_cpu, 'gpu': resources_gpu},
    #    verbose=3
    # )

    best_trial_lst.append(result.get_best_trial("loss", "min", "last"))
    best_config_lst.append(result.get_best_config("loss", "min"))

# Analyze the results:
val_loss = []
test_loss = []
open_loop_mse = []
open_loop_std = []

for best_trial in best_trial_lst:
    # Extract validation loss:
    val_loss.append(best_trial.last_result["loss"])

    # Calculate test loss:
    net = KoopmanNetAut(best_trial.config)
    best_model = KoopDnn(net)
    checkpoint_path = os.path.join(best_trial.checkpoint.value, 'checkpoint')
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_model.net.load_state_dict(model_state)
    test_loss.append(best_model.test_loss(xs_test, t_eval_test))

    # Calculate open loop mse and std:
    n_tot = net_params['state_dim'] + net_params['encoder_output_dim'] + int(net_params['first_obs_const'])
    best_model.construct_koopman_model()
    sys_kdnn = LinearLiftedDynamics(best_model.A, None, best_model.C, best_model.basis_encode, continuous_mdl=False, dt=dt)
    mse, std = evaluate_ol_pred(sys_kdnn, xs_test, t_eval_test)
    open_loop_mse.append(mse)
    open_loop_std.append(std)

plt.figure()
plt.plot(n_multistep_lst, val_loss, label='validation loss')
plt.plot(n_multistep_lst, test_loss, label='test loss')
plt.plot(n_multistep_lst, open_loop_mse, label='open loop mse')
plt.plot(n_multistep_lst, open_loop_std, label='open loop std')
plt.xlabel('# of multistep prediction steps')
plt.ylabel('Loss')
plt.title('Best tuned model performance VS multistep horizon')
plt.legend()
plt.savefig(directory + '/figures/' + 'tuning_summary_' + sys_name + '.pdf')

outfile = open(directory + '/data/' + sys_name + '_best_params.pickle', 'wb')
data_list_tuning = [best_config_lst, val_loss, test_loss, open_loop_mse, open_loop_std]
dill.dump(data_list_tuning, outfile)
outfile.close()
