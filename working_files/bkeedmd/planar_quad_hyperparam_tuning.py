#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../../')
import os
import numpy as np
import torch
import scipy as sc
import dill
from core.dynamics import ConfigurationDynamics
from core.controllers import PDController
from koopman_core.util import evaluate_ol_pred, fit_standardizer
from koopman_core.controllers import MPCController, PerturbedController
from koopman_core.dynamics import LinearLiftedDynamics, BilinearLiftedDynamics
from koopman_core.learning import KoopDnn, KoopmanNetCtrl
from koopman_core.systems import PlanarQuadrotorForceInput
from koopman_core.learning.utils import differentiate_vec
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import random as rand
from sklearn import preprocessing

class QuadrotorPdOutput(ConfigurationDynamics):
    def __init__(self, dynamics, xd, t_d, n, m):
        ConfigurationDynamics.__init__(self, dynamics, 1)
        self.xd = xd
        self.t_d = t_d
        self.xd_dot = differentiate_vec(self.xd, self.t_d)
        self.n = n
        self.m = m

    def proportional(self, x, t):
        q, q_dot = x[:int(n/2)], x[int(n/2):]
        return self.y(q) - self.y_d(t)

    def derivative(self, x, t):
        q, q_dot = x[:int(n/2)], x[int(n/2):]
        return self.dydq(q)@q_dot - self.y_d_dot(t)

    def y(self, q):
        return q

    def dydq(self, q):
        return np.eye(int(self.n/2))

    def d2ydq2(self, q):
        return np.zeros((int(self.n/2), int(self.n/2), int(self.n/2)))

    def y_d(self, t):
        return self.desired_state_(t)[:int(self.n/2)]

    def y_d_dot(self, t):
        return self.desired_state_(t)[int(self.n/2):]

    def y_d_ddot(self, t):
        return self.desired_state_dot_(t)[int(self.n/2):]

    def desired_state_(self, t):
        return [np.interp(t, self.t_d.flatten(),self.xd[:,ii].flatten()) for ii in range(self.xd.shape[1])]

    def desired_state_dot_(self, t):
        return [np.interp(t, self.t_d.flatten(),self.xd_dot[:,ii].flatten()) for ii in range(self.xd_dot.shape[1])]


def run_experiment(system, n, n_traj, n_pred_dc, t, x0_max, m=None, K_p=None, K_d=None, noise_var=None):
    xd = np.empty((n_traj, n_pred_dc + 1, n))
    xs = np.empty((n_traj, n_pred_dc + 1, n))
    us = np.empty((n_traj, n_pred_dc, m))

    for ii in range(n_traj):
        x0 = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
        set_pt_dc = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
        mpc_trajgen = MPCController(nominal_sys, n_pred_dc, dt, umin, umax, xmin, xmax, QN_trajgen, R_trajgen,
                                    QN_trajgen, set_pt_dc)
        mpc_trajgen.eval(x0, 0)
        xd[ii, :, :] = mpc_trajgen.parse_result().T
        while abs(x0[0]) + abs(x0[1]) < 1 or np.any(np.isnan(xd[ii, :, :])):
            x0 = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
            set_pt_dc = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
            mpc_trajgen = MPCController(nominal_sys, n_pred_dc, dt, umin, umax, xmin, xmax, QN_trajgen, R_trajgen,
                                        QN_trajgen, set_pt_dc)
            mpc_trajgen.eval(x0, 0)
            xd[ii, :, :] = mpc_trajgen.parse_result().T

        output = QuadrotorPdOutput(system, xd[ii, :, :], t, n, m)
        pd_controller = PDController(output, K_dc_p, K_dc_d)
        perturbed_pd_controller = PerturbedController(system, pd_controller, noise_var, const_offset=hover_thrust)
        xs[ii, :, :], us[ii, :, :] = system.simulate(x0, perturbed_pd_controller, t)

    return xs, us, t

# Define system and system linearization:
mass = 2.
inertia = 1.
prop_arm = 0.2
gravity = 9.81
sys_name = 'planar_quad'
system = PlanarQuadrotorForceInput(mass, inertia, prop_arm, g=gravity)

# Linearized system specification:
n, m = 6, 2                                                         # Number of states, number of control inputs
A_nom = np.array([[0., 0., 0., 1., 0., 0.],                         # Linearization of the true system around the origin
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 1.],
                  [0., 0., -gravity, 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.]])
B_nom = np.array([[0., 0.],                                         # Linearization of the true system around the origin
                  [0., 0.],
                  [0., 0.],
                  [0., 0.],
                  [1./mass, 1./mass],
                  [-prop_arm/inertia, prop_arm/inertia]])

hover_thrust = mass*gravity/m

# Define LQR controller for data collection:
q_dc, r_dc = 5e2, 1                                                 # State and actuation penalty values, data collection
Q_dc = q_dc * np.identity(n)                                        # State penalty matrix, data collection
R_dc = r_dc*np.identity(m)                                          # Actuation penalty matrix, data collection
P_dc = sc.linalg.solve_continuous_are(A_nom, B_nom, Q_dc, R_dc)     # Algebraic Ricatti equation solution, data collection
K_dc = np.linalg.inv(R_dc)@B_nom.T@P_dc                             # LQR feedback gain matrix, data collection
K_dc_p = K_dc[:,:int(n/2)]                                          # Proportional control gains, data collection
K_dc_d = K_dc[:,int(n/2):]                                          # Derivative control gains, data collection
nominal_sys = LinearLiftedDynamics(A_nom, B_nom, np.eye(n), lambda x: x)

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
directory = os.path.abspath("working_files/bkeedmd/")                                                  # Path to save learned models

xmax = np.array([2, 2, np.pi/3, 2.,2.,2.])                          # State constraints, trajectory generation
xmin = -xmax
umax = np.array([2*hover_thrust, 2*hover_thrust]) - hover_thrust    # Actuation constraint, trajectory generation
umin = np.array([0., 0.]) - hover_thrust
x0_max = np.array([xmax[0], xmax[1], xmax[2], 1., 1., 1.])          # Initial value limits
Q_trajgen = sc.sparse.diags([0,0,0,0,0,0])                          # State penalty matrix, trajectory generation
QN_trajgen = sc.sparse.diags([5e1,5e1,5e1,1e1,1e1,1e1])             # Final state penalty matrix, trajectory generation
R_trajgen = sc.sparse.eye(m)                                        # Actuation penalty matrix, trajectory generation

# Model configuration parameters:
net_params = {}
net_params['state_dim'] = n
net_params['ctrl_dim'] = m
net_params['first_obs_const'] = True
net_params['override_kinematics'] = True
net_params['dt'] = dt
net_params['data_dir'] = directory + '/data'

# DNN architecture parameters:
net_params['epochs'] = 500
net_params['optimizer'] = 'adam'
net_params['lin_loss_penalty'] = 1.

# DNN tunable parameters:
net_params['encoder_hidden_width'] = tune.choice([20, 50, 100])
net_params['encoder_hidden_depth'] = tune.choice([1, 2, 4])
net_params['encoder_output_dim'] = tune.choice([1, 5, 10, 20])
net_params['lr'] = tune.loguniform(1e-5, 1e-2)
net_params['l2_reg'] = tune.loguniform(1e-5, 1e-2)
net_params['l1_reg'] = tune.loguniform(1e-5, 1e-2)
net_params['batch_size'] = tune.choice([64, 128, 256])
net_params['activation_type'] = tune.choice(['tanh'])


# Hyperparameter tuning parameters:
lin_loss_penalty_lst = np.linspace(0, 1, 11)
num_samples = -1
time_budget_s = 6*60*60                                    # Time budget for tuning process for each n_multistep value
if torch.cuda.is_available():
    resources_cpu = 2
    resources_gpu = 0.2
else:
    resources_cpu = 1
    resources_gpu = 0

# Collect/load datasets:
if collect_data:
    xs_train, us_train, t_eval_train = run_experiment(system, n, n_traj_train, n_pred_dc, t_eval, x0_max, m=m, K_p=K_dc_p,
                                                      K_d=K_dc_d, noise_var=noise_var)
    xs_val, us_val, t_eval_val = run_experiment(system, n, int(0.2*n_traj_train), n_pred_dc, t_eval, x0_max, m=m,
                                                      K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var)
    xs_test, us_test, t_eval_test = run_experiment(system, n, n_traj_test, n_pred_dc, t_eval,
                                                      x0_max, m=m, K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var)

    data_list = [xs_train, us_train, t_eval_train, xs_val, us_val, t_eval_val, n_traj_train, xs_test, us_test, t_eval_test, n_traj_test]
    outfile = open(directory + '/data/' + sys_name + '_data.pickle', 'wb')
    dill.dump(data_list, outfile)
    outfile.close()
else:
    infile = open(directory + '/data/' + sys_name + '_data.pickle', 'rb')
    xs_train, us_train, t_eval_train, xs_val, us_val, t_eval_val, n_traj_train, xs_test, us_test, \
            t_eval_test, n_traj_test = dill.load(infile)
    infile.close()

# Define Koopman DNN model:
standardizer_x_kdnn = fit_standardizer(xs_train, preprocessing.StandardScaler())
standardizer_u_kdnn = fit_standardizer(us_train, preprocessing.StandardScaler())

net = KoopmanNetCtrl(net_params, standardizer_x=standardizer_x_kdnn, standardizer_u=standardizer_u_kdnn)
model_kdnn = KoopDnn(net)
model_kdnn.set_datasets(xs_train, t_eval_train, u_train=us_train, x_val=xs_val, u_val=us_val, t_val=t_eval_val)

# Set up hyperparameter tuning:
trainable = lambda config: model_kdnn.model_pipeline(config, print_epoch=False, tune_run=True)
tune.register_trainable('trainable_pipeline', trainable)

best_trial_lst, best_config_lst = [], []
for lin_loss in lin_loss_penalty_lst:
    net_params['lin_loss_penalty'] = lin_loss
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
    best_trial_lst.append(result.get_best_trial("loss", "min", "last"))
    best_config_lst.append(result.get_best_config("loss", "min"))

# Analyze the results:
val_loss = []
test_loss = []
open_loop_mse = []
open_loop_std = []
for best_trial, best_config in zip(best_trial_lst, best_config_lst):
    # Extract validation loss:
    val_loss.append(best_trial.last_result["loss"])

    # Calculate test loss:
    best_net = KoopmanNetCtrl(best_trial.config, standardizer_x=standardizer_x_kdnn, standardizer_u=standardizer_u_kdnn)
    best_net.construct_net()
    best_model = KoopDnn(best_net)
    checkpoint_path = os.path.join(best_trial.checkpoint.value, 'checkpoint')
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_model.net.load_state_dict(model_state)
    test_loss.append(best_model.test_loss(xs_test, t_eval_test, u_test=us_test).cpu())

    # Calculate open loop mse and std:
    n_tot = net_params['state_dim'] + best_config['encoder_output_dim'] + int(net_params['first_obs_const'])
    best_model.construct_koopman_model()
    sys_kdnn = BilinearLiftedDynamics(n_tot, m, best_model.A, best_model.B, best_model.C,
                                          best_model.basis_encode,
                                          continuous_mdl=False, dt=dt, standardizer_x=standardizer_x_kdnn,
                                          standardizer_u=standardizer_u_kdnn)
    _, mse, std = evaluate_ol_pred(sys_kdnn, xs_test, t_eval_test, us=us_test)
    open_loop_mse.append(mse)
    open_loop_std.append(std)

print('Tuning procedure finalized.')

outfile = open(directory + '/data/' + sys_name + '_best_params.pickle', 'wb')
data_list_tuning = [best_config_lst, val_loss, test_loss, open_loop_mse, open_loop_std]
dill.dump(data_list_tuning, outfile)
outfile.close()
