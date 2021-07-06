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
from koopman_core.util import evaluate_ol_pred
from koopman_core.controllers import MPCController, PerturbedController
from koopman_core.dynamics import LinearLiftedDynamics, BilinearLiftedDynamics
from koopman_core.learning import KoopDnn, KoopmanNetCtrl
from koopman_core.systems import PlanarQuadrotorForceInput
from koopman_core.learning.utils import differentiate_vec
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import random as rand

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

# Experiment parameters:
n_data_exp = [10, 20, 50, 100, 200]
n_lift_exp = [1, 5, 10, 15, 20]
n_exp = 10
tune_mdls = True

# Data collection parameters:
collect_data = True
test_frac = 0.2
val_frac = 0.2
dt = 1.0e-2                                                         # Time step length
traj_length_dc = 2.                                                 # Trajectory length, data collection
n_pred_dc = int(traj_length_dc/dt)                                  # Number of time steps, data collection
t_eval = dt * np.arange(n_pred_dc + 1)                              # Simulation time points
n_traj_train = n_data_exp[-1]                                                      # Number of trajectories to execute, data collection
n_traj_val = int(0.2*n_traj_train)                                                      # Number of trajectories to execute, data collection
n_traj_test = 100                                                      # Number of trajectories to execute, data collection
n_traj_tune = 200
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
net_params['epochs'] = 200
net_params['optimizer'] = 'adam'

# DNN tunable parameters:
net_params['encoder_hidden_width'] = tune.choice([20, 50, 100, 200])
net_params['encoder_hidden_depth'] = tune.choice([1, 2, 3, 4, 10])
net_params['lr'] = tune.loguniform(1e-5, 1e-2)
net_params['l2_reg'] = tune.loguniform(1e-6, 1e-1)
net_params['l1_reg'] = tune.loguniform(1e-6, 1e-1)
net_params['batch_size'] = tune.choice([16, 32, 64, 128])
net_params['lin_loss_penalty'] = tune.uniform(0, 1)

# Hyperparameter tuning parameters:
num_samples = -1
time_budget_s = 60 #60*60                                    # Time budget for tuning process for each n_multistep value
if torch.cuda.is_available():
    resources_cpu = 2
    resources_gpu = 0.2
else:
    resources_cpu = 1
    resources_gpu = 0

config_lst = []
result_lst = []

# Collect/load datasets:
if collect_data:
    xs_train = np.zeros((n_exp, n_traj_train, n_pred_dc+1, n))
    us_train = np.zeros((n_exp, n_traj_train, n_pred_dc, m))
    t_eval_train = np.zeros((n_exp, n_pred_dc+1))
    xs_val = np.zeros((n_exp, n_traj_val, n_pred_dc+1, n))
    us_val = np.zeros((n_exp, n_traj_val, n_pred_dc, m))
    t_eval_val = np.zeros((n_exp, n_pred_dc+1))
    xs_test = np.zeros((n_exp, n_traj_test, n_pred_dc+1, n))
    us_test = np.zeros((n_exp, n_traj_test, n_pred_dc, m))
    t_eval_test = np.zeros((n_exp, n_pred_dc+1))

    for ii in range(n_exp):
        xs_train[ii, :, :, :], us_train[ii, :, :, :], t_eval_train[ii, :] = run_experiment(system, n, n_traj_train, n_pred_dc, t_eval, x0_max, m=m, K_p=K_dc_p,
                                                          K_d=K_dc_d, noise_var=noise_var)
        xs_val[ii, :, :, :], us_val[ii, :, :, :], t_eval_val[ii, :] = run_experiment(system, n, int(0.2*n_traj_train), n_pred_dc, t_eval, x0_max, m=m,
                                                          K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var)
        xs_test[ii, :, :, :], us_test[ii, :, :, :], t_eval_test[ii, :] = run_experiment(system, n, n_traj_test, n_pred_dc, t_eval,
                                                          x0_max, m=m, K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var)

    xs_tune, us_tune, t_eval_tune = run_experiment(system, n, n_traj_tune, n_pred_dc, t_eval, x0_max, m=m,
                                                      K_p=K_dc_p, K_d=K_dc_d, noise_var=noise_var)



    data_list = [xs_train, us_train, t_eval_train, xs_val, us_val, t_eval_val, n_traj_val,
                 xs_test, us_test, t_eval_test, n_traj_test, xs_tune, us_tune, t_eval_train, n_traj_tune]
    outfile = open(directory + '/data/' + sys_name + '_data.pickle', 'wb')
    dill.dump(data_list, outfile)
    outfile.close()
else:
    infile = open(directory + '/data/' + sys_name + '_data.pickle', 'rb')
    xs_train, us_train, t_eval_train, xs_val, us_val, t_eval_val, n_traj_train, \
    xs_test, us_test, t_eval_test, n_traj_test, xs_tune, us_tune, t_eval_tune, n_traj_tune = dill.load(infile)
    infile.close()


for n_lift in n_lift_exp:
    net_params['encoder_output_dim'] = n_lift
    if tune_mdls:
        # Define Koopman DNN model:
        net = KoopmanNetCtrl(net_params)
        model_kdnn = KoopDnn(net)
        model_kdnn.set_datasets(xs_tune, t_eval_tune, u_train=us_tune-hover_thrust, x_val=xs_val, u_val=us_val-hover_thrust, t_val=t_eval_val)

        # Set up hyperparameter tuning:
        trainable = lambda config: model_kdnn.model_pipeline(config, print_epoch=False, tune_run=True)
        tune.register_trainable('trainable_pipeline', trainable)

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

        config_dict = {'n_lift': n_lift,
                       'config': config}
        config_lst.append(config_dict)

    else:
        infile = open(directory + '/data/' + sys_name + 'best_params.pickle', 'rb')
        config_lst = dill.load(infile)
        infile.close()
        config = [c['config'] for c in config_lst if c['n_lift'] == n_lift][0]

    for n_data in n_data_exp:
        for ii in range(n_exp):
            # Set datasets:
            xs_tmp, us_tmp = xs_train[ii, :n_data, :, :], us_train[ii, :n_data, :, :]
            xs_val_tmp, us_val_tmp = xs_val[ii, :, :, :], us_val[ii, :, :, :]
            xs_test_tmp, us_test_tmp = xs_test[ii, :, :, :], us_test[ii, :, :, :]

            # Initialize and train model:
            net = KoopmanNetCtrl(config)
            model_kdnn = KoopDnn(net)
            model_kdnn.set_datasets(xs_tmp, t_eval_train, u_train=us_tmp - hover_thrust, x_val=xs_val_tmp,
                                    u_val=us_val_tmp - hover_thrust, t_val=t_eval_val[ii, :])
            model_kdnn.model_pipeline(config, early_stop=True)

            # Evaluate performance:
            model_kdnn.construct_koopman_model()
            n_tot = net_params['state_dim'] + config['encoder_output_dim'] + int(net_params['first_obs_const'])
            sys_kdnn = BilinearLiftedDynamics(n_tot, m, model_kdnn.A, model_kdnn.B, model_kdnn.C,
                                                  model_kdnn.basis_encode, continuous_mdl=False, dt=dt)

            train_loss_tmp = model_kdnn.test_loss(xs_tmp, t_eval_test, u_test=us_tmp - hover_thrust).cpu()
            test_loss_tmp = model_kdnn.test_loss(xs_test_tmp, t_eval_test[ii, :], u_test=us_test_tmp - hover_thrust).cpu()
            _, mse_tmp, std_tmp = evaluate_ol_pred(sys_kdnn, xs_test_tmp, t_eval_test, us=us_test_tmp - hover_thrust)

            result_dict = {'n_lift': n_lift,
                        'n_data': n_data,
                        'train_loss_kdnn': train_loss_tmp,
                        'test_loss_kdnn': test_loss_tmp,
                        'mse_kdnn': mse_tmp,
                        'std_kdnn': std_tmp}
            result_lst.append(result_dict)

if tune_mdls:
    outfile = open(directory + '/data/' + sys_name + 'best_params.pickle', 'wb')
    dill.dump(config_lst, outfile)
    outfile.close()

outfile = open(directory + '/data/' + sys_name + 'num_exp.pickle', 'wb')
dill.dump(result_lst, outfile)
outfile.close()


