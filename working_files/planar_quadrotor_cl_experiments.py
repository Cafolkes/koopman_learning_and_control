#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sc
from scipy import ndimage
import random as rand
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tabulate import tabulate
import dill
import copy

from core.controllers import PDController
from core.dynamics import ConfigurationDynamics

from koopman_core.controllers import OpenLoopController, MPCController, PerturbedController, NonlinearMPCController, BilinearMPCController
from koopman_core.dynamics import LinearLiftedDynamics, BilinearLiftedDynamics
from koopman_core.learning import Edmd, BilinearEdmd
from koopman_core.basis_functions import PlanarQuadBasis
from koopman_core.learning.utils import differentiate_vec
from koopman_core.systems import PlanarQuadrotorForceInput

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

class PlanarQuadrotorForceInputDiscrete(PlanarQuadrotorForceInput):
    def __init__(self, mass, inertia, prop_arm, g=9.81, dt=1e-2):
        PlanarQuadrotorForceInput.__init__(self, mass, inertia, prop_arm, g=g)
        self.dt=dt
        
    def eval_dot(self, x, u, t):
        return x + self.dt*self.drift(x, t) + self.dt*np.dot(self.act(x, t),u)

    def get_linearization(self, x0, x1, u0, t):
        m, J, b, g = self.params
        A_lin = np.eye(self.n) + self.dt*np.array([[0, 0, 0, 1, 0, 0],
                                                   [0, 0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 0, 1],
                                                   [0, 0, -(1/m)*np.cos(x0[2])*u0[0] -(1/m)*np.cos(x0[2])*u0[1], 0, 0, 0],
                                                   [0, 0, -(1/m)*np.sin(x0[2])*u0[0] -(1/m)*np.sin(x0[2])*u0[1], 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0],])

        B_lin = self.dt*np.array([[0, 0],
                                  [0, 0],
                                  [0, 0],
                                  [-(1/m)*np.sin(x0[2]), -(1/m)*np.sin(x0[2])],
                                  [(1/m)*np.cos(x0[2]), (1/m)*np.cos(x0[2])],
                                  [-b/J, b/J]])

        if x1 is None:
            x1 = A_lin@x0 + B_lin@u0

        f_d = self.eval_dot(x0,u0,t)
        r_lin = f_d - x1

        return A_lin, B_lin, r_lin

# Cart pole system parameters
mass = 2.
inertia = 1.
prop_arm = 0.2
gravity = 9.81
quadrotor = PlanarQuadrotorForceInput(mass, inertia, prop_arm, g=gravity)

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

# Data collection parameters:
q_dc, r_dc = 5e2, 1                                                 # State and actuation penalty values, data collection
Q_dc = q_dc * np.identity(n)                                        # State penalty matrix, data collection
R_dc = r_dc*np.identity(m)                                          # Actuation penalty matrix, data collection
P_dc = sc.linalg.solve_continuous_are(A_nom, B_nom, Q_dc, R_dc)     # Algebraic Ricatti equation solution, data collection
K_dc = np.linalg.inv(R_dc)@B_nom.T@P_dc                             # LQR feedback gain matrix, data collection
K_dc_p = K_dc[:,:int(n/2)]                                          # Proportional control gains, data collection
K_dc_d = K_dc[:,int(n/2):]                                          # Derivative control gains, data collection
nominal_sys = LinearLiftedDynamics(A_nom, B_nom, np.eye(n), lambda x: x)

dt = 1.0e-2                                                         # Time step length
traj_length_dc = 2.                                                 # Trajectory length, data collection
n_pred_dc = int(traj_length_dc/dt)                                  # Number of time steps, data collection
t_eval = dt * np.arange(n_pred_dc + 1)                              # Simulation time points
n_traj_dc = 100                                                     # Number of trajectories to execute, data collection
noise_var = 5.                                                      # Exploration noise to perturb controller, data collection

xmax = np.array([2, 2, np.pi/3, 2.,2.,2.])                          # State constraints, trajectory generation
xmin = -xmax
umax = np.array([2*hover_thrust, 2*hover_thrust]) - hover_thrust    # Actuation constraint, trajectory generation
umin = np.array([0., 0.]) - hover_thrust
x0_max = np.array([xmax[0], xmax[1], xmax[2], 1., 1., 1.])          # Initial value limits
Q_trajgen = sc.sparse.diags([0,0,0,0,0,0])                          # State penalty matrix, trajectory generation
QN_trajgen = sc.sparse.diags([5e1,5e1,5e1,1e1,1e1,1e1])             # Final state penalty matrix, trajectory generation
R_trajgen = sc.sparse.eye(m)                                        # Actuation penalty matrix, trajectory generation
sub_sample_rate = 1                                                 # Rate to subsample data for training
n_cols = 10                                                         # Number of columns in training data plot
folder_plots = 'working_files/figures/'
#dropbox_folder = '/Users/carlaxelfolkestad/Dropbox/Apps/Overleaf/Koopman NMPC (ICRA21)/'

# Model and learning parameters:
learn_models = False
model_fname = 'working_files/data/planar_quad_models.pickle'
alpha_dmd = 9.8e-5                                                  # Regularization strength (LASSO) DMD
tune_mdl_dmd = False
alpha_edmd = 2.22e-4                                                # Regularization strength (LASSO) EDMD
tune_mdl_edmd = False
alpha_bedmd = 6.9e-5                                                # Regularization strength (LASSO) bEDMD
tune_mdl_bedmd = False

# Trajectory generation performance evaluation parameters:
solver_settings = {}
solver_settings['gen_embedded_ctrl'] = False
solver_settings['warm_start'] = True
solver_settings['polish'] = True
solver_settings['polish_refine_iter'] = 3
solver_settings['scaling'] = True
solver_settings['adaptive_rho'] = False
solver_settings['check_termination'] = 25
solver_settings['max_iter'] = 4000
solver_settings['eps_abs'] = 1e-6
solver_settings['eps_rel'] = 1e-6
solver_settings['eps_prim_inf'] = 1e-4
solver_settings['eps_dual_inf'] = 1e-4
solver_settings['linsys_solver'] = 'qdldl'

traj_length_trajgen = 250
t_eval_trajgen = dt * np.arange(traj_length_trajgen+1)              # Simulation time points, closed loop
Q_mpc = sc.sparse.diags([0,0,0,0,0,0])                              # State penalty matrix, trajectory generation
QN_mpc = sc.sparse.diags([1e5,1e5,1e5,1e5,1e5,1e5])                 # Final state penalty matrix, trajectory generation
R_mpc = sc.sparse.eye(m)                                            # Actuation penalty matrix, trajectory generation
ctrl_offset = np.array([[hover_thrust], [hover_thrust]])            # Control offset, learned models (center around hover)
x0f_trajgen = np.array([2., 2., 0.1, 1., 1., 1.])                   # Initial value limits
xmax_trajgen = np.array([2, 2, np.pi/3, 2.,2.,2.])                  # State constraints, trajectory generation
xmin_trajgen = -xmax
term_constraint = False
max_iter_sqp = 100
add_slack_trajgen = False
q_slack_trajgen = 1e3

# Closed loop control performance evaluation parameters:
Q_mpc_cl = sc.sparse.diags([1e3,1e3,1e3,1e2,1e2,1e2])
QN_mpc_cl = Q_mpc_cl
R_mpc_cl = sc.sparse.eye(m)
traj_duration_cl = 0.5
traj_length_cl = int(traj_duration_cl/dt)
t_eval_cl = np.arange(250)*dt
add_slack_cl = True
q_slack_cl = q_slack_trajgen

solver_settings_cl = copy.deepcopy(solver_settings)
solver_settings_cl['polish'] = False
solver_settings_cl['check_termination'] = 10
solver_settings_cl['max_iter'] = 10
solver_settings_cl['eps_abs'] = 1e-2
solver_settings_cl['eps_rel'] = 1e-2
solver_settings_cl['eps_prim_inf'] = 1e-3
solver_settings_cl['eps_dual_inf'] = 1e-3

# Experiment parameters:
n_exp = 100

#==================================================== COLLECT DATA ====================================================:
if learn_models:
    xd = np.empty((n_traj_dc, n_pred_dc + 1, n))
    xs = np.empty((n_traj_dc, n_pred_dc + 1, n))
    us = np.empty((n_traj_dc, n_pred_dc, m))

    plt.figure(figsize=(12, 12 * n_traj_dc / (n_cols ** 2)))
    for ii in range(n_traj_dc):
        x0 = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
        set_pt_dc = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
        mpc_trajgen = MPCController(nominal_sys, n_pred_dc, dt, umin, umax, xmin, xmax, QN_trajgen, R_trajgen,
                                    QN_trajgen, set_pt_dc)
        mpc_trajgen.eval(x0, 0)
        xd[ii, :, :] = mpc_trajgen.parse_result().T
        while np.linalg.norm(x0[:3] - set_pt_dc[:3]) < 2 or np.any(np.isnan(xd[ii, :, :])):
            x0 = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
            set_pt_dc = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
            mpc_trajgen = MPCController(nominal_sys, n_pred_dc, dt, umin, umax, xmin, xmax, QN_trajgen, R_trajgen,
                                        QN_trajgen, set_pt_dc)
            mpc_trajgen.eval(x0, 0)
            xd[ii, :, :] = mpc_trajgen.parse_result().T

        output = QuadrotorPdOutput(quadrotor, xd[ii, :, :], t_eval, n, m)
        pd_controller = PDController(output, K_dc_p, K_dc_d)
        perturbed_pd_controller = PerturbedController(quadrotor, pd_controller, noise_var, const_offset=hover_thrust)
        xs[ii, :, :], us[ii, :, :] = quadrotor.simulate(x0, perturbed_pd_controller, t_eval)

        plt.subplot(int(np.ceil(n_traj_dc / n_cols)), n_cols, ii + 1)
        plt.plot(t_eval, xs[ii, :, 0], 'b', label='$y$')
        plt.plot(t_eval, xs[ii, :, 1], 'g', label='$z$')
        plt.plot(t_eval, xs[ii, :, 2], 'r', label='$\\theta$')
        plt.plot(t_eval, xd[ii, :, 0], '--b', label='$y_d$')
        plt.plot(t_eval, xd[ii, :, 1], '--g', label='$z_d$')
        plt.plot(t_eval, xd[ii, :, 2], '--r', label='$\\theta_d$')

    suptitle = plt.suptitle(
        'Training data \nx-axis: time (sec), y-axis: state value, $x$ - blue, $xd$ - dotted blue, $\\theta$ - red, '
        '$\\theta_d$ - dotted red',
        y=0.94)
    plt.tight_layout()


#==================================================== LEARN MODELS ====================================================:
if learn_models:
    #Learn linear model with DMD:
    basis = lambda x: x
    C_dmd = np.eye(n)

    optimizer_dmd = linear_model.MultiTaskLasso(alpha=alpha_dmd, fit_intercept=False, selection='random')
    cv_dmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
    standardizer_dmd = preprocessing.StandardScaler(with_mean=False)

    model_dmd = Edmd(n, m, basis, n, n_traj_dc, optimizer_dmd, cv=cv_dmd, standardizer=standardizer_dmd, C=C_dmd,
                     first_obs_const=False, continuous_mdl=False, dt=dt)
    xdmd, y_dmd = model_dmd.process(xs, us-hover_thrust, np.tile(t_eval,(n_traj_dc,1)), downsample_rate=sub_sample_rate)
    model_dmd.fit(xdmd, y_dmd, cv=tune_mdl_dmd, override_kinematics=True)
    sys_dmd = LinearLiftedDynamics(model_dmd.A, model_dmd.B, model_dmd.C, model_dmd.basis, continuous_mdl=False, dt=dt)
    if tune_mdl_dmd:
        print('$\\alpha$ DMD: ', model_dmd.cv.alpha_)


    #Learn lifted linear model with EDMD:
    basis = PlanarQuadBasis(n, poly_deg=3)
    basis.construct_basis()
    planar_quad_features = preprocessing.FunctionTransformer(basis.basis)
    planar_quad_features.fit(np.zeros((1,n)))
    n_lift_edmd = planar_quad_features.transform((np.zeros((1,n)))).shape[1]
    C_edmd = np.zeros((n,n_lift_edmd))
    C_edmd[:,1:n+1] = np.eye(n)

    optimizer_edmd = linear_model.MultiTaskLasso(alpha=alpha_edmd, fit_intercept=False, selection='random', max_iter=2000)
    cv_edmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random', max_iter=2000)
    standardizer_edmd = preprocessing.StandardScaler(with_mean=False)

    model_edmd = Edmd(n, m, basis.basis, n_lift_edmd, n_traj_dc, optimizer_edmd, cv=cv_edmd,
                      standardizer=standardizer_edmd, C=C_edmd, continuous_mdl=False, dt=dt)
    X_edmd, y_edmd = model_edmd.process(xs, us-hover_thrust, np.tile(t_eval,(n_traj_dc,1)),
                                        downsample_rate=sub_sample_rate)
    model_edmd.fit(X_edmd, y_edmd, cv=tune_mdl_edmd, override_kinematics=True)
    #model_edmd.reduce_mdl()
    #sys_edmd = LinearLiftedDynamics(model_edmd.A, model_edmd.B, model_edmd.C, model_edmd.basis_reduced, continuous_mdl=False, dt=dt)
    sys_edmd = LinearLiftedDynamics(model_edmd.A, model_edmd.B, model_edmd.C, model_edmd.basis, continuous_mdl=False, dt=dt)
    if tune_mdl_edmd:
        print('$\\alpha$ EDMD: ',model_edmd.cv.alpha_)

    #Learn lifted bilinear model with bEDMD
    n_lift_bedmd = n_lift_edmd
    C_bedmd = np.zeros((n,n_lift_bedmd))
    C_bedmd[:,1:n+1] = np.eye(n)

    basis_bedmd = lambda x: planar_quad_features.transform(x)
    optimizer_bedmd = linear_model.MultiTaskLasso(alpha=alpha_bedmd, fit_intercept=False, selection='random',
                                                  max_iter=1e4)
    cv_bedmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
    standardizer_bedmd = preprocessing.StandardScaler(with_mean=False)

    model_bedmd = BilinearEdmd(n, m, basis_bedmd, n_lift_bedmd, n_traj_dc, optimizer_bedmd, cv=cv_bedmd,
                               standardizer=standardizer_bedmd, C=C_bedmd, continuous_mdl=False, dt=dt)
    X_bedmd, y_bedmd = model_bedmd.process(xs, us-hover_thrust, np.tile(t_eval,(n_traj_dc,1)),
                                           downsample_rate=sub_sample_rate)
    model_bedmd.fit(X_bedmd, y_bedmd, cv=tune_mdl_bedmd, override_kinematics=True)
    sys_bedmd = BilinearLiftedDynamics(model_bedmd.n_lift, m, model_bedmd.A, model_bedmd.B, model_bedmd.C,
                                       model_bedmd.basis, continuous_mdl=False, dt=dt)
    if tune_mdl_bedmd:
        print('$\\alpha$ bilinear EDMD: ', model_bedmd.cv.alpha_)

    # Store learned models
    data_list = [sys_dmd, sys_edmd, sys_bedmd]
    outfile = open(model_fname, 'wb')
    dill.dump(data_list, outfile)
    outfile.close()
else:
    infile = open(model_fname, 'rb')
    sys_dmd, sys_edmd, sys_bedmd = dill.load(infile)
    infile.close()
    basis = PlanarQuadBasis(n, poly_deg=3)
    basis.construct_basis()
    planar_quad_features = preprocessing.FunctionTransformer(basis.basis)
    planar_quad_features.fit(np.zeros((1, n)))

#============================== RUN TRAJECTORY GENERATION AND CLOSED LOOP EXPERIMENTS =============================:
exp = 0
sol_stats = []
while exp < n_exp:
    print('Running experiment ', exp+1, ' out of ', n_exp)
    # Reset parameters:
    dmd_trajgen_success, edmd_trajgen_success, bedmd_trajgen_success, nmpc_trajgen_success = True, True, True, True
    dmd_cl_success, edmd_cl_success, bedmd_cl_success, nmpc_cl_success = True, True, True, True

    # Sample initial and terminal condition:
    x0_exp = np.asarray([rand.uniform(l, u) for l, u in zip(-x0f_trajgen, x0f_trajgen)])
    xf_exp = np.asarray([rand.uniform(l, u) for l, u in zip(-x0f_trajgen, x0f_trajgen)])
    while np.linalg.norm(x0_exp - xf_exp) < 4:
        x0_exp = np.asarray([rand.uniform(l, u) for l, u in zip(-x0f_trajgen, x0f_trajgen)])
        xf_exp = np.asarray([rand.uniform(l, u) for l, u in zip(-x0f_trajgen, x0f_trajgen)])

    x_init = np.linspace(x0_exp, xf_exp, int(traj_length_trajgen) + 1)  # Initial state guess for SQP-algorithm
    u_init = np.zeros((m, traj_length_trajgen)).T  # Initial control guess for SQP-algorithm

    # ========================================== TRAJECTORY GENERATION =========================================:
    # Generate trajectory with DMD model:
    controller_dmd = MPCController(sys_dmd, traj_length_trajgen, dt, umin, umax, xmin_trajgen, xmax_trajgen, Q_mpc,
                                   R_mpc, QN_mpc, xf_exp, add_slack=add_slack_trajgen, terminal_constraint=term_constraint,
                                   const_offset=ctrl_offset.squeeze())
    try:
        controller_dmd.eval(x0_exp, 0)
        xr_dmd = controller_dmd.parse_result()
        ur_dmd = controller_dmd.get_control_prediction() + hover_thrust

        ol_controller_dmd = OpenLoopController(quadrotor, ur_dmd.T, t_eval_trajgen[:-1])
        xs_dmd, _ = quadrotor.simulate(x0_exp, ol_controller_dmd, t_eval_trajgen)
        xs_dmd = xs_dmd.T
    except:
        xr_dmd, ur_dmd, xs_dmd = np.nan, np.nan, np.nan
        dmd_trajgen_success = False

    # Generate trajectory with EDMD model:
    controller_edmd = MPCController(sys_edmd, traj_length_trajgen, dt, umin, umax, xmin_trajgen, xmax_trajgen, Q_mpc,
                                    R_mpc, QN_mpc, xf_exp, add_slack=add_slack_trajgen, terminal_constraint=term_constraint,
                                    const_offset=ctrl_offset.squeeze())
    try:
        controller_edmd.eval(x0_exp, 0)
        xr_edmd = sys_edmd.C@controller_edmd.parse_result()
        ur_edmd = controller_edmd.get_control_prediction() + hover_thrust

        ol_controller_edmd = OpenLoopController(quadrotor, ur_edmd.T, t_eval_trajgen[:-1])
        xs_edmd, _ = quadrotor.simulate(x0_exp, ol_controller_edmd, t_eval_trajgen)
        xs_edmd = xs_edmd.T
    except:
        xr_edmd, ur_edmd, xs_edmd = np.nan, np.nan, np.nan
        edmd_trajgen_success = False

    # Generate trajectory with bEDMD model:
    controller_bedmd = BilinearMPCController(sys_bedmd, traj_length_trajgen, dt, umin, umax, xmin_trajgen, xmax_trajgen,
                                             Q_mpc, R_mpc, QN_mpc, xf_exp, solver_settings, add_slack=add_slack_trajgen,
                                             q_slack=q_slack_trajgen, terminal_constraint=term_constraint, const_offset=ctrl_offset)
    z0_exp = sys_bedmd.basis(x0_exp.reshape((1,-1))).squeeze()
    z_init = sys_bedmd.basis(x_init)
    controller_bedmd.construct_controller(z_init, u_init)
    try:
        controller_bedmd.solve_to_convergence(z0_exp, 0., z_init, u_init, max_iter=max_iter_sqp)
        xr_bedmd = sys_bedmd.C@controller_bedmd.get_state_prediction().T
        ur_bedmd = controller_bedmd.get_control_prediction().T + hover_thrust

        ol_controller_bedmd = OpenLoopController(quadrotor, ur_bedmd.T, t_eval_trajgen[:-1])
        xs_bedmd, _ = quadrotor.simulate(x0_exp, ol_controller_bedmd, t_eval_trajgen)
        xs_bedmd = xs_bedmd.T
    except:
        xr_bedmd, ur_bedmd, xs_bedmd = np.nan, np.nan, np.nan
        bedmd_trajgen_success = False

    # Generate trajectory with nonlinear MPC with full model knowledge (benchmark):
    quadrotor_d = PlanarQuadrotorForceInputDiscrete(mass, inertia, prop_arm, g=gravity, dt=dt)
    controller_nmpc = NonlinearMPCController(quadrotor_d, traj_length_trajgen, dt, umin+hover_thrust, umax+hover_thrust,
                                             xmin_trajgen, xmax_trajgen, Q_mpc, R_mpc, QN_mpc, xf_exp, solver_settings,
                                             add_slack=add_slack_trajgen, q_slack=q_slack_trajgen,
                                             terminal_constraint=term_constraint)
    controller_nmpc.construct_controller(x_init, u_init+hover_thrust)
    try:
        controller_nmpc.solve_to_convergence(x0_exp, 0., x_init, u_init + ctrl_offset.reshape(1,-1),
                                             max_iter=max_iter_sqp)
        xr_nmpc = controller_nmpc.get_state_prediction().T
        ur_nmpc = controller_nmpc.get_control_prediction().T

        ol_controller_nmpc = OpenLoopController(quadrotor, ur_nmpc.T, t_eval_trajgen[:-1])
        xs_nmpc, _ = quadrotor.simulate(x0_exp, ol_controller_nmpc, t_eval_trajgen)
        xs_nmpc = xs_nmpc.T
    except:
        xr_nmpc, ur_nmpc, xs_nmpc = np.nan, np.nan, np.nan
        nmpc_trajgen_success = False

    trajgen_success_lst = [dmd_trajgen_success, edmd_trajgen_success, bedmd_trajgen_success, nmpc_trajgen_success]
    xr_lst = [xr_dmd, xr_edmd, xr_bedmd, xr_nmpc]
    ur_lst = [ur_dmd, ur_edmd, ur_bedmd, ur_nmpc]
    xs_lst = [xs_dmd, xs_edmd, xs_bedmd, xs_nmpc]
    n_iter_sqp_lst = [np.nan, np.nan, len(controller_bedmd.x_iter), len(controller_nmpc.x_iter)]
    """
    # TODO: Remove plotting once functinality is verified
    # Analyze and compare performance:
    plot_inds = [0, 1, 2, 3, 4, 5, 0, 1]
    subplot_inds = [1, 2, 3, 5, 6, 7, 4, 8]
    labels = ['$y$ (m)', '$z$ (m)', '$\\theta$ (rad)', '$\\dot{y}$ (m/s)','$\\dot{z}$ (m/s)', '$\\dot{\\theta}$',
              '$T_1$ (N)','$T_2$ (N)']
    titles = ['y-coordinates', 'z-coordinates', '$\\theta$-coordinates', 'Control inputs']
    colors = ['tab:blue', 'tab:orange', 'tab:brown', 'tab:cyan']

    plt.figure(figsize=(12,4))
    #plt.suptitle('Trajectory designed with model predictive controllers\nsolid lines - designed trajectory |
    # dashed lines - open loop simulated trajectory | black dotted lines - state/actuation bounds')
    for ii in range(8):
        ind = plot_inds[ii]
        if ii < 6:
            ax = plt.subplot(2,4,subplot_inds[ii])
            if dmd_trajgen_success:
                plt.plot(t_eval_trajgen, xr_dmd[ind,:], colors[0], label='DMD MPC')
                plt.plot(t_eval_trajgen, xs_dmd[ind, :], '--', color=colors[0], linewidth=1)

            if edmd_trajgen_success:
                plt.plot(t_eval_trajgen, xr_edmd[ind, :], colors[1], label='EDMD MPC')
                plt.plot(t_eval_trajgen, xs_edmd[ind, :], '--', color=colors[1], linewidth=1)

            if bedmd_trajgen_success:
                plt.plot(t_eval_trajgen, xr_bedmd[ind, :], colors[2], label='K-MPC')
                plt.plot(t_eval_trajgen, xs_bedmd[ind, :], '--', color=colors[2], linewidth=1)

            if nmpc_trajgen_success:
                plt.plot(t_eval_trajgen, xr_nmpc[ind,:], colors[3], label='NMPC')
                plt.plot(t_eval_trajgen, xs_nmpc[ind, :], '--', color=colors[3], linewidth=1)

            plt.scatter(t_eval_trajgen[0], x0_exp[ind], color='g')
            plt.scatter(t_eval_trajgen[-1], xf_exp[ind], color='r')
            plt.ylabel(labels[ind])
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if ii >= 3:
                plt.plot([0, t_eval_trajgen[-1]], [xmax[ind], xmax[ind]], ':k')
                plt.plot([0, t_eval_trajgen[-1]], [xmin[ind], xmin[ind]], ':k')
                #plt.ylim(xmin[ind]-0.1,xmax[ind]+0.1)
            if subplot_inds[ii]==1:
                plt.legend(loc='upper left', frameon=False)
        elif ii < 8:
            ax = plt.subplot(2,4,subplot_inds[ii])
            if dmd_trajgen_success:
                plt.plot(t_eval_trajgen[:-1],ur_dmd[ind,:], color=colors[0], label='DMD MPC')
            if edmd_trajgen_success:
                plt.plot(t_eval_trajgen[:-1], ur_edmd[ind, :], color=colors[1], label='EDMD MPC')
            if bedmd_trajgen_success:
                plt.plot(t_eval_trajgen[:-1], ur_bedmd[ind, :], color=colors[2], label='K-NMPC')
            if nmpc_trajgen_success:
                plt.plot(t_eval_trajgen[:-1],ur_nmpc[ind,:], color=colors[3], label='NMPC')
            plt.plot([0, t_eval_trajgen[-1]], [umax[ind]+hover_thrust, umax[ind]+hover_thrust], ':k')
            plt.plot([0, t_eval_trajgen[-1]], [umin[ind]+hover_thrust, umin[ind]+hover_thrust], ':k')
            plt.ylabel(labels[ii])
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if subplot_inds[ii] > 4:
            plt.xlabel('Time (sec)')
        else:
            plt.title(titles[subplot_inds[ii]-1])

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.tight_layout()
    """
    sol_stats_exp = {}
    model_names = ['dmd', 'edmd', 'bedmd', 'nmpc']
    for name, trajgen_success, xr, ur, xs, n_iter_sqp in \
            zip(model_names, trajgen_success_lst, xr_lst, ur_lst, xs_lst, n_iter_sqp_lst):
        if trajgen_success:
            cost_ref = (xr[:,-1]-xf_exp).T@QN_mpc@(xr[:,-1]-xf_exp) + np.sum(np.diag(ur.T@R_mpc@ur))
            dist_ol = np.linalg.norm(xs[:, -1] - xf_exp)
            ctrl_ref = np.linalg.norm(ur)
        else:
            cost_ref, dist_ol, ctrl_ref, n_iter_sqp = np.nan, np.nan, np.nan, np.nan

        sol_stats_exp[name] = {'trajgen_success': trajgen_success, 'x0': x0_exp, 'xf': xf_exp, 'cost_traj': cost_ref,
                              'xf_dist_traj': dist_ol, 'ctrl_traj': ctrl_ref, 'n_iter_sqp': n_iter_sqp}

    # ========================================== CLOSED LOOP  =========================================:
    # Define and initialize controller based on DMD model:
    controller_dmd_cl = MPCController(sys_dmd, traj_length_cl, dt, umin, umax, xmin_trajgen, xmax_trajgen, Q_mpc_cl,
                                      R_mpc_cl, QN_mpc_cl, xf_exp, add_slack=add_slack_cl, const_offset=ctrl_offset.squeeze())
    controller_dmd_cl = PerturbedController(sys_dmd, controller_dmd_cl, 0., const_offset=hover_thrust,
                                            umin=umin, umax=umax)
    try:
        xs_dmd_cl, us_dmd_cl = quadrotor.simulate(x0_exp, controller_dmd_cl, t_eval_cl)
        xs_dmd_cl, us_dmd_cl = xs_dmd_cl.T, us_dmd_cl.T
        dmd_cl_runtime_mean = np.mean(controller_dmd_cl.nom_controller.comp_time)
        dmd_cl_runtime_std = np.std(controller_dmd_cl.nom_controller.comp_time)
    except:
        xs_dmd_cl, us_dmd_cl = np.nan, np.nan
        dmd_cl_runtime_mean, dmd_cl_runtime_std = np.nan, np.nan
        dmd_cl_success = False

    # Define and initialize controller based on EDMD model:
    controller_edmd_cl = MPCController(sys_edmd, traj_length_cl, dt, umin, umax, xmin_trajgen, xmax_trajgen, Q_mpc_cl,
                                       R_mpc_cl, QN_mpc_cl, xf_exp, add_slack=add_slack_cl, const_offset=ctrl_offset.squeeze())
    controller_edmd_cl = PerturbedController(sys_edmd,controller_edmd_cl,0.,const_offset=hover_thrust,
                                             umin=umin, umax=umax)
    try:
        xs_edmd_cl, us_edmd_cl = quadrotor.simulate(x0_exp, controller_edmd_cl, t_eval_cl)
        xs_edmd_cl, us_edmd_cl = xs_edmd_cl.T, us_edmd_cl.T
        edmd_cl_runtime_mean = np.mean(controller_edmd_cl.nom_controller.comp_time)
        edmd_cl_runtime_std = np.std(controller_edmd_cl.nom_controller.comp_time)
    except:
        xs_edmd_cl, us_edmd_cl = np.nan, np.nan
        edmd_cl_runtime_mean, edmd_cl_runtime_std = np.nan, np.nan
        edmd_cl_success = False

    # Define and initialize controller based on bEDMD model:
    if bedmd_trajgen_success:
        z_init_bedmd = controller_bedmd.cur_z[:traj_length_cl+1,:]
        u_init_bedmd = controller_bedmd.cur_u[:traj_length_cl, :]
    else:
        z_init_bedmd = z_init[:traj_length_cl + 1, :]
        u_init_bedmd = u_init[:traj_length_cl, :]

    controller_bedmd_cl = BilinearMPCController(sys_bedmd, traj_length_cl, dt, umin, umax, xmin_trajgen, xmax_trajgen,
                                                Q_mpc_cl, R_mpc_cl, QN_mpc_cl, xf_exp, solver_settings, add_slack=add_slack_cl,
                                                q_slack=q_slack_cl, const_offset=ctrl_offset)
    controller_bedmd_cl.construct_controller(z_init_bedmd, u_init_bedmd)
    try:
        controller_bedmd_cl.solve_to_convergence(z0_exp, 0., controller_bedmd.cur_z[:traj_length_cl+1,:],
                                                 controller_bedmd.cur_u[:traj_length_cl,:], max_iter=max_iter_sqp)
        controller_bedmd_cl = PerturbedController(sys_bedmd,controller_bedmd_cl,0.,const_offset=hover_thrust,
                                                  umin=umin, umax=umax)
        controller_bedmd_cl.nom_controller.comp_time, controller_bedmd_cl.nom_controller.prep_time, \
        controller_bedmd_cl.nom_controller.qp_time,  = [], [], []
        controller_bedmd_cl.nom_controller.update_solver_settings(solver_settings_cl)
        xs_bedmd_cl, us_bedmd_cl = quadrotor.simulate(x0_exp, controller_bedmd_cl, t_eval_cl)
        xs_bedmd_cl, us_bedmd_cl = xs_bedmd_cl.T, us_bedmd_cl.T
        bedmd_cl_runtime_mean = np.mean(controller_bedmd_cl.nom_controller.comp_time)
        bedmd_cl_runtime_std = np.std(controller_bedmd_cl.nom_controller.comp_time)
    except:
        xs_bedmd_cl, us_bedmd_cl = np.nan, np.nan
        bedmd_cl_runtime_mean, bedmd_cl_runtime_std = np.nan, np.nan
        bedmd_cl_success = False

    # Define and initialize controller based on full model knowledge (benchmark):
    if nmpc_trajgen_success:
        x_init_nmpc = controller_nmpc.cur_z[:traj_length_cl+1,:]
        u_init_nmpc = controller_nmpc.cur_u[:traj_length_cl, :]
    else:
        x_init_nmpc = x_init[:traj_length_cl + 1, :]
        u_init_nmpc = u_init[:traj_length_cl, :]

    controller_nmpc_cl = NonlinearMPCController(quadrotor_d, traj_length_cl, dt, umin+hover_thrust, umax+hover_thrust,
                                                xmin_trajgen, xmax_trajgen, Q_mpc_cl, R_mpc_cl, QN_mpc_cl, xf_exp,
                                                solver_settings, add_slack=add_slack_cl, q_slack=q_slack_cl)
    controller_nmpc_cl.construct_controller(x_init_nmpc, u_init_nmpc)
    try:
        controller_nmpc_cl.solve_to_convergence(x0_exp, 0., x_init_nmpc, u_init_nmpc, max_iter=max_iter_sqp)
        controller_nmpc_cl.comp_time, controller_nmpc_cl.prep_time, controller_nmpc_cl.qp_time,  = [], [], []
        controller_nmpc_cl.update_solver_settings(solver_settings_cl)
        xs_nmpc_cl, us_nmpc_cl = quadrotor.simulate(x0_exp, controller_nmpc_cl, t_eval_cl)
        xs_nmpc_cl, us_nmpc_cl = xs_nmpc_cl.T, us_nmpc_cl.T
        nmpc_cl_runtime_mean = np.mean(controller_nmpc_cl.comp_time)
        nmpc_cl_runtime_std = np.std(controller_nmpc_cl.comp_time)
    except:
        xs_nmpc_cl, us_nmpc_cl = np.nan, np.nan
        nmpc_cl_runtime_mean, nmpc_cl_runtime_std = np.nan, np.nan
        nmpc_cl_success = False

    cl_success_lst = [dmd_cl_success, edmd_cl_success, bedmd_cl_success, nmpc_cl_success]
    xs_cl_lst = [xs_dmd_cl, xs_edmd_cl, xs_bedmd_cl, xs_nmpc_cl]
    us_cl_lst = [us_dmd_cl, us_edmd_cl, us_bedmd_cl, us_nmpc_cl]
    cl_comp_time_mean = [dmd_cl_runtime_mean, edmd_cl_runtime_mean, bedmd_cl_runtime_mean, nmpc_cl_runtime_mean]
    cl_comp_time_std = [dmd_cl_runtime_std, edmd_cl_runtime_std, bedmd_cl_runtime_std, nmpc_cl_runtime_std]

    for name, cl_success, xs, us, comp_time_mean, comp_time_std in \
            zip(model_names, cl_success_lst, xs_cl_lst, us_cl_lst, cl_comp_time_mean, cl_comp_time_std):
        if cl_success:
            cost_cl = np.sum(np.diag(
                (xs[:, :-1] - xf_exp.reshape(-1, 1)).T @ Q_mpc_cl @ (xs[:, :-1] - xf_exp.reshape(-1, 1)))) +\
                      (xs[:, -1] - xf_exp).T @ QN_mpc_cl @ (xs[:, -1] - xf_exp) + np.sum(np.diag(us.T @ R_mpc_cl @ us))
        else:
            cost_cl = np.nan

        sol_stats_exp[name]['cl_success'] = cl_success
        sol_stats_exp[name]['cost_cl'] = cost_cl
        sol_stats_exp[name]['cl_comp_time_mean'] = comp_time_mean
        sol_stats_exp[name]['cl_comp_time_std'] = comp_time_std

    sol_stats.append(sol_stats_exp)
    exp += 1
    """
    # TODO: Remove plotting once functinoality verified
    # Plot and analyze the results:
    plot_inds = [0, 1, 2, 0, 1]
    subplot_inds = [1, 2, 3, 4, 8]

    plt.figure(figsize=(12,2.5))
    for ii in range(5):
        ind = plot_inds[ii]
        if ii < 3:
            ax = plt.subplot(1,4,subplot_inds[ii])
            if dmd_cl_success:
                plt.plot(t_eval_cl, xs_dmd_cl[ind,:], colors[0], label='DMD MPC')
            if edmd_cl_success:
                plt.plot(t_eval_cl, xs_edmd_cl[ind, :], colors[1], label='EDMD MPC')
            if bedmd_cl_success:
                plt.plot(t_eval_cl, xs_bedmd_cl[ind, :], colors[2], label='K-NMPC')
            if nmpc_cl_success:
                plt.plot(t_eval_cl, xs_nmpc_cl[ind,:], colors[3], label='NMPC')

            plt.scatter(t_eval_cl[0], x0_exp[ind], color='g')
            plt.scatter(t_eval_cl[-1], xf_exp[ind], color='r')
            plt.ylabel(labels[ind])
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.title(titles[subplot_inds[ii]-1])
            plt.xlabel('Time (sec)')
            if subplot_inds[ii]==1:
                plt.legend(loc='upper left', frameon=False)
                #plt.ylim(-0.15,2)
        else:
            bx = plt.subplot(2,4,subplot_inds[ii])
            if dmd_cl_success:
                plt.plot(t_eval_cl[:-1],us_dmd_cl[ind,:], color=colors[0], label='DMD MPC')
            if edmd_cl_success:
                plt.plot(t_eval_cl[:-1], us_edmd_cl[ind, :], color=colors[1], label='EDMD MPC')
            if bedmd_cl_success:
                plt.plot(t_eval_cl[:-1], us_bedmd_cl[ind, :], color=colors[2], label='K-NMPC')
            if nmpc_cl_success:
                plt.plot(t_eval_cl[:-1],us_nmpc_cl[ind,:], color=colors[3], label='NMPC')
            plt.plot([0, t_eval_cl[-1]], [umax[ind]+hover_thrust, umax[ind]+hover_thrust], ':k')
            plt.plot([0, t_eval_cl[-1]], [umin[ind]+hover_thrust, umin[ind]+hover_thrust], ':k')
            plt.ylabel(labels[ii+3])
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if subplot_inds[ii] == 4:
                plt.title('Control inputs')
            else:
                plt.xlabel('Time (sec)')

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.tight_layout()
    """
# =================================================== CLOSED LOOP  ==================================================:
dmd_stats, edmd_stats, bedmd_stats, nmpc_stats = {}, {}, {}, {}
stat_fields = sol_stats[0]['dmd'].keys()
normalize_fields = ['cost_traj', 'ctrl_traj', 'cost_cl']
normalize_method = 'nmpc'
stats_lst = [dmd_stats, edmd_stats, bedmd_stats, nmpc_stats]

for method_stats in stats_lst:
    for field in stat_fields:
        method_stats[field] = []

for stats in sol_stats:
    for name, method_stats in zip(model_names, stats_lst):
        for field in stat_fields:
            if field in normalize_fields:
                if field[-4:] == 'traj' and stats[normalize_method]['trajgen_success']:
                    method_stats[field].append(stats[name][field]/stats[normalize_method][field])
                elif field[-2:] == 'cl' and stats[normalize_method]['cl_success']:
                    method_stats[field].append(stats[name][field] / stats[normalize_method][field])
                else:
                    method_stats[field].append(np.nan)
            else:
                method_stats[field].append(stats[name][field])

trajectory_stats_table = []
cl_stats_table = []
for name, method_stats in zip(model_names, stats_lst):
    trajectory_stats_table.append([name,
                                   '{:.4f}'.format(100*np.sum(method_stats['trajgen_success']) / n_exp),
                                   '{:.4f}'.format(np.nanmean(method_stats['ctrl_traj'])),
                                   '{:.4f}'.format(np.nanstd(method_stats['ctrl_traj'])),
                                   '{:.4f}'.format(np.nanmean(method_stats['xf_dist_traj'])),
                                   '{:.4f}'.format(np.nanstd(method_stats['xf_dist_traj'])),
                                   '{:.4f}'.format(np.nanmean(method_stats['n_iter_sqp'])),
                                   '{:.4f}'.format(np.nanstd(method_stats['n_iter_sqp']))
                                   ])

    cl_stats_table.append([name,
                           '{:.4f}'.format(100*np.sum(method_stats['cl_success'])/n_exp),
                           '{:.4f}'.format(np.nanmean(method_stats['cost_cl'])),
                           '{:.4f}'.format(np.nanstd(method_stats['cost_cl'])),
                           '{:.4f}'.format(np.nanmean(method_stats['cl_comp_time_mean'])),
                           '{:.4f}'.format(np.nanmean(method_stats['cl_comp_time_std']))
                           ])

print(tabulate(trajectory_stats_table, headers=['% of initial conditions solved', 'avg ctrl effort', 'std ctrl effort',
                                                'avg terminal error', 'std terminal error',
                                                'avg sqp iter', 'std sqp iter']))
print(tabulate(cl_stats_table, headers=['% of initial conditions solved', 'avg realized cost', 'std realized cost',
                                        'avg comp time', 'avg std comp time']))

plt.show()