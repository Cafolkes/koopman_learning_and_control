import numpy as np
import scipy as sc
import random as rand
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import dill

from core.controllers import PDController
from core.dynamics import LinearSystemDynamics, ConfigurationDynamics

from koopman_core.controllers import OpenLoopController, MPCController,BilinearFBLinController, PerturbedController, LinearLiftedController
from koopman_core.dynamics import LinearLiftedDynamics, BilinearLiftedDynamics
from koopman_core.learning import Edmd, BilinearEdmd
from koopman_core.basis_functions import PolySineBasis
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

class QuadrotorTrajectoryOutput(ConfigurationDynamics):
    def __init__(self, bilinear_dynamics, y_d, y_d_dot, y_d_ddot, dt, z_d=None, z_d_dot=None, z_d_ddot=None, C_h=None):
        ConfigurationDynamics.__init__(self, bilinear_dynamics, 2)
        self.bilinear_dynamics = bilinear_dynamics
        self.ref = y_d
        self.ref_dot = y_d_dot
        self.ref_ddot = y_d_ddot

        self.ref_z = z_d
        self.ref_dot_z = z_d_dot
        self.ref_ddot_z = z_d_ddot

        self.C_h = C_h
        self.dt = dt
        self.t_d = self.dt * np.arange(0, self.ref.shape[1])

    def eval_z(self, x, t):
        z = self.bilinear_dynamics.phi_fun(x.reshape(1,-1)).squeeze()
        return z - self.z_d(t)

    def y(self, q):
        return q

    def dydq(self, q):
        return np.array([[1, 0], [0, 1]])

    def d2ydq2(self, q):
        return np.zeros((1, 2, 2))

    def y_d(self, t):
        return self.interpolate_ref_(self.ref, t)

    def y_d_dot(self, t):
        return self.interpolate_ref_(self.ref_dot, t)

    def y_d_ddot(self, t):
        return self.interpolate_ref_(self.ref_ddot, t)

    def z_d(self, t):
        return self.interpolate_ref_(self.ref_z, t)

    def z_d_dot(self, t):
        return self.interpolate_ref_(self.ref_dot_z, t)

    def z_d_ddot(self, t):
        return self.interpolate_ref_(self.ref_ddot_z, t)

    def interpolate_ref_(self, ref, t):
        return np.array([np.interp(t, self.t_d, ref[ii, :]) for ii in range(ref.shape[0])])

#================================================== DEFINE PARAMETERS ==================================================
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
q_dc, r_dc = 1e2, 1                                                 # State and actuation penalty values, data collection
Q_dc = q_dc * np.identity(n)                                        # State penalty matrix, data collection
R_dc = r_dc*np.identity(m)                                          # Actuation penalty matrix, data collection
P_dc = sc.linalg.solve_continuous_are(A_nom, B_nom, Q_dc, R_dc)     # Algebraic Ricatti equation solution, data collection
K_dc = np.linalg.inv(R_dc)@B_nom.T@P_dc                             # LQR feedback gain matrix, data collection
K_dc_p = K_dc[:,:int(n/2)]                                          # Proportional control gains, data collection
K_dc_d = K_dc[:,int(n/2):]                                          # Derivative control gains, data collection
nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)

# Data collection parameters:
dt = 1.0e-2                                                         # Time step length
traj_length_dc = 2.                                                 # Trajectory length, data collection
n_pred_dc = int(traj_length_dc/dt)                                  # Number of time steps, data collection
t_eval = dt * np.arange(n_pred_dc + 1)                              # Simulation time points
n_traj_dc = 50                                                      # Number of trajectories to execute, data collection
noise_var = 1.                                                     # Exploration noise to perturb controller, data collection

xmax = np.array([2, 2, np.pi/3, 2.,2.,2.])                          # State constraints, trajectory generation
xmin = -xmax
umax = np.array([50., 50.]) - hover_thrust                          # Actuation constraint, trajectory generation
umin = np.array([0., 0.]) - hover_thrust
x0_max = np.array([xmax[0], xmax[1], xmax[2], 1., 1., 1.])          # Initial value limits
Q_trajgen = sc.sparse.diags([0,0,0,0,0,0])                          # State penalty matrix, trajectory generation
QN_trajgen = sc.sparse.diags([5e1,5e1,5e1,1e1,1e1,1e1])             # Final state penalty matrix, trajectory generation
R_trajgen = sc.sparse.eye(m)                                        # Actuation penalty matrix, trajectory generation
sub_sample_rate = 1                                                 # Rate to subsample data for training
model_fname = 'examples/planar_quad_models_sim'                     # Path to save learned models
n_cols = 10                                                         # Numb  er of columns in training data plot
learn_models = False                                                # Learn models (True), load models (False)

#DMD parameters:
alpha_dmd = 4.9e-3                                                    # Regularization strength (LASSO) DMD
tune_mdl_dmd = False

#EDMD parameters:
alpha_edmd = 1.4e-1                                                   # Regularization strength (LASSO) EDMD
tune_mdl_edmd = False

#Bilinear EDMD parameters:
alpha_bedmd = 2e-2                                                  # Regularization strength (LASSO) bEDMD
tune_mdl_bedmd = False

# Prediction performance evaluation parameters:
folder_plots = 'examples/figures/'                                  # Path to save plots
n_traj_ol = 100                                                      # Number of trajectories to execute, open loop

#Closed loop performance evaluation parameters:
x0_cl = np.array([-1.75, 0., 0., 0., 0., 0.])                       # Initial value, closed loop trajectory
set_pt_cl = np.array([1.75, 1., 0., 0., 0., 0.])                    # Desired final value, closed loop trajectory
t_eval_cl = dt * np.arange(201)                                     # Simulation time points, closed loop
Q_trajgen_cl = sc.sparse.diags([0,0,0,0,0,0])                       # State penalty matrix, trajectory generation
QN_trajgen_cl = sc.sparse.diags([1e2,1e2,1e2,1e1,1e1,1e1])          # Final state penalty matrix, trajectory generation
R_trajgen_cl = sc.sparse.eye(m)                                     # Actuation penalty matrix, trajectory generation
mpc_trajgen_cl = MPCController(nominal_sys,t_eval_cl.size,dt,umin,umax,xmin,xmax,QN_trajgen_cl,R_trajgen_cl,QN_trajgen_cl,set_pt_cl)
q_cl, r_cl = 10, 1                                                  # State and actuation penalty values, closed loop
output_inds = np.array([1, 2])                                      # Output states, feedback linearizing controller

#================================================== COLLECT DATA ==================================================

if learn_models:
    xd = np.empty((n_traj_dc, n_pred_dc + 1, n))
    xs = np.empty((n_traj_dc, n_pred_dc + 1, n))
    us = np.empty((n_traj_dc, n_pred_dc, m))

    plt.figure(figsize=(12,12*n_traj_dc/(n_cols**2)))
    for ii in range(n_traj_dc):
        x0 = np.asarray([rand.uniform(l,u) for l, u in zip(-x0_max, x0_max)])
        set_pt_dc = np.asarray([rand.uniform(l,u) for l, u in zip(-x0_max, x0_max)])
        mpc_trajgen = MPCController(nominal_sys, n_pred_dc, dt, umin, umax, xmin, xmax, QN_trajgen, R_trajgen,
                                    QN_trajgen, set_pt_dc)
        mpc_trajgen.eval(x0, 0)
        xd[ii, :, :] = mpc_trajgen.parse_result().T
        while abs(x0[0]-set_pt_dc[0]) < 1 or np.any(np.isnan(xd[ii,:,:])):
            x0 = np.asarray([rand.uniform(l,u) for l, u in zip(-x0_max, x0_max)])
            set_pt_dc = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
            mpc_trajgen = MPCController(nominal_sys, n_pred_dc, dt, umin, umax, xmin, xmax, QN_trajgen, R_trajgen,
                                        QN_trajgen, set_pt_dc)
            mpc_trajgen.eval(x0, 0)
            xd[ii, :, :] = mpc_trajgen.parse_result().T

        output = QuadrotorPdOutput(quadrotor, xd[ii,:,:], t_eval, n, m)
        pd_controller = PDController(output, K_dc_p, K_dc_d)
        perturbed_pd_controller = PerturbedController(quadrotor, pd_controller, noise_var, const_offset=hover_thrust)
        xs[ii,:,:], us[ii,:,:] = quadrotor.simulate(x0, perturbed_pd_controller, t_eval)

        plt.subplot(int(np.ceil(n_traj_dc/n_cols)),n_cols,ii+1)
        plt.plot(t_eval, xs[ii,:,0], 'b', label='$y$')
        plt.plot(t_eval, xs[ii, :, 1], 'g', label='$z$')
        plt.plot(t_eval, xs[ii,:,2], 'r', label='$\\theta$')
        plt.plot(t_eval, xd[ii,:,0], '--b', label='$y_d$')
        plt.plot(t_eval, xd[ii, :, 1], '--g', label='$z_d$')
        plt.plot(t_eval, xd[ii,:,2], '--r', label='$\\theta_d$')
    plt.suptitle('Training data \nx-axis: time (sec), y-axis: state value, $x$ - blue, $xd$ - dotted blue, $\\theta$ - red, $\\theta_d$ - dotted red',y=0.94)
    plt.show()

    # ================================================== LEARN MODELS ==================================================

    #DMD:
    basis = lambda x: x
    C_dmd = np.eye(n)

    optimizer_dmd = linear_model.MultiTaskLasso(alpha=alpha_dmd, fit_intercept=False, selection='random')
    cv_dmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
    standardizer_dmd = preprocessing.StandardScaler(with_mean=False)

    model_dmd = Edmd(n, m, basis, n, n_traj_dc, optimizer_dmd, cv=cv_dmd, standardizer=standardizer_dmd, C=C_dmd, first_obs_const=False)
    xdmd, y_dmd = model_dmd.process(xs, us-hover_thrust, np.tile(t_eval,(n_traj_dc,1)), downsample_rate=sub_sample_rate)
    model_dmd.fit(xdmd, y_dmd, cv=tune_mdl_dmd, override_kinematics=True)
    sys_dmd = LinearLiftedDynamics(model_dmd.A, model_dmd.B, model_dmd.C, model_dmd.basis)
    if tune_mdl_dmd:
        print('$\\alpha$ DMD: ',model_dmd.cv.alpha_)

    #EDMD:
    basis = PolySineBasis(n, poly_deg=2, cross_terms=False)
    basis.construct_basis()
    poly_sine_features = preprocessing.FunctionTransformer(basis.basis)
    poly_sine_features.fit(np.zeros((1,n)))
    n_lift_edmd = poly_sine_features.transform((np.zeros((1,n)))).shape[1]
    C_edmd = np.zeros((n,n_lift_edmd))
    C_edmd[:,1:n+1] = np.eye(n)

    optimizer_edmd = linear_model.MultiTaskLasso(alpha=alpha_edmd, fit_intercept=False, selection='random')
    cv_edmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
    standardizer_edmd = preprocessing.StandardScaler(with_mean=False)

    model_edmd = Edmd(n, m, basis.basis, n_lift_edmd, n_traj_dc, optimizer_edmd, cv=cv_edmd, standardizer=standardizer_edmd, C=C_edmd)
    X_edmd, y_edmd = model_edmd.process(xs, us-hover_thrust, np.tile(t_eval,(n_traj_dc,1)), downsample_rate=sub_sample_rate)
    model_edmd.fit(X_edmd, y_edmd, cv=tune_mdl_edmd, override_kinematics=True)
    model_edmd.reduce_mdl()
    sys_edmd = LinearLiftedDynamics(model_edmd.A, model_edmd.B, model_edmd.C, model_edmd.basis_reduced)
    if tune_mdl_edmd:
        print('$\\alpha$ EDMD: ',model_edmd.cv.alpha_)

    #Bilinear EDMD:
    n_lift_bedmd = n_lift_edmd
    C_bedmd = np.zeros((n,n_lift_bedmd))
    C_bedmd[:,1:n+1] = np.eye(n)

    basis_bedmd = lambda x: poly_sine_features.transform(x)
    optimizer_bedmd = linear_model.MultiTaskLasso(alpha=alpha_bedmd, fit_intercept=False, selection='random')
    cv_bedmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
    standardizer_bedmd = preprocessing.StandardScaler(with_mean=False)

    model_bedmd = BilinearEdmd(n, m, basis_bedmd, n_lift_bedmd, n_traj_dc, optimizer_bedmd, cv=cv_bedmd, standardizer=standardizer_bedmd, C=C_bedmd)
    X_bedmd, y_bedmd = model_bedmd.process(xs, us-hover_thrust, np.tile(t_eval,(n_traj_dc,1)), downsample_rate=sub_sample_rate)
    model_bedmd.fit(X_bedmd, y_bedmd, cv=tune_mdl_bedmd, override_kinematics=True)
    sys_bedmd = BilinearLiftedDynamics(model_bedmd.n_lift, m, model_bedmd.A, model_bedmd.B, model_bedmd.C, model_bedmd.basis)
    if tune_mdl_bedmd:
        print('$\\alpha$ bilinear EDMD: ', model_bedmd.cv.alpha_)

    #Save models:
    model_dict = {'sys_dmd': sys_dmd,'model_dmd': model_dmd,'sys_edmd': sys_edmd,'model_edmd': model_edmd, 'sys_bedmd': sys_bedmd,'model_bedmd': model_bedmd}
    with open(model_fname, 'wb') as handle:
        dill.dump(model_dict, handle)
else:
    with open(model_fname, 'rb') as handle:
        p = dill.load(handle)
    sys_dmd = p['sys_dmd']
    model_dmd = p['model_dmd']
    sys_edmd = p['sys_edmd']
    model_edmd = p['model_edmd']
    sys_bedmd = p['sys_bedmd']
    model_bedmd = p['model_bedmd']

    basis = PolySineBasis(n, poly_deg=2, cross_terms=False)
    basis.construct_basis()
    poly_sine_features = preprocessing.FunctionTransformer(basis.basis)
    poly_sine_features.fit(np.zeros((1, n)))

#=========================================== EVALUATE OPEN LOOP PERFORMANCE ===========================================

#Compare open loop performance:
xs_ol = np.empty((n_traj_ol, t_eval.shape[0], n))
xs_dmd_ol = np.empty((n_traj_ol, t_eval.shape[0]-1, n))
xs_edmd_ol = np.empty((n_traj_ol, t_eval.shape[0]-1, n))
xs_bedmd_ol = np.empty((n_traj_ol, t_eval.shape[0]-1, n))
us_test = np.empty((n_traj_ol, t_eval.shape[0]-1, m))

for ii in range(n_traj_ol):
    x0 = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
    set_pt_dc = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
    mpc_trajgen = MPCController(nominal_sys, n_pred_dc, dt, umin, umax, xmin, xmax, QN_trajgen, R_trajgen,
                                QN_trajgen, set_pt_dc)
    mpc_trajgen.eval(x0, 0)
    xd = mpc_trajgen.parse_result().T

    while xd[0,0] is None:
        x0 = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
        set_pt_dc = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
        mpc_trajgen = MPCController(nominal_sys, n_pred_dc, dt, umin, umax, xmin, xmax, QN_trajgen, R_trajgen,
                                    QN_trajgen, set_pt_dc)
        mpc_trajgen.eval(x0, 0)
        xd = mpc_trajgen.parse_result().T

    output = QuadrotorPdOutput(quadrotor, xd, t_eval, n, m)
    pd_controller = PDController(output, K_dc_p, K_dc_d)
    perturbed_pd_controller = PerturbedController(quadrotor, pd_controller, noise_var, const_offset=mass * gravity / 2)

    xs_ol[ii,:,:], us_test[ii,:,:] = quadrotor.simulate(x0, perturbed_pd_controller, t_eval)
    ol_controller_nom = OpenLoopController(sys_bedmd, us_test[ii,:,:]-hover_thrust, t_eval[:-1])

    xs_dmd_ol[ii,:,:], _ = sys_dmd.simulate(x0, ol_controller_nom, t_eval[:-1])

    z_0_edmd = sys_edmd.phi_fun(np.atleast_2d(x0)).squeeze()
    zs_edmd_tmp, _ = sys_edmd.simulate(z_0_edmd, ol_controller_nom, t_eval[:-1])
    xs_edmd_ol[ii,:,:] = np.dot(model_edmd.C, zs_edmd_tmp.T).T

    z_0_bedmd = sys_bedmd.phi_fun(np.atleast_2d(x0)).squeeze()
    zs_bedmd_tmp, _ = sys_bedmd.simulate(z_0_bedmd, ol_controller_nom, t_eval[:-1])
    xs_bedmd_ol[ii,:,:] = np.dot(model_bedmd.C, zs_bedmd_tmp.T).T

error_dmd = xs_ol[:,:-1,:] - xs_dmd_ol
error_dmd_mean = np.mean(error_dmd, axis=0).T
error_dmd_std = np.std(error_dmd, axis=0).T
mse_dmd = np.mean(np.mean(np.mean(np.square(error_dmd))))

error_edmd = xs_ol[:,:-1,:] - xs_edmd_ol
error_edmd_mean = np.mean(error_edmd, axis=0).T
error_edmd_std = np.std(error_edmd, axis=0).T
mse_edmd = np.mean(np.mean(np.mean(np.square(error_edmd))))

error_bedmd = xs_ol[:,:-1,:] - xs_bedmd_ol
error_bedmd_mean = np.mean(error_bedmd, axis=0).T
error_bedmd_std = np.std(error_bedmd, axis=0).T
mse_bedmd = np.mean(np.mean(np.mean(np.square(error_bedmd))))

print('\nOpen loop performance statistics:')
print('   MSE DMD:   ', "{:.3f}".format(mse_dmd),
      '\n   MSE EDMD:  ', "{:.3f}".format(mse_edmd),
      '\n   MSE bEDMD: ', "{:.3f}".format(mse_bedmd))
print('   Improvement DMD -> EDMD:   ', "{:.2f}".format((1 - mse_edmd / mse_dmd) * 100), ' %'
      '\n   Improvement DMD -> bEDMD:  ', "{:.2f}".format((1 - mse_bedmd / mse_dmd) * 100), ' %'
      '\n   Improvement EDMD -> bEDMD: ', "{:.2f}".format((1 - mse_bedmd / mse_edmd) * 100), ' %')

#========================================== EVALUATE CLOSED LOOP PERFORMANCE ==========================================

#Compare closed loop performance:
# Generate trajectory:
mpc_trajgen_cl.eval(x0_cl, 0)
xr_cl = mpc_trajgen_cl.parse_result()[:,:-1]
ur_cl = mpc_trajgen_cl.get_control_prediction()
xr_cl_dot = nominal_sys.eval_dot(xr_cl,ur_cl,0.)

# Define outputs:
y_d = xr_cl[output_inds,:]
y_d_dot = xr_cl[output_inds+int(n/2),:]
y_d_ddot = xr_cl_dot[output_inds+int(n/2),:]

# Design LQR controller for DMD model:
Q_dmd = q_cl*np.identity(n)
R_dmd = r_cl*np.identity(m)
P_dmd = sc.linalg.solve_continuous_are(model_dmd.A, model_dmd.B, Q_dmd, R_dmd)
K_dmd = np.linalg.inv(R_dmd)@model_dmd.B.T@P_dmd
K_dmd_p, K_dmd_d = K_dmd[:,:int(n/2)], K_dmd[:,int(n/2):]
output_dmd = QuadrotorPdOutput(sys_dmd, xr_cl.T, t_eval_cl, n, m)
controller_dmd = PDController(output_dmd, K_dmd_p, K_dmd_d)
controller_dmd = PerturbedController(sys_dmd,controller_dmd,0.,const_offset=hover_thrust, umin=umin, umax=umax)

# Design LQR controller for EDMD model:
z_d_edmd = np.array([sys_edmd.phi_fun(x.reshape(1,-1)).squeeze() for x in xr_cl.T]).T
z_d_dot_edmd = differentiate_vec(z_d_edmd.T, t_eval_cl).T
z_d_ddot_edmd = differentiate_vec(z_d_dot_edmd.T, t_eval_cl).T
output_edmd = QuadrotorTrajectoryOutput(sys_edmd, y_d, y_d_dot, y_d_ddot, dt, z_d_edmd, z_d_dot_edmd, z_d_ddot_edmd, model_edmd.C[output_inds,:])

Q_edmd = q_cl*np.identity(sys_edmd.n)
R_edmd = r_cl*np.identity(m)
P_edmd = sc.linalg.solve_continuous_are(sys_edmd.A, sys_edmd.B, Q_edmd, R_edmd)
K_edmd = np.linalg.inv(R_edmd)@model_edmd.B.T@P_edmd
controller_edmd = LinearLiftedController(output_edmd, K_edmd)
controller_edmd = PerturbedController(quadrotor, controller_edmd,0.,const_offset=hover_thrust, umin=umin, umax=umax)

# Design feedback linearizing controller for bilinear EDMD model:
k = m
n_lift_bedmd = sys_bedmd.n
Q_bedmd = q_cl*np.eye(int(2*n_lift_bedmd))
R_bedmd = r_cl*np.eye(n_lift_bedmd)
C_h = model_bedmd.C[output_inds,:]

z_d_bedmd = np.array([sys_bedmd.phi_fun(x.reshape(1,-1)).squeeze() for x in xr_cl.T]).T
z_d_dot_bedmd = differentiate_vec(z_d_bedmd.T, t_eval_cl).T
z_d_ddot_bedmd = differentiate_vec(z_d_dot_bedmd.T, t_eval_cl).T
output_bedmd = QuadrotorTrajectoryOutput(sys_bedmd, y_d, y_d_dot, y_d_ddot, dt, z_d_bedmd, z_d_dot_bedmd, z_d_ddot_bedmd, C_h)

f_eta = np.concatenate((np.zeros((n_lift_bedmd,n_lift_bedmd)), np.eye(n_lift_bedmd)), axis=1)
f_eta_dot = np.concatenate((sys_bedmd.F@sys_bedmd.F, np.zeros((n_lift_bedmd,n_lift_bedmd))), axis=1)
F_lin = np.concatenate((f_eta, f_eta_dot), axis=0)
G_lin = np.concatenate((np.zeros((n_lift_bedmd,n_lift_bedmd)), np.eye(n_lift_bedmd)), axis=0)

P_bedmd = sc.linalg.solve_continuous_are(F_lin, G_lin, Q_bedmd, R_bedmd)
K_bedmd = np.linalg.inv(R_bedmd)@G_lin.T@P_bedmd
controller_bedmd = BilinearFBLinController(sys_bedmd, output_bedmd, K_bedmd)
controller_bedmd = PerturbedController(sys_bedmd, controller_bedmd,0.,const_offset=hover_thrust, umin=umin, umax=umax)

# Simulate the system under closed loop control:
xs_cl_dmd, us_cl_dmd = quadrotor.simulate(x0_cl, controller_dmd, t_eval_cl)
xs_cl_edmd, us_cl_edmd = quadrotor.simulate(x0_cl, controller_edmd, t_eval_cl)
xs_cl_bedmd, us_cl_bedmd = quadrotor.simulate(x0_cl, controller_bedmd, t_eval_cl)

#hover_cost = 2*hover_thrust**2*t_eval_cl.size
hover_cost = 0.
mse_cl_dmd = np.linalg.norm(xs_cl_dmd[1:,output_inds]-xr_cl[output_inds,1:].T, ord='fro')**2
mse_cl_edmd = np.linalg.norm(xs_cl_edmd[1:,output_inds]-xr_cl[output_inds,1:].T, ord='fro')**2
mse_cl_bedmd = np.linalg.norm(xs_cl_bedmd[1:,output_inds]-xr_cl[output_inds,1:].T, ord='fro')**2
ctrl_cost_dmd = np.linalg.norm(us_cl_dmd, ord='fro')**2
ctrl_cost_edmd = np.linalg.norm(us_cl_edmd, ord='fro')**2
ctrl_cost_bedmd = np.linalg.norm(us_cl_bedmd, ord='fro')**2

print('\nClosed loop performance statistics:')
print(' -Tracking error:')
print('   Tracking MSE DMD:   ', "{:.3f}".format(mse_cl_dmd),
      '\n   Tracking MSE EDMD:  ', "{:.3f}".format(mse_cl_edmd),
      '\n   Tracking MSE bEDMD: ', "{:.3f}".format(mse_cl_bedmd))
print('   Improvement DMD -> EDMD:   ', "{:.2f}".format(100*(1-(mse_cl_edmd)/(mse_cl_dmd))), ' %'
      '\n   Improvement DMD -> bEDMD:  ', "{:.2f}".format(100*(1-(mse_cl_bedmd)/(mse_cl_dmd))), ' %'
      '\n   Improvement EDMD -> bEDMD: ', "{:.2f}".format(100*(1-(mse_cl_bedmd)/(mse_cl_edmd))), ' %')
print(' -Control effort:')
print('   Control effort DMD:   ', "{:.3f}".format(ctrl_cost_dmd-hover_cost),
      '\n   Control effort EDMD:  ', "{:.3f}".format(ctrl_cost_edmd-hover_cost),
      '\n   Control effort bEDMD: ', "{:.3f}".format(ctrl_cost_bedmd-hover_cost))
print('   Improvement DMD -> EDMD:   ', "{:.2f}".format(100*(1-(ctrl_cost_edmd-hover_cost)/(ctrl_cost_dmd-hover_cost))), ' %'
      '\n   Improvement DMD -> bEDMD:  ', "{:.2f}".format(100*(1-(ctrl_cost_bedmd-hover_cost)/(ctrl_cost_dmd-hover_cost))), ' %'
      '\n   Improvement EDMD -> bEDMD: ', "{:.2f}".format(100*(1-(ctrl_cost_bedmd-hover_cost)/(ctrl_cost_edmd-hover_cost))), ' %')


def plot_paper(folder_name, show_plots=False):
    import matplotlib
    figwidth = 12
    lw = 2
    fs = 16
    y_lim_gain = 1.2

    #Plot open loop results:
    ylabels = ['$e_{y}$', '$e_z$', '$e_{\\theta}$']
    plt.figure(figsize=(figwidth,3))
    for ii in range(3):
        plt.subplot(1,3,ii+1)
        plt.plot(t_eval[:-1], error_dmd_mean[ii,:], linewidth=lw, label='DMD')
        plt.fill_between(t_eval[:-1], error_dmd_mean[ii,:] - error_dmd_std[ii,:], error_dmd_mean[ii,:] + error_dmd_std[ii,:], alpha=0.2)
        plt.plot(t_eval[:-1], error_edmd_mean[ii, :], linewidth=lw, label='EDMD')
        plt.fill_between(t_eval[:-1], error_edmd_mean[ii, :] - error_edmd_std[ii, :],error_edmd_mean[ii, :] + error_edmd_std[ii, :], alpha=0.2)
        plt.plot(t_eval[:-1], error_bedmd_mean[ii, :], linewidth=lw, label='bEDMD')
        plt.fill_between(t_eval[:-1], error_bedmd_mean[ii, :] - error_bedmd_std[ii, :],error_bedmd_mean[ii, :] + error_bedmd_std[ii, :], alpha=0.2)
        ylim = max(max(np.abs(error_bedmd_mean[ii, :] - error_bedmd_std[ii, :])), max(np.abs(error_bedmd_mean[ii, :] + error_bedmd_std[ii, :])))
        plt.ylim([-ylim * y_lim_gain, ylim * y_lim_gain])
        plt.xlabel('$t$ (sec)', fontsize=fs)
        plt.ylabel(ylabels[ii], fontsize=fs)
        plt.grid()

    plt.legend(loc='upper left', fontsize=fs-4)
    suptitle = plt.suptitle('Open loop prediction error of DMD, EDMD and bilinear EDMD models', y=1.05, fontsize=18)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.tight_layout()
    plt.savefig(folder_name + 'planar_quad_prediction.pdf', format='pdf', dpi=2400, bbox_extra_artists=(suptitle,), bbox_inches="tight")

    if show_plots:
        plt.show()

    #Plot closed loop results:
    x_index = output_inds[0]
    y_index = output_inds[1]
    plt.figure(figsize=(figwidth, 4))

    plt.subplot(2, 2, 1)
    plt.plot(t_eval_cl, xr_cl[x_index,:], '--r', linewidth=2, label='Reference')
    plt.plot(t_eval_cl,xs_cl_dmd[:, x_index], linewidth=lw, label='DMD')
    plt.plot(t_eval_cl,xs_cl_edmd[:, x_index], linewidth=lw, label='EDMD')
    plt.plot(t_eval_cl,xs_cl_bedmd[:, x_index], linewidth=lw, label='bEDMD')
    plt.ylabel('$z$', fontsize=fs)
    plt.title('Output states', fontsize=fs)
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(t_eval_cl, xr_cl[y_index, :], '--r', linewidth=2, label='Reference')
    plt.plot(t_eval_cl, xs_cl_dmd[:, y_index], linewidth=lw, label='DMD')
    plt.plot(t_eval_cl, xs_cl_edmd[:, y_index], linewidth=lw, label='EDMD')
    plt.plot(t_eval_cl, xs_cl_bedmd[:, y_index], linewidth=lw, label='bEDMD')
    plt.ylabel('$\\theta$', fontsize=fs)
    plt.xlabel('Time (sec)')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(t_eval_cl[:-1], us_cl_dmd[:, 0], linewidth=lw, label='DMD')
    plt.plot(t_eval_cl[:-1], us_cl_edmd[:, 0], linewidth=lw, label='EDMD')
    plt.plot(t_eval_cl[:-1], us_cl_bedmd[:, 0], linewidth=lw, label='bEDMD')
    plt.ylabel('$u_1$', fontsize=fs)
    plt.title('Control action', fontsize=fs)
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(t_eval_cl[:-1], us_cl_dmd[:, 1], linewidth=lw, label='DMD')
    plt.plot(t_eval_cl[:-1], us_cl_edmd[:, 1], linewidth=lw, label='EDMD')
    plt.plot(t_eval_cl[:-1], us_cl_bedmd[:, 1], linewidth=lw, label='bEDMD')
    plt.xlabel('Time (sec)', fontsize=fs)
    plt.ylabel('$u_2$', fontsize=fs)
    plt.grid()

    suptitle = plt.suptitle('Trajectory tracking based on DMD, EDMD and bilinear EDMD models', y=1.05,fontsize=18)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.tight_layout()
    plt.savefig(folder_name + 'planar_quad_closedloop.pdf', format='pdf', dpi=2400, bbox_extra_artists=(suptitle,),
                bbox_inches="tight")
    if show_plots:
        plt.show()

plot_paper(folder_plots, show_plots=True)

