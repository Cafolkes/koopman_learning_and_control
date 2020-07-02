import numpy as np
import scipy as sc
import random as rand
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt

from core.controllers import PDController, LQRController
from core.dynamics import LinearSystemDynamics, ConfigurationDynamics

from koopman_core.controllers import OpenLoopController, MPCController, LinearMpcController, MPCControllerDense, \
    BilinearMpcController, PerturbedController
from koopman_core.dynamics import LinearLiftedDynamics, BilinearLiftedDynamics
from koopman_core.learning import Edmd, BilinearEdmd, FlBilinearLearner
from koopman_core.basis_functions import PolySineBasis
from koopman_core.learning.utils import differentiate_vec
from koopman_core.systems import PlanarQuadrotorForceInput
import dill as pickle

class QuadrotorTrajectoryOutput(ConfigurationDynamics):
    def __init__(self, dynamics, x_d, t_d, n, m):
        ConfigurationDynamics.__init__(self, dynamics, 1)
        self.x_d = x_d
        self.t_d = t_d
        self.x_d_dot = differentiate_vec(self.x_d, self.t_d)
        self.n = n
        self.m = m

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

    def z_d(self, t):
        return self.robotic_dynamics.phi_fun(np.array([[0, 0, 0, 0, 0, 0]])).squeeze()

    def z_d_dot(self, t):
        return np.zeros(self.robotic_dynamics.n)

    def z_d_ddot(self, t):
        return np.zeros(self.robotic_dynamics.n)

    def desired_state_(self, t):
        return [np.interp(t, self.t_d.flatten(),self.x_d[:,ii].flatten()) for ii in range(self.x_d.shape[1])]

    def desired_state_dot_(self, t):
        return [np.interp(t, self.t_d.flatten(),self.x_d_dot[:,ii].flatten()) for ii in range(self.x_d_dot.shape[1])]

# Cart pole system parameters
mass = 2.
inertia = 1.
prop_arm = 0.2
gravity = 9.81
quadrotor = PlanarQuadrotorForceInput(mass, inertia, prop_arm, g=gravity)

# Linearized system specification:
n, m = 6, 2                                             # Number of states, number of control inputs
A_nom = np.array([[0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 1.],
                  [0., 0., -gravity, 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.]])                # Linearization of the true system around the origin
B_nom = np.array([[0., 0.],
                  [0., 0.],
                  [0., 0.],
                  [0., 0.],
                  [1./mass, 1./mass],
                  [-prop_arm/inertia, prop_arm/inertia]])                               # Linearization of the true system around the origin

hover_thrust = mass*gravity/m
q, r = 1e2, 1
Q = q * np.identity(n)
R = r*np.identity(m)
P = sc.linalg.solve_continuous_are(A_nom, B_nom, Q, R)
K = np.linalg.inv(R)@B_nom.T@P
K_p = K[:,:int(n/2)]                                            # Proportional control gains
K_d = K[:,int(n/2):]                                            # Derivative control gains
nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)

# Data collection parameters:
n_traj = 50                                             # Number of trajectories to collect data from
dt = 1.0e-2                                             # Time step length
mpc_horizon = 2.                                        # MPC time horizon [sec] (trajectory generation)
n_pred = int(mpc_horizon/dt)                            # Number of time steps
t_eval = dt * np.arange(n_pred + 1)                     # Simulation time points
noise_var = 5.                                          # Exploration noise to perturb controller
xmax = np.array([2, 2, np.pi/3, 2.,2.,2.])              # State constraints
xmin = -xmax
set_pt = np.zeros(n)                                    # Desired trajectories (initialization)
Q = sc.sparse.diags([0,0,0,0,0,0])                          # MPC state penalty matrix (trajectory generation)
QN = sc.sparse.diags([1e3,1e3,1e3,1e1,1e1,1e1])                 # MPC final state penalty matrix (trajectory generation)
R = sc.sparse.eye(m)                                    # MPC control penalty matrix (trajectory generation)
umax = np.array([np.inf, np.inf])                                   # MPC actuation constraint (trajectory generation)
umin = np.array([-mass*gravity/2., -mass*gravity/2])
sub_sample_rate = 5
model_fname = 'examples/planar_quad_models_sim'

#EDMD parameters:
alpha_edmd = 5e-1
tune_mdl_edmd = False

#Bilinear EDMD parameters:
alpha_bedmd = 1e-1
tune_mdl_bedmd = False

learn_models = False

# Prediction performance evaluation parameters:
folder_plots = 'examples/figures/'
n_traj_test = 50
mpc_controller = MPCController(nominal_sys,n_pred,dt,umin,umax,xmin,xmax,Q,R,QN,set_pt)
x_0_max = np.array([xmax[0], xmax[1], xmax[2], 1., 1., 1.])
x_d = np.empty((n_traj,n_pred+1,n))
xs = np.empty((n_traj,n_pred+1,n))
us = np.empty((n_traj, n_pred, m))

#Closed loop performance evaluation parameters:

if learn_models:
    n_cols = 10
    plt.figure(figsize=(12,12*n_traj/(n_cols**2)))
    for ii in range(n_traj):
        x_0 = np.asarray([rand.uniform(l,u) for l, u in zip(-x_0_max, x_0_max)])
        mpc_controller.eval(x_0, 0)
        x_d[ii, :, :] = mpc_controller.parse_result().T
        while abs(x_0[0]) < 1.25 or np.any(np.isnan(x_d[ii,:,:])):
            x_0 = np.asarray([rand.uniform(l,u) for l, u in zip(-x_0_max, x_0_max)])
            mpc_controller.eval(x_0, 0)
            x_d[ii, :, :] = mpc_controller.parse_result().T

        output = QuadrotorTrajectoryOutput(quadrotor, x_d[ii,:,:], t_eval, n, m)
        pd_controller = PDController(output, K_p, K_d)
        perturbed_pd_controller = PerturbedController(quadrotor, pd_controller, noise_var, const_offset=hover_thrust)
        xs[ii,:,:], us[ii,:,:] = quadrotor.simulate(x_0, perturbed_pd_controller, t_eval)

        plt.subplot(int(np.ceil(n_traj/n_cols)),n_cols,ii+1)
        plt.plot(t_eval, xs[ii,:,0], 'b', label='$y$')
        plt.plot(t_eval, xs[ii, :, 1], 'g', label='$z$')
        plt.plot(t_eval, xs[ii,:,2], 'r', label='$\\theta$')
        plt.plot(t_eval, x_d[ii,:,0], '--b', label='$y_d$')
        plt.plot(t_eval, x_d[ii, :, 1], '--g', label='$z_d$')
        plt.plot(t_eval, x_d[ii,:,2], '--r', label='$\\theta_d$')
    plt.suptitle('Training data \nx-axis: time (sec), y-axis: state value, $x$ - blue, $x_d$ - dotted blue, $\\theta$ - red, $\\theta_d$ - dotted red',y=0.94)
    plt.show()

    #EDMD:
    basis = PolySineBasis(n, poly_deg=2, cross_terms=False)
    basis.construct_basis()
    poly_sine_features = preprocessing.FunctionTransformer(basis.basis)
    poly_sine_features.fit(np.zeros((1,n)))
    n_lift_edmd = poly_sine_features.transform((np.zeros((1,n)))).shape[1]
    C_edmd = np.zeros((n,n_lift_edmd))
    C_edmd[:,1:n+1] = np.eye(n)

    from koopman_core.learning import LassoQp
    optimizer_edmd = linear_model.MultiTaskLasso(alpha=alpha_edmd, fit_intercept=False, selection='random')
    cv_edmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
    standardizer_edmd = preprocessing.StandardScaler(with_mean=False)

    model_edmd = Edmd(n, m, basis.basis, n_lift_edmd, n_traj, optimizer_edmd, cv=cv_edmd, standardizer=standardizer_edmd, C=C_edmd)
    X_edmd, y_edmd = model_edmd.process(xs, us-hover_thrust, np.tile(t_eval,(n_traj,1)), downsample_rate=sub_sample_rate)
    model_edmd.fit(X_edmd, y_edmd, cv=tune_mdl_edmd, override_kinematics=True)
    sys_edmd = LinearLiftedDynamics(model_edmd.A, model_edmd.B, model_edmd.C, basis.basis)
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

    model_bedmd = BilinearEdmd(n, m, basis_bedmd, n_lift_bedmd, n_traj, optimizer_bedmd, cv=cv_bedmd, standardizer=standardizer_bedmd, C=C_bedmd)
    X_bedmd, y_bedmd = model_bedmd.process(xs, us-hover_thrust, np.tile(t_eval,(n_traj,1)), downsample_rate=sub_sample_rate)
    model_bedmd.fit(X_bedmd, y_bedmd, cv=tune_mdl_bedmd, override_kinematics=True)
    #model_bedmd.reduce_mdl()
    sys_bedmd = BilinearLiftedDynamics(model_bedmd.n_lift, m, model_bedmd.A, model_bedmd.B, model_bedmd.C, model_bedmd.basis)
    if tune_mdl_bedmd:
        print('$\\alpha$ bilinear EDMD: ', model_bedmd.cv.alpha_)

    #Save models:
    model_dict = {'sys_edmd': sys_edmd,'model_edmd': model_edmd, 'sys_bedmd': sys_bedmd,'model_bedmd': model_bedmd}
    with open(model_fname, 'wb') as handle:
        pickle.dump(model_dict, handle)
else:
    with open(model_fname, 'rb') as handle:
        p = pickle.load(handle)
    sys_edmd = p['sys_edmd']
    model_edmd = p['model_edmd']
    sys_bedmd = p['sys_bedmd']
    model_bedmd = p['model_bedmd']

    basis = PolySineBasis(n, poly_deg=2, cross_terms=False)
    basis.construct_basis()
    poly_sine_features = preprocessing.FunctionTransformer(basis.basis)
    poly_sine_features.fit(np.zeros((1, n)))
    basis_edmd = lambda x: poly_sine_features.transform(x)
    basis_bedmd = lambda x: poly_sine_features.transform(x)

#Compare open loop performance:
xs_test = np.empty((n_traj_test, t_eval.shape[0], n))
xs_nom_test = np.empty((n_traj_test, t_eval.shape[0]-1, n))
xs_edmd_test = np.empty((n_traj_test, t_eval.shape[0]-1, n))
xs_bedmd_test = np.empty((n_traj_test, t_eval.shape[0]-1, n))
us_test = np.empty((n_traj_test, t_eval.shape[0]-1, m))

for ii in range(n_traj_test):
    x_0 = np.asarray([rand.uniform(l, u) for l, u in zip(-x_0_max, x_0_max)])
    mpc_controller.eval(x_0, 0)
    x_d = mpc_controller.parse_result().T

    while x_d[0,0] is None:
        x_0 = np.asarray([rand.uniform(l, u) for l, u in zip(-x_0_max, x_0_max)])
        mpc_controller.eval(x_0, 0)
        x_d = mpc_controller.parse_result().T

    output = QuadrotorTrajectoryOutput(quadrotor, x_d, t_eval, n, m)
    pd_controller = PDController(output, K_p, K_d)
    perturbed_pd_controller = PerturbedController(quadrotor, pd_controller, noise_var, const_offset=mass * gravity / 2)

    xs_test[ii,:,:], us_test[ii,:,:] = quadrotor.simulate(x_0, perturbed_pd_controller, t_eval)
    ol_controller_nom = OpenLoopController(sys_bedmd, us_test[ii,:,:]-hover_thrust, t_eval[:-1])

    xs_nom_test[ii,:,:], _ = nominal_sys.simulate(x_0, ol_controller_nom, t_eval[:-1])

    z_0_edmd = basis.basis(np.atleast_2d(x_0)).squeeze()
    zs_edmd_tmp, _ = sys_edmd.simulate(z_0_edmd, ol_controller_nom, t_eval[:-1])
    xs_edmd_test[ii,:,:] = np.dot(model_edmd.C, zs_edmd_tmp.T).T

    z_0_bedmd = model_bedmd.basis(np.atleast_2d(x_0)).squeeze()
    zs_bedmd_tmp, _ = sys_bedmd.simulate(z_0_bedmd, ol_controller_nom, t_eval[:-1])
    xs_bedmd_test[ii,:,:] = np.dot(model_bedmd.C, zs_bedmd_tmp.T).T

error_nom = xs_test[:,:-1,:] - xs_nom_test
error_nom_mean = np.mean(error_nom, axis=0).T
error_nom_std = np.std(error_nom, axis=0).T
mse_nom = np.mean(np.mean(np.mean(np.square(error_nom))))

error_edmd = xs_test[:,:-1,:] - xs_edmd_test
error_edmd_mean = np.mean(error_edmd, axis=0).T
error_edmd_std = np.std(error_edmd, axis=0).T
mse_edmd = np.mean(np.mean(np.mean(np.square(error_edmd))))

error_bedmd = xs_test[:,:-1,:] - xs_bedmd_test
error_bedmd_mean = np.mean(error_bedmd, axis=0).T
error_bedmd_std = np.std(error_bedmd, axis=0).T
mse_bedmd = np.mean(np.mean(np.mean(np.square(error_bedmd))))

#Compare closed loop performance:
# MPC parameters:
x_0 = np.array([-1.75, 0., 0., 0., 0., 0.])
t_eval_cl = dt * np.arange(201)
umax = np.array([15, 15]) - hover_thrust
umin = np.array([0., 0.]) - hover_thrust
q, r = 1e3, 1
Q = q*np.identity(n)
R = r*np.identity(m)
n_pred = 100

set_pt = np.array([1.75, 1., 0., 0., 0., 0.])
mpc_controller_trajgen = MPCController(nominal_sys,t_eval_cl.size,dt,umin,umax,xmin,xmax,np.zeros((n,n)),R,QN,set_pt)
mpc_controller_trajgen.eval(x_0, 0)
xr = mpc_controller_trajgen.parse_result()[:,:-1]

# Design MPC for linearized nominal model:
controller_nom = MPCController(nominal_sys,n_pred,dt,umin,umax,xmin,xmax,Q,R,Q,xr)
controller_nom = PerturbedController(quadrotor,controller_nom,0.,const_offset=hover_thrust)

# Design MPC for lifted linear EDMD model:
controller_edmd = MPCController(sys_edmd, n_pred, dt, umin, umax, xmin, xmax, Q, R, Q, xr, lifting=True,
                                     edmd_object=model_edmd, plotMPC=False)
controller_edmd = PerturbedController(quadrotor, controller_edmd,0.,const_offset=hover_thrust)

# Design MPC for lifted bilinear EDMD model:
k = m
n_lift_bedmd = model_bedmd.n_lift
Q_fl = q*np.eye(int(2*n_lift_bedmd))
R_fl = r*np.eye(n_lift_bedmd)
C_h = model_bedmd.C[1:k+1,:]

f_eta = np.concatenate((np.zeros((n_lift_bedmd,n_lift_bedmd)), np.eye(n_lift_bedmd)), axis=1)
f_eta_dot = np.concatenate((sys_bedmd.F@sys_bedmd.F, np.zeros((n_lift_bedmd,n_lift_bedmd))), axis=1)
F_lin = np.concatenate((f_eta, f_eta_dot), axis=0)
G_lin = np.concatenate((np.zeros((n_lift_bedmd,n_lift_bedmd)), np.eye(n_lift_bedmd)), axis=0)
fb_sys = sc.signal.StateSpace(F_lin, G_lin, np.eye(int(2*n_lift_bedmd)), np.zeros((int(2*n_lift_bedmd),n_lift_bedmd)))
fb_sys_d = fb_sys.to_discrete(dt)

controller_bedmd = BilinearMpcController(n, m, k, n_lift_bedmd, n_pred, fb_sys_d, sys_bedmd, model_bedmd.C, C_h, xmin,
                                         xmax, umin, umax,
                                         Q_fl, Q_fl, R_fl, xr, t_eval_cl, const_offset=hover_thrust)

xs_mpc_nom, us_mpc_nom = quadrotor.simulate(x_0, controller_nom, t_eval_cl)
xs_mpc_edmd, us_mpc_edmd = quadrotor.simulate(x_0, controller_edmd, t_eval_cl)
xs_mpc_bedmd, us_mpc_bedmd = quadrotor.simulate(x_0, controller_bedmd, t_eval_cl)

outputs = np.array([1,2])
Q_cost = q*np.eye(outputs.size)

cost_nom = np.cumsum(np.diag((xs_mpc_nom[1:,outputs]-xr[outputs,1:].T) @ Q_cost @ (xs_mpc_nom[1:,outputs]-xr[outputs,1:].T).T) + np.diag((us_mpc_nom) @ R @ (us_mpc_nom).T))
cost_edmd = np.cumsum(np.diag((xs_mpc_edmd[1:,outputs]-xr[outputs,1:].T) @ Q_cost @ (xs_mpc_edmd[1:,outputs]-xr[outputs,1:].T).T) + np.diag((us_mpc_edmd) @ R @ (us_mpc_edmd).T))
cost_bedmd = np.cumsum(np.diag((xs_mpc_bedmd[1:,outputs]-xr[outputs,1:].T) @ Q_cost @ (xs_mpc_bedmd[1:,outputs]-xr[outputs,1:].T).T) + np.diag((us_mpc_bedmd) @ R @ (us_mpc_bedmd).T))


def plot_paper(folder_name, show_plots=False):
    import matplotlib
    figwidth = 12
    lw = 2
    y_lim_gain = 1.2

    #Plot open loop results:

    ylabels = ['$e_{y}$', '$e_z$', '$e_{\\theta}$']
    plt.figure(figsize=(figwidth,3))
    for ii in range(3):
        plt.subplot(1,3,ii+1)
        plt.plot(t_eval[:-1], error_nom_mean[ii,:], linewidth=lw, label='Nominal (linearized)')
        plt.fill_between(t_eval[:-1], error_nom_mean[ii,:] - error_nom_std[ii,:], error_nom_mean[ii,:] + error_nom_std[ii,:], alpha=0.2)
        plt.plot(t_eval[:-1], error_edmd_mean[ii, :], linewidth=lw, label='EDMD')
        plt.fill_between(t_eval[:-1], error_edmd_mean[ii, :] - error_edmd_std[ii, :],error_edmd_mean[ii, :] + error_edmd_std[ii, :], alpha=0.2)
        plt.plot(t_eval[:-1], error_bedmd_mean[ii, :], linewidth=lw, label='bEDMD')
        plt.fill_between(t_eval[:-1], error_bedmd_mean[ii, :] - error_bedmd_std[ii, :],error_bedmd_mean[ii, :] + error_bedmd_std[ii, :], alpha=0.2)
        ylim = max(max(np.abs(error_bedmd_mean[ii, :] - error_bedmd_std[ii, :])), max(np.abs(error_bedmd_mean[ii, :] + error_bedmd_std[ii, :])))
        plt.ylim([-ylim * y_lim_gain, ylim * y_lim_gain])
        plt.xlabel('$t$ (sec)', fontsize=16)
        plt.ylabel(ylabels[ii], fontsize=16)
        plt.grid()

    plt.legend(loc='upper left')
    suptitle = plt.suptitle('Open loop prediction error of linearized, EDMD and bilinear EDMD models', y=1.05, fontsize=18)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.tight_layout()
    plt.savefig(folder_name + 'planar_quad_prediction.pdf', format='pdf', dpi=2400, bbox_extra_artists=(suptitle,), bbox_inches="tight")

    if show_plots:
        plt.show()
        print('MSE nominal: ', mse_nom, '\nMSE EDMD: ', mse_edmd, '\nMSE bilinear EDMD: ', mse_bedmd)
        print('Improvement: ', (1 - mse_bedmd / mse_edmd) * 100, ' percent')

    #Plot closed loop results:
    x_index = 1
    y_index = 2
    plt.figure(figsize=(figwidth, 4))
    plt.subplot(1, 3, 1)
    plt.plot(xr[x_index,:], xr[y_index,:], '--r', linewidth=2, label='Reference')
    plt.plot(xs_mpc_nom[:, x_index], xs_mpc_nom[:, y_index], linewidth=lw, label='Nominal (linearized)')
    plt.plot(xs_mpc_edmd[:, x_index], xs_mpc_edmd[:, y_index], linewidth=lw, label='EDMD')
    plt.plot(xs_mpc_bedmd[:, x_index], xs_mpc_bedmd[:, y_index], linewidth=lw, label='bEDMD')
    plt.xlabel('$z$')
    plt.ylabel('$\\theta$')
    plt.title('Phase plot')
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(t_eval_cl[:-1], us_mpc_nom[:, 0], linewidth=lw, label='Nominal (linearized)')
    plt.plot(t_eval_cl[:-1], us_mpc_edmd[:, 0], linewidth=lw, label='EDMD')
    plt.plot(t_eval_cl[:-1], us_mpc_bedmd[:, 0], linewidth=lw, label='bEDMD')
    plt.plot([t_eval_cl[0], t_eval_cl[-1]], [umax[0]+hover_thrust, umax[0]+hover_thrust], ':k', linewidth=lw)
    plt.plot([t_eval_cl[0], t_eval_cl[-1]], [umin[1] + hover_thrust, umin[1] + hover_thrust], ':k', linewidth=lw)
    plt.ylabel('$u_1$')
    plt.title('Control action')
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(t_eval_cl[:-1], us_mpc_nom[:, 1], linewidth=lw, label='Nominal (linearized)')
    plt.plot(t_eval_cl[:-1], us_mpc_edmd[:, 1], linewidth=lw, label='EDMD')
    plt.plot(t_eval_cl[:-1], us_mpc_bedmd[:, 1], linewidth=lw, label='bEDMD')
    plt.plot([t_eval_cl[0], t_eval_cl[-1]], [umax[1]+hover_thrust, umax[1]+hover_thrust], ':k', linewidth=lw)
    plt.plot([t_eval_cl[0], t_eval_cl[-1]], [umin[1] + hover_thrust, umin[1] + hover_thrust], ':k', linewidth=lw)
    plt.xlabel('Time (sec)')
    plt.ylabel('$u_2$')
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(t_eval_cl[:-1], cost_nom / cost_nom[-1], linewidth=lw, label='Nominal (linearized)')
    plt.plot(t_eval_cl[:-1], cost_edmd / cost_nom[-1], linewidth=lw, label='EDMD')
    plt.plot(t_eval_cl[:-1], cost_bedmd / cost_nom[-1], linewidth=lw, label='bEDMD')
    plt.xlabel('Time (sec)')
    plt.ylabel('J (normalized)')
    plt.title('Normalized cost')
    plt.grid()
    plt.legend(loc='upper left')

    suptitle = plt.suptitle('Trajectory tracking with MPC based on linearized, EDMD and bilinear EDMD models', y=1.05,fontsize=18)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.tight_layout()
    plt.savefig(folder_name + 'planar_quad_closedloop.pdf', format='pdf', dpi=2400, bbox_extra_artists=(suptitle,),
                bbox_inches="tight")
    if show_plots:
        plt.show()

def plot_debug(show_plots=False):

    #Show open loop results:
    y_lim_gain = 1.2
    fig, axs = plt.subplots(2, int(n / 2), figsize=(10, 6))
    ylabels = ['$e_{y}$', '$e_z$', '$e_{\\theta}$', '$e_{\dot{y}}$', '$e_{\dot{z}}$', '$e_{\dot{\\theta}}$']
    fig.suptitle('Open loop predicition error of EDMD and bilinear EDMD models', y=1.025, fontsize=18)
    for ax, err_nom_mean, err_nom_std, err_edmd_mean, err_edmd_std, err_bedmd_mean, err_bedmd_std, ylabel in zip(
            axs.flatten(), error_nom_mean, error_nom_std, error_edmd_mean, error_edmd_std, error_bedmd_mean,
            error_bedmd_std, ylabels):
        ax.plot(t_eval[:-1], err_nom_mean, linewidth=3, label='mean, nominal')
        ax.fill_between(t_eval[:-1], err_nom_mean - err_nom_std, err_nom_mean + err_nom_std, alpha=0.2)
        ax.plot(t_eval[:-1], err_edmd_mean, linewidth=3, label='mean, EDMD')
        ax.fill_between(t_eval[:-1], err_edmd_mean - err_edmd_std, err_edmd_mean + err_edmd_std, alpha=0.2)
        ax.plot(t_eval[:-1], err_bedmd_mean, linewidth=3, label='mean, bEDMD')
        ax.fill_between(t_eval[:-1], err_bedmd_mean - err_bedmd_std, err_bedmd_mean + err_bedmd_std, alpha=0.2)
        ax.set_ylabel(ylabel, fontsize=16)
        ylim = max(max(np.abs(err_bedmd_mean - err_bedmd_std)), max(np.abs(err_bedmd_mean + err_bedmd_std)))
        ax.set_ylim([-ylim * y_lim_gain, ylim * y_lim_gain])
        ax.grid()
        ax.set_xlabel('$t$ (sec)', fontsize=16)

    ax.legend()
    # plt.tight_layout()
    if show_plots:
        plt.show()
        print('MSE nominal: ', mse_nom, '\nMSE EDMD: ', mse_edmd, '\nMSE bilinear EDMD: ', mse_bedmd)
        print('Improvement: ', (1 - mse_bedmd / mse_edmd) * 100, ' percent')

    #Plot closed loop results:
    _, axs = plt.subplots(2, int(n / 2), figsize=(12, 8))
    ylabels = ['$y$', '$z$', '$\\theta$']
    legend_labels = ['Nominal (linearized)', 'EDMD', 'bEDMD']

    for ax, data_nom, data_edmd, data_bedmd, ylabel in zip(axs[:-1].flatten(), xs_mpc_nom.T, xs_mpc_edmd.T,
                                                           xs_mpc_bedmd.T, ylabels):
        # for ax, data_nom, data_bedmd, ylabel in zip(axs[:-1].flatten(), xs_mpc_nom.T, xs_mpc_bedmd.T, ylabels):
        ax.plot(t_eval_cl, data_nom, linewidth=3, label=legend_labels[0])
        ax.plot(t_eval_cl, data_edmd, linewidth=3, label=legend_labels[1])
        ax.plot(t_eval_cl, data_bedmd, linewidth=3, label=legend_labels[2])
        ax.set_ylabel(ylabel, fontsize=16)
        ax.grid()
        ax.set_xlabel('$t$ (sec)', fontsize=16)
        ax.legend()

    axs[0, 0].plot(t_eval_cl, xr[0, :], '--r', linewidth=2)
    axs[0, 1].plot(t_eval_cl, xr[1, :], '--r', linewidth=2)
    axs[0, 2].plot(t_eval_cl, xr[2, :], '--r', linewidth=2)

    ax = axs[1, 0]
    ax.plot(t_eval_cl[:-1], us_mpc_nom[:, 0], linewidth=3, label='$u$, ' + legend_labels[0])
    ax.plot(t_eval_cl[:-1], us_mpc_nom[:, 1], linewidth=3, label='$u$, ' + legend_labels[0])
    ax.plot(t_eval_cl[:-1], us_mpc_edmd[:, 0], linewidth=3, label='$u$, ' + legend_labels[1])
    ax.plot(t_eval_cl[:-1], us_mpc_edmd[:, 1], linewidth=3, label='$u$, ' + legend_labels[1])
    ax.plot(t_eval_cl[:-1], us_mpc_bedmd[:, 0], linewidth=3, label='$u_1$, ' + legend_labels[2])
    ax.plot(t_eval_cl[:-1], us_mpc_bedmd[:, 1], linewidth=3, label='$u_2$, ' + legend_labels[2])
    ax.grid()
    ax.set_xlabel('$t$ (sec)', fontsize=16)
    ax.set_ylabel('$u$', fontsize=16)
    ax.legend()

    ax = axs[1, 1]
    ax.plot(t_eval_cl[:-1], cost_nom / cost_nom[-1], linewidth=3, label=legend_labels[0])
    ax.plot(t_eval_cl[:-1], cost_edmd / cost_nom[-1], linewidth=3, label=legend_labels[1])
    ax.plot(t_eval_cl[:-1], cost_bedmd / cost_nom[-1], linewidth=3, label=legend_labels[2])
    ax.grid()
    ax.set_xlabel('$t$ (sec)', fontsize=16)
    ax.set_ylabel('Normalized cost, $J$', fontsize=16)
    ax.legend(loc='lower right')

    plt.tight_layout()
    if show_plots:
        plt.show()

#plot_debug(show_plots=False)
plot_paper(folder_plots, show_plots=True)

