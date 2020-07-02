import numpy as np
from core.dynamics import RoboticDynamics
from koopman_core.controllers import MPCController, PerturbedController
from core.controllers import PDController
from core.dynamics import LinearSystemDynamics
import scipy as sc
import random as rand
import matplotlib.pyplot as plt
from koopman_core.dynamics import LinearLiftedDynamics
from koopman_core.learning import Edmd
from sklearn import preprocessing, linear_model
from koopman_core.dynamics import BilinearLiftedDynamics
from koopman_core.learning.bilinear_edmd import BilinearEdmd
from koopman_core.controllers.openloop_controller import OpenLoopController
from koopman_core.controllers import LinearMpcController, MPCController, MPCControllerDense, BilinearMpcController
from core.dynamics import ConfigurationDynamics
from koopman_core.learning.utils import differentiate_vec
from koopman_core.learning import FlBilinearLearner
from pysindy.optimizers.stlsq import STLSQ
import dill as pickle

class FiniteDimKoopSys(RoboticDynamics):
    def __init__(self, lambd, mu, c):
        RoboticDynamics.__init__(self, 2, 2)
        self.params = lambd, mu, c

    def D(self, q):
        return np.array([[1, 0], [0, (q[0] + 2) ** (-1)]])

    def C(self, q, q_dot):
        labmd, mu, c = self.params
        return -np.array([[lambd, 0], [(q[0] + 2) ** (-1) * (2 * lambd - mu) * c * q_dot[0], (q[0] + 2) ** (-1) * mu]])

    def G(self, q):
        return np.array([0, 0])

    def B(self, q):
        return np.array([[1, 0], [0, 1]])

class BilinearSysOutput(ConfigurationDynamics):
    def __init__(self, bilinear_dynamics, C_h):
        ConfigurationDynamics.__init__(self, bilinear_dynamics, 1)
        self.bilinear_dynamics = bilinear_dynamics
        self.C_h = C_h

    def y(self, x):
        z = self.bilinear_dynamics.phi_fun(x)
        return np.dot(self.C_h, z)

    def dydx(self, x):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    def d2ydz2(self, x):
        return np.zeros((1, self.k, 4))

    def y_d(self, t):
        return np.dot(self.C_h, self.z_d(t))

    def y_d_dot(self, t):
        return np.dot(self.C_h, self.z_d_dot(t))

    def y_d_ddot(self, t):
        return np.dot(self.C_h, self.z_d_ddot(t))

    def z_d(self, t):
        return self.bilinear_dynamics.phi_fun([[0, 0, 0, 0]]).squeeze()

    def z_d_dot(self, t):
        return np.zeros(self.bilinear_dynamics.n)

    def z_d_ddot(self, t):
        return np.zeros(self.bilinear_dynamics.n)

# Bilinearizable system parameters:
lambd, mu, c = .3, .2, -.5
n, m = 4, 2
finite_dim_koop_sys = FiniteDimKoopSys(lambd, mu, c)

# Linearized system specification:
A_nom = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, lambd, 0],
                  [0, 0, 0, mu]])
B_nom = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 2]])

q, r = 5, 1
Q_fb = q * np.identity(4)
R_fb = r*np.identity(2)
P = sc.linalg.solve_continuous_are(A_nom, B_nom, Q_fb, R_fb)
K = np.linalg.inv(R_fb)@B_nom.T@P
K_p, K_d = K[:,:int(n/2)], K[:,int(n/2):]
pd_controller = PDController(finite_dim_koop_sys, K_p, K_d)
nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)

# Data collection parameters:
n_traj = 50                                                 # Number of trajectories to collect data from
dt = 1.0e-2                                                 # Time step length
N = int(4./dt)                                              # Number of time steps
t_eval = dt * np.arange(N + 1)                              # Simulation time points
traj_max = [2., 2., 2., 2.]                                 # State constraints, [q, q_dot]
traj_min = [-.5, -2., -1., -2.]                                 # State constraints, [q, q_dot]
pert_noise_var = 2.                                        # Variance of controller perturbation noise
downsample_rate = 3

#xmax = np.array([2, 0.35,2.,2.])                       # State constraints, [x, theta, x_dot, theta_dot]
#xmin = -xmax
#set_pt = np.zeros(n)                                    # Desired trajectories (initialization)
#umax = np.array([10])                                   # MPC actuation constraint (trajectory generation)
#umin = -umax

#EDMD parameters:
alpha_edmd = 7e-2
tune_mdl_edmd = False

#Bilinear EDMD parameters:
alpha_bedmd = 4e-2
tune_mdl_bedmd = False

#Feedback linearizable bilinear EDMD parameters:
alpha_fl_bedmd = 4e-2
#alpha_fl_bedmd = 1e-3
tune_mdl_fl_bedmd = False


learn_models = True
model_fname = 'bilinearizable_sys_models'

# Prediction performance evaluation parameters:
n_traj_test = 25
#mpc_controller = MPCController(nominal_sys,n_pred,dt,umin,umax,xmin,xmax,Q,R,QN,set_pt)
#x_0_max = np.array([xmax[0], xmax[1], 0.2, 0.2])

#Closed loop performance evaluation parameters:

if learn_models:
    pert_lqr_controller = PerturbedController(finite_dim_koop_sys, pd_controller, pert_noise_var)
    xs = np.empty((n_traj,N+1,n))
    us = np.empty((n_traj, N, m))

    n_cols = 10
    plt.figure(figsize=(12,12*n_traj/(n_cols**2)))
    for ii in range(n_traj):
        x_0 = np.asarray([rand.uniform(i,j) for i,j in zip(traj_min, traj_max)])
        xs[ii,:,:], us[ii,:,:] = finite_dim_koop_sys.simulate(x_0, pert_lqr_controller, t_eval)

        plt.subplot(int(np.ceil(n_traj/n_cols)),n_cols,ii+1)
        plt.plot(t_eval, xs[ii,:,0], 'b', label='$x$')
        plt.plot(t_eval, xs[ii,:,1], 'r', label='$\\theta$')
    plt.suptitle('Training data',y=0.94)
    plt.show()

    #EDMD:
    print('Fitting EDMD...')
    poly_features_edmd = preprocessing.PolynomialFeatures(degree=3)
    poly_features_edmd.fit(np.zeros((1,n)))
    n_lift_edmd = poly_features_edmd.n_output_features_
    C_edmd = np.zeros((n,n_lift_edmd))
    C_edmd[:,1:n+1] = np.eye(n)

    basis_edmd = lambda x: poly_features_edmd.transform(x)
    optimizer_edmd = linear_model.MultiTaskLasso(alpha=alpha_edmd, fit_intercept=False, selection='random')
    cv_edmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
    standardizer_edmd = preprocessing.StandardScaler(with_mean=False)

    model_edmd = Edmd(n, m, basis_edmd, n_lift_edmd, n_traj, optimizer_edmd, cv=cv_edmd, standardizer=standardizer_edmd, C=C_edmd)
    X_edmd, y_edmd = model_edmd.process(xs, us, np.tile(t_eval,(n_traj,1)), downsample_rate=downsample_rate)
    model_edmd.fit(X_edmd, y_edmd, cv=tune_mdl_bedmd, override_kinematics=True)
    #sys_edmd = LinearSystemDynamics(model_edmd.A, model_edmd.B)
    sys_edmd = LinearLiftedDynamics(model_edmd.A, model_edmd.B, C_edmd, basis_edmd)
    if tune_mdl_edmd:
        print(model_edmd.cv.alpha_)

    #TODO: Remove after debug -->
    plt.figure()
    plt.plot(X_edmd[:,1])
    plt.show()

    #Bilinear EDMD:
    print('Fitting bilinear EDMD...')
    poly_features_bedmd = preprocessing.PolynomialFeatures(degree=2)
    poly_features_bedmd.fit(np.zeros((1,n)))
    n_lift_bedmd = poly_features_bedmd.n_output_features_
    C_bedmd = np.zeros((n,n_lift_bedmd))
    C_bedmd[:,1:n+1] = np.eye(n)

    basis_bedmd = lambda x: poly_features_bedmd.transform(x)
    optimizer_bedmd = linear_model.MultiTaskLasso(alpha=alpha_bedmd, fit_intercept=False, selection='random')
    cv_bedmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
    standardizer_bedmd = preprocessing.StandardScaler(with_mean=False)

    model_bedmd = BilinearEdmd(n, m, basis_bedmd, n_lift_bedmd, n_traj, optimizer_bedmd, cv=cv_bedmd, standardizer=standardizer_bedmd, C=C_bedmd)
    X_bedmd, y_bedmd = model_bedmd.process(xs, us, np.tile(t_eval,(n_traj,1)), downsample_rate=downsample_rate)
    model_bedmd.fit(X_bedmd, y_bedmd, cv=tune_mdl_bedmd, override_kinematics=True)
    sys_bedmd = BilinearLiftedDynamics(n_lift_bedmd, m, model_bedmd.A, model_bedmd.B, C_bedmd, basis_bedmd)
    if tune_mdl_bedmd:
        print(model_bedmd.cv.alpha_)
        alpha_bedmd = model_bedmd.cv.alpha_

    #Feedback linearizable bilinear EDMD:
    print('Fitting feedback linearizable bilinear EDMD...')
    poly_features_fl_bedmd = preprocessing.PolynomialFeatures(degree=2)
    poly_features_fl_bedmd.fit(np.zeros((1,n)))
    n_lift_fl_bedmd = poly_features_fl_bedmd.n_output_features_
    C_fl_bedmd = np.zeros((n,n_lift_fl_bedmd))
    C_fl_bedmd[:,1:n+1] = np.eye(n)
    C_h_fl_bedmd = C_fl_bedmd[:m,:]

    basis_fl_bedmd = lambda x: poly_features_fl_bedmd.transform(x)
    optimizer_fl_bedmd = linear_model.MultiTaskLasso(alpha=alpha_bedmd, fit_intercept=False, selection='random')
    standardizer_fl_bedmd = preprocessing.StandardScaler(with_mean=False)

    cv_fl_bedmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
    model_fl_bedmd = FlBilinearLearner(n, m, basis_fl_bedmd, n_lift_fl_bedmd, n_traj, optimizer_fl_bedmd, C_h_fl_bedmd, cv=cv_fl_bedmd, standardizer=standardizer_fl_bedmd, C=C_fl_bedmd)
    X_fl_bedmd, y_fl_bedmd = model_fl_bedmd.process(xs, us, np.tile(t_eval,(n_traj,1)), downsample_rate=downsample_rate)
    model_fl_bedmd.fit(X_fl_bedmd, y_fl_bedmd, cv=tune_mdl_fl_bedmd, override_kinematics=True, l1_reg=alpha_fl_bedmd)
    sys_fl_bedmd = BilinearLiftedDynamics(n_lift_fl_bedmd, m, model_fl_bedmd.A, model_fl_bedmd.B, C_fl_bedmd, basis_fl_bedmd)

    #Save models:
    model_dict = {'sys_edmd': sys_edmd,'model_edmd': model_edmd, 'sys_bedmd': sys_bedmd,'model_bedmd': model_bedmd, 'sys_fl_bedmd': sys_fl_bedmd,'model_fl_bedmd': model_fl_bedmd}
    with open(model_fname, 'wb') as handle:
        pickle.dump(model_dict, handle)
else:
    with open(model_fname, 'rb') as handle:
        p = pickle.load(handle)
    sys_edmd = p['sys_edmd']
    model_edmd = p['model_edmd']
    sys_bedmd = p['sys_bedmd']
    model_bedmd = p['model_bedmd']
    sys_fl_bedmd = p['sys_fl_bedmd']
    model_fl_bedmd = p['model_fl_bedmd']

    poly_features_edmd = preprocessing.PolynomialFeatures(degree=3)
    poly_features_edmd.fit(np.zeros((1, n)))
    poly_features_bedmd = preprocessing.PolynomialFeatures(degree=2)
    poly_features_bedmd.fit(np.zeros((1, n)))
    poly_features_fl_bedmd = preprocessing.PolynomialFeatures(degree=2)
    poly_features_fl_bedmd.fit(np.zeros((1, n)))

#Compare open loop performance:
pert_lqr_controller = PerturbedController(finite_dim_koop_sys, pd_controller, pert_noise_var)
xs_test, xs_edmd_test, xs_bedmd_test, xs_fl_bedmd_test, us_test, ts_test = [], [], [], [], [], []
for ii in range(n_traj_test):
    x_0 = np.asarray([rand.uniform(i,j) for i,j in zip(traj_min, traj_max)])
    xs_tmp, us_tmp = finite_dim_koop_sys.simulate(x_0, pert_lqr_controller, t_eval)
    ol_controller = OpenLoopController(sys_bedmd, us_tmp, t_eval[:-1])

    z_0_edmd = sys_edmd.phi_fun(np.atleast_2d(x_0)).squeeze()
    zs_edmd_tmp, _ = sys_edmd.simulate(z_0_edmd, ol_controller, t_eval[:-1])
    xs_edmd_tmp = np.dot(sys_edmd.Cx, zs_edmd_tmp.T)

    z_0_bedmd = sys_bedmd.phi_fun(np.atleast_2d(x_0)).squeeze()
    zs_bedmd_tmp, _ = sys_bedmd.simulate(z_0_bedmd, ol_controller, t_eval[:-1])
    xs_bedmd_tmp = np.dot(sys_bedmd.Cx, zs_bedmd_tmp.T)

    z_0_fl_bedmd = sys_fl_bedmd.phi_fun(np.atleast_2d(x_0)).squeeze()
    zs_fl_bedmd_tmp, _ = sys_fl_bedmd.simulate(z_0_fl_bedmd, ol_controller, t_eval[:-1])
    xs_fl_bedmd_tmp = np.dot(sys_fl_bedmd.Cx, zs_fl_bedmd_tmp.T)

    xs_test.append(xs_tmp.T)
    xs_edmd_test.append(xs_edmd_tmp)
    xs_bedmd_test.append(xs_bedmd_tmp)
    xs_fl_bedmd_test.append(xs_fl_bedmd_tmp)
    us_test.append(us_tmp)
    ts_test.append(t_eval)

xs_test, xs_edmd_test, xs_bedmd_test, xs_fl_bedmd_test, us_test, ts_test = \
    np.array(xs_test), np.array(xs_edmd_test), np.array(xs_bedmd_test), np.array(xs_fl_bedmd_test), np.array(us_test), np.array(ts_test)

error_edmd = xs_test[:,:,:-1] - xs_edmd_test
error_edmd_mean = np.mean(error_edmd, axis=0)
error_edmd_std = np.std(error_edmd, axis=0)
mse_edmd = np.mean(np.mean(np.mean(np.square(error_edmd))))

error_bedmd = xs_test[:,:,:-1] - xs_bedmd_test
error_bedmd_mean = np.mean(error_bedmd, axis=0)
error_bedmd_std = np.std(error_bedmd, axis=0)
mse_bedmd = np.mean(np.mean(np.mean(np.square(error_bedmd))))

error_fl_bedmd = xs_test[:,:,:-1] - xs_fl_bedmd_test
error_fl_bedmd_mean = np.mean(error_fl_bedmd, axis=0)
error_fl_bedmd_std = np.std(error_fl_bedmd, axis=0)
mse_fl_bedmd = np.mean(np.mean(np.mean(np.square(error_fl_bedmd))))

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
ylabels = ['$e_{q_1}$', '$e_{q_2}$', '$e_{\dot{q}_1}$', '$e_{\dot{q}_2}$']
fig.suptitle('Open loop predicition error of EDMD and bilinear EDMD models', y=1.025, fontsize=18)
for ax, err_edmd_mean, err_edmd_std, err_bedmd_mean, err_bedmd_std, err_fl_bedmd_mean, err_fl_bedmd_std, ylabel in zip(axs.flatten(), error_edmd_mean, error_edmd_std, error_bedmd_mean, error_bedmd_std, error_fl_bedmd_mean, error_fl_bedmd_std, ylabels):
    ax.plot(ts_test[0,:-1], err_edmd_mean, linewidth=3, label='mean, EDMD')
    ax.fill_between(ts_test[0,:-1], err_edmd_mean-err_edmd_std, err_edmd_mean+err_edmd_std, alpha=0.2, label='std, EDMD')
    ax.plot(ts_test[0,:-1], err_bedmd_mean, linewidth=3, label='mean, bEDMD')
    ax.fill_between(ts_test[0,:-1], err_bedmd_mean-err_bedmd_std, err_bedmd_mean+err_bedmd_std, alpha=0.2, label='std, bEDMD')
    ax.plot(ts_test[0,:-1], err_fl_bedmd_mean, linewidth=3, label='mean, FL bEDMD')
    ax.fill_between(ts_test[0,:-1], err_fl_bedmd_mean-err_fl_bedmd_std, err_fl_bedmd_mean+err_fl_bedmd_std, alpha=0.2, label='std, FL bEDMD')
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid()
    ax.set_xlabel('$t$ (sec)', fontsize=16)

ax.legend(ncol=2)
plt.tight_layout()
plt.show()
#print('Lifting dimension EDMD: ', sys_edmd.n_li, 'Lifting dimension bEDMD: ', n_lift_bedmd*(m+1) , 'Lifting dimension FL bEDMD: ', n_lift_fl_bedmd*(m+1))
print('MSE EDMD: ', mse_edmd, '\nMSE bilinear EDMD: ', mse_bedmd, '\nMSE FL bilinear EDMD: ', mse_fl_bedmd)
print('Improvement EDMD -> bEDMD: ', (1-mse_bedmd/mse_edmd)*100, ' percent\nImprovement bEDMD -> FL bEDMD: ', (1-mse_fl_bedmd/mse_bedmd)*100, ' percent')

#Compare closed loop performance:
# MPC parameters:
x_0 = np.array([.8, 0.2, 0., 0.])
umax = np.array([20., 20.])
umin = -umax
xmax = np.array([10., 10., 10., 10.])
xmin = -xmax
q, r = 1e1, 1
Q = q*np.identity(n)
R = r*np.identity(m)
n_pred = 200
set_pt = np.zeros(n)
x_d = np.tile(set_pt.reshape(-1,1), (1,t_eval.shape[0]))

# Design MPC for linearized nominal model:
lin_sys = sc.signal.StateSpace(A_nom, B_nom, np.eye(n), np.zeros((n,m)))
lin_sys_d = lin_sys.to_discrete(dt)
A_d, B_d = lin_sys_d.A, lin_sys_d.B
controller_nom = LinearMpcController(n, m, n, n_pred, lin_sys_d, xmin, xmax, umin, umax, Q, Q, R, set_pt)
controller_nom.construct_controller()
#controller_nom = MPCController(nominal_sys, n_pred, dt, umin, umax, xmin, xmax, Q, R, Q, x_d)

# Design MPC for lifted linear EDMD model:
controller_edmd = MPCControllerDense(sys_edmd, n_pred, dt, umin, umax, xmin, xmax, Q, R, Q, x_d, lifting=True,
                                     edmd_object=model_edmd, plotMPC=False, name='EDMD')

# Design MPC for lifted bilinear EDMD model:
k = m
Q_fl = q*np.eye(int(2*model_bedmd.n_lift))
#Q_fl = np.zeros((int(2*model_bedmd.n_lift),int(2*model_bedmd.n_lift)))
#Q_fl[:model_bedmd.n_lift, :model_bedmd.n_lift] = model_bedmd.C.T@Q@model_bedmd.C
R_fl = r*np.eye(model_bedmd.n_lift)
C_h = model_bedmd.C[:k,:]

f_eta = np.concatenate((np.zeros((model_fl_bedmd.n_lift,model_fl_bedmd.n_lift)), np.eye(model_fl_bedmd.n_lift)), axis=1)
f_eta_dot = np.concatenate((sys_fl_bedmd.F@sys_fl_bedmd.F, np.zeros((model_fl_bedmd.n_lift,model_fl_bedmd.n_lift))), axis=1)
F_lin = np.concatenate((f_eta, f_eta_dot), axis=0)
G_lin = np.concatenate((np.zeros((model_fl_bedmd.n_lift,model_fl_bedmd.n_lift)), np.eye(model_fl_bedmd.n_lift)), axis=0)
fb_sys = sc.signal.StateSpace(F_lin, G_lin, np.eye(int(2*model_fl_bedmd.n_lift)), np.zeros((int(2*model_fl_bedmd.n_lift),model_bedmd.n_lift)))
fb_sys_d = fb_sys.to_discrete(dt)

controller_bedmd = BilinearMpcController(n, m, k, model_fl_bedmd.n_lift, n_pred, fb_sys_d, sys_fl_bedmd, model_fl_bedmd.C, C_h, xmin, xmax, umin, umax,
                                          Q_fl, Q_fl, R_fl, set_pt.reshape((1,-1)))
controller_bedmd.construct_controller()

# Design FL for lifted bilinear EDMD model: #TODO: Remove after debug...
from scipy.linalg import solve_continuous_are
from koopman_core.controllers import BilinearFBLinController
#output_fl = BilinearSysOutput(sys_fl_bedmd, C_h)
#P = solve_continuous_are(F_lin, G_lin, Q_fl, R_fl)
#K = -np.linalg.inv(R_fl)@G_lin.T@P
#bl_fb_lin = BilinearFBLinController(sys_fl_bedmd, output_fl, K)

xs_mpc_nom, us_mpc_nom = finite_dim_koop_sys.simulate(x_0, controller_nom, t_eval)
xs_mpc_edmd, us_mpc_edmd = finite_dim_koop_sys.simulate(x_0, controller_edmd, t_eval)
print('Simulating fl beedmd controller...')
xs_mpc_bedmd, us_mpc_bedmd = finite_dim_koop_sys.simulate(x_0.reshape((1,-1)), controller_bedmd, t_eval)
#xs_mpc_bedmd, us_mpc_bedmd = finite_dim_koop_sys.simulate(x_0.reshape((1,-1)), bl_fb_lin, t_eval)

cost_nom = np.cumsum(np.diag(xs_mpc_nom[1:,:] @ Q @ xs_mpc_nom[1:,:].T) + np.diag(us_mpc_nom @ R @ us_mpc_nom.T))
cost_edmd = np.cumsum(np.diag(xs_mpc_edmd[1:,:] @ Q @ xs_mpc_edmd[1:,:].T) + np.diag(us_mpc_edmd @ R @ us_mpc_edmd.T))
cost_bedmd = np.cumsum(np.diag(xs_mpc_bedmd[1:,:] @ Q @ xs_mpc_bedmd[1:,:].T) + np.diag(us_mpc_bedmd @ R @ us_mpc_bedmd.T))

_, axs = plt.subplots(2, 2, figsize=(12, 8))
ylabels = ['$x$', '$\\theta$']
legend_labels=['Nominal (linearized)', 'EDMD', 'bEDMD']

for ax, data_nom, data_edmd, data_bedmd, ylabel in zip(axs[:-1].flatten(), xs_mpc_nom.T, xs_mpc_edmd.T, xs_mpc_bedmd.T, ylabels):
#for ax, data_nom, data_edmd, ylabel in zip(axs[:-1].flatten(), xs_mpc_nom.T, xs_mpc_edmd.T, ylabels):
    ax.plot(t_eval, data_nom, linewidth=3, label=legend_labels[0])
    ax.plot(t_eval, data_edmd, linewidth=3, label=legend_labels[1])
    ax.plot(t_eval, data_bedmd, linewidth=3, label=legend_labels[2])
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid()
    ax.set_xlabel('$t$ (sec)', fontsize=16)
    ax.legend()

ax = axs[1,0]
ax.plot(t_eval[:-1], us_mpc_nom[:,0], linewidth=3, label='$u$, '+ legend_labels[0])
ax.plot(t_eval[:-1], us_mpc_edmd[:,0], linewidth=3, label='$u$, '+ legend_labels[1])
ax.plot(t_eval[:-1], us_mpc_bedmd[:,0], linewidth=3, label='$u$, '+ legend_labels[2])
ax.grid()
ax.set_xlabel('$t$ (sec)', fontsize=16)
ax.set_ylabel('$u$', fontsize=16)
ax.legend()

ax = axs[1,1]
ax.plot(t_eval[:-1], cost_nom/cost_nom[-1], linewidth=3, label=legend_labels[0])
ax.plot(t_eval[:-1], cost_edmd/cost_nom[-1], linewidth=3, label=legend_labels[1])
ax.plot(t_eval[:-1], cost_bedmd/cost_nom[-1], linewidth=3, label=legend_labels[2])
ax.grid()
ax.set_xlabel('$t$ (sec)', fontsize=16)
ax.set_ylabel('Normalized cost, $J$', fontsize=16)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.subplot(3,1,1)
plt.hist(controller_nom.comp_time)
#plt.xlim(1e-4, 1e-1)
plt.subplot(3,1,2)
#plt.hist(controller_edmd.comp_time)
#plt.xlim(1e-4, 1e-1)
plt.subplot(3,1,3)
#plt.hist(controller_bedmd.comp_time)
#plt.xlim(1e-4, 1e-1)
plt.tight_layout()
plt.show()