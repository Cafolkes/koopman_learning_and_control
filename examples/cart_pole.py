import numpy as np
import scipy as sc
import random as rand
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
from core.controllers import PDController
from core.dynamics import LinearSystemDynamics, ConfigurationDynamics
from koopman_core.controllers import OpenLoopController, MPCController, LinearMpcController, MPCControllerDense, \
    BilinearMpcController, PerturbedController
from koopman_core.dynamics import LinearLiftedDynamics, BilinearLiftedDynamics
from koopman_core.learning import Edmd, BilinearEdmd
from koopman_core.basis_functions import CartPoleBasis


class CartPole(RoboticDynamics):
    def __init__(self, m_c, m_p, l, g=9.81):
        RoboticDynamics.__init__(self, 2, 1)
        self.params = m_c, m_p, l, g

    def D(self, q):
        m_c, m_p, l, _ = self.params
        _, theta = q
        return np.array([[m_c + m_p, m_p * l * np.cos(theta)], [m_p * l * np.cos(theta), m_p * (l ** 2)]])

    def C(self, q, q_dot):
        _, m_p, l, _ = self.params
        _, theta = q
        _, theta_dot = q_dot
        return np.array([[0, -m_p * l * theta_dot * np.sin(theta)], [0, 0]])

    def U(self, q):
        _, m_p, l, g = self.params
        _, theta = q
        return m_p * g * l * np.cos(theta)

    def G(self, q):
        _, m_p, l, g = self.params
        _, theta = q
        return np.array([0, -m_p * g * l * np.sin(theta)])

    def B(self, q):
        return np.array([[1], [0]])
class CartPoleOutput(ConfigurationDynamics):
    def __init__(self, cart_pole, x_d, t_d):
        ConfigurationDynamics.__init__(self, cart_pole, 1)
        self.cart_pole = cart_pole
        self.x_d = x_d
        self.t_d = t_d
        self.x_d_dot = differentiate_vec(self.x_d, self.t_d)

    def y(self, q):
        return q

    def dydq(self, q):
        return np.eye(2)

    def d2ydq2(self, q):
        return np.zeros((2, 2, 2))

    def y_d(self, t):
        return self.desired_state_(t)[:2]

    def y_d_dot(self, t):
        return self.desired_state_(t)[2:]

    def y_d_ddot(self, t):
        return self.desired_state_dot_(t)[2:]

    def desired_state_(self, t):
        return [np.interp(t, self.t_d.flatten(),self.x_d[:,ii].flatten()) for ii in range(self.x_d.shape[1])]

    def desired_state_dot_(self, t):
        return [np.interp(t, self.t_d.flatten(),self.x_d_dot[:,ii].flatten()) for ii in range(self.x_d_dot.shape[1])]

# Cart pole system parameters
m_c = 0.5
m_p = 0.2
l = 0.4
cart_pole = CartPole(m_c, m_p, l)

# Linearized system specification:
n, m = 4, 1                                             # Number of states, number of control inputs
A_nom = np.array([[0., 0., 1., 0.],
                  [0., 0., 0., 1.],
                  [0., -3.924, 0., 0.],
                  [0., 34.335, 0., 0.]])                # Linearization of the true system around the origin
B_nom = np.array([[0.],
                  [0.],
                  [2.],
                  [-5.]])                               # Linearization of the true system around the origin
K_p = -np.array([[7.3394, 39.0028]])                    # Proportional control gains
K_d = -np.array([[8.0734, 7.4294]])                     # Derivative control gains
nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)

# Data collection parameters:
n_traj = 50                                             # Number of trajectories to collect data from
dt = 1.0e-2                                             # Time step length
mpc_horizon = 2.                                        # MPC time horizon [sec] (trajectory generation)
n_pred = int(mpc_horizon/dt)                            # Number of time steps
t_eval = dt * np.arange(n_pred + 1)                     # Simulation time points
noise_var = 2.                                         # Exploration noise to perturb controller
xmax = np.array([2, 0.35,2.,2.])                       # State constraints, [x, theta, x_dot, theta_dot]
xmin = -xmax
set_pt = np.zeros(n)                                    # Desired trajectories (initialization)
Q = sc.sparse.diags([0,0,0,0])                          # MPC state penalty matrix (trajectory generation)
QN = sc.sparse.diags([1e3,1e3,1e1,1e1])                 # MPC final state penalty matrix (trajectory generation)
R = sc.sparse.eye(m)                                    # MPC control penalty matrix (trajectory generation)
umax = np.array([10])                                   # MPC actuation constraint (trajectory generation)
umin = -umax

#EDMD parameters:
alpha_edmd = 2e-1
tune_mdl_edmd = False

#Bilinear EDMD parameters:
alpha_bedmd = 2e-1
tune_mdl_bedmd = False

# Prediction performance evaluation parameters:
n_traj_test = 25
mpc_controller = MPCController(nominal_sys,n_pred,dt,umin,umax,xmin,xmax,Q,R,QN,set_pt)
x_0_max = np.array([xmax[0], xmax[1], 0.2, 0.2])
x_d = np.empty((n_traj,n_pred+1,n))
xs = np.empty((n_traj,n_pred+1,n))
us = np.empty((n_traj, n_pred, m))

#Closed loop performance evaluation parameters:


n_cols = 5
plt.figure(figsize=(12,12*n_traj/(n_cols**2)))
for ii in range(n_traj):
    x_0 = np.asarray([rand.uniform(l,u) for l, u in zip(-x_0_max, x_0_max)])
    while abs(x_0[0]) < 1.25:
        x_0 = np.asarray([rand.uniform(l,u) for l, u in zip(-x_0_max, x_0_max)])

    mpc_controller.eval(x_0, 0)
    x_d[ii,:,:] = mpc_controller.parse_result().transpose()

    output = CartPoleOutput(cart_pole, x_d[ii,:,:], t_eval)
    pd_controller = PDController(output, K_p, K_d)
    perturbed_pd_controller = PerturbedController(cart_pole, pd_controller, noise_var)
    xs[ii,:,:], us[ii,:,:] = cart_pole.simulate(x_0, perturbed_pd_controller, t_eval)

    plt.subplot(int(np.ceil(n_traj/n_cols)),n_cols,ii+1)
    plt.plot(t_eval, xs[ii,:,0], 'b', label='$x$')
    plt.plot(t_eval, xs[ii,:,1], 'r', label='$\\theta$')
    plt.plot(t_eval, x_d[ii,:,0], '--b', label='$x_d$')
    plt.plot(t_eval, x_d[ii,:,1], '--r', label='$\\theta_d$')
plt.suptitle('Training data \nx-axis: time (sec), y-axis: state value, $x$ - blue, $x_d$ - dotted blue, $\\theta$ - red, $\\theta_d$ - dotted red',y=0.94)
plt.show()


#EDMD:
cart_pole_basis = CartPoleBasis(n, 0)
cart_pole_basis.construct_basis()
cart_pole_features = preprocessing.FunctionTransformer(cart_pole_basis.basis)
cart_pole_features.fit(np.zeros((1,n)))
n_lift_edmd = cart_pole_features.transform((np.zeros((1,n)))).shape[1]
C_edmd = np.zeros((n,n_lift_edmd))
C_edmd[:,1:n+1] = np.eye(n)

optimizer_edmd = linear_model.MultiTaskLasso(alpha=alpha_edmd, fit_intercept=False, selection='random')
cv_edmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
standardizer_edmd = preprocessing.StandardScaler(with_mean=False)

model_edmd = Edmd(n, m, cart_pole_basis.basis, n_lift_edmd, n_traj, optimizer_edmd, cv=cv_edmd, standardizer=standardizer_edmd, C=C_edmd)
X_edmd, y_edmd = model_edmd.process(xs, us, np.tile(t_eval,(n_traj,1)))
model_edmd.fit(X_edmd, y_edmd, cv=tune_mdl_edmd, override_kinematics=True)
sys_edmd = LinearLiftedDynamics(model_edmd.A, model_edmd.B, model_edmd.C)
#print(model_edmd.cv.alpha_)

#Bilinear EDMD:
n_lift_bedmd = n_lift_edmd
C_bedmd = np.zeros((n,n_lift_bedmd))
C_bedmd[:,1:n+1] = np.eye(n)

basis_bedmd = lambda x: cart_pole_features.transform(x)
optimizer_bedmd = linear_model.MultiTaskLasso(alpha=alpha_bedmd, fit_intercept=False, selection='random')
cv_bedmd = linear_model.MultiTaskLassoCV(fit_intercept=False, n_jobs=-1, cv=3, selection='random')
standardizer_bedmd = preprocessing.StandardScaler(with_mean=False)

model_bedmd = BilinearEdmd(n, m, basis_bedmd, n_lift_bedmd, n_traj, optimizer_bedmd, cv=cv_bedmd, standardizer=standardizer_bedmd, C=C_bedmd)
X_bedmd, y_bedmd = model_bedmd.process(xs, us, np.tile(t_eval,(n_traj,1)))
model_bedmd.fit(X_bedmd, y_bedmd, cv=tune_mdl_bedmd, override_kinematics=True)
sys_bedmd = BilinearLiftedDynamics(n_lift_bedmd, m, model_bedmd.A, model_bedmd.B, C_bedmd, basis_bedmd)
#print(model_bedmd.cv.alpha_)


#Compare open loop performance:
t_eval_test = t_eval[:101]
xs_test = np.empty((n_traj_test, t_eval_test.shape[0], n))
xs_nom_test = np.empty((n_traj_test, t_eval_test.shape[0]-1, n))
xs_edmd_test = np.empty((n_traj_test, t_eval_test.shape[0]-1, n))
xs_bedmd_test = np.empty((n_traj_test, t_eval_test.shape[0]-1, n))
us_test = np.empty((n_traj_test, t_eval_test.shape[0]-1, m))

for ii in range(n_traj_test):
    x_0 = np.asarray([rand.uniform(l,u)  for l,u in zip(-x_0_max, x_0_max)])
    xs_test[ii,:,:], us_test[ii,:,:] = cart_pole.simulate(x_0, perturbed_pd_controller, t_eval_test)
    ol_controller = OpenLoopController(sys_bedmd, us_test[ii,:,:], t_eval_test[:-1])

    xs_nom_test[ii,:,:], _ = nominal_sys.simulate(x_0, ol_controller, t_eval_test[:-1])

    z_0_edmd = cart_pole_basis.basis(np.atleast_2d(x_0)).squeeze()
    zs_edmd_tmp, _ = sys_edmd.simulate(z_0_edmd, ol_controller, t_eval_test[:-1])
    xs_edmd_test[ii,:,:] = np.dot(C_edmd, zs_edmd_tmp.T).T

    z_0_bedmd = basis_bedmd(np.atleast_2d(x_0)).squeeze()
    zs_bedmd_tmp, _ = sys_bedmd.simulate(z_0_bedmd, ol_controller, t_eval_test[:-1])
    xs_bedmd_test[ii,:,:] = np.dot(C_bedmd, zs_bedmd_tmp.T).T

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

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
ylabels = ['$e_{x}$', '$e_{\\theta}$', '$e_{\dot{x}}$', '$e_{\dot{\\theta}}$']
fig.suptitle('Open loop predicition error of EDMD and bilinear EDMD models', y=1.025, fontsize=18)
for ax, err_nom_mean, err_nom_std, err_edmd_mean, err_edmd_std, err_bedmd_mean, err_bedmd_std, ylabel in zip(axs.flatten(), error_nom_mean, error_nom_std, error_edmd_mean, error_edmd_std, error_bedmd_mean, error_bedmd_std, ylabels):
    ax.plot(t_eval_test[:-1], err_nom_mean, linewidth=3, label='mean, nominal')
    ax.fill_between(t_eval_test[:-1], err_nom_mean-err_nom_std, err_nom_mean+err_nom_std, alpha=0.2, label='std, nominal')
    ax.plot(t_eval_test[:-1], err_edmd_mean, linewidth=3, label='mean, EDMD')
    ax.fill_between(t_eval_test[:-1], err_edmd_mean-err_edmd_std, err_edmd_mean+err_edmd_std, alpha=0.2, label='std, EDMD')
    ax.plot(t_eval_test[:-1], err_bedmd_mean, linewidth=3, label='mean, bEDMD')
    ax.fill_between(t_eval_test[:-1], err_bedmd_mean-err_bedmd_std, err_bedmd_mean+err_bedmd_std, alpha=0.2, label='std, bEDMD')
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid()
    ax.set_xlabel('$t$ (sec)', fontsize=16)

ax.legend()
plt.tight_layout()
plt.show()
print('Lifting dimension EDMD: ', n_lift_edmd, 'Lifting dimension bEDMD: ', n_lift_bedmd*(m+1))
print('MSE nominal: ', mse_nom, '\nMSE EDMD: ', mse_edmd, '\nMSE bilinear EDMD: ', mse_bedmd, '\nImprovement: ', (1-mse_bedmd/mse_edmd)*100, ' percent')

#Compare closed loop performance:
# MPC parameters:
x_0 = np.array([2., 0., 0., 0.])
umax = np.array([20.])
umin = -umax
xmax = np.array([10., 10., 10., 10.])
xmin = -xmax
q, r = 1e2, 1
Q = q*np.identity(4)
R = r*np.identity(1)
n_pred = 100
set_pt = np.zeros(n)
x_d = np.tile(set_pt.reshape(-1,1), (1,t_eval.shape[0]))

# Design MPC for linearized nominal model:
lin_sys = sc.signal.StateSpace(A_nom, B_nom, np.eye(n), np.zeros((n,m)))
lin_sys_d = lin_sys.to_discrete(dt)
A_d, B_d = lin_sys_d.A, lin_sys_d.B
controller_nom = LinearMpcController(n, m, n, n_pred, lin_sys_d, xmin, xmax, umin, umax, Q, Q, R, set_pt)
controller_nom.construct_controller()

# Design MPC for lifted linear EDMD model:
controller_edmd = MPCControllerDense(sys_edmd, n_pred, dt, umin, umax, xmin, xmax, Q, R, Q, x_d, lifting=True,
                                     edmd_object=model_edmd, plotMPC=False, name='EDMD')

# Design MPC for lifted bilinear EDMD model:
k = m
Q_fl = q*np.eye(int(2*n_lift_bedmd))
R_fl = r*np.eye(n_lift_bedmd)
C_h = C_bedmd[1,:]
#C_stacked = np.zeros((int(2*k), int(2*n_lift_bedmd)))
#C_stacked[:k, :n_lift_bedmd] = C_h
#C_stacked[k:, n_lift_bedmd:] = C_h

f_eta = np.concatenate((np.zeros((n_lift_bedmd,n_lift_bedmd)), np.eye(n_lift_bedmd)), axis=1)
f_eta_dot = np.concatenate((sys_bedmd.F@sys_bedmd.F, np.zeros((n_lift_bedmd,n_lift_bedmd))), axis=1)
F_lin = np.concatenate((f_eta, f_eta_dot), axis=0)
G_lin = np.concatenate((np.zeros((n_lift_bedmd,n_lift_bedmd)), np.eye(n_lift_bedmd)), axis=0)
fb_sys = sc.signal.StateSpace(F_lin, G_lin, np.eye(int(2*n_lift_bedmd)), np.zeros((int(2*n_lift_bedmd),n_lift_bedmd)))
fb_sys_d = fb_sys.to_discrete(dt)

controller_bedmd = BilinearMpcController(n, m, k, n_lift_bedmd, n_pred, fb_sys_d, sys_bedmd, C_bedmd, C_h, xmin, xmax, umin, umax,
                                          Q_fl, Q_fl, R_fl, set_pt.reshape((1,-1)))
controller_bedmd.construct_controller()

xs_mpc_nom, us_mpc_nom = cart_pole.simulate(x_0, controller_nom, t_eval)
#xs_mpc_edmd, us_mpc_edmd = cart_pole.simulate(x_0, controller_edmd, t_eval)
xs_mpc_bedmd, us_mpc_bedmd = cart_pole.simulate(x_0.reshape((1,-1)), controller_bedmd, t_eval)

cost_nom = np.cumsum(np.diag(xs_mpc_nom[1:,:] @ Q @ xs_mpc_nom[1:,:].T) + np.diag(us_mpc_nom @ R @ us_mpc_nom.T))
#cost_edmd = np.cumsum(np.diag(xs_mpc_edmd[1:,:] @ Q @ xs_mpc_edmd[1:,:].T) + np.diag(us_mpc_edmd @ R @ us_mpc_edmd.T))
cost_bedmd = np.cumsum(np.diag(xs_mpc_bedmd[1:,:] @ Q @ xs_mpc_bedmd[1:,:].T) + np.diag(us_mpc_bedmd @ R @ us_mpc_bedmd.T))

_, axs = plt.subplots(2, 2, figsize=(12, 8))
ylabels = ['$x$', '$\\theta$']
legend_labels=['Nominal (linearized)', 'EDMD', 'bEDMD']

for ax, data_nom, data_edmd, data_bedmd, ylabel in zip(axs[:-1].flatten(), xs_mpc_nom.T, xs_mpc_edmd.T, xs_mpc_bedmd.T, ylabels):
    ax.plot(t_eval, data_nom, linewidth=3, label=legend_labels[0])
    #ax.plot(t_eval, data_edmd, linewidth=3, label=legend_labels[1])
    ax.plot(t_eval, data_bedmd, linewidth=3, label=legend_labels[2])
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid()
    ax.set_xlabel('$t$ (sec)', fontsize=16)
    ax.legend()

ax = axs[1,0]
ax.plot(t_eval[:-1], us_mpc_nom[:,0], linewidth=3, label='$u$, '+ legend_labels[0])
#ax.plot(t_eval[:-1], us_mpc_edmd[:,0], linewidth=3, label='$u$, '+ legend_labels[1])
ax.plot(t_eval[:-1], us_mpc_bedmd[:,0], linewidth=3, label='$u$, '+ legend_labels[2])
ax.grid()
ax.set_xlabel('$t$ (sec)', fontsize=16)
ax.set_ylabel('$u$', fontsize=16)
ax.legend()

ax = axs[1,1]
ax.plot(t_eval[:-1], cost_nom/cost_nom[-1], linewidth=3, label=legend_labels[0])
#ax.plot(t_eval[:-1], cost_edmd/cost_nom[-1], linewidth=3, label=legend_labels[1])
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
plt.hist(controller_edmd.comp_time)
#plt.xlim(1e-4, 1e-1)
plt.subplot(3,1,3)
plt.hist(controller_bedmd.comp_time)
#plt.xlim(1e-4, 1e-1)
plt.tight_layout()
plt.show()

