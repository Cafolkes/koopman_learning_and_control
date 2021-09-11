import matplotlib.pyplot as plt
import numpy as np
import random as rand
from sklearn import preprocessing
from ..core.dynamics import ConfigurationDynamics
from ..core.controllers import ConstantController, PDController
from ..koopman_core.controllers import PerturbedController, OpenLoopController

class KoopPdOutput(ConfigurationDynamics):
    def __init__(self, dynamics, xd, n, m):
        ConfigurationDynamics.__init__(self, dynamics, 1)
        self.xd = xd
        self.n = n
        self.m = m

    def proportional(self, x, t):
        q = x[:int(self.n/2)]
        q_d = self.xd[:int(self.n/2)]

        return q - q_d

    def derivative(self, x, t):
        q_dot = x[int(self.n/2):]
        q_dot_d = self.xd[int(self.n/2):]

        return q_dot - q_dot_d


def run_experiment(system, n, n_traj, n_pred, t_eval, x0_max, plot_experiment_data=False, n_cols_plot=10, m=None, K_p=None, K_d=None, noise_var=None):
    xs = np.empty((n_traj, n_pred + 1, n))
    if m is not None:
        us = np.empty((n_traj, n_pred, m))

    plt.figure(figsize=(12, 12 * n_traj / (n_cols_plot ** 2)))
    for ii in range(n_traj):
        x0 = np.asarray([rand.uniform(l, u) for l, u in zip(-x0_max, x0_max)])
        set_pt_dc = np.zeros(n)

        if m is None and K_p is None and K_d is None:
            ctrl = ConstantController(system, 0.)
            xs[ii, :, :], _ = system.simulate(x0, ctrl, t_eval)
        else:
            output = KoopPdOutput(system, set_pt_dc, n, m)
            pd_controller = PDController(output, K_p, K_d)
            ctrl = PerturbedController(system, pd_controller, noise_var)
            xs[ii, :, :], us[ii, :, :] = system.simulate(x0, ctrl, t_eval)

        if plot_experiment_data:
            plt.subplot(int(np.ceil(n_traj / n_cols_plot)), n_cols_plot, ii + 1)
            clrs = ['b', 'g', 'r']
            for jj in range(min(int(n/2), 3)):
                plt.plot(t_eval, xs[ii, :, jj], clrs[jj])
                plt.plot(t_eval, set_pt_dc[jj] * np.ones_like(xs[ii, :, 0]), '--' + clrs[jj])

    if plot_experiment_data:
        plt.suptitle(
            'Training data \nx-axis: time (sec), y-axis: state value, $x_1$ - blue, $\tau_1$ - dotted blue, $x_2$ - green, $\\tau_2$ - dotted green',
            y=0.94)
        plt.show()

    if m is None and K_p is None and K_d is None:
        return xs, t_eval
    else:
        return xs, us, t_eval

def evaluate_ol_pred(sys, xs, t_eval, us=None, n_eval_states=None):
    n_traj = xs.shape[0]
    traj_length = xs.shape[1]
    n = xs.shape[2]
    if n_eval_states is None:
        n_eval_states = n

    if us is not None:
        us_scaled = us.copy()
    xs_pred = np.empty((n_traj, traj_length-1, n_eval_states))
    for ii in range(n_traj):
        if us is None:
            ctrl = ConstantController(sys, 0.)
        else:
            if sys.standardizer_u is not None:
                us_scaled[ii, :, :] = sys.standardizer_u.transform(us[ii, :, :])
            ctrl = OpenLoopController(sys, us_scaled[ii, :, :], t_eval[ii, :-1])

        x0 = xs[ii, 0, :]
        z0 = sys.basis(np.atleast_2d(x0)).squeeze()
        zs_tmp, _ = sys.simulate(z0, ctrl, t_eval[ii, :-1])
        xs_pred[ii, :, :] = np.dot(sys.C, zs_tmp.T).T

        # TODO: Evaluate need for standardization
        if sys.standardizer_x is not None:
            if n_eval_states is not None:
                standardizer_x = preprocessing.StandardScaler()
                standardizer_x.n_features_in_ = n_eval_states
                standardizer_x.scale_ = sys.standardizer_x.scale_[:n_eval_states]
                standardizer_x.var_ = sys.standardizer_x.var_[:n_eval_states]
                standardizer_x.mean_ = sys.standardizer_x.mean_[:n_eval_states]
            else:
                standardizer_x = sys.standardizer_x

            xs_pred[ii, :, :] = standardizer_x.inverse_transform(xs_pred[ii, :, :])
            if sys.standardizer_x.with_mean:
                xs_pred[ii,: , :int(n/2)] += np.multiply(t_eval[ii, :-1], sys.standardizer_x.mean_[int(n/2):].reshape(-1,1)).T

    error = xs[:, :-1, :xs_pred.shape[2]] - xs_pred
    mse = np.mean(np.square(error))
    std = np.std(error)

    return error, mse, std

def fit_standardizer(data, standardizer, flattened=False):
    if flattened:
        data_flat = data
    else:
        n_traj, traj_length, n = data.shape
        data_flat = data.T.reshape((n, n_traj * traj_length), order='F').T

    standardizer.fit(data_flat)

    return standardizer

def split_dataset(x_test, u_test, t_test, dataset_length):
    x_tests, u_tests, t_tests = [], [], []
    for x, u, t in zip(x_test, u_test, t_test):
        cur_index = 0
        while cur_index+dataset_length < t.shape[0]:
            x_tests.append(x[cur_index:cur_index+dataset_length, :])
            u_tests.append(u[cur_index:cur_index + dataset_length-1, :])
            t_tests.append(t[cur_index:cur_index + dataset_length] - t[cur_index])
            cur_index += dataset_length

    return np.array(x_tests), np.array(u_tests), np.array(t_tests)



