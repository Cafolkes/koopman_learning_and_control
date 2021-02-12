from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, savefig, ylim, ylabel, xlabel
from numpy import array, gradient, zeros, tile
import numpy as np
import torch
from torch.nn.utils import prune


def plot_trajectory(X, X_d, U, U_nom, t, display=True, save=False, filename=''):
    """ Plots the position, velocity and control input

    # Inputs:
    - state X, numpy 2d array [number of time steps 'N', number of states 'n'] 
    - desired state X_d, numpy 2d array [number of time steps 'N', number of states 'n']
    - control input U, numpy 2d array [number of time steps, number of inputs]
    - nominal control input U_nom, numpy 2d array [number of time steps, number of inputs]
    - time t, numpy 1d array [number of time steps 'N']
    """
    figure()
    subplot(2, 1, 1)
    plot(t, X[:,0], linewidth=2, label='$x$')
    plot(t, X[:,2], linewidth=2, label='$\\dot{x}$')
    plot(t, X_d[:,0], '--', linewidth=2, label='$x_d$')
    plot(t, X_d[:,2], '--', linewidth=2, label='$\\dot{x}_d$')
    title('Trajectory Tracking with PD controller')
    legend(fontsize=12)
    grid()
    subplot(2, 1, 2)
    plot(t[:-1], U[:,0], label='$u$')
    plot(t[:-1], U_nom[:,0], label='$u_{nom}$')
    legend(fontsize=12)
    grid()
    if display:
        show()
    if save:
        savefig(filename)

def plot_trajectory_ep(X, X_d, U, U_nom, t, display=True, save=False, filename='', episode=0):
    # Plot the first simulated trajectory
    figure(figsize=(4.7,5.5))
    subplot(3, 1, 1)
    title('Trajectory tracking with MPC, episode ' + str(episode))
    plot(t, X[0,:], linewidth=2, label='$x$')
    plot(t, X[2,:], linewidth=2, label='$\\dot{x}$')
    plot(t, X_d[0,:], '--', linewidth=2, label='$x_d$')
    plot(t, X_d[2,:], '--', linewidth=2, label='$\\dot{x}_d$')
    legend(fontsize=10, loc='lower right', ncol=4)
    ylim((-4.5, 2.5))
    ylabel('$x$, $\\dot{x}$')
    grid()
    subplot(3, 1, 2)
    plot(t, X[1,:], linewidth=2, label='$\\theta$')
    plot(t, X[3,:], linewidth=2, label='$\\dot{\\theta}$')
    plot(t, X_d[1,:], '--', linewidth=2, label='$\\theta_d$')
    plot(t, X_d[3,:], '--', linewidth=2, label='$\\dot{\\theta}_d$')
    legend(fontsize=10, loc='lower right', ncol=4)
    ylim((-2.25,1.25))
    ylabel('$\\theta$, $\\dot{\\theta}$')
    grid()

    subplot(3, 1, 3)
    plot(t[:-1], U[0,:], label='$u$')
    plot(t[:-1], U_nom[0,:], label='$u_{nom}$')
    legend(fontsize=10, loc='upper right', ncol=2)
    ylabel('u')
    xlabel('Time (sec)')
    grid()
    if save:
        savefig(filename)
    if display:
        show()

def differentiate_vec(xs, ts):
    """differentiate_vec Numerically differencitate a vector
    
    Arguments:
        xs {numpy array [Nt,Ns]} -- state as a block matrix
        ts {numpy array [Nt,]} -- time vecotr
    
    Keyword Arguments:
        L {integer} -- differenciation order, only L=3 (default: {3})
    
    Returns:
        numpy array [Nt,Ns] -- numerical derivative
    """
    assert(xs.shape[0] == ts.shape[0])
    return array([differentiate(xs[:,ii], ts) for ii in range(xs.shape[1])]).transpose()

def differentiate(xs, ts):
    """differentiate     Compute the discrete derivative of a Python function
    f on [a,b] using n intervals. Internal points apply
    a centered difference, while end points apply a one-sided
    difference. Vectorized version.
    
    
    Arguments:
        xs {numpy array [Nt,]} -- state as a vector
        ts {numpy array [Nt,]} -- time vecotr
    
    Returns:
        numpy array [Nt,] -- numerical derivative
    """

    dt = ts[1] - ts[0]
    dx = gradient(xs, dt, edge_order=2)
    return dx

def rbf(X, C, type='gauss', eps=1.):
    """rbf Radial Basis Function
    
    Arguments:
        X {numpy array [Ns,Nz]} -- state
        C {numpy array [Ns,Nc]} -- centers. 
    
    Keyword Arguments:
        type {str} -- RBF type (default: {'gauss'})
        eps {float} -- epsilon for gauss (default: {1.})
    
    Returns:
        numpy array [] -- [description]
    """
    N = X.shape[1]
    n = X.shape[0]
    Cbig = C
    Y = zeros((C.shape[1],N))
    for ii in range(C.shape[1]):
        C = Cbig[:,ii]
        C = tile(C.reshape((C.size,1)), (1, N))
        r_sq = np.sum((X-C)**2,axis=0)
        if type == 'gauss':
            y = np.exp(-eps**2*r_sq)

        Y[ii,:] = y

    return Y

def calc_koopman_modes(A, output, x_0, t_eval):
    d_w, w = np.linalg.eig(A.T)
    d, v = np.linalg.eig(A)

    sort_ind_w = np.argsort(np.abs(d_w))
    w = w[:, sort_ind_w]
    d_w = d_w[sort_ind_w]

    sort_ind_v = np.argsort(np.abs(d))
    v = v[:, sort_ind_v]
    d = d[sort_ind_v]

    non_zero_cols = np.where(np.diag(np.dot(w.T,v)) > 0)
    w = w[:,non_zero_cols].squeeze()
    v = v[:,non_zero_cols].squeeze()
    d = d[non_zero_cols].squeeze()

    eigfuncs = lambda x, t: np.divide(np.dot(w.T, output(x, t)), np.diag(np.dot(w.T, v)))
    eigvals = np.exp(d)

    koop_mode = lambda t: [eigvals[ii] ** t * eigfuncs(x_0, t)[ii] * v[:, ii] for ii in range(d.size)]
    xs_koop = array([koop_mode(t) for t in t_eval])  # Evolution of each mode [n_time, n_modes, n_outputs]

    return xs_koop, v, w, d

def calc_reduced_mdl(model):
    A = model.A
    C = model.C
    useful_rows = np.argwhere(np.abs(C) > 0)
    useful_rows = np.unique(useful_rows[:,1])
    useful_inds = np.argwhere(np.abs(A[useful_rows,:]) > 0)
    useful_cols = np.unique(useful_inds[:,1])
    useful_coords = np.unique(np.concatenate((useful_rows, useful_cols)))


    A_red = model.A[useful_coords, :]
    A_red = A_red[:, useful_coords]
    if model.B is not None:
        B_red = model.B[useful_coords,:]
    else:
        B_red = None
    C_red = C[:,useful_coords]

    return A_red, B_red, C_red, useful_coords

class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold
