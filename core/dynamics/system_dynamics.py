from matplotlib.pyplot import figure
from numpy import array, zeros
from scipy.integrate import solve_ivp

from .dynamics import Dynamics
from ..util import default_fig

class SystemDynamics(Dynamics):
    """Abstract dynamics class for simulation.

    Override eval_dot.
    """

    def __init__(self, n, m):
        """Create a SystemDynamics object.

        Inputs:
        Number of states, n: int
        Number of actions, m: int
        """

        self.n = n
        self.m = m

    def eval(self, x, t):
        return x

    def step(self, x_0, u_0, t_0, t_f, atol=1e-6, rtol=1e-6):
        """Simulate system from initial state with constant action over a
        time interval.

        Approximated using Runge-Kutta 4,5 solver.

        Inputs:
        Initial state, x_0: numpy array
        Control action, u_0: numpy array
        Initial time, t_0: float
        Final time, t_f: float
        Absolute tolerance, atol: float
        Relative tolerance, rtol: float

        Outputs:
        State at final time: numpy array
        """

        x_dot = lambda t, x: self.eval_dot(x, u_0, t)
        t_span = [t_0, t_f]
        res = solve_ivp(x_dot, t_span, x_0, atol=atol, rtol=rtol)
        return res.y[:, -1]

    def simulate(self, x_0, controller, ts, processed=True, atol=1e-6, rtol=1e-6):
        """Simulate system from initial state with specified controller.

        Approximated using Runge-Kutta 4,5 solver.

        Actions computed at time steps and held constant over sample period.

        Inputs:
        Initial state, x_0: numpy array
        Control policy, controller: Controller
        Time steps, ts: numpy array
        Flag to process actions, processed: bool
        Absolute tolerance, atol: float
        Relative tolerance, rtol: float

        Outputs:
        State history: numpy array
        Action history: numpy array
        """

        N = len(ts)
        xs = zeros((N, self.n))
        us = [None] * (N - 1)

        controller.reset()

        xs[0] = x_0
        for j in range(N - 1):
            x = xs[j]
            t = ts[j]
            u = controller.eval(x, t)
            us[j] = u
            u = controller.process(u)
            xs[j + 1] = self.step(x, u, t, ts[j + 1])

        if processed:
            us = array([controller.process(u) for u in us])

        return xs, us

    def plot_timeseries(self, ts, data, fig=None, ax=None, title=None, labels=None):
        fig, ax = default_fig(fig, ax)

        if title is not None:
            ax.set_title(title, fontsize=16)

        ax.set_xlabel('$t$ (sec)', fontsize=16)
        ax.plot(ts, data, linewidth=3)

        if labels is not None:
            ax.legend(labels, fontsize=16)

        return fig, ax

    def plot_states(self, ts, xs, fig=None, ax=None, labels=None):
        if labels is None:
            labels = [f'$x_{i}$' for i in range(self.n)]

        return self.plot_timeseries(ts, xs, fig, ax, 'States', labels)

    def plot_actions(self, ts, us, fig=None, ax=None, labels=None):
        if labels is None:
            labels = [f'$u_{j}$' for j in range(self.m)]

        return self.plot_timeseries(ts[:-1], us, fig, ax, 'Actions', labels)

    def plot(self, xs, us, ts, fig=None, state_labels=None, action_labels=None):
        if fig is None:
            fig = figure(figsize=(12, 6), tight_layout=True)

        state_ax = fig.add_subplot(1, 2, 1)
        fig, state_ax = self.plot_states(ts, xs, fig, state_ax, state_labels)

        action_ax = fig.add_subplot(1, 2, 2)
        fig, action_ax = self.plot_actions(ts, us, fig, action_ax, action_labels)

        return fig, (state_ax, action_ax)
