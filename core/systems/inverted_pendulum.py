from matplotlib.pyplot import figure
from numpy import array, cos, sin

from core.dynamics import FullyActuatedRoboticDynamics
from core.util import default_fig

class InvertedPendulum(FullyActuatedRoboticDynamics):
    def __init__(self, m, l, g=9.81):
        FullyActuatedRoboticDynamics.__init__(self, 1, 1)
        self.params = m, l, g

    def D(self, q):
        m, l, _ = self.params
        return array([[m * (l ** 2)]])

    def C(self, q, q_dot):
        return array([[0]])

    def U(self, q):
        m, l, g = self.params
        theta, = q
        return m * g * l * cos(theta)

    def G(self, q):
        m, l, g = self.params
        theta, = q
        return array([-m * g * l * sin(theta)])

    def B(self, q):
        return array([[1]])

    def plot_states(self, ts, xs, fig=None, ax=None, labels=None):
        fig, ax = default_fig(fig, ax)

        ax.set_title('States', fontsize=16)
        ax.set_xlabel('$\\theta$ (rad)', fontsize=16)
        ax.set_ylabel('$\\dot{\\theta}$ (rad / sec)', fontsize=16)
        ax.plot(*xs.T, linewidth=3)

        return fig, ax

    def plot_physical(self, ts, xs, fig=None, ax=None, skip=1):
        fig, ax = default_fig(fig, ax)

        _, l, g = self.params
        thetas = xs[:, 0]
        rs = l * array([sin(thetas), cos(thetas)])
        zs = 0 * thetas[::skip]

        ax.set_title('Physical space', fontsize=16)
        ax.set_xlabel('$x$ (m)', fontsize=16)
        ax.set_ylabel('$z$ (m)', fontsize=16)
        ax.plot([zs, rs[0, ::skip]], [zs, rs[1, ::skip]], 'k')
        ax.plot(*rs, linewidth=3)
        ax.axis('equal')

        return fig, ax

    def plot(self, xs, us, ts, fig=None, action_labels=None, skip=1):
        if fig is None:
            fig = figure(figsize=(12, 6), tight_layout=True)

        physical_ax = fig.add_subplot(1, 2, 1)
        fig, physical_ax = self.plot_physical(ts, xs, fig, physical_ax, skip)

        action_ax = fig.add_subplot(1, 2, 2)
        fig, action_ax = self.plot_actions(ts, us, fig, action_ax, action_labels)

        return fig, (physical_ax, action_ax)
