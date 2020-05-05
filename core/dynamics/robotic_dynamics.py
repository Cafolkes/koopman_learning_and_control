from matplotlib.pyplot import figure
from numpy import array, concatenate, dot, reshape, zeros, atleast_1d
from numpy.linalg import solve

from .affine_dynamics import AffineDynamics
from .pd_dynamics import PDDynamics
from .system_dynamics import SystemDynamics

class RoboticDynamics(SystemDynamics, AffineDynamics, PDDynamics):
    """Abstract class for unconstrained Euler-Lagrange systems.

    State represented as x = (q, q_dot), where q are generalized coordinates and
    q_dot are corresponding rates.

    Dynamics represented as D(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = B(q) * u + F_ext(q, q_dot).

    Override D, C, B, U, G.
    """

    def __init__(self, n, m):
        """Create a RoboticDynamics object.

        Inputs:
        Configuration space dimension, n: int
        Action space dimension, m: int
        """

        SystemDynamics.__init__(self, 2 * n, m)
        self.k = n

    def D(self, q):
        """Compute positive-definite inertia matrix.

        Inputs:
        Coordinates, q: numpy array

        Outputs:
        Positive-definite inertia matrix: numpy array
        """

        pass

    def C(self, q, q_dot):
        """Compute Coriolis terms.

        Inputs:
        Coordinates, q: numpy array
        Coordinate rates, q_dot, numpy array

        Outputs:
        Coriolis terms matrix: numpy array
        """

        pass

    def B(self, q):
        """Compute actuation terms.

        Inputs:
        Coordinates, q: numpy array

        Outputs:
        Actuation matrix: numpy array
        """

        pass

    def U(self, q):
        """Compute potential energy.

        Inputs:
        Coordinates, q: numpy array

        Outputs:
        Potential energy: float
        """

        pass

    def G(self, q):
        """Compute potential energy gradient.

        Inputs:
        Coordinates, q: numpy array

        Outputs:
        Potential energy gradient: numpy array
        """

        pass

    def F_ext(self, q, qdot):
        """Compute non-conservative generalized forces.

        Inputs:
        Coordinates, q: numpy array
        Coordinate rates, q_dot: numpy array

        Outputs:
        Generalized forces: numpy array
        """

        return zeros(self.k)

    def T(self, q, q_dot):
        """Compute kinetic energy.

        Inputs:
        Coordinates, q: numpy array
        Coordinate rates, q_dot: numpy array

        Outputs:
        Kinetic energy: float
        """

        return dot(q_dot, self.D(q), q_dot) / 2

    def H(self, q, q_dot):
        """Compute Coriolis and potential terms.

        Inputs:
        Coordinates, q: numpy array
        Coordinate rates, q_dot: numpy array

        Outputs:
        Coriolis and potential terms: numpy array
        """

        return dot(self.C(q, q_dot), q_dot) + self.G(q) -  self.F_ext(q, q_dot)

    def drift(self, x, t):
        q, q_dot = reshape(x, (2, -1))
        return concatenate([q_dot, atleast_1d(-solve(self.D(q), self.H(q, q_dot)).squeeze())])

    def act(self, x, t):
        q = self.proportional(x, t)
        return concatenate([zeros((self.k, self.m)), solve(self.D(q), self.B(q))])

    def proportional(self, x, t):
        return self.eval(x, t)[:self.k]

    def derivative(self, x, t):
        return self.eval(x, t)[self.k:]

    def plot_coordinates(self, ts, qs, fig=None, ax=None, labels=None):
        if labels is None:
            labels = [f'$q_{i}$' for i in range(self.k)]

        return self.plot_timeseries(ts, qs, fig, ax, 'Coordinates', labels)

    def plot(self, xs, us, ts, fig=None, coordinate_labels=None, action_labels=None):
        if fig is None:
            fig = figure(figsize=(12, 6), tight_layout=True)

        qs = xs[:, :self.k]

        coordinate_ax = fig.add_subplot(1, 2, 1)
        fig, coordinate_ax = self.plot_coordinates(ts, qs, fig, coordinate_ax, coordinate_labels)

        action_ax = fig.add_subplot(1, 2, 2)
        fig, action_ax = self.plot_actions(ts, us, fig, action_ax, action_labels)

        return fig, (coordinate_ax, action_ax)
