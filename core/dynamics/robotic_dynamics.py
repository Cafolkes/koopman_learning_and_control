from numpy import array, concatenate, dot, reshape, zeros
from numpy.linalg import solve

from .fb_lin_dynamics import FBLinDynamics
from .pd_dynamics import PDDynamics

class RoboticDynamics(FBLinDynamics, PDDynamics):
    """Abstract class for Euler-Lagrange systems.

    State represented as x = (q, q_dot), where q are generalized coordinates and
    q_dot are corresponding rates.

    Dynamics represented as D(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = B * u.

    Override D, C, U, G.
    """

    def __init__(self, B):
        """Create a RoboticDynamics object.

        Inputs:
        Full rank static actuation matrix, B: numpy array
        """

        self.B = B
        self.k, self.m = B.shape
        relative_degrees = [2] * self.k
        perm = concatenate([array([j, j + self.k]) for j in range(self.k)])
        FBLinDynamics.__init__(self, relative_degrees, perm)

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

        return dot(self.C(q, q_dot), q_dot) + self.G(q)

    def drift(self, x, t):
        q, q_dot = reshape(x, (2, -1))
        return concatenate([q_dot, -solve(self.D(q), self.H(q, q_dot))])

    def act(self, x, t):
        q = self.proportional(x, t)
        return concatenate([zeros((self.k, self.m)), solve(self.D(q), self.B)])

    def proportional(self, x, t):
        return self.eval(x, t)[:self.k]

    def derivative(self, x, t):
        return self.eval(x, t)[self.k:]