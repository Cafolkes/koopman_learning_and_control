from numpy import dot

from .affine_dynamics import AffineDynamics
from .linearizable_dynamics import LinearizableDynamics
from .system_dynamics import SystemDynamics

class LinearSystemDynamics(SystemDynamics, AffineDynamics, LinearizableDynamics):
    """Class for linear dynamics of the form x_dot = A * x + B * u."""

    def __init__(self, A, B):
        """Create a LinearSystemDynamics object.

        Inputs:
        State matrix, A: numpy array
        Input matrix, B: numpy array
        """

        n, m = B.shape
        assert A.shape == (n, n)

        SystemDynamics.__init__(self, n, m)
        self.A = A
        self.B = B

    def drift(self, x, t):
        return dot(self.A, x)

    def act(self, x, t):
        return self.B

    def linear_system(self):
        return self.A, self.B