from numpy import dot, reshape, zeros
from numpy.linalg import solve

from .controller import Controller

class EnergyController(Controller):
    """Class for energy-based controllers."""

    def __init__(self, robotic_dynamics, K_p, K_d, q_d=None):
        """Create an EnergyController object.

        Requires fully actuated robotic system with relative degree 2.

        Inputs:
        Robotic system, robotic_dynamics: RoboticDynamics
        Positive-definite proportional gain matrix, K_p: numpy array
        Positive-definite derivative gain matrix, K_d: numpy array
        """

        Controller.__init__(self, robotic_dynamics)
        self.K_p = K_p
        self.K_d = K_d
        if q_d is None:
            q_d = zeros(len(K_p))
        self.q_d = q_d

    def eval(self, x, t):
        q, q_dot = reshape(x, (2, -1))
        e = q - self.q_d
        return solve(self.dynamics.B, self.dynamics.G(q) - dot(self.K_p, e) - dot(self.K_d, q_dot))