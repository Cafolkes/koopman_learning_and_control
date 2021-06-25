from numpy import dot
from numpy.linalg import solve

from .controller import Controller
from ..dynamics import AffineQuadCLF

class LQRController(Controller):
    """Class for Linear Quadratic Regulator (LQR) policies.

    Policy is u = -1 / 2 * R^-1 P * B' * x for dynamics x_dot = A * x + B * u

    Policy optimizes infinite-horizon continuous-time LQR problem with stage
    cost x(t)' * Q * x(t) + u(t)' * R * u(t) and cost-to-go x(t)' * P * x(t).
    """

    def __init__(self, affine_linearizable_dynamics, P, R):
        """Create an LQRController object.

        Inputs:
        Positive-definite cost-to-go matrix, P: numpy array
        Positive-definite action cost matrix, R: numpy array
        """

        Controller.__init__(self, affine_linearizable_dynamics)
        self.P = P
        self.R = R

    def eval(self, x, t):
        _, B = self.dynamics.linear_system()
        return -solve(self.R, dot(B.T, dot(self.P, self.dynamics.eval(x, t)))) / 2

    def build(affine_linearizable_dynamics, Q, R):
        """Create an LQRController from state and action matrices.

        Inputs:
        Affine and linearizable dynamics, affine_linearizable_dynamics: AffineDynamics, LinearizableDynamics
        Positive-definite state cost matrix, Q: numpy array
        Positive-definite action cost matrix, R: numpy array

        Outputs:
        LQR policy: LQRController
        """

        lyap = AffineQuadCLF.build_care(affine_linearizable_dynamics, Q, R)
        return LQRController(affine_linearizable_dynamics, lyap.P, R)