from numpy import dot

from core.controllers import Controller

class LinearLiftedController(Controller):
    """Class for linear policies."""

    def __init__(self, affine_dynamics, K):
        """Create a LinearController object.

        Policy is u = -K * x.

        Inputs:
        Affine dynamics, affine_dynamics: AffineDynamics
        Gain matrix, K: numpy array
        """

        Controller.__init__(self, affine_dynamics)
        self.K = K

    def eval(self, x, t):
        return -dot(self.K, self.dynamics.eval_z(x, t))