from numpy import dot
from numpy.linalg import solve

from .controller import Controller

class FBLinController(Controller):
    """Class for linearizing feedback policies."""

    def __init__(self, fb_lin_dynamics, linear_controller):
        """Create an FBLinController object.

        Policy is u = (act)^-1 * (-drift + aux), where drift and act are
        components of drift vector and actuation matrix corresponding to
        highest-order derivatives of each output coordinate and aux is an
        auxilliary linear controller.

        Inputs:
        Feedback linearizable dynamics, fb_lin_dynamics: FBLinDynamics
        Auxilliary linear controller, linear_controller: LinearController
        """

        Controller.__init__(self, fb_lin_dynamics)
        self.linear_controller = linear_controller
        self.select = fb_lin_dynamics.select
        self.permute = fb_lin_dynamics.permute

    def eval(self, x, t):
        drift = self.select(self.permute(self.dynamics.drift(x, t)))
        act = self.select(self.permute(self.dynamics.act(x, t)))
        return solve(act, -drift + self.linear_controller.eval(x, t))

