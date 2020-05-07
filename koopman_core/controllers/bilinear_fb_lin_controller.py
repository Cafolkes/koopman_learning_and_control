from numpy import dot, atleast_2d
from numpy.linalg import solve, pinv

from core.controllers.controller import Controller

class BilinearFBLinController(Controller):
    """Class for bilinear linearizing feedback policies."""

    def __init__(self, bilinear_dynamics, output, lin_controller_gain):
        """Create an FBLinController object.

        Policy is u = (act)^-1 * (aux), where drift and act are
        components of drift vector and actuation matrix corresponding to
        highest-order derivatives of each output coordinate and aux is an
        auxilliary linear controller.

        Inputs:
        Bilinear dynamics, fb_lin_dynamics: FBLinDynamics
        Auxilliary linear controller, linear_controller: LinearController
        """

        Controller.__init__(self, bilinear_dynamics)
        self.dynamics = bilinear_dynamics
        self.output = output
        self.lin_controller_gain = lin_controller_gain

    def eval(self, x, t):
        z = self.dynamics.phi_fun(x)
        eta = dot(self.output.C_h, z - self.output.z_d(t))

        act = self.dynamics.act(z, t).T
        nu = dot(self.lin_controller_gain, z-self.output.z_d(t))

        return dot(pinv(act), self.output.z_d_dot(t) - self.dynamics.drift(self.output.z_d(t),t) + nu)

