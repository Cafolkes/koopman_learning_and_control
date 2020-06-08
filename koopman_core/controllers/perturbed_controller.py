from numpy.random import normal
from core.controllers.controller import Controller

class PerturbedController(Controller):
    """Class for proportional-derivative policies."""

    def __init__(self, dynamics, nom_controller, pert_noise_var, const_offset=0):
        """Create a PDController object.

        Policy is u = -K_p * e_p - K_d * e_d, where e_p and e_d are propotional
        and derivative components of error.

        Inputs:
        Dynamics, dynamics: Dynamics
        Nominal controller, controller: Controller
        """

        Controller.__init__(self, dynamics)
        self.nom_controller = nom_controller
        self.pert_noise_var = pert_noise_var
        self.const_offset = const_offset

    def eval(self, x, t):
        u_nom = self.nom_controller.eval(x,t)
        return self.const_offset + u_nom + normal(size=u_nom.shape, scale=self.pert_noise_var)
