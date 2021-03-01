import numpy as np
from core.controllers.controller import Controller

class PerturbedController(Controller):
    """Class for proportional-derivative policies."""

    def __init__(self, dynamics, nom_controller, pert_noise_var, const_offset=0, umin=None, umax=None):
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
        self.umin = umin
        self.umax = umax

    def eval(self, x, t):
        u_nom = self.nom_controller.eval(x,t)
        if self.umin is not None:
            u_nom = np.maximum(u_nom, self.umin)

        if self.umax is not None:
            u_nom = np.minimum(u_nom, self.umax)

        # TODO: Add support for multiple distributions
        #return self.const_offset + u_nom + np.random.normal(size=u_nom.shape, scale=self.pert_noise_var)
        return self.const_offset + u_nom + np.random.uniform(-self.pert_noise_var, self.pert_noise_var,size=u_nom.shape)
