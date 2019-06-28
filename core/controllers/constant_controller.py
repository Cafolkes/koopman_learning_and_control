from .controller import Controller

class ConstantController(Controller):
    """Class for constant action policies."""

    def __init__(self, dynamics, u_const):
        """Create a ConstantController object.

        Inputs:
        Dynamics, dynamics: Dynamics
        Constant action, u_const: numpy array
        """

        Controller.__init__(self, dynamics)
        self.u_const = u_const

    def eval(self, x, t):
        return self.u_const