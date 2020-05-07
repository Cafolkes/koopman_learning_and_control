from .controller import Controller

class ConstantController(Controller):
    """Class for constant action policies."""

    def __init__(self, dynamics, u_const):
        """__init__ Create a constant controller
        
        Arguments:
            dynamics {dynamical system} -- dynamics for the controller
            u_const {numpy array [nu,]} -- value for the contant controller
        """

        Controller.__init__(self, dynamics)
        self.u_const = u_const

    def eval(self, x, t):
        """eval Function to evaluate the constant controller
        
        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time
        
        Returns:
            control action -- numpy array [Nu,]
        """
        return self.u_const