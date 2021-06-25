from ...core.controllers.controller import Controller
from numpy import array, interp, atleast_1d
class OpenLoopController(Controller):
    """Class for open loop action policies."""

    def __init__(self, dynamics, u_open_loop, t_open_loop):
        """__init__ Create controller
        
        Arguments:
            dynamics {dynamical system} -- dynamics for the controller
            u_open_loop {numpy array [Nu,Nt]} -- open loop time series
            t_open_loop {numpy array [Nt,} -- time vector
        """

        Controller.__init__(self, dynamics)
        self.u_open_loop = u_open_loop
        self.t_open_loop = t_open_loop
        self.m = u_open_loop.shape[1]

    def eval(self, x, t):
        """eval Function to evaluate controller
        
        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time
        
        Returns:
            control action -- numpy array [Nu,]
        """
        return atleast_1d(array([interp(t, self.t_open_loop.flatten(), self.u_open_loop[:,ii].flatten()) for ii in range(self.m)]).squeeze())