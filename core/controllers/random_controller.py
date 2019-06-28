from numpy import zeros
from numpy.random import multivariate_normal

from .controller import Controller

class RandomController(Controller):
    """Class for i.i.d. Gaussian perturbations around policies.

    Perturbations are held constant for multiple time steps to reduce the effect
    of aliasing.
    """

    def __init__(self, controller, cov, reps=2):
        """Create a RandomController object.

        Mean policy, controller: Controller
        Covariance matrix, cov: numpy array
        Number of time steps held, reps: int
        """

        Controller.__init__(self, controller.dynamics)
        self.controller = controller
        self.cov = cov
        self.m, _ = cov.shape
        self.reps = reps
        self.counter = None
        self.pert = self.sample()

    def sample(self):
        """Sample a perturbation.

        Outputs:
        Zero-mean Gaussian sample: numpy array
        """

        return multivariate_normal(zeros(self.m), self.cov)

    def eval(self, x, t):
        if self.counter == 0:
            self.pert = self.sample()
            self.counter = self.reps + 1
        self.counter = self.counter - 1
        return self.controller.eval(x, t), self.pert

    def process(self, u):
        u_nom, u_pert = u
        return u_nom + u_pert

    def reset(self):
        self.counter = self.reps
        self.controller.reset()