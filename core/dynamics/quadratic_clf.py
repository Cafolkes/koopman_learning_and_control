from numpy import dot

from .scalar_dynamics import ScalarDynamics

class QuadraticCLF(ScalarDynamics):
    """Class for Lyapunov functions of the form V(z) = z' * P * z."""

    def __init__(self, dynamics, P):
        """Create a QuadraticCLF object.

        Inputs:
        Dynamics, dynamics: Dynamics
        Positive-definite matrix, P: numpy array
        """

        self.dynamics = dynamics
        self.P = P

    def eval(self, x, t):
        """Compute V(z).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Lyapunov function value: float
        """

        z = self.dynamics.eval(x, t)
        return dot(z, dot(self.P, z))

    def eval_grad(self, x, t):
        z = self.dynamics.eval(x, t)
        return 2 * dot(self.P, z)

    def eval_dot(self, x, u, t):
        """Compute dV/dt.

        Inputs:
        State, x: numpy array
        Action, u: numpy array
        Time, t: float

        Outputs:
        Lyapunov function time derivative: float
        """

        return dot(self.eval_grad(x, t), self.dynamics.eval_dot(x, u, t))

