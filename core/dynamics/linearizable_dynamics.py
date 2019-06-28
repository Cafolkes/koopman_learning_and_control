from numpy import dot

from .dynamics import Dynamics

class LinearizableDynamics(Dynamics):
    """Abstract class for dynamics with representations as x_dot = A * x + B * u.

    Override eval, eval_dot, linear_system.
    """

    def linear_system(self):
        """Compute matrices A and B in linear representation of dynamics.

        Outputs:
        A and B matrices: numpy array * numpy array
        """

        pass

    def closed_loop_linear_system(self, K):
        """Compute matrix A - B * K in linear representation of closed-loop dynamics.

        Inputs:
        Gain matrix, K: numpy array

        Outputs:
        Closed-loop matrix: numpy array
        """

        A, B = self.linear_system()
        return A - dot(B, K)