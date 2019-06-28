from numpy import dot
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov

from .affine_dynamics import AffineDynamics
from .quadratic_clf import QuadraticCLF

class AffineQuadCLF(AffineDynamics, QuadraticCLF):
    """Class for quadratic Lyapunov functions for affine dynamics."""

    def __init__(self, affine_dynamics, P):
        """Build an AffineQuadCLF object.

        Inputs:
        Affine dynamics, affine_dynamics: AffineDynamics
        Positive-definite matrix, P: numpy array
        """

        QuadraticCLF.__init__(self, affine_dynamics, P)

    def drift(self, x, t):
        return dot(self.eval_grad(x, t), self.dynamics.drift(x, t))

    def act(self, x, t):
        return dot(self.eval_grad(x, t), self.dynamics.act(x, t))

    def build_care(affine_linearizable_dynamics, Q, R):
        """Build AffineQuadCLF from affine and linearizable dynamics by solving continuous-time algebraic Riccati equation (CARE).

        CARE is F' * P + P * F - P * G * R^-1 * G' * P = -Q.

        Inputs:
        Affine and linearizable dynamics, affine_linearizable_dynamics: AffineDynamics, LinearizableDynamics
        Positive-definite state cost matrix, Q: numpy array
        Positive-definite action cost matrix, R: numpy array
        """

        F, G = affine_linearizable_dynamics.linear_system()
        P = solve_continuous_are(F, G, Q, R)
        return AffineQuadCLF(affine_linearizable_dynamics, P)

    def build_ctle(affine_linearizable_dynamics, K, Q):
        """Build AffineQuadCLF from affine and linearizable dynamics with gain matrix by solving continuous-time Lyapunov equation (CTLE).

        CTLE is A' * P + P * A = -Q, where A = F - G * K is closed-loop matrix.

        Inputs:
        Affine and linearizable dynamics, affine_linearizable_dynamics: AffineDynamics, LinearizableDynamics
        Positive-definite state cost matrix, Q: numpy array
        Gain matrix, K: numpy array
        """

        A = affine_linearizable_dynamics.closed_loop_linear_system(K)
        P = solve_continuous_lyapunov(A.T, -Q)
        return AffineQuadCLF(affine_linearizable_dynamics, P)