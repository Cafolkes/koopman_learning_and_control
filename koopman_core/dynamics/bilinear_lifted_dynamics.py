from numpy import array, concatenate, dot, reshape, zeros, atleast_1d
from core.dynamics import AffineDynamics, SystemDynamics
from koopman_core.learning.bilinear_edmd import BilinearEdmd

#class BilinearLiftedDynamics(SystemDynamics, AffineDynamics, BilinearEdmd):
class BilinearLiftedDynamics(SystemDynamics, AffineDynamics):
    """Class for unconstrained bilinear dynamics

    State represented as x = (q, q_dot), where q are generalized coordinates and
    q_dot are corresponding rates.

    Override drift, act.
    """

    def __init__(self, n, m, A, B, C, basis, continuous=True, dt=1e-2):
        """Create a RoboticDynamics object.

        Inputs:
        Configuration space dimension, n: int
        Action space dimension, m: int
        Unactuated dynamics matrix, F: array (n,n)
        Actuated dynamics matrix, G: list(array(n,n))
        """

        assert m == len(B)
        assert n == A.shape[0]

        SystemDynamics.__init__(self, n, m)
        self.k = n
        self.A = A
        self.B = B
        self.C = C
        self.basis = basis

        self.continuous = continuous
        if not self.continuous:
            self.dt = dt
        else:
            self.dt = None

    def drift(self, x, t):
        return dot(self.A, x)

    def act(self, x, t):
        return array([b@x for b in self.B]).T

    def lift(self, x, u):
        return self.basis(x)