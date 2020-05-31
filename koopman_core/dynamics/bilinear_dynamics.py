from numpy import array, concatenate, dot, reshape, zeros, atleast_1d

from core.dynamics.affine_dynamics import AffineDynamics
from core.dynamics.system_dynamics import SystemDynamics

class BilinearDynamics(SystemDynamics, AffineDynamics):
    """Class for unconstrained bilinear dynamics

    State represented as x = (q, q_dot), where q are generalized coordinates and
    q_dot are corresponding rates.

    Override drift, act.
    """

    def __init__(self, n, m, F, G, Cx, phi_fun):
        """Create a RoboticDynamics object.

        Inputs:
        Configuration space dimension, n: int
        Action space dimension, m: int
        Unactuated dynamics matrix, F: array (n,n)
        Actuated dynamics matrix, G: list(array(n,n))
        """

        assert m == len(G)
        assert n == F.shape[0]

        SystemDynamics.__init__(self, n, m)
        self.k = n
        self.F = F
        self.G = G
        self.Cx = Cx
        self.phi_fun = phi_fun

    def drift(self, x, t):
        return dot(self.F, x)

    def act(self, x, t):
        return array([dot(g, x) for g in self.G]).T