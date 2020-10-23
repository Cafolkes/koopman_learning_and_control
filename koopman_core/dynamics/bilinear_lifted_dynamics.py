import numpy as np
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
        return np.dot(self.A, x)

    def act(self, x, t):
        return np.array([b@x for b in self.B]).T

    def lift(self, x, u):
        return self.basis(x)

    def get_linearization(self, z0, z1, u0, t):
        A_lin = self.A + np.sum(np.array([b*u for b,u in zip(self.B, u0)]),axis=0)
        B_lin = np.array([b @ z0 for b in self.B]).T

        if z1 is None:
            z1 = A_lin@z0 + B_lin@u0

        f_d = self.drift(z0, t) + np.dot(self.act(z0,t),u0)
        r_lin = f_d - z1

        return A_lin, B_lin, r_lin