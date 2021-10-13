import numpy as np
from numpy import dot, zeros, array, sum

from core.dynamics import AffineDynamics, SystemDynamics


class BilinearLiftedDynamics(SystemDynamics, AffineDynamics):
    """Class for unconstrained bilinear dynamics

    State represented as x = (q, q_dot), where q are generalized coordinates and
    q_dot are corresponding rates.

    Override drift, act.
    """

    def __init__(self, n, m, A, B, C, basis, continuous_mdl=True, dt=None, standardizer_x=None, standardizer_u=None):
        """
        Initialize bilinear lifted dynamics class
        :param n: (int) State dimension
        :param m: (int) Actuation dimension
        :param A: (np.array) Lifted dynamics matrix (autonomous part)
        :param B: (list(np.array)) List of lifted dynamics matrices (actuated part)
        :param C: (np.array) Projection matrix
        :param basis: (lambda function) Function dictionary
        :param continuous_mdl: (boolean) System matrices provided are for continuous dynamics
        :param dt: (float) Sampling interval for discrete-time dynamics
        """

        assert m == len(B)
        assert n == A.shape[0]

        SystemDynamics.__init__(self, n, m)
        self.k = n
        self.A = A
        self.B = B
        self.C = C
        self.basis = basis

        self.B_tensor = np.empty((self.m, self.n, self.n))
        for ii, b in enumerate(self.B):
            self.B_tensor[ii] = b

        self.continuous_mdl = continuous_mdl
        self.dt = dt
        self.standardizer_x = standardizer_x
        #if standardizer_x is not None:
        #    assert standardizer_x.with_mean is False, 'Mean offset of data not supported'
        self.standardizer_u = standardizer_u
        #if standardizer_u is not None:
        #    assert standardizer_u.with_mean is False, 'Mean offset of data not supported'

    def drift(self, x, t):
        return dot(self.A, x)

    def act(self, x, t):
        return (self.B_tensor@x).T

    def lift(self, x, u):
        return self.basis(x)

    def get_linearization(self, z0, z1, u0, t):
        A_lin = self.A + sum(array([b*u for b,u in zip(self.B, u0)]),axis=0)
        B_lin = array([b @ z0 for b in self.B]).T

        if z1 is None:
            z1 = A_lin@z0 + B_lin@u0

        f_d = self.drift(z0, t) + dot(self.act(z0,t),u0)
        r_lin = f_d - z1

        return A_lin, B_lin, r_lin

    def simulate(self, x_0, controller, ts, processed=True, atol=1e-6, rtol=1e-6):
        if self.continuous_mdl:
            xs, us = SystemDynamics.simulate(self, x_0, controller, ts, processed=True, atol=1e-6, rtol=1e-6)
        else:
            N = len(ts)
            xs = zeros((N, self.n))
            us = [None] * (N - 1)

            controller.reset()

            xs[0] = x_0
            for j in range(N - 1):
                x = xs[j]
                t = ts[j]
                u = controller.eval(x, t)
                us[j] = u
                u = controller.process(u)
                xs[j + 1] = self.eval_dot(x, u, t)
            if processed:
                us = array([controller.process(u) for u in us])

        return xs, us