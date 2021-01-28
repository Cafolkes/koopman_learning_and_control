from numpy import dot, zeros, array
from core.dynamics import AffineDynamics, LinearizableDynamics, SystemDynamics

class LinearLiftedDynamics(SystemDynamics, AffineDynamics, LinearizableDynamics):
    """Class for linear dynamics of the form x_dot = A * x + B * u."""

    def __init__(self, A, B, C, basis, continuous_mdl=True, dt=None):
        """Create a LinearSystemDynamics object.

        Inputs:
        State matrix, A: numpy array
        Input matrix, B: numpy array
        """

        if B is not None:
            n, m = B.shape
        else:
            n, m = A.shape[0], None

        assert A.shape == (n, n)

        SystemDynamics.__init__(self, n, m)
        self.A = A
        self.B = B
        self.C = C
        self.basis = basis

        self.continuous_mdl = continuous_mdl
        self.dt = dt

    def drift(self, x, t):
        return dot(self.A, x)

    def act(self, x, t):
        return self.B

    def eval_dot(self, x, u, t):
        if self.B is None:
            return self.drift(x,t)
        else:
            return AffineDynamics.eval_dot(self, x, u, t)

    def linear_system(self):
        return self.A, self.B

    def lift(self, x, u):
        return self.basis(x)

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