from numpy import array, zeros
from scipy.integrate import solve_ivp

from .dynamics import Dynamics

class SystemDynamics(Dynamics):
    """Abstract dynamics class for simulation.

    Override eval_dot.
    """

    def __init__(self, n, m):
        """Create a SystemDynamics object.

        Inputs:
        Number of states, n: int
        Number of actions, m: int
        """

        self.n = n
        self.m = m

    def eval(self, x, t):
        return x

    def simulate(self, x_0, controller, ts, processed=True, atol=1e-6, rtol=1e-6):
        """Simulate system from initial state with specified controller.

        Approximated using Runge-Kutta 4,5 solver.

        Actions computed at time steps and held constant over sample period.

        Inputs:
        Initial state, x_0: numpy array
        Control policy, controller: Controller
        Time steps, ts: numpy array
        Flag to process actions, processed: bool
        Absolute tolerance, atol: float
        Relative tolerance, rtol: float

        Outputs:
        State history: numpy array
        Action history: numpy array
        """

        assert len(x_0) == self.n

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
            x_dot = lambda t, x: self.eval_dot(x, u, t)
            t_span = [t, ts[j + 1]]
            res = solve_ivp(x_dot, t_span, x, atol=atol, rtol=rtol)
            xs[j + 1] = res.y[:, -1]

        if processed:
            us = array([controller.process(u) for u in us])

        return xs, us

