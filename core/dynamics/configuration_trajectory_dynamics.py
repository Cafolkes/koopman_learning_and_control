from numpy import array, dot, arange, where
from numpy.linalg import solve

from .configuration_dynamics import ConfigurationDynamics

class ConfigurationTrajectoryDynamics(ConfigurationDynamics):
    def __init__(self, robotic_dynamics, k, _y_d=None, _y_d_dot=None, _y_d_ddot=None):
        ConfigurationDynamics.__init__(self, robotic_dynamics, k)
        self._y_d = _y_d
        self._y_d_dot = _y_d_dot
        self._y_ddot = _y_d_ddot

    def y(self, q):
        raise NotImplementedError

    def dydq(self, q):
        raise NotImplementedError

    def d2ydq2(self, q):
        raise NotImplementedError

    def y_d(self, t):
        return self._y_d(t)

    def y_d_dot(self, t):
        return self._y_d_dot(t)

    def y_d_ddot(self, t):
        return self._y_d_ddot(t)

    def add_trajectory(self, qs, q_dots, ts):
        ys = array([self.y(q) for q in qs])
        y_dots = array([dot(self.dydq(q), q_dot) for q, q_dot in zip(qs, q_dots)])

        def cubic_spline(t):
            before, = where(ts <= t)
            after, = where(ts > t)

            if len(after) == 0:
                idx_0 = before[-2]
                idx_1 = before[-1]
            else:
                idx_0 = before[-1]
                idx_1 = after[0]

            t_0, y_0, y_dot_0 = ts[idx_0], ys[idx_0], y_dots[idx_0]
            t_1, y_1, y_dot_1 = ts[idx_1], ys[idx_1], y_dots[idx_1]

            A = array([
                [t_0 ** 3, t_0 ** 2, t_0, 1],
                [t_1 ** 3, t_1 ** 2, t_1, 1],
                [3 * (t_0 ** 2), 2 * t_0, 1, 0],
                [3 * (t_1 ** 2), 2 * t_1, 1, 0]
            ])

            bs = array([y_0, y_1, y_dot_0, y_dot_1])

            alphas_0 = solve(A, bs)
            alphas_1 = array([3 * alphas_0[0], 2 * alphas_0[1], alphas_0[2]])
            alphas_2 = array([2 * alphas_1[0], alphas_1[1]])

            ts_0 = t ** arange(3, -1, -1)
            ts_1 = ts_0[1:]
            ts_2 = ts_1[1:]

            y = dot(ts_0, alphas_0)
            y_dot = dot(ts_1, alphas_1)
            y_ddot = dot(ts_2, alphas_2)

            return y, y_dot, y_ddot

        def _y_d(t):
            y, _, _ = cubic_spline(t)
            return y

        def _y_d_dot(t):
            _, y_dot, _ = cubic_spline(t)
            return y_dot

        def _y_d_ddot(t):
            _, _, y_ddot = cubic_spline(t)
            return y_ddot

        self._y_d = _y_d
        self._y_d_dot = _y_d_dot
        self._y_d_ddot = _y_d_ddot
