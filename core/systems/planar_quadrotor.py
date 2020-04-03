from numpy import append, arange, arctan, array, concatenate, cos, reshape, sin, zeros

from core.dynamics import FBLinDynamics, RoboticDynamics, SystemDynamics

class PlanarQuadrotor(RoboticDynamics):
    def __init__(self, m, J, g=9.81):
        RoboticDynamics.__init__(self, 3, 2)
        self.params = m, J, g

    def D(self, q):
        m, J, _ = self.params
        return array([[m, 0, 0], [0, m, 0], [0, 0, J]])

    def C(self, q, q_dot):
        return zeros((3, 3))

    def U(self, q):
        m, _, g = self.params
        _, z, _ = q
        return m * g * z

    def G(self, q):
        m, _, g = self.params
        return array([0, m * g, 0])

    def B(self, q):
        _, _, theta = q
        return array([[sin(theta), 0], [cos(theta), 0], [0, 1]])

    class Extension(SystemDynamics):
        def __init__(self, planar_quadrotor):
            SystemDynamics.__init__(self, n=8, m=2)
            self.quad = planar_quadrotor

        def step(self, x_0, u_0, t_0, t_f, atol=1e-6, rtol=1e-6):
            x = x_0[:6]
            f, f_dot = x_0[-2:]
            f_ddot, tau = u_0
            u = array([f, tau])

            dt = t_f - t_0
            f += f_dot * dt
            f_dot += f_ddot * dt
            x = self.quad.step(x, u, t_0, t_f, atol, rtol)

            return concatenate([x, array([f, f_dot])])

    class Output(FBLinDynamics):
        def __init__(self, extension):
            relative_degrees = [4, 4]
            perm = concatenate([2 * arange(4), 2 * arange(4) + 1])
            FBLinDynamics.__init__(self, relative_degrees, perm)
            self.params = extension.quad.params

        def r_ddot(self, f, theta):
            m, _, g = self.params
            x_ddot = f * sin(theta) / m
            z_ddot = f * cos(theta) / m - g
            return array([x_ddot, z_ddot])

        def r_dddot(self, f, f_dot, theta, theta_dot):
            m, _, _ = self.params
            x_dddot = (f_dot * sin(theta) + f * theta_dot * cos(theta)) / m
            z_dddot = (f_dot * cos(theta) - f * theta_dot * sin(theta)) / m
            return array([x_dddot, z_dddot])

        def r_ddddot_drift(self, f, f_dot, theta, theta_dot):
            m, _, _ = self.params
            x_ddddot_drift =  (2 * f_dot * theta_dot * cos(theta) - f * (theta_dot ** 2) * sin(theta)) / m
            z_ddddot_drift = -(2 * f_dot * theta_dot * sin(theta) + f * (theta_dot ** 2) * cos(theta)) / m
            return array([x_ddddot_drift, z_ddddot_drift])

        def r_ddddot_act(self, f, theta):
            m, J, _ = self.params
            x_ddddot_act = array([sin(theta),  f * cos(theta) / J]) / m
            z_ddddot_act = array([cos(theta), -f * sin(theta) / J]) / m
            return array([x_ddddot_act, z_ddddot_act])

        def eval(self, x, t):
            q, q_dot = reshape(x[:6], (2, 3))
            f, f_dot = x[-2:]
            r, theta = q[:2], q[-1]
            r_dot, theta_dot = q_dot[:2], q_dot[-1]
            r_ddot = self.r_ddot(f, theta)
            r_dddot = self.r_dddot(f, f_dot, theta, theta_dot)
            return concatenate([r, r_dot, r_ddot, r_dddot])

        def drift(self, x, t):
            eta = self.eval(x, t)
            theta, theta_dot, f, f_dot = x[array([2, 5, -2, -1])]
            r_ddddot_drift = self.r_ddddot_drift(f, f_dot, theta, theta_dot)
            return concatenate([eta[2:], r_ddddot_drift])

        def act(self, x, t):
            theta, f = x[2], x[-2]
            r_ddddot_act = self.r_ddddot_act(f, theta)
            return concatenate([zeros((6, 2)), r_ddddot_act])

        def to_state(self, eta):
            m, _, g = self.params

            r, r_dot = reshape(eta[:4], (2, 2))
            x_ddot, z_ddot, x_dddot, z_dddot = eta[-4:]
            theta = arctan(x_ddot / (z_ddot + g))
            theta_dot = ((z_ddot + g) * x_dddot - x_ddot * z_dddot) / ((z_ddot + g) ** 2) * (cos(theta) ** 2)
            q = append(r, theta)
            q_dot = append(r_dot, theta_dot)

            f = m * (z_ddot + g) / cos(theta)
            f_dot = m * (z_dddot + x_ddot * theta_dot) / cos(theta)

            return concatenate([q, q_dot, array([f, f_dot])])
