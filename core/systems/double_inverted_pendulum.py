from numpy import array, cos, identity, sin

from core.dynamics import FullyActuatedRoboticDynamics

class DoubleInvertedPendulum(FullyActuatedRoboticDynamics):
    def __init__(self, m_1, m_2, l_1, l_2, g=9.81):
        FullyActuatedRoboticDynamics.__init__(self, 2, 2)
        self.params = m_1, m_2, l_1, l_2, g

    def D(self, q):
        m_1, m_2, l_1, l_2, _ = self.params
        _, theta_2 = q
        D_11 = (m_1 + m_2) * (l_1 ** 2) + 2 * m_2 * l_1 * l_2 * cos(theta_2) + m_2 * (l_2 ** 2)
        D_12 = m_2 * l_1 * l_2 * cos(theta_2) + m_2 * (l_2 ** 2)
        D_21 = D_12
        D_22 = m_2 * (l_2 ** 2)
        return array([[D_11, D_12], [D_21, D_22]])

    def C(self, q, q_dot):
        _, m_2, l_1, l_2, _ = self.params
        _, theta_2 = q
        theta_1_dot, theta_2_dot = q_dot
        C_11 = 0
        C_12 = -m_2 * l_1 * l_2 * (2 * theta_1_dot + theta_2_dot) * sin(theta_2)
        C_21 = -C_12 / 2
        C_22 = -m_2 * l_1 * l_2 * theta_1_dot * sin(theta_2) / 2
        return array([[C_11, C_12], [C_21, C_22]])

    def B(self, q):
        return identity(2)

    def U(self, q):
        m_1, m_2, l_1, l_2, g = self.params
        theta_1, theta_2 = q
        return (m_1 + m_2) * g * l_1 * cos(theta_1) + m_2 * g * l_2 * cos(theta_1 + theta_2)

    def G(self, q):
        m_1, m_2, l_1, l_2, g = self.params
        theta_1, theta_2 = q
        G_1 = -(m_1 + m_2) * g * l_1 * sin(theta_1) - m_2 * g * l_2 * sin(theta_1 + theta_2)
        G_2 = -m_2 * g * l_2 * sin(theta_1 + theta_2)
        return array([G_1, G_2])
