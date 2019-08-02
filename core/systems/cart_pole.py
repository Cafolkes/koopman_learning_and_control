from numpy import array, concatenate, cos, dot, reshape, sin, zeros
from numpy.linalg import solve

from core.dynamics import RoboticDynamics

class CartPole(RoboticDynamics):
    def __init__(self, m_c, m_p, l, g=9.81):
        RoboticDynamics.__init__(self, 2, 1)
        self.params = m_c, m_p, l, g

    def D(self, q):
        m_c, m_p, l, _ = self.params
        _, theta = q
        return array([[m_c + m_p, m_p * l * cos(theta)], [m_p * l * cos(theta), m_p * (l ** 2)]])

    def C(self, q, q_dot):
        _, m_p, l, _ = self.params
        _, theta = q
        return array([[0, -m_p * l * sin(theta)], [0, 0]])

    def G(self, q):
        _, m_p, l, g = self.params
        _, theta = q
        return array([0, -m_p * g * l * sin(theta)])

    def U(self, q):
        _, m_p, l, g = self.params
        _, theta = q
        return m_p * g * l * cos(theta)

    def B(self, q):
        return array([[1], [0]])
