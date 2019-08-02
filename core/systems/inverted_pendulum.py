from numpy import array, cos, sin

from core.dynamics import FullyActuatedRoboticDynamics

class InvertedPendulum(FullyActuatedRoboticDynamics):
    def __init__(self, m, l, g=9.81):
        FullyActuatedRoboticDynamics.__init__(self, 1, 1)
        self.params = m, l, g

    def D(self, q):
        m, l, _ = self.params
        return array([[m * (l ** 2)]])

    def C(self, q, q_dot):
        return array([[0]])

    def U(self, q):
        m, l, g = self.params
        theta, = q
        return m * g * l * cos(theta)

    def G(self, q):
        m, l, g = self.params
        theta, = q
        return array([-m * g * l * sin(theta)])

    def B(self, q):
        return array([[1]])
