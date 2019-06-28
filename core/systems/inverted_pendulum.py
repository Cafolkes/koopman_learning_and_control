from numpy import array, cos, sin

from core.dynamics import RoboticDynamics, SystemDynamics

class InvertedPendulum(SystemDynamics, RoboticDynamics):
    def __init__(self, m, l, g=9.81):
        SystemDynamics.__init__(self, 2, 1)
        RoboticDynamics.__init__(self, array([[1]]))
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