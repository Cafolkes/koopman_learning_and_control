from numpy import array, cos, sin, zeros

from core.dynamics import RoboticDynamics

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