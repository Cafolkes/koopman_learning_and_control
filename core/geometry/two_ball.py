from numpy import array, cos, linspace, pi, sin

from .ball import Ball
from .visual import Visual

class TwoBall(Ball, Visual):
    def __init__(self):
        Ball.__init__(self, dim=2)

    def boundary(self, N):
        thetas = linspace(0, 2 * pi, N)
        return array([cos(thetas), sin(thetas)]).T
