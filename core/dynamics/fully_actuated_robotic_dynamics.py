from .configuration_dynamics import ConfigurationDynamics
from .fb_lin_dynamics import FBLinDynamics
from .robotic_dynamics import RoboticDynamics

class FullyActuatedRoboticDynamics(FBLinDynamics, RoboticDynamics):
    def __init__(self, n, m):
        RoboticDynamics.__init__(self, n, m)
        config = ConfigurationDynamics(self, n)
        FBLinDynamics.__init__(self, config.relative_degrees, config.perm)
