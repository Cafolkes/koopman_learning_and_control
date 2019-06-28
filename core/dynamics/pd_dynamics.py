from .dynamics import Dynamics

class PDDynamics(Dynamics):
    """Abstract class for dynamics with proportional and derivative components.

    Override eval, eval_dot, proportional, derivative.
    """

    def proportional(self, x, t):
        """Compute proportional component.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Proportional component: numpy array
        """

        pass

    def derivative(self, x, t):
        """Compute derivative component.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Derivative component: numpy array
        """

        pass