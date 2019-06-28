class Controller:
    """Abstract policy class for control.

    Override eval.
    """

    def __init__(self, dynamics):
        """Create a Controller object.

        Inputs:
        Dynamics, dynamics: Dynamics
        """

        self.dynamics = dynamics

    def eval(self, x, t):
        """Compute general representation of an action.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Action: object
        """

        pass

    def process(self, u):
        """Transform general representation of an action to a numpy array.

        Inputs:
        Action, u: object

        Outputs:
        Action: numpy array
        """

        return u

    def reset(self):
        """Reset any controller state."""

        pass