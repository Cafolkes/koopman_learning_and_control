class Dynamics:
    """Abstract dynamics class.

    Override eval, eval_dot.
    """

    def eval(self, x, t):
        """Compute representation.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Representation: numpy array
        """

        pass

    def eval_dot(self, x, u, t):
        """Compute dynamics (time derivative of representation).

        Inputs:
        State, x: numpy array
        Action, u: numpy array
        Time, t: float

        Outputs:
        Time-derivative: numpy array
        """

        pass