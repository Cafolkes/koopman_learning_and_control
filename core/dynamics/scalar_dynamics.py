from .dynamics import Dynamics

class ScalarDynamics(Dynamics):
    """Abstract scalar dynamics class.

    Override eval, eval_dot, eval_grad.
    """

    def eval_grad(self, x, t):
        """Compute gradient of representation.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Representation: float
        """

        pass