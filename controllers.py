from cvxpy import Minimize, Problem, quad_form, sum_squares, Variable
from numpy import dot, reshape, zeros
from numpy.linalg import norm, solve
from numpy.random import multivariate_normal

from dynamics import AffineQuadCLF

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

class ConstantController(Controller):
    """Class for constant action policies."""

    def __init__(self, dynamics, u_const):
        """Create a ConstantController object.

        Inputs:
        Dynamics, dynamics: Dynamics
        Constant action, u_const: numpy array
        """

        Controller.__init__(self, dynamics)
        self.u_const = u_const

    def eval(self, x, t):
        return self.u_const

class LinearController(Controller):
    """Class for linear policies."""

    def __init__(self, affine_dynamics, K):
        """Create a LinearController object.

        Policy is u = -K * x.

        Inputs:
        Affine dynamics, affine_dynamics: AffineDynamics
        Gain matrix, K: numpy array
        """

        Controller.__init__(self, affine_dynamics)
        self.K = K

    def eval(self, x, t):
        return -dot(self.K, self.dynamics.eval(x, t))

class PDController(Controller):
    """Class for proportional-derivative policies."""

    def __init__(self, pd_dynamics, K_p, K_d):
        """Create a PDController object.

        Policy is u = -K_p * e_p - K_d * e_d, where e_p and e_d are propotional
        and derivative components of error.

        Inputs:
        Proportional-derivative dynamics, pd_dynamics: PDDynamics
        Proportional gain matrix, K_p: numpy array
        Derivative gain matrix, K_d: numpy array
        """

        Controller.__init__(self, pd_dynamics)
        self.K_p = K_p
        self.K_d = K_d

    def eval(self, x, t):
        e_p = dynamics.proportional(x, t)
        e_d = dynamics.derivative(x, t)
        return -dot(self.K_p, e_p) - dot(self.K_d, e_d)

class FBLinController(Controller):
    """Class for linearizing feedback policies."""

    def __init__(self, fb_lin_dynamics, linear_controller):
        """Create an FBLinController object.

        Policy is u = (act)^-1 * (-drift + aux), where drift and act are
        components of drift vector and actuation matrix corresponding to
        highest-order derivatives of each output coordinate and aux is an
        auxilliary linear controller.

        Inputs:
        Feedback linearizable dynamics, fb_lin_dynamics: FBLinDynamics
        Auxilliary linear controller, linear_controller: LinearController
        """

        Controller.__init__(self, fb_lin_dynamics)
        self.linear_controller = linear_controller
        self.select = fb_lin_dynamics.select
        self.permute = fb_lin_dynamics.permute

    def eval(self, x, t):
        drift = self.select(self.permute(self.dynamics.drift(x, t)))
        act = self.select(self.permute(self.dynamics.act(x, t)))
        return solve(act, -drift + self.linear_controller.eval(x, t))

class RandomController(Controller):
    """Class for i.i.d. Gaussian perturbations around policies.

    Perturbations are held constant for multiple time steps to reduce the effect
    of aliasing.
    """

    def __init__(self, controller, cov, reps=2):
        """Create a RandomController object.

        Mean policy, controller: Controller
        Covariance matrix, cov: numpy array
        Number of time steps held, reps: int
        """

        Controller.__init__(self, controller.dynamics)
        self.controller = controller
        self.cov = cov
        self.m, _ = cov.shape
        self.reps = reps
        self.counter = None
        self.pert = self.sample()

    def sample(self):
        """Sample a perturbation.

        Outputs:
        Zero-mean Gaussian sample: numpy array
        """

        return multivariate_normal(zeros(self.m), self.cov)

    def eval(self, x, t):
        if self.counter == 0:
            self.pert = self.sample()
            self.counter = self.reps + 1
        self.counter = self.counter - 1
        return self.controller.eval(x, t), self.pert

    def process(self, u):
        u_nom, u_pert = u
        return u_nom + u_pert

    def reset(self):
        self.counter = self.reps
        self.controller.reset()

class LQRController(Controller):
    """Class for Linear Quadratic Regulator (LQR) policies.

    Policy is u = -1 / 2 * R^-1 P * B' * x for dynamics x_dot = A * x + B * u

    Policy optimizes infinite-horizon continuous-time LQR problem with stage
    cost x(t)' * Q * x(t) + u(t)' * R * u(t) and cost-to-go x(t)' * P * x(t).
    """

    def __init__(self, affine_linearizable_dynamics, P, R):
        """Create an LQRController object.

        Inputs:
        Positive-definite cost-to-go matrix, P: numpy array
        Positive-definite action cost matrix, R: numpy array
        """

        Controller.__init__(self, affine_linearizable_dynamics)
        self.P = P
        self.R = R

    def eval(self, x, t):
        _, B = self.dynamics.linear_system()
        return -solve(self.R, dot(B.T, dot(self.P, self.dynamics.eval(x, t)))) / 2

    def build(affine_linearizable_dynamics, Q, R):
        """Create an LQRController from state and action matrices.

        Inputs:
        Affine and linearizable dynamics, affine_linearizable_dynamics: AffineDynamics, LinearizableDynamics
        Positive-definite state cost matrix, Q: numpy array
        Positive-definite action cost matrix, R: numpy array

        Outputs:
        LQR policy: LQRController
        """

        lyap = AffineQuadCLF.build_care(affine_linearizable_dynamics, Q, R)
        return LQRController(affine_linearizable_dynamics, lyap.P, R)

class QPController(Controller):
    """Class for QP-based controllers."""

    def __init__(self, affine_dynamics, affine_quad_clf, Q, cost_mat, reg_coeff=0, reg_controller=None):
        """Construct a QPController object.

        Policy minimizes u' * cost_mat * u + reg_coeff * || u - reg_controller || ^ 2
        subject to V_dot <= -x' * P * x.

        Inputs:
        Affine dynamics, affine_dynamics: AffineDynamics
        Quadratic CLF, affine_quad_clf: AffineQuadCLF
        Positive-definite state cost, Q: numpy array
        Positive-definite action cost, cost_mat: numpy array
        Non-negative regularization coefficient, reg_coeff: float
        Regularizing controller, reg_controller: Controller
        """

        Controller.__init__(self, affine_dynamics)
        self.lyap = affine_quad_clf
        self.Q = Q
        self.cost_mat = cost_mat
        self.reg_coeff = reg_coeff
        if reg_controller is None:
            reg_controller = ConstantController(affine_dynamics, zeros(len(cost_mat)))
        self.reg_controller = reg_controller
        self.u = Variable(len(cost_mat))

    def eval(self, x, t):
        z = self.dynamics.eval(x, t)
        obj = Minimize(quad_form(self.u, self.cost_mat) + self.reg_coeff * sum_squares(self.u - self.reg_controller.eval(x, t)) ** 2)
        cons = [self.lyap.drift(x, t) + self.lyap.act(x, t) * self.u <= -dot(z, dot(self.Q, z))]
        prob = Problem(obj, cons)
        prob.solve(warm_start=True)
        return self.u.value

    def build_care(affine_linearizable_dynamics, Q, R, cost_mat, reg_coeff=0, reg_controller=None):
        """Create a QPController by solving continuous-time algebraic Riccati equation (CARE).

        CARE is F' * P + P * F - P * G * R^-1 * G' * P = -Q.

        Inputs:
        Affine and linearizable dynamics, affine_linearizable_dynamics: AffineDynamics, LinearizableDynamics
        Positive-definite state cost matrix, Q: numpy array
        Positive-definite auxilliary action cost matrix, R: numpy array
        Positive-definite action cost matrix, cost_mat: numpy array
        Non-negative regularization coefficient, reg_coeff: float
        Regularizing controller, reg_controller: Controller
        """

        lyap = AffineQuadCLF.build_care(affine_linearizable_dynamics, Q, R)
        return QPController(affine_linearizable_dynamics, lyap, Q, cost_mat, reg_coeff, reg_controller)

    def build_ctle(affine_linearizable_dynamics, K, Q, cost_mat, reg_coeff=0, reg_controller=None):
        """Create a QPController by solving continuous-time Lyapunov equation (CTLE).

        CTLE is A' * P + P * A = -Q, where A = F - G * K is closed-loop matrix.

        Inputs:
        Affine and linearizable dynamics, affine_linearizable_dynamics: AffineDynamics, LinearizableDynamics
        Gain matrix, K: numpy array
        Positive-definite state cost matrix, Q: numpy array
        Positive-definite action cost matrix, cost_mat: numpy array
        Non-negative regularization coefficient, reg_coeff: float
        Regularizing controller, reg_controller: Controller
        """

        lyap = AffineQuadCLF.build_ctle(affine_linearizable_dynamics, K, Q)
        return QPController(affine_linearizable_dynamics, lyap, Q, cost_mat, reg_coeff, reg_controller)

class EnergyController(Controller):
    """Class for energy-based controllers."""

    def __init__(self, robotic_dynamics, K_p, K_d, q_d=None):
        """Create an EnergyController object.

        Requires fully actuated robotic system with relative degree 2.

        Inputs:
        Robotic system, robotic_dynamics: RoboticDynamics
        Positive-definite proportional gain matrix, K_p: numpy array
        Positive-definite derivative gain matrix, K_d: numpy array
        """

        Controller.__init__(self, robotic_dynamics)
        self.K_p = K_p
        self.K_d = K_d
        if q_d is None:
            q_d = zeros(len(K_p))
        self.q_d = q_d

    def eval(self, x, t):
        q, q_dot = reshape(x, (2, -1))
        e = q - self.q_d
        return solve(self.dynamics.B, self.dynamics.G(q) - dot(self.K_p, e) - dot(self.K_d, q_dot))
