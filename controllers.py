from cvxpy import Minimize, Problem, quad_form, square, sum_squares, Variable
from numpy import dot, identity, reshape, zeros
from numpy.linalg import eigvals, norm, solve
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
    """Class for controllers solving quadratic programs (QPs).

    QPs are solved using cvxpy.
    """

    def __init__(self, affine_dynamics, m):
        """Create a QPController object.

        Inputs:
        Affine dynamics, affine_dynamics: AffineDynamics
        Number of actions, m: int
        """

        Controller.__init__(self, affine_dynamics)
        self.m = m
        self.u = Variable(m)
        self.variables = []
        self.static_costs = []
        self.dynamic_costs = []
        self.constraints = []

    def add_static_cost(self, P=None, q=None, r=0):
        """Add term to cost of the form u' * P * u + q' * u + r

        Inputs:
        Quadratic term, P: numpy array
        Linear term, q: numpy array
        Constant term, r: float
        """

        if P is None:
            P = zeros((self.m, self.m))
        if q is None:
            q = zeros(self.m)
        cost = quad_form(self.u, P) + q * self.u + r
        self.static_costs.append(cost)

    def add_dynamic_cost(self, P, q, r):
        """Add term to cost of the form u' * P(x, t) * u + q(x, t)' * u + r(x, t)

        Inputs:
        Quadratic term, P: numpy array * float -> numpy array
        Linear term, q: numpy array * float -> numpy array
        Constant term, r: numpy array * float -> float
        """

        if P is None:
            P = lambda x, t: zeros((self.m, self.m))
        if q is None:
            q = lambda x, t: zeros(self.m)
        if r is None:
            r = lambda x, t: 0
        cost = lambda x, t: quad_form(self.u, P(x, t)) + q(x, t) * self.u + r(x, t)
        self.dynamic_costs.append(cost)

    def add_regularizer(self, controller, coeff=1):
        """Add 2-norm regularizer about another controller to cost

        Inputs:
        Controller, controller: Controller
        Regularizer weight, coeff: float
        """

        cost = lambda x, t: coeff * sum_squares(self.u - controller.process(controller.eval(x, t)))
        self.dynamic_costs.append(cost)

    def add_stability_constraint(self, aff_lyap, comp=None, slacked=False, coeff=0):
        """Add Lyapunov function derivative constraint

        Inputs:
        Affine Lyapunov function: AffineDynamics, ScalarDynamics
        Class-K comparison function, comp: float -> float
        Flag for slacking constraint, slacked: bool
        Coefficient for slack variable in cost function, coeff: float
        """

        if comp is None:
            comp = lambda r: 0
        if slacked:
            delta = Variable()
            self.variables.append(delta)
            self.static_costs.append(coeff * square(delta))
            constraint = lambda x, t: aff_lyap.drift(x, t) + aff_lyap.act(x, t) * self.u <= -comp(aff_lyap.eval(x, t)) + delta
        else:
            constraint = lambda x, t: aff_lyap.drift(x, t) + aff_lyap.act(x, t) * self.u <= -comp(aff_lyap.eval(x, t))
        self.constraints.append(constraint)

    def add_safety_constraint(self, aff_safety, comp=None, slacked=False, coeff=0):
        """Add safety function derivative constraint

        Inputs:
        Affine safety function: AffineDynamics, ScalarDynamics
        Class-K comparison function, comp: float -> float
        Flag for slacking constraint, slacked: bool
        Coefficient for slack variable in cost function, coeff: float
        """

        if comp is None:
            comp = lambda r: 0
        if slacked:
            delta = Variable()
            self.variables.append(delta)
            self.static_costs.append(coeff * square(delta))
            constraint = lambda x, t: aff_safety.drift(x, t) + aff_safety.act(x, t) * self.u >= -comp(aff_safety.eval(x, t)) - delta
        else:
            constraint = lambda x, t: aff_safety.drift(x, t) + aff_safety.act(x, t) * self.u >= -comp(aff_safety.eval(x, t))
        self.constraints.append(constraint)

    def build_care(aff_dynamics, Q, R):
        """Build minimum-norm controller with stability constraint from solving CARE

        Inputs:
        Affine dynamics, aff_dynamics: AffineDynamics
        Positive-definite state cost matrix, Q: numpy array
        Positive-definite action cost matrix, R: numpy array

        Outputs:
        QP-based controller: QPController
        """

        return QPController._build(aff_dynamics, None, Q, R, 'CARE')

    def build_ctle(aff_dynamics, K, Q):
        """Build minimum-norm controller with stability constraint from solving CTLE

        Inputs:
        Affine dynamics, aff_dynamics: AffineDynamics
        Gain matrix, K: numpy array
        Positive-definite state cost matrix, Q: numpy array

        Outputs:
        QP-based controller: QPController
        """

        return QPController._build(aff_dynamics, K, Q, None, 'CTLE')

    def _build(aff_dynamics, K, Q, R, method):
        """Helper function for build_care and build_ctle"""

        if method is 'CARE':
            m = len(R)
            lyap = AffineQuadCLF.build_care(aff_dynamics, Q, R)
        elif method is 'CTLE':
            m = len(K)
            R = identity(m)
            lyap = AffineQuadCLF.build_ctle(affine_dynamics, K, Q)
        qp = QPController(aff_dynamics, m)
        qp.add_static_cost(R)
        alpha = min(eigvals(Q)) / max(eigvals(lyap.P))
        comp = lambda r: alpha * r
        qp.add_stability_constraint(lyap, comp)
        return qp

    def eval(self, x, t):
        static_cost = sum(self.static_costs)
        dynamic_cost = sum([cost(x, t) for cost in self.dynamic_costs])
        obj = Minimize(static_cost + dynamic_cost)
        cons = [constraint(x, t) for constraint in self.constraints]
        prob = Problem(obj, cons)
        prob.solve(warm_start=True)
        return self.u.value, [variable.value for variable in self.variables]

    def process(self, u):
        u, _ = u
        return u

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
