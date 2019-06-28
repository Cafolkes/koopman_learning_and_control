from cvxpy import Minimize, Problem, quad_form, square, sum_squares, Variable
from numpy import zeros
from numpy.linalg import eigvals

from .controller import Controller
from core.dynamics import AffineQuadCLF

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