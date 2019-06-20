from numpy import arange, argsort, array, concatenate, cumsum, diag, dot, identity, ones, reshape, zeros
from numpy.linalg import solve
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, solve_continuous_are, solve_continuous_lyapunov

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

class SystemDynamics(Dynamics):
    """Abstract dynamics class for simulation.

    Override eval_dot.
    """

    def __init__(self, n, m):
        """Create a SystemDynamics object.

        Inputs:
        Number of states, n: int
        Number of actions, m: int
        """

        self.n = n
        self.m = m

    def eval(self, x, t):
        return x

    def simulate(self, x_0, controller, ts, atol=1e-6, rtol=1e-6):
        """Simulate system from initial state with specified controller.

        Approximated using Runge-Kutta 4,5 solver.

        Actions computed at time steps and held constant over sample period.

        Inputs:
        Initial state, x_0: numpy array
        Control policy, controller: Controller
        Time steps, ts: numpy array
        Absolute tolerance, atol: float
        Relative tolerance, rtol: float

        Outputs:
        State history: numpy array
        Action history: numpy array
        """

        assert len(x_0) == self.n

        N = len(ts)
        xs = zeros((N, self.n))
        us = [None] * (N - 1)

        controller.reset()

        xs[0] = x_0
        for j in range(N - 1):
            x = xs[j]
            t = ts[j]
            u = controller.eval(x, t)
            us[j] = u
            u = controller.process(u)
            x_dot = lambda t, x: self.eval_dot(x, u, t)
            t_span = [t, ts[j + 1]]
            res = solve_ivp(x_dot, t_span, x, atol=atol, rtol=rtol)
            xs[j + 1] = res.y[:, -1]

        return xs, us

class AffineDynamics(Dynamics):
    """Abstract class for dynamics of the form x_dot = f(x, t) + g(x, t) * u.

    Override eval, drift, act.
    """

    def drift(self, x, t):
        """Compute drift vector f(x, t).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Drift vector: numpy array
        """

        pass

    def act(self, x, t):
        """Compute actuation matrix g(x, t).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Actuation matrix: numpy array
        """

        pass

    def eval_dot(self, x, u, t):
        return self.drift(x, t) + dot(self.act(x, t), u)

class LinearizableDynamics(Dynamics):
    """Abstract class for dynamics with representations as x_dot = A * x + B * u.

    Override eval, eval_dot, linear_system.
    """

    def linear_system(self):
        """Compute matrices A and B in linear representation of dynamics.

        Outputs:
        A and B matrices: numpy array * numpy array
        """

        pass

    def closed_loop_linear_system(self, K):
        """Compute matrix A - B * K in linear representation of closed-loop dynamics.

        Inputs:
        Gain matrix, K: numpy array

        Outputs:
        Closed-loop matrix: numpy array
        """

        A, B = self.linear_system()
        return A - dot(B, K)

class LinearSystemDynamics(SystemDynamics, AffineDynamics, LinearizableDynamics):
    """Class for linear dynamics of the form x_dot = A * x + B * u."""

    def __init__(self, A, B):
        """Create a LinearSystemDynamics object.

        Inputs:
        State matrix, A: numpy array
        Input matrix, B: numpy array
        """

        n, m = B.shape
        assert A.shape == (n, n)

        SystemDynamics.__init__(self, n, m)
        self.A = A
        self.B = B

    def drift(self, x, t):
        return dot(self.A, x)

    def act(self, x, t):
        return self.B

    def linear_system(self):
        return self.A, self.B

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

class FBLinDynamics(AffineDynamics, LinearizableDynamics):
    """Abstract class for feedback linearizable affine dynamics.

    Representation must be block form, with each block corresponding to an
    output coordinate. If an output has relative degree gamma, then the
    corresponding block must express derivatives of order 0 through gamma - 1,
    in that order.

    If dynamics are specified in a different order, specify a permutation into
    block form.

    Override eval, drift, act.
    """

    def __init__(self, relative_degrees, perm=None):
        """Create an FBLinDynamics object.

        Inputs:
        Relative degrees of each output coordinate, relative_degrees: int list
        Indices of coordinates that make up each block, perm: numpy array
        """

        self.relative_degrees = relative_degrees
        self.relative_degree_idxs = cumsum(relative_degrees) - 1
        if perm is None:
            perm = arange(sum(relative_degrees))
        self.perm = perm
        self.inv_perm = argsort(perm)

    def select(self, arr):
        """Select coordinates of block order corresponding to highest-order derivatives.

        Inputs:
        Array, arr: numpy array

        Outputs:
        Array of selected coordinates: numpy array
        """

        return arr[self.relative_degree_idxs]

    def permute(self, arr):
        """Permute array into block order.

        Inputs:
        Array, arr: numpy array

        Outputs:
        Array permuted into block form: numpy array
        """

        return arr[self.perm]

    def inv_permute(self, arr):
        """Permute array out of block order.

        Inputs:
        Array in block form, arr: numpy array

        Outputs:
        Array out of block form: numpy array
        """

        return arr[self.inv_perm]

    def linear_system(self):
        F = block_diag(*[diag(ones(gamma - 1), 1) for gamma in self.relative_degrees])
        G = (identity(sum(self.relative_degrees))[self.relative_degree_idxs]).T

        F = (self.inv_permute((self.inv_permute(F)).T)).T
        G = self.inv_permute(G)

        return F, G

class RoboticDynamics(FBLinDynamics, PDDynamics):
    """Abstract class for Euler-Lagrange systems.

    State represented as x = (q, q_dot), where q are generalized coordinates and
    q_dot are corresponding rates.

    Dynamics represented as D(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = B * u.

    Override D, C, U, G.
    """

    def __init__(self, B):
        """Create a RoboticDynamics object.

        Inputs:
        Full rank static actuation matrix, B: numpy array
        """

        self.B = B
        self.k, self.m = B.shape
        relative_degrees = [2] * self.k
        perm = concatenate([array([j, j + self.k]) for j in range(self.k)])
        FBLinDynamics.__init__(self, relative_degrees, perm)

    def D(self, q):
        """Compute positive-definite inertia matrix.

        Inputs:
        Coordinates, q: numpy array

        Outputs:
        Positive-definite inertia matrix: numpy array
        """

        pass

    def C(self, q, q_dot):
        """Compute Coriolis terms.

        Inputs:
        Coordinates, q: numpy array
        Coordinate rates, q_dot, numpy array

        Outputs:
        Coriolis terms matrix: numpy array
        """

        pass

    def U(self, q):
        """Compute potential energy.

        Inputs:
        Coordinates, q: numpy array

        Outputs:
        Potential energy: float
        """

        pass

    def G(self, q):
        """Compute potential energy gradient.

        Inputs:
        Coordinates, q: numpy array

        Outputs:
        Potential energy gradient: numpy array
        """

        pass

    def T(self, q, q_dot):
        """Compute kinetic energy.

        Inputs:
        Coordinates, q: numpy array
        Coordinate rates, q_dot: numpy array

        Outputs:
        Kinetic energy: float
        """

        return dot(q_dot, self.D(q), q_dot) / 2

    def H(self, q, q_dot):
        """Compute Coriolis and potential terms.

        Inputs:
        Coordinates, q: numpy array
        Coordinate rates, q_dot: numpy array

        Outputs:
        Coriolis and potential terms: numpy array
        """

        return dot(self.C(q, q_dot), q_dot) + self.G(q)

    def drift(self, x, t):
        q, q_dot = reshape(x, (2, -1))
        return concatenate([q_dot, -solve(self.D(q), self.H(q, q_dot))])

    def act(self, x, t):
        q = self.proportional(x, t)
        return concatenate([zeros((self.k, self.m)), solve(self.D(q), self.B)])

    def proportional(self, x, t):
        return self.eval(x, t)[:self.k]

    def derivative(self, x, t):
        return self.eval(x, t)[self.k:]

class QuadraticCLF(Dynamics):
    """Class for Lyapunov functions of the form V(z) = z' * P * z."""

    def __init__(self, dynamics, P):
        """Create a QuadraticCLF object.

        Inputs:
        Dynamics, dynamics: Dynamics
        Positive-definite matrix, P: numpy array
        """

        self.dynamics = dynamics
        self.P = P

    def eval(self, x, t):
        """Compute V(z).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Lyapunov function value: float
        """

        z = self.dynamics.eval(x, t)
        return dot(z, dot(self.P, z))

    def eval_grad(self, x, t):
        """Compute (dV/dz)'.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Lyapunov function gradient: numpy array
        """

        z = self.dynamics.eval(x, t)
        return 2 * dot(self.P, z)

    def eval_dot(self, x, u, t):
        """Compute dV/dt.

        Inputs:
        State, x: numpy array
        Action, u: numpy array
        Time, t: float

        Outputs:
        Lyapunov function time derivative: float
        """

        return dot(self.eval_grad(x, t), self.dynamics.eval_dot(x, u, t))

class AffineQuadCLF(AffineDynamics, QuadraticCLF):
    """Class for quadratic Lyapunov functions for affine dynamics."""

    def __init__(self, affine_dynamics, P):
        """Build an AffineQuadCLF object.

        Inputs:
        Affine dynamics, affine_dynamics: AffineDynamics
        Positive-definite matrix, P: numpy array
        """

        QuadraticCLF.__init__(self, affine_dynamics, P)

    def drift(self, x, t):
        return dot(self.eval_grad(x, t), self.dynamics.drift(x, t))

    def act(self, x, t):
        return dot(self.eval_grad(x, t), self.dynamics.act(x, t))

    def build_care(affine_linearizable_dynamics, Q, R):
        """Build AffineQuadCLF from affine and linearizable dynamics by solving continuous-time algebraic Riccati equation (CARE).

        CARE is F' * P + P * F - P * G * R^-1 * G' * P = -Q.

        Inputs:
        Affine and linearizable dynamics, affine_linearizable_dynamics: AffineDynamics, LinearizableDynamics
        Positive-definite state cost matrix, Q: numpy array
        Positive-definite action cost matrix, R: numpy array
        """

        F, G = affine_linearizable_dynamics.linear_system()
        P = solve_continuous_are(F, G, Q, R)
        return AffineQuadCLF(affine_linearizable_dynamics, P)

    def build_ctle(affine_linearizable_dynamics, K, Q):
        """Build AffineQuadCLF from affine and linearizable dynamics with gain matrix by solving continuous-time Lyapunov equation (CTLE).

        CTLE is A' * P + P * A = -Q, where A = F - G * K is closed-loop matrix.

        Inputs:
        Affine and linearizable dynamics, affine_linearizable_dynamics: AffineDynamics, LinearizableDynamics
        Positive-definite state cost matrix, Q: numpy array
        Gain matrix, K: numpy array
        """

        A = affine_linearizable_dynamics.closed_loop_linear_system(K)
        P = solve_continuous_lyapunov(A.T, -Q)
        return AffineQuadCLF(affine_linearizable_dynamics, P)
