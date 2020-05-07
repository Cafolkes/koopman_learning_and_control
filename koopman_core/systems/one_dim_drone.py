from numpy import array, ones_like, zeros
from scipy.integrate import solve_ivp

from core.dynamics import RoboticDynamics


class OneDimDrone(RoboticDynamics):

    def __init__(self, mass, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover):
        """One dimensional drone model
        Arguments:
            RoboticDynamics {dynamical system} -- dynamics
            mass {float} -- mass
            rotor_rad {float} -- rotor radius
            drag_coeff {float} -- drag coefficient
            air_dens {float} -- air density
            area {float} -- drone surface area in xy plane
        Keyword Arguments:
            gravity {float} -- gravity (default: {9.81})
        """
        RoboticDynamics.__init__(self, 1, 1)
        self.mass = mass
        self.rotor_rad = rotor_rad
        self.drag_coeff = drag_coeff
        self.air_dens = air_dens
        self.area = area
        self.gravity = gravity
        self.ground_altitude = ground_altitude
        self.T_hover = T_hover

    def D(self, q):
        return array([[self.mass]])

    def C(self, q, q_dot):
        return array([[0.5*self.drag_coeff*self.air_dens*self.area*q_dot]])

    def G(self, q):
        return array([self.mass*self.gravity])

    def B(self, q):
        return array([1/max((1-(self.rotor_rad/(4*q))**2), 0.5*ones_like(q))])

    def simulate(self, x_0, controller, ts, processed=True, atol=1e-6, rtol=1e-6):
        """Simulate system from initial state with specified controller.
        Approximated using Runge-Kutta 4,5 solver.
        Actions computed at time steps and held constant over sample period.
        Inputs:
        Initial state, x_0: numpy array
        Control policy, controller: Controller
        Time steps, ts: numpy array
        Flag to process actions, processed: bool
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
            if x[0] < self.ground_altitude:
                x[0] = self.ground_altitude
                x[1] = -x[1]

            t = ts[j]
            u = controller.eval(x, t) + self.T_hover
            us[j] = u
            u = controller.process(u)
            x_dot = lambda t, x: self.eval_dot(x, u, t)
            t_span = [t, ts[j + 1]]
            res = solve_ivp(x_dot, t_span, x, atol=atol, rtol=rtol)
            xs[j + 1] = res.y[:, -1]

        if processed:
            us = array([controller.process(u) for u in us])

        return xs, us