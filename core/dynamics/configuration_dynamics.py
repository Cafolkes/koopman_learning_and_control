from numpy import array, concatenate, dot, reshape, tensordot, zeros

from .fb_lin_dynamics import FBLinDynamics
from .pd_dynamics import PDDynamics

class ConfigurationDynamics(FBLinDynamics, PDDynamics):
    def __init__(self, robotic_dynamics, k):
        relative_degrees = [2] * k
        perm = concatenate([array([j, j + k]) for j in range(k)])
        FBLinDynamics.__init__(self, relative_degrees, perm)
        self.robotic_dynamics = robotic_dynamics
        self.k = k

    def y(self, q):
        pass

    def dydq(self, q):
        pass

    def d2ydq2(self, q):
        pass

    def y_d(self, t):
        return zeros(self.k)

    def y_d_dot(self, t):
        return zeros(self.k)

    def y_d_ddot(self, t):
        return zeros(self.k)

    def eval(self, x, t):
        return concatenate([self.proportional(x, t), self.derivative(x, t)])

    def drift(self, x, t):
        q, q_dot = reshape(x, (2, -1))
        q_ddot_drift = self.robotic_dynamics.drift(x, t)[self.robotic_dynamics.k:]
        err_ddot_drift = dot(tensordot(self.d2ydq2(q), q_dot, 1), q_dot) + dot(self.dydq(q), q_ddot_drift) - self.y_d_ddot(t)
        return concatenate([self.derivative(x, t), err_ddot_drift])

    def act(self, x, t):
        q = self.robotic_dynamics.proportional(x, t)
        q_ddot_act = self.robotic_dynamics.act(x, t)[self.robotic_dynamics.k:]
        return concatenate([zeros((self.k, self.robotic_dynamics.m)), dot(self.dydq(q), q_ddot_act)])

    def proportional(self, x, t):
        q = self.robotic_dynamics.proportional(x, t)
        return self.y(q) - self.y_d(t)

    def derivative(self, x, t):
        q, q_dot = reshape(x, (2, -1))
        return dot(self.dydq(q), q_dot) - self.y_d_dot(t)
