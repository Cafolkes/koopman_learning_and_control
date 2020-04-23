from .box import Box
from .convex_body import ConvexBody
from ..dynamics import AffineDynamics, ScalarDynamics

from numpy import cos, dot, identity, linspace, pi, reshape, sin, zeros
from numpy.linalg import norm
from numpy.ma import masked_array
from numpy.random import multivariate_normal, rand

class Ball(ConvexBody):
    def __init__(self, dim):
        ConvexBody.__init__(self, dim)

    def sample(self, N):
        mean = zeros(self.dim)
        cov = identity(self.dim)
        normal_samples = multivariate_normal(mean, cov, N)
        sphere_samples = normal_samples / norm(normal_samples, axis=1, keepdims=True)
        radius_samples = rand(N, 1) ** (1 / self.dim)
        return radius_samples * sphere_samples

    def is_member(self, xs):
        return norm(xs, axis=1) <= 1

    def uniform(self, N):
        xs = Box(self.dim).uniform_list(N)
        idxs = self.is_member(xs)
        return xs, idxs

    def uniform_grid(self, N):
        shape = [N] * self.dim
        xs, idxs = self.uniform(N)
        mask = reshape(~idxs, shape)
        return [masked_array(reshape(arr, shape), mask) for arr in xs.T]

    def uniform_list(self, N):
        xs, idxs = self.uniform(N)
        return xs[idxs]

    def safety(self, affine_dynamics):
        return self.SafetyDynamics(affine_dynamics)

    class SafetyDynamics(AffineDynamics, ScalarDynamics):
        def __init__(self, affine_dynamics):
            self.dynamics = affine_dynamics

        def eval(self, x, t):
            return 1 - norm(self.dynamics.eval(x, t)) ** 2

        def eval_grad(self, x, t):
            return -2 * self.dynamics.eval(x, t)

        def drift(self, x, t):
            return dot(self.eval_grad(x, t), self.dynamics.drift(x, t))

        def act(self, x, t):
            return dot(self.eval_grad(x, t), self.dynamics.act(x, t))
