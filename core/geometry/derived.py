from numpy import dot, ones, zeros
from numpy.linalg import inv
from numpy.ma import masked_array

from .convex_body import ConvexBody
from ..dynamics import AffineDynamics
from ..util import arr_map

class Derived(ConvexBody):
    def __init__(self, primitive, T, c):
        ConvexBody.__init__(self, primitive.dim)
        self.primitive = primitive
        self.T = T
        self.T_inv = inv(T)
        self.c = c

    def from_primitive(self, xs):
        return dot(self.T_inv, xs.T).T + self.c

    def to_primitive(self, xs):
        return dot(self.T, (xs - self.c).T).T

    def sample(self, N):
        return self.from_primitive(self.primitive.sample(N))

    def is_member(self, xs):
        return self.primitive.is_member(self.to_primitive(xs))

    def uniform_grid(self, N):
        grids = self.primitive.uniform_grid(N)
        primitive_list = arr_map(lambda grid: grid.compressed(), grids).T
        derived_list = self.from_primitive(primitive_list)

        mask = grids[0].mask
        idxs = ~mask
        shape = [N] * self.dim
        derived_grids = [masked_array(zeros(shape), mask) for _ in range(self.dim)]
        for i in range(self.dim):
            derived_grids[i][idxs] = derived_list[:, i]
        return derived_grids

    def uniform_list(self, N):
        return self.from_primitive(self.primitive.uniform_list(N))

    def safety(self, affine_dynamics):
        derived_dynamics = self.DerivedDynamics(self, affine_dynamics)
        return self.primitive.safety(derived_dynamics)

    class DerivedDynamics(AffineDynamics):
        def __init__(self, derived, affine_dynamics):
            self.derived = derived
            self.dynamics = affine_dynamics

        def eval(self, x, t):
            return self.derived.to_primitive(self.dynamics.eval(x, t))

        def drift(self, x, t):
            return dot(self.derived.T, self.dynamics.drift(x, t))

        def act(self, x, t):
            return dot(self.derived.T, self.dynamics.act(x, t))
