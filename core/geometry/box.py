from numpy import Inf, linspace, meshgrid, reshape
from numpy.linalg import norm
from numpy.ma import masked_array
from numpy.random import rand

from .convex_body import ConvexBody
from ..util import arr_map

class Box(ConvexBody):
    def sample(self, N):
        return 2 * rand(N, self.dim) - 1

    def is_member(self, xs):
        return norm(xs, Inf, axis=1) <= 1

    def meshgrid(self, N):
        interval = linspace(-1, 1, N)
        return meshgrid(*([interval] * self.dim), indexing='ij')

    def uniform_grid(self, N):
        grids = self.meshgrid(N)
        return [masked_array(grid, mask=(False * grid)) for grid in grids]

    def uniform_list(self, N):
        grids = self.meshgrid(N)
        return arr_map(lambda grid: reshape(grid, -1), grids).T
