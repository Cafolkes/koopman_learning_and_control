from numpy import argmin, array, Inf, mean, sqrt, zeros
from numpy.linalg import norm
from numpy.ma import masked_array
from numpy.random import permutation
from sys import stdout

from ..util import arr_map

class ConvexBody:
    def __init__(self, dim):
        self.dim = dim

    def sample(self, N):
        raise NotImplementedError

    def is_member(self, xs):
        raise NotImplementedError

    def uniform_grid(self, N):
        raise NotImplementedError

    def uniform_list(self, N):
        raise NotImplementedError

    def safety(self, affine_dynamics):
        raise NotImplementedError

    def voronoi_iteration(self, N, k, tol, verbose=False):
        samples = self.sample(N)

        def centers_to_clusters(centers):
            cluster_idxs = argmin([norm(samples - center, axis=1) for center in centers], axis=0)
            clusters = [samples[cluster_idxs == idx] for idx in range(k)]
            return clusters

        centers = samples[permutation(N)[:k]]
        clusters = centers_to_clusters(centers)

        def total_distance(clusters, centers):
            return sqrt(sum(sum(norm(cluster - center, axis=1) ** 2) for cluster, center in zip(clusters, centers)))

        prev_dist = Inf
        dist = total_distance(centers, clusters)
        iters = 0
        while abs(prev_dist - dist) > tol:
            iters += 1
            if verbose:
                stdout.write('Iterations: ' + str(iters) + '\tChange in total distance: ' + str(abs(prev_dist - dist)) + '\r')
                stdout.flush()
            prev_dist = dist
            centers = array([mean(cluster, axis=0) for cluster in clusters])
            clusters = centers_to_clusters(centers)
            dist = total_distance(centers, clusters)

        return clusters, centers

    def grid_map(self, func, grids):
        xs = array([grid.compressed() for grid in grids]).T
        vals = arr_map(func, xs)

        mask = grids[0].mask
        idxs = ~mask
        shape = grids[0].shape
        val_grid = masked_array(zeros(shape), mask)
        val_grid[idxs] = vals

        return val_grid
