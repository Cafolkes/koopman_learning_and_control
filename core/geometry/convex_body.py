from numpy import argmin, array, Inf, mean, sqrt
from numpy.linalg import norm
from numpy.random import permutation
from sys import stdout

class ConvexBody:
    def __init__(self, dim):
        self.dim = dim

    def sample(self, N):
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
