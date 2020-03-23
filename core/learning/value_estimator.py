from numpy.linalg import solve

from .gaussian_process_estimator import GaussianProcessEstimator
from .. import arr_map

class ValueEstimator(GaussianProcessEstimator):
    def __init__(self, kernel, gamma, states, next_states, rewards):
        GaussianProcessEstimator.__init__(self, kernel, states)
        embedded_next_states = arr_map(self.embedding, next_states)
        self.weights = solve( self.kernel_mat - gamma * embedded_next_states, (1 - gamma) * rewards )

    def build(kernel, f, pi, R, gamma, states):
        actions = arr_map(pi, states)
        next_states = arr_map(f, states, actions)
        rewards = arr_map(R, states, actions, next_states)
        return ValueEstimator(kernel, gamma, states, next_states, rewards)
