from numpy.linalg import solve

from .gaussian_process_estimator import GaussianProcessEstimator
from .. import arr_map

class ValueEstimator(GaussianProcessEstimator):
    def __init__(self, kernel, gamma, states, next_states, rewards):
        GaussianProcessEstimator.__init__(self, kernel, states)
        embedded_next_states = arr_map(self.embedding, next_states)
        self.weights = solve( self.kernel_mat - gamma * embedded_next_states, (1 - gamma) * rewards )
        
    def transition_map(system, dt, atol=1e-6, rtol=1e-6):
        
        def f(s, a):
            return system.step(s, a, 0, dt, atol, rtol)
        
        return f
    
    def policy(controller):
        
        def pi(s):
            return controller.process(controller.eval(s, 0))
        
        return pi
        
    def build(kernel, system, controller, R, gamma, states, dt, atol=1e-6, rtol=1e-6):
        f = ValueEstimator.transition_map(system, dt, atol, rtol)
        pi = ValueEstimator.policy(controller)
        return ValueEstimator._build(kernel, f, pi, R, gamma, states)
        
    def _build(kernel, f, pi, R, gamma, states):
        _, next_states, rewards = ValueEstimator.gen_data(f, pi, R, states)
        return ValueEstimator(kernel, gamma, states, next_states, rewards)
    
    def gen_data(f, pi, R, states):
        actions = arr_map(pi, states)
        next_states = arr_map(f, states, actions)
        rewards = arr_map(R, states, actions, next_states)
        return actions, next_states, rewards
    
    
        
