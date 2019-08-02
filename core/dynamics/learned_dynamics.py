from numpy import array, concatenate, zeros

from .. import differentiate
from .dynamics import Dynamics

class LearnedDynamics(Dynamics):
    def __init__(self, dynamics, res_model):
        self.dynamics = dynamics
        self.res_model = res_model

    def process(self, x, u, t):
        return concatenate([x, u, array([t])])

    def eval(self, x, t):
        return self.dynamics.eval(x, t)

    def eval_dot(self, x, u, t):
        return self.dynamics.eval_dot(x, u, t) + self.res_model.eval_dot(self.process(x, u, t))

    def process_episode(self, xs, us, ts, window=3):
        half_window = (window - 1) // 2
        xs = xs[:len(us)]
        ts = ts[:len(us)]

        inputs = array([self.process(x, u, t) for x, u, t in zip(xs, us, ts)])

        reps = array([self.dynamics.eval(x, t) for x, t in zip(xs, ts)])
        rep_dots = differentiate(reps, ts)

        rep_dot_noms = array([self.dynamics.eval_dot(x, u, t) for x, u, t in zip(xs, us, ts)])

        inputs = inputs[half_window:-half_window]
        rep_dot_noms = rep_dot_noms[half_window:-half_window]

        residuals = rep_dots - rep_dot_noms

        return inputs, residuals

    def init_data(self, d_in, d_out):
        return [zeros((0, d_in)), zeros((0, d_out))]

    def aggregate_data(self, old_data, new_data):
        return [concatenate(old, new) for old, new in zip(old_data, new_data)]

    def fit(self, data, batch_size=1, num_epochs=1, validation_split=0):
        inputs, residuals = data
        self.res_model.fit(inputs, residuals, batch_size, num_epochs, validation_split)
