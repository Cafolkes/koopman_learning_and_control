from numpy import array, concatenate, zeros

from .. import differentiate
from .affine_dynamics import AffineDynamics

class LearnedAffineDynamics(AffineDynamics):
    def __init__(self, affine_dynamics, res_aff_model):
        self.dynamics = affine_dynamics
        self.res_model = res_aff_model

    def process_drift(self, x, t):
        return concatenate([x, array([t])])

    def process_act(self, x, t):
        return concatenate([x, array([t])])

    def eval(self, x, t):
        return self.dynamics.eval(x, t)

    def drift(self, x, t):
        return self.dynamics.drift(x, t) + self.res_model.eval_drift(self.process_drift(x, t))

    def act(self, x, t):
        return self.dynamics.act(x, t) + self.res_model.eval_act(self.process_act(x, t))

    def process_episode(self, xs, us, ts, window=3):
        half_window = (window - 1) // 2
        xs = xs[:len(us)]
        ts = ts[:len(us)]

        drift_inputs = array([self.process_drift(x, t) for x, t in zip(xs, ts)])
        act_inputs = array([self.process_act(x, t) for x, t in zip(xs, ts)])

        reps = array([self.dynamics.eval(x, t) for x, t in zip(xs, ts)])
        rep_dots = differentiate(reps, ts)

        rep_dot_noms = array([self.dynamics.eval_dot(x, u, t) for x, u, t in zip(xs, us, ts)])

        drift_inputs = drift_inputs[half_window:-half_window]
        act_inputs = act_inputs[half_window:-half_window]
        rep_dot_noms = rep_dot_noms[half_window:-half_window]

        residuals = rep_dots - rep_dot_noms

        return drift_inputs, act_inputs, us, residuals

    def init_data(self, d_drift_in, d_act_in, m, d_out):
        return [zeros((0, d_drift_in)), zeros((0, d_act_in)), zeros((0, m)), zeros((0, d_out))]

    def fit(self, data, batch_size=1, num_epochs=1, validation_split=0):
        drift_inputs, act_inputs, us, residuals = data
        self.res_model.fit(drift_inputs, act_inputs, us, residuals, batch_size, num_epochs, validation_split)
