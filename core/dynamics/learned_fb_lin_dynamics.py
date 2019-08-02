from numpy import array, zeros

from .. import differentiate
from .fb_lin_dynamics import FBLinDynamics
from .learned_affine_dynamics import LearnedAffineDynamics

class LearnedFBLinDynamics(LearnedAffineDynamics, FBLinDynamics):
    def __init__(self, fb_lin_dynamics, res_aff_model):
        LearnedAffineDynamics.__init__(self, fb_lin_dynamics, res_aff_model)
        FBLinDynamics.__init__(self, fb_lin_dynamics.relative_degrees, fb_lin_dynamics.perm)

    def drift(self, x, t):
        drift_nom = self.dynamics.drift(x, t)
        drift_res = zeros(drift_nom.shape)
        drift_res[self.dynamics.relative_degree_idxs] = self.res_model.eval_drift(self.process_drift(x, t))
        drift_res = self.inv_permute(drift_res)
        return drift_nom + drift_res

    def act(self, x, t):
        act_nom = self.dynamics.act(x, t)
        act_res = zeros(act_nom.shape)
        act_res[self.dynamics.relative_degree_idxs] = self.res_model.eval_act(self.process_act(x, t))
        act_res = self.dynamics.inv_permute(act_res)
        return act_nom + act_res

    def process_episode(self, xs, us, ts, window=3):
        drift_inputs, act_inputs, us, residuals = super().process_episode(xs, us, ts, window)
        residuals = array([self.dynamics.select(self.dynamics.permute(residual)) for residual in residuals])
        return drift_inputs, act_inputs, us, residuals
