from .learned_affine_dynamics import LearnedAffineDynamics
from numpy import zeros

class LearnedScalarAffineDynamics(LearnedAffineDynamics):
    def __init__(self, affine_dynamics, res_aff_model):
        LearnedAffineDynamics.__init__(self, affine_dynamics, res_aff_model)
        
    def init_data(self, d_drift_in, d_act_in, m, d_out):
        return [zeros((0, d_drift_in)), zeros((0, d_act_in)), zeros((0, m)), zeros(0)]