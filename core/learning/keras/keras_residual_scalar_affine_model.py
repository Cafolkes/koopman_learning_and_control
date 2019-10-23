from .keras_residual_affine_model import KerasResidualAffineModel
from numpy import array

class KerasResidualScalarAffineModel(KerasResidualAffineModel):
    def __init__(self, d_drift_in, d_act_in, d_hidden, m, d_out, optimizer='sgd', loss='mean_absolute_error'):
        KerasResidualAffineModel.__init__(self, d_drift_in, d_act_in, d_hidden, m, d_out, optimizer, loss)

    def eval_drift(self, drift_input):
        return self.drift_model.predict(array([drift_input]))[0][0]

    def eval_act(self, act_input):
        return self.act_model.predict(array([act_input]))[0][0]