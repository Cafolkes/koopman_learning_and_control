from keras import Model, Sequential
from keras.layers import Add, Dense, Dot, Input, Reshape
from numpy import array
from numpy.random import permutation

from .. import ResidualAffineModel

class KerasResidualAffineModel(ResidualAffineModel):
    def __init__(self, d_drift_in, d_act_in, d_hidden, m, d_out, optimizer='sgd', loss='mean_absolute_error'):
        drift_model = Sequential()
        drift_model.add(Dense(d_hidden, input_shape=(d_drift_in,), activation='relu'))
        drift_model.add(Dense(d_out))
        self.drift_model = drift_model

        drift_inputs = Input((d_drift_in,))
        drift_residuals = self.drift_model(drift_inputs)

        act_model = Sequential()
        act_model.add(Dense(d_hidden, input_shape=(d_act_in,), activation='relu'))
        act_model.add(Dense(d_out * m))
        act_model.add(Reshape((d_out, m)))
        self.act_model = act_model

        act_inputs = Input((d_act_in,))
        act_residuals = self.act_model(act_inputs)

        us = Input((m,))

        residuals = Add()([drift_residuals, Dot([2, 1])([act_residuals, us])])
        model = Model([drift_inputs, act_inputs, us], residuals)
        model.compile(optimizer, loss)
        self.model = model

    def eval_drift(self, drift_input):
        return self.drift_model.predict(array([drift_input]))[0]

    def eval_act(self, act_input):
        return self.act_model.predict(array([act_input]))[0]

    def shuffle(self, drift_inputs, act_inputs, us, residuals):
        perm = permutation(len(residuals))
        return drift_inputs[perm], act_inputs[perm], us[perm], residuals[perm]

    def fit(self, drift_inputs, act_inputs, us, residuals, batch_size=1, num_epochs=1, validation_split=0):
        drift_inputs, act_inputs, us, residuals = self.shuffle(drift_inputs, act_inputs, us, residuals)
        self.model.fit([drift_inputs, act_inputs, us], residuals, batch_size=batch_size, epochs=num_epochs, validation_split=validation_split)
