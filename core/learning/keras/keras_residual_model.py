from keras import Sequential
from keras.layers import Dense
from numpy import array
from numpy.random import permutation

from .. import ResidualModel

class KerasResidualModel(ResidualModel):
    def __init__(self, d_in, d_hidden, d_out, optimizer='sgd', loss='mean_absolute_error'):
        model = Sequential()
        model.add(Dense(d_hidden, input_shape=(d_in,), activation='relu'))
        model.add(Dense(d_out))
        model.compile(optimizer, loss)
        self.model = model

    def eval_dot(self, input):
        return self.model.predict(array([input]))[0]

    def shuffle(self, inputs, residuals):
        perm = permutation(len(inputs))
        return inputs[perm], residuals[perm]

    def fit(self, inputs, residuals, batch_size=1, num_epochs=1, validation_split=0):
        inputs, residuals = self.shuffle(inputs, residuals)
        self.model.fit(inputs, residuals, batch_size=batch_size, epochs=num_epochs, validation_split=validation_split)
