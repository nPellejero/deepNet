from time import time
import numpy as np

import theano
import theano.tensor as T

from lib import updates
from lib import inits
from lib.theano_utils import floatX, sharedX
from lib.costs import mse, mae
from lib.rng import np_rng


class Z_reconstructor(object):

    def __init__(self, model, batch_size,
                 z_updater = updates.Adam(lr=sharedX(0.0002), b1=0.5, regularizer=updates.Regularizer(l2=1e-5))):

        ifn = inits.Constant()

        self.batch_size = batch_size
        self.gen_dim = model.gen_dim
        self.Z = ifn((batch_size, self.gen_dim), 'zsh')

        self.gen_fn, _ = model.gen_function(True, self.Z, *model.gen_params)
        # self.cost = mse(model.X, self.gen_fn)
        self.cost = mae(model.X, self.gen_fn)
        z_updates = z_updater([self.Z], self.cost)

        self._train_z = theano.function([model.X], self.cost, updates=z_updates)
        self._reconstruct = theano.function([], self.gen_fn)

    def set_Z_value(self, value):
        self.Z.set_value(value)

    def set_Z_uniform(self):
        self.Z.set_value(floatX(np_rng.uniform(-1., 1, size=(self.batch_size, self.gen_dim))))

    def set_Z_zero(self):
        self.Z.set_value(floatX(np.zeros((self.batch_size, self.gen_dim))))

    def train_z_on_batch(self, X_batch):
        return self._train_z(X_batch)

    def reconstruct(self):
        return self._reconstruct()

    def get_Z_value(self):
        return self.Z.get_value()

class partial_reconstructor(object):

    def __init__(self, model, layer, layer_shape, batch_size,
                 h_updater = updates.Adam(lr=sharedX(0.0002), b1=0.5, regularizer=updates.Regularizer(l2=1e-5))):

        ifn = inits.Constant()

        self.batch_size = batch_size
        self.gen_dim = model.gen_dim
        self.H_shape = (batch_size,) + layer_shape
        self.H = ifn(self.H_shape, 'Hsh')

        self.gen_fn = model.gen_function_partial(layer, self.H, *model.gen_params)
        # self.cost = mse(model.X, self.gen_fn)
        self.cost = mae(model.X, self.gen_fn)
        h_updates = h_updater([self.H], self.cost)

        self._train_h = theano.function([model.X], self.cost, updates=h_updates)
        self._reconstruct = theano.function([], self.gen_fn)

    def set_H_value(self, value):
        self.H.set_value(value)

    def set_H_uniform(self):
        self.H.set_value(floatX(np_rng.uniform(0., 1, size=self.H_shape)))

    def set_H_zero(self):
        self.H.set_value(floatX(np.zeros(self.H_shape)))

    def train_h_on_batch(self, X_batch):
        return self._train_h(X_batch)

    def reconstruct(self):
        return self._reconstruct()

    def get_H_value(self):
        return self.H.get_value()
