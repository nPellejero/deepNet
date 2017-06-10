from time import time

import theano
import theano.tensor as T

from lib import updates
from lib import inits
from lib.theano_utils import sharedX

bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy

class GAN_trainer(object):

    def __init__(self,model,
                 dis_updater = updates.Adam(lr=sharedX(0.0002), b1=0.5, regularizer=updates.Regularizer(l2=1e-5)),
                 gen_updater = updates.Adam(lr=sharedX(0.0002), b1=0.5, regularizer=updates.Regularizer(l2=1e-5))):

        X = model.X
        Z = model.Z
        targets = T.matrix()

        genX = model.genX

        disX = model.disX
        disgenX = model.disgenX

        disX_loss = bce(disX, T.ones(disX.shape)).mean()
        disgenX_loss = bce(disgenX, T.zeros(disgenX.shape)).mean()
        genX_loss = bce(disgenX, T.ones(disgenX.shape)).mean()

        dis_loss = disX_loss + disgenX_loss
        gen_loss = genX_loss

        trainable_discrim_params = model.trainable_discrim_params
        trainable_gen_params = model.trainable_gen_params

        dis_updates = dis_updater(trainable_discrim_params, dis_loss) + model.other_discrim_updates
        gen_updates = gen_updater(trainable_gen_params, gen_loss) + model.other_gen_updates

        print 'COMPILING'
        t = time()
        self._train_gen = theano.function([Z], gen_loss, updates=gen_updates)
        self._train_dis = theano.function([X, Z], dis_loss, updates=dis_updates)
        self._gen = theano.function([Z], genX)
        print '%.2f seconds to compile theano functions'%(time()-t)

    def train_on_batch(self,X_batch,Z_batch):
        cost_gen = self._train_gen(X_batch, Z_batch)
        cost_dis_real, cost_dis_gen = self._train_dis(X_batch, Z_batch)
        return cost_gen,cost_dis_real,cost_dis_gen

    def train_discriminator_on_batch(self,X_batch,Z_batch):
        return self._train_dis(X_batch, Z_batch)

    def train_generator_on_batch(self,Z_batch):
        return self._train_gen(Z_batch)

