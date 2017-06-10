from time import time

import numpy as np
import theano
import theano.tensor as T

from lib import updates
from lib import inits
from lib.theano_utils import floatX, sharedX

bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy

class GAN_trainer(object):

    def __init__(self,model,
                 dis_updater = updates.Adam(lr=sharedX(0.0002), b1=0.5, regularizer=updates.Regularizer(l2=1e-5)),
                 gen_updater = updates.Adam(lr=sharedX(0.0002), b1=0.5, regularizer=updates.Regularizer(l2=1e-5)),
                 cls_updater = updates.Adam(lr=sharedX(0.0002), b1=0.5, regularizer=updates.Regularizer(l2=1e-5))):

        X = model.X
        Z = model.Z
        y = T.matrix() # input con ceros o unos segun si X es gen o real respectivamente
        targets = T.matrix()

        genX = model.genX

        disX = model.disX
        classX = model.classX
        disgenX = model.disgenX
        classgenX = model.classgenX
        classXTest = model.classXTest

        disX_loss = bce(disX, y).mean()
        disgenX_loss = bce(disgenX, T.zeros(disgenX.shape)).mean()
        genX_loss = bce(disgenX, T.ones(disgenX.shape)).mean()
        cls_loss = cce(classX, targets).mean()
        cls_err = T.mean(T.neq(T.argmax(classXTest,axis=1),T.argmax(targets,axis=1)))

        dis_loss = disX_loss + disgenX_loss
        gen_loss = genX_loss

        trainable_discrim_params = model.trainable_discrim_params
        trainable_gen_params = model.trainable_gen_params
        trainable_classif_params = model.trainable_classif_params

        dis_updates = dis_updater(trainable_discrim_params, dis_loss) + model.other_discrim_updates
        gen_updates = gen_updater(trainable_gen_params, gen_loss) + model.other_gen_updates
        cls_updates = cls_updater(trainable_classif_params, cls_loss) + model.other_classif_updates

        print 'COMPILING'
        t = time()
        self._train_gen = theano.function([Z], gen_loss, updates=gen_updates)
        self._train_dis = theano.function([X, y, Z], dis_loss, updates=dis_updates)
        self._train_cls = theano.function([X, targets], cls_loss, updates=cls_updates)
        self._gen = theano.function([Z], genX)
        self._cls_predict = theano.function([X],classXTest)
        self._cls_error = theano.function([X,targets], cls_err)
        print '%.2f seconds to compile theano functions'%(time()-t)

    def train_on_batch(self,X_batch,Z_batch):
        cost_gen = self._train_gen(X_batch, Z_batch)
        cost_dis_real, cost_dis_gen = self._train_dis(X_batch, Z_batch)
        return cost_gen,cost_dis_real,cost_dis_gen

    def train_discriminator_on_batch(self,X_batch,Z_batch,y_batch=None):
        if y_batch is None: # Supongo en este caso que X_batch son todas reales
            y_batch = floatX(np.ones((X_batch.shape[0],1)))
        return self._train_dis(X_batch, y_batch, Z_batch)

    def train_generator_on_batch(self,Z_batch):
        return self._train_gen(Z_batch)

    def train_classifier_on_batch(self,X_batch,y_batch):
        return self._train_cls(X_batch,y_batch)

    def test_classifier_on_batch(self, X_batch, y_batch):
        return self._cls_error(X_batch, y_batch)
