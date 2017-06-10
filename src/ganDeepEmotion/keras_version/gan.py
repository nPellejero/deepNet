from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import theano
np.random.seed(1337)  # for reproducibility

from copy import copy, deepcopy

from keras.datasets import mnist
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils, generic_utils
from keras.objectives import binary_crossentropy
from keras.callbacks import Callback
from keras.layers.advanced_activations import LeakyReLU as lrelu
from keras.layers.convolutional import Convolution2D, MaxPooling2D

batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
dim = 300  # dim of the random number vector
mnist_dim = 784

detector = Sequential()
detector.add(Dense(dim, input_dim=mnist_dim))
detector.add(lrelu())
detector.add(Dropout(.3))
detector.add(Dense(dim))
detector.add(Activation('tanh'))
detector.add(Dropout(.3))
detector.add(Dense(1)) # 1: Yes, it belongs to S, 0: fake!
detector.add(Activation('sigmoid'))

# Fully Connected model

sampler = Sequential()
sampler.add(Dense(dim, input_dim=dim))
sampler.add(lrelu())
sampler.add(Dense(dim))
sampler.add(lrelu())
sampler.add(Dense(mnist_dim))
sampler.add(Activation('sigmoid'))

# This is G itself!!!
sample_fake = theano.function([sampler.get_input()], sampler.get_output())

# We add the detector G on top, but it won't be adapted with this cost function.
# But here is a dirty hack: Theano shared variables on the GPU are the same for
# `detector` and `detector_no_grad`, so, when we adapt `detector` the values of
# `detector_no_grad` will be updated as well. But this only happens following the
# correct gradients.
# Don't you love pointers? Aliasing can be our friend sometimes.
detector.trainable = False
sampler.add(detector)

opt_g = Adam(lr=.001) # I got better results when
                      # detector's learning rate is faster
sampler.compile(loss='binary_crossentropy', optimizer=opt_g)

# debug
opt_d = Adam(lr=.002)
detector.trainable = True
detector.compile(loss='binary_crossentropy', optimizer=opt_d)
detector.predict(np.ones((3, mnist_dim))).shape



nb_epoch = 1000 # it takes some time to get something recognizable.
batch_size = 128
num_batches = X_train.shape[0] / 128
fig = plt.figure()
#plt.show()
print("Starting..")
fixed_noise = np.random.uniform(-1, 1, (9, dim)).astype('float32') # Let us visualize how these sampes evolve with training

progbar = generic_utils.Progbar(1000)
#try:
for e in range(nb_epoch):
        loss0 = 0
        loss1 = 0

        for (first, last) in zip(range(0, X_train.shape[0]-batch_size, batch_size),
                                 range(batch_size, X_train.shape[0], batch_size)):
            noise_batch = np.random.uniform(-1, 1, (batch_size, dim)).astype('float32')
            fake_samples = sample_fake(noise_batch)
            true_n_fake = np.concatenate([X_train[first: last],
                                          fake_samples], axis=0)
            y_batch = np.concatenate([np.ones((batch_size, 1)),
                                      np.zeros((batch_size, 1))], axis=0).astype('float32')
            all_fake = np.ones((batch_size, 1)).astype('float32')
            # We take turns adapting G and D. We may give D an upper hand,
            #  letting it train for more turns, keeping G fixed.
            #  Do that increasing the upper hand `uh` variable.
            uh = 2
            if e % uh == 0:
                loss0 += sampler.train_on_batch(noise_batch, all_fake)[0]
            else:
                loss1 += detector.train_on_batch(true_n_fake, y_batch)[0]
            loss = loss0 + loss1

        progbar.add(1, values=[("train loss", loss),
                               ("G loss", loss0),
                               ("D loss", loss1)])

        if e % 10 == 0: # visualize results once in a while
            fixed_fake = sample_fake(fixed_noise)
            plt.clf()
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow(fixed_fake[i].reshape((28,28)), cmap='gray')
                plt.axis('off')
            fig.canvas.draw()
            #plt.show()
            plt.savefig(str(e)+'.png')
#except:
