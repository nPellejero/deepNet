from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
import numpy as np
import argparse
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from keras import backend as K
import images
from custom_layers import UpSampling2D, Convolution2D, Dense, Deconvolution2D
from keras.regularizers import l2
#from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
#from normalization import BatchNormalization


parser = argparse.ArgumentParser()
parser.add_argument("--gendim", type = int, default = 100)
parser.add_argument("--dataset", type = str, default = 'stl10')
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--n_epoch", type = int, default = 100)
parser.add_argument("--k_iter", type = int, default = 1)
parser.add_argument("--monitor_size", type = int, default = 5)
parser.add_argument("--init_scale", type = float, default = 0.02)
parser.add_argument("--folds", type = int, default = 5)
parser.add_argument("--valid_fold", type = int, default = 0)
parser.add_argument("--iter_save", type = int, default = 100)
parser.add_argument("--l2wdecay", type = float, default = 1e-5)
parser.add_argument('--small', action='store_true')
parser.add_argument('--classify', action='store_true')
args = parser.parse_args()
print args

gen_dim = args.gendim
n_epoch = args.n_epoch
batch_size = args.batch_size
dataset = args.dataset
k_iter = args.k_iter
monitor_size = args.monitor_size
smallmodel = args.small
folds = args.folds
valid_fold = args.valid_fold
iter_save = args.iter_save
classify = args.classify
l2wdecay = l2(args.l2wdecay)

def my_normal(shape, scale=args.init_scale, name=None):
    return K.variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                      name=name)

def my_normal4(shape, scale=args.init_scale*16, name=None):
    return K.variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                      name=name)
def get_models(dataset, small=False):
    if dataset == "mnist":
        return mnist_gen(),mnist_dis()
    elif dataset == "stl10":
        if small:
            return image_gen_small(), image_dis_small()
        else:
            return image_gen(),image_dis()
    elif dataset == "stl10_64x64":
        return image64x64_gen(),image64x64_dis()


def image_gen():
    x = Input(shape=(gen_dim,))
    h = Dense(output_dim=1024*4*4, init=my_normal, W_regularizer=l2wdecay)(x)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = Reshape(target_shape=(1024, 4, 4))(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Convolution2D(512, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Convolution2D(256, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Convolution2D(128, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = UpSampling2D(size=(3, 3))(h)
    h = Convolution2D(3, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    y = Activation('tanh')(h)
    return Model(input=x, output=y)

def image_dis():
    x = Input(shape=(3, 96, 96))
    h = Convolution2D(128, 5, 5, subsample=(3, 3), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(x)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(axis=1)(h)
    h = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(1024, 5, 5, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Flatten()(h)
    h1 = Dense(output_dim=1, init=my_normal, W_regularizer=l2wdecay)(h)
    y = Activation('sigmoid')(h1)
    h2 = Dense(output_dim=10, init=my_normal, W_regularizer=l2wdecay)(h)
    y2 = Activation('softmax')(h2)
    return Model(input=x, output=y), Model(input=x, output=y2)

def image64x64_gen():
    x = Input(shape=(gen_dim,))
    h = Dense(output_dim=1024*4*4, init=my_normal, W_regularizer=l2wdecay)(x)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = Reshape(target_shape=(1024, 4, 4))(h)
    #h = UpSampling2D(size=(2, 2))(h)
    #h = Convolution2D(512, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = Deconvolution2D(512, 5, 5, border_mode='same', subsample=(2,2), init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    #h = UpSampling2D(size=(2, 2))(h)
    #h = Convolution2D(256, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = Deconvolution2D(256, 5, 5, border_mode='same', subsample=(2,2), init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    #h = UpSampling2D(size=(2, 2))(h)
    #h = Convolution2D(128, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = Deconvolution2D(128, 5, 5, border_mode='same', subsample=(2,2), init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    #h = UpSampling2D(size=(2, 2))(h)
    #h = Convolution2D(3, 5, 5, border_mode='same', init=my_normal4, W_regularizer=l2wdecay)(h)
    h = Deconvolution2D(3, 5, 5, border_mode='same', subsample=(2,2), init=my_normal, W_regularizer=l2wdecay)(h)
    y = Activation('tanh')(h)
    return Model(input=x, output=y)

def image64x64_dis():
    x = Input(shape=(3, 64, 64))
    h = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(x)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(axis=1)(h)
    h = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(1024, 5, 5, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Flatten()(h)
    h1 = Dense(output_dim=1, init=my_normal, W_regularizer=l2wdecay)(h)
    y = Activation('sigmoid')(h1)
    h2 = Dense(output_dim=10, init=my_normal, W_regularizer=l2wdecay)(h)
    y2 = Activation('softmax')(h2)
    return Model(input=x, output=y), Model(input=x, output=y2)

def image_gen_small():
    x = Input(shape=(gen_dim,))
    h = Dense(output_dim=256*4*4, init=my_normal, W_regularizer=l2wdecay)(x)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = Reshape(target_shape=(256, 4, 4))(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Convolution2D(128, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Convolution2D(64, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Convolution2D(32, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = UpSampling2D(size=(3, 3))(h)
    h = Convolution2D(3, 5, 5, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    y = Activation('tanh')(h)
    return Model(input=x, output=y)

def image_dis_small():
    x = Input(shape=(3, 96, 96))
    h = Convolution2D(32, 5, 5, subsample=(3, 3), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(x)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(axis=1)(h)
    h = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Flatten()(h)
    h2 = Dense(output_dim=10, init=my_normal, W_regularizer=l2wdecay)(h)
    y2 = Activation('softmax')(h2)
    h1 = Dense(output_dim=1, init=my_normal, W_regularizer=l2wdecay)(h2)
    y = Activation('sigmoid')(h1)
    return Model(input=x, output=y), Model(input=x, output=y2)


def mnist_gen():
    x = Input(shape=(gen_dim,))
    h = Dense(output_dim=gen_dim, init=my_normal, W_regularizer=l2wdecay)(x)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = Dense(output_dim=64*7*7, init=my_normal, W_regularizer=l2wdecay)(x)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = Reshape(target_shape=(64, 7, 7))(h)
    h = Convolution2D(64, 3, 3, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Convolution2D(32, 3, 3, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Convolution2D(1, 3, 3, border_mode='same', init=my_normal, W_regularizer=l2wdecay)(h)
    y = Activation('sigmoid')(h)
    return Model(input=x, output=y)

def mnist_dis():
    x = Input(shape=(1, 28, 28))
    h = Convolution2D(16, 7, 7, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(x)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(axis=1)(h)
    h = Convolution2D(32, 3, 3, subsample=(2, 2), border_mode = 'same', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode = 'valid', init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h = Flatten()(h)
    h = Dense(64*3*3, init=my_normal, W_regularizer=l2wdecay)(h)
    h = BatchNormalization(axis=1)(h)
    h = LeakyReLU(0.2)(h)
    h1 = Dense(output_dim=1, init=my_normal, W_regularizer=l2wdecay)(h)
    y = Activation('sigmoid')(h1)
    h2 = Dense(output_dim=10, init=my_normal, W_regularizer=l2wdecay)(h)
    y2 = Activation('softmax')(h2)
    return Model(input=x, output=y), Model(input=x, output=y2)

## DATASET

def get_batch(X, index, batch_size):
    size = X.shape[0]
    n1 = (index*batch_size)%size
    n2 = ((index+1)*batch_size)%size
    if n1>n2:
        return np.concatenate((X[n1:], X[:n2]))
    else:
        return X[n1:n2]

def scale_img(X,scale,bias):
    return X / scale - bias

if dataset == "mnist":
    from keras.datasets import mnist
    # input image dimensions
    img_rows, img_cols = 28, 28
    img_shape = (28,28)
    cmap = 'gray'
    img_scale = np.float32(255.0)
    img_bias =  np.float32(0.0)
    nb_classes=10

    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')
    #X_train /= 255
    #X_test /= 255
    X_unlab = X_train
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

elif dataset == "stl10":
    import stl10
    img_rows, img_cols = 96, 96
    img_shape = (96, 96, 3)
    cmap = None
    img_scale = np.float32(127.5)
    img_bias =  np.float32(1.0)
    nb_classes=10

    X_train, y_train = stl10.load_data('train')
    X_test,  y_test  = stl10.load_data('test')
    X_unlab, y_unlab  = stl10.load_data('unlabeled')

    #X_train = X_train / np.float32(127.5) - np.float32(1.0)
    #X_test = X_test / np.float32(127.5) - np.float32(1.0)
    #X_unlab = X_unlab / np.float32(127.5) - np.float32(1.0)

elif dataset == "stl10_64x64":
    import stl10
    img_rows, img_cols = 64, 64
    img_shape = (64, 64, 3)
    cmap = None
    img_scale = np.float32(127.5)
    img_bias =  np.float32(1.0)
    nb_classes=10

    X_train, y_train = stl10.load_data('train')
    X_test,  y_test  = stl10.load_data('test')
    X_unlab, y_unlab  = stl10.load_data('unlabeled')

    X_train = stl10.crop64x64(X_train)
    X_test = stl10.crop64x64(X_test)
    X_unlab = stl10.crop64x64(X_unlab)


elif dataset == "random":
    img_rows, img_cols = 96, 96
    img_shape = (96, 96, 3)
    cmap = None
    img_scale = np.float32(1.0)
    img_bias =  np.float32(0.0)
    X_unlab = np.random.uniform(-1, 1, (100,3,96,96)).astype('float32')
    X_train = np.random.uniform(-1, 1, (100,3,96,96)).astype('float32')
    X_test  = np.random.uniform(-1, 1, (100,3,96,96)).astype('float32')
    y_train = np.concatenate([np.ones((50, 1)),np.zeros((50, 1))], axis=0).astype('float32')
    y_test  = np.concatenate([np.ones((50, 1)),np.zeros((50, 1))], axis=0).astype('float32')

## OPTIMIZER
dis_op = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
cls_op = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
gen_op = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
#gen_op = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

## MODEL
(gen, (dis, cls)) = get_models(dataset,smallmodel)
dis.trainable = False
x = Input(shape=(gen_dim,))
h = gen(x)
y = dis(h)
disgen = Model(input=x, output=y)
disgen.compile(loss='binary_crossentropy', optimizer=gen_op, metrics=['accuracy'])
disgen._make_train_function()
disgen._make_test_function()
disgen._make_predict_function()

#noise_batch = np.random.uniform(-1, 1, (batch_size, gen_dim)).astype('float32')
#labels2 = np.ones((batch_size, 1)).astype('float32')
#g_loss = disgen.train_on_batch(noise_batch, labels2)

dis.trainable = True
dis.compile(loss='binary_crossentropy', optimizer=dis_op, metrics=['accuracy'])

if classify:
    cls.trainable = True
    cls.compile(loss='categorical_crossentropy', optimizer=cls_op, metrics=['accuracy'])

gen.compile(loss='mean_squared_error', optimizer=gen_op)


# TRAINING

n_unlab = X_unlab.shape[0]
n_train = X_train.shape[0]
n_fold = n_train / folds
valid_mask = np.zeros(n_train, dtype=bool)
valid_mask[valid_fold*n_fold:(valid_fold+1)*n_fold] = True
if dataset == 'stl10':
    valid_mask = -valid_mask
X_valid = scale_img(X_train[valid_mask],img_scale,img_bias)
y_valid = np_utils.to_categorical(y_train[valid_mask],nb_classes)
X_train = X_train[-valid_mask]
y_train = y_train[-valid_mask]
X_test = scale_img(X_test,img_scale,img_bias)
y_test = np_utils.to_categorical(y_test,nb_classes)

data_size = max(n_unlab,n_train)
n_batches = data_size / batch_size
n_iter = data_size * n_epoch
labels = np.concatenate([np.ones((batch_size, 1)),
                    np.zeros((batch_size, 1))], axis=0).astype('float32')

labels2 = np.ones((batch_size, 1)).astype('float32')
fig = plt.figure()

fixed_noise = np.random.uniform(-1, 1, (monitor_size*monitor_size, gen_dim)).astype('float32')
fixed_class = disgen.predict(fixed_noise)
best_acc = 0.0
print "starting training"
with open('errors.log', 'w') as f:
    f.write('# iter data_seen epoch d_loss c_loss g_loss d_acc c_acc g_acc ')
    f.write('c_val_loss c_val_acc c_test_loss c_test_acc\n')
with open('best.log', 'w') as f:
    f.write('# iter data_seen epoch c_val_loss c_val_acc c_test_loss c_test_acc\n')

for it in xrange(n_iter):
    epoch = it/data_size

    data_batch = scale_img(get_batch(X_unlab,it,batch_size),img_scale,img_bias)
    noise_batch = np.random.uniform(-1, 1, (batch_size, gen_dim)).astype('float32')
    gen_batch = gen.predict(noise_batch)
    X = np.concatenate((data_batch, gen_batch))
    train_batch = scale_img(get_batch(X_train,it,batch_size),img_scale,img_bias)
    ytrain_batch = np_utils.to_categorical(get_batch(y_train,it,batch_size), nb_classes)

    if (it % iter_save == 0) or (it % 10 == 0 and it < iter_save):
        fixed_gen = gen.predict(fixed_noise)
        images.save_images(fixed_gen,monitor_size,'gen'+str(it).zfill(6)+'.png',img_scale,img_bias)


        if (it == 0) or (it >= iter_save):
            if classify:
                cls_valid = cls.evaluate(X_valid,y_valid,batch_size)
                cls_test = cls.evaluate(X_test,y_test,batch_size)
            else:
                cls_valid = (1,0)
                cls_test = (1,0)

            if (it == 0):
                d_loss = dis.evaluate(X, labels, batch_size)
                if classify:
                    c_loss = cls.evaluate(train_batch, ytrain_batch, batch_size)
                else:
                    c_loss = (1,0)
                g_loss = disgen.evaluate(noise_batch, labels2, batch_size)


            with open('errors.log', 'a') as f:
                f.write( " ".join(map(str, (it,it*batch_size,epoch) ))+" ")
                f.write( " ".join(map(str, (d_loss[0],c_loss[0],g_loss[0],d_loss[1],c_loss[1],g_loss[1]) ))+" ")
                f.write( " ".join(map(str, (cls_valid[0],cls_valid[1]) ))+" ")
                f.write( " ".join(map(str, (cls_test[0], cls_test[1] ) ))+"\n")

            if cls_valid[1]>best_acc:
                best_acc = cls_valid[1]
                with open('best.log', 'a') as f:
                    f.write( " ".join(map(str, (it,it*batch_size,epoch) ))+" ")
                    f.write( " ".join(map(str, (cls_valid[0],cls_valid[1]) ))+" ")
                    f.write( " ".join(map(str, (cls_test[0],cls_test[1]) ))+"\n")

                #gen.save_weights('gen_wts_best.hdf5', True)
                #dis.save_weights('dis_wts_best.hdf5', True)
                #disgen.save_weights('disgen_wts_best.hdf5', True)


            #gen.save_weights('gen_wts'+str(it).zfill(6)+'.hdf5', True)
            #dis.save_weights('dis_wts'+str(it).zfill(6)+'.hdf5', True)
            #disgen.save_weights('disgen_wts'+str(it).zfill(6)+'.hdf5', True)

    concat = False
    if concat:
        d_loss = dis.train_on_batch(X, labels)
    else:
        d_loss_1 = dis.train_on_batch(data_batch, labels[:batch_size])
        d_loss_2 = dis.train_on_batch(gen_batch,  labels[batch_size:])
        for i in range(len(d_loss)):
            d_loss[i] = (d_loss_1[i] + d_loss_2[i])*0.5

    if classify:
        c_loss = cls.train_on_batch(train_batch, ytrain_batch)
    else:
        c_loss = (1,0)

    g_loss = disgen.train_on_batch(noise_batch, labels2)

    giter = 0
    while (g_loss[0] > 1.01*d_loss[0]) and (giter<k_iter):
        noise_batch = np.random.uniform(-1, 1, (batch_size, gen_dim)).astype('float32')
        g_loss = disgen.train_on_batch(noise_batch, labels2)
        giter = giter+1



