import sys
import argparse
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as LL
import time
import nn
from theano.sandbox.rng_mrg import MRG_RandomStreams

# settings
CANT_CLASSES = 7
TAM_IMG = 64
NUM_EPOCH = 10000
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--count', type=int, default=7)
args = parser.parse_args()
print(args)

# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
data_rng = np.random.RandomState(args.seed_data)

# specify generative model
noise = theano_rng.uniform(size=(args.batch_size, 100))
gen_layers = [LL.InputLayer(shape=(args.batch_size, 100), input_var=noise)]
gen_layers.append(nn.batch_norm(LL.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.batch_norm(LL.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
# OJO en esta capa va el tamanio de la imagen de entrada
gen_layers.append(nn.l2normalize(LL.DenseLayer(gen_layers[-1], num_units=TAM_IMG**2, nonlinearity=T.nnet.sigmoid)))
gen_dat = LL.get_output(gen_layers[-1], deterministic=False)

# specify supervised model
# OJO aca tambien
layers = [LL.InputLayer(shape=(None, TAM_IMG**2))]
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.3))
layers.append(nn.DenseLayer(layers[-1], num_units=1000))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=500))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
# OJO , aca va la cantidad de clases
layers.append(nn.DenseLayer(layers[-1], num_units=CANT_CLASSES, nonlinearity=None, train_scale=True))

# costs
labels = T.ivector()
x_lab = T.matrix()
x_unl = T.matrix()

temp = LL.get_output(gen_layers[-1], init=True)
temp = LL.get_output(layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in gen_layers+layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = LL.get_output(layers[-1], x_lab, deterministic=False)
output_before_softmax_unl = LL.get_output(layers[-1], x_unl, deterministic=False)
output_before_softmax_fake = LL.get_output(layers[-1], gen_dat, deterministic=False)

z_exp_lab = T.mean(nn.log_sum_exp(output_before_softmax_lab))
z_exp_unl = T.mean(nn.log_sum_exp(output_before_softmax_unl))
z_exp_fake = T.mean(nn.log_sum_exp(output_before_softmax_fake))
l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]
l_unl = nn.log_sum_exp(output_before_softmax_unl)
#--- funciones de loss a minimizar
loss_lab = -T.mean(l_lab) + T.mean(z_exp_lab)
loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_before_softmax_unl))) + 0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_before_softmax_fake)))

train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

mom_gen = T.mean(LL.get_output(layers[-3], gen_dat), axis=0)
mom_real = T.mean(LL.get_output(layers[-3], x_unl), axis=0)
loss_gen = T.mean(T.square(mom_gen - mom_real))

# test error
output_before_softmax = LL.get_output(layers[-1], x_lab, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

# Theano functions for training and testing
lr = T.scalar()
disc_params = LL.get_all_params(layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
gen_params = LL.get_all_params(gen_layers[-1], trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates)
#--- train_batch_disc ----
# funcion de entrenamiento del discrim. toma un batch etiquetado y uno sin etiquetar. 
train_batch_disc = th.function(inputs=[x_lab,labels,x_unl,lr], outputs=[loss_lab, loss_unl, train_err], updates=disc_param_updates+disc_avg_updates)
#--- train_batch_gen ----
# funcion de entrenamiento del gen. toma un batch unlab.
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=[loss_gen], updates=gen_param_updates)
#--- test_batch ----
# toman un batch y sus labels y calcula el error como la media de los errores de cada ejemplo del batch
test_batch = th.function(inputs=[x_lab,labels], outputs=test_err, givens=disc_avg_givens)
# load MNIST data
dataTrain = np.load('../../../data/afew100/afewBD.npz')
dataVal = np.load('../../../data/afew100/afewValBD.npz')
dataUnlab = np.load('../../../data/casia100/casiaBD.npz')
data_unl    = dataUnlab['x']
print "shape data unlab"
print np.shape(data_unl) 
dataX_train = dataTrain['x']
dataY_train = dataTrain['y']
dataX_valid = dataVal['x'][250:]
dataY_valid = dataVal['y'][250:]
dataX_test = dataVal['x'][:250]
dataY_test = dataVal['y'][:250]

trainx = np.concatenate([dataX_train, dataX_valid], axis=0).astype(th.config.floatX)
trainy = np.concatenate([dataY_train, dataY_valid]).astype(np.int32)
trainx_unl = data_unl
trainx_unl2 = data_unl
testx = dataX_test.astype(th.config.floatX)
testy = dataY_test.astype(np.int32)

nr_batches_train = int(trainx.shape[0]/args.batch_size)
print "numero de batches de train: (lab)"
print nr_batches_train
print "tamanio batch train: (lab)"
print args.batch_size
print "numero de batches de train: (unlab)"
print trainx_unl.shape[0]/args.batch_size
print "tamanio batch train: (unlab)"
print args.batch_size
print "tamanio batch test"
print args.batch_size

nr_batches_test = int(testx.shape[0]/args.batch_size)
print "numero de batches de test: (lab)"
print nr_batches_test
print "shape labeled data train"
print trainx.shape
print "shape labeled data test"
print testx.shape


# select labeled data
inds = data_rng.permutation(trainx.shape[0])
txs = trainx[inds]
tys = trainy[inds]
# args.count es un argumento. por defecto es 10. 
#for j in range(CANT_CLASSES):
#    txs.append(trainx[trainy==j][:args.count])
#    tys.append(trainy[trainy==j][:args.count])
#txs = np.concatenate(txs, axis=0)
#tys = np.concatenate(tys, axis=0)
init_param( txs ) #trainx[:500]) # data dependent initialization

# //////////// perform training //////////////
lr = 0.003
for epoch in range(NUM_EPOCH):
    begin = time.time()

    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(trainx_unl.shape[0]/txs.shape[0]):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    for t in range(nr_batches_train):
        ll, lu, te = train_batch_disc(trainx[t*args.batch_size:(t+1)*args.batch_size],trainy[t*args.batch_size:(t+1)*args.batch_size],
                                        trainx_unl[t*args.batch_size:(t+1)*args.batch_size],lr)
        loss_lab += ll
        loss_unl += lu
        train_err += te
        e = train_batch_gen(trainx_unl2[t*args.batch_size:(t+1)*args.batch_size],lr)
    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train

    # test
    test_err = 0.
		# tanto en train como en test, el nro de ejemplos debe ser multiplo del tamanio del batch. caso contrario se desperdician ejemplos.
		# se testea en todas las iteraciones, y se testea en cada it. el conjunto completo de test. 
    for t in range(nr_batches_test):
        test_err += test_batch(testx[t*args.batch_size:(t+1)*args.batch_size],testy[t*args.batch_size:(t+1)*args.batch_size])
    test_err /= nr_batches_test

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, test err = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl, train_err, test_err))
    sys.stdout.flush()
