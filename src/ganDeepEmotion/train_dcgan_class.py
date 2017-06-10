
import os
import argparse
from time import time
import numpy as np
from sklearn.externals import joblib

from lib.vis import color_grid_vis
from lib.rng import np_rng
from lib.theano_utils import floatX

from src.model_32x_sub import GAN_model

parser = argparse.ArgumentParser()
parser.add_argument("--gen_dim", type = int, default = 100)
parser.add_argument("--dataset", type = str, default = 'faces')
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--n_epochs", type = int, default = 30)
#parser.add_argument("--k_iter", type = int, default = 1)
parser.add_argument("--monitor_size", type = int, default = 196)
parser.add_argument("--init_scale", type = float, default = 0.02)
#parser.add_argument("--folds", type = int, default = 5)
#parser.add_argument("--valid_fold", type = int, default = 0)
parser.add_argument("--iter_save", type = int, default = 100)
parser.add_argument("--img_size", type = int, default = 64)
args = parser.parse_args()
print args
globals().update(vars(args))



from src.gan_class import GAN_trainer


model = GAN_model(img_shape=(img_size,img_size),gen_dim=gen_dim,init_scale=init_scale)
trainer = GAN_trainer(model)

if dataset == 'stl10':
    import datasets.stl10_data as dataset
    data = dataset.stl10()
elif dataset == 'faces':
    import dataset.faces_semi as dataset
    data = dataset.faces_semi()




desc = 'dcgan'
model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

X_sample = data.get_unlab_batch(0,monitor_size)
X_sample = data.center_crop(X_sample,img_size)
color_grid_vis(X_sample.transpose(0, 2, 3, 1), (14, 14), 'samples/%s_etl_test.png'%desc)


Z_sample = floatX(np_rng.uniform(-1., 1., size=(monitor_size, model.gen_dim)))


print desc.upper()

print "starting training"
with open('errors.log', 'w') as f:
    f.write('# iter data_seen epoch dis_loss g_loss')
    f.write(' c_loss c_val_err c_test_err\n')
    
with open('best.log', 'w') as f:
    f.write('# iter data_seen epoch c_val_err c_test_err\n')

n_iter = n_epochs*(data.unlab_size/batch_size+1)

best_err = 1e6
last_it = 0
t = time()
for it in xrange(n_iter):


    Z_batch = floatX(np_rng.uniform(-1., 1., size=(batch_size, gen_dim)))
    gen_loss = trainer.train_generator_on_batch(Z_batch)

    epoch = it*batch_size/data.unlab_size
    
    X_batch = data.get_unlab_batch(it,batch_size)
    X_batch = data.scale_data(data.center_crop(X_batch,img_size))
    y_batch = None

    dis_loss = trainer.train_discriminator_on_batch(X_batch, Z_batch, y_batch)

    X_batch, y_batch = data.get_train_batch(it,batch_size)
    X_batch = data.scale_data(data.center_crop(X_batch,img_size))
    cls_loss = trainer.train_classifier_on_batch(X_batch, y_batch)

    if (it % iter_save == 0) or (it % 10 == 0 and it < iter_save):
        cls_test_err = 0.0
        for it2 in xrange(data.test_size/batch_size):
            X_batch, y_batch = data.get_test_batch(it2,batch_size)
            X_batch = data.scale_data(data.center_crop(X_batch,img_size))
            cls_test_err += trainer._cls_error(X_batch, y_batch)
        cls_test_err /= data.test_size/batch_size
        cls_valid_err = 0.0
        for it2 in xrange(data.valid_size/batch_size):
            X_batch, y_batch = data.get_valid_batch(it2,batch_size)
            X_batch = data.scale_data(data.center_crop(X_batch,img_size))
            cls_valid_err += trainer._cls_error(X_batch, y_batch)
        cls_valid_err /= data.valid_size/batch_size

        samples = np.asarray(trainer._gen(Z_sample))
        color_grid_vis(data.inv_scale_data(samples).transpose(0, 2, 3, 1), (14, 14), 'samples/%s/%d.png'%(desc, it))

        with open('errors.log', 'a') as f:
            f.write( " ".join(map(str, (it,it*batch_size,epoch) ))+" ")
            f.write( " ".join(map(str, (dis_loss,gen_loss) ))+" ")
            f.write( " ".join(map(str, (cls_loss,cls_valid_err,cls_test_err) ))+"\n")
            
        if cls_valid_err<best_err:
            best_err = cls_valid_err
            with open('best.log', 'a') as f:
                f.write( " ".join(map(str, (it,it*batch_size,epoch) ))+" ")
                f.write( " ".join(map(str, (cls_valid_err,cls_test_err) ))+"\n")

            model.dump('models/%s/best_gen_params.jl'%(desc))

        t2 = time()-t
        t += t2
        print "iter:%d/%d; epoch:%d;    %f sec. per iteration"%(it,n_iter,epoch,t2/(1+it-last_it))
        last_it = it+1

    if epoch in [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 200, n_epochs]:
        if (it*batch_size)%data.unlab_size<batch_size:
            model_dir = 'models/%s/%d'%(desc, it)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.dump('%s/params.jl'%(model_dir))

model_dir = 'models/%s/last'%(desc)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.dump('%s/params.jl' % (model_dir))

