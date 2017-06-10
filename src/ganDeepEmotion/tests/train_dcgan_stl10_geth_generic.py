# No existe model_module_file. Hay que definirlo antes
import sys
sys.path.append('..')

import os
import argparse
from time import time

from lib.vis import color_grid_vis

import datasets.stl10_data as dataset
from src.gan_reconstructor import partial_reconstructor
import importlib
from sklearn.externals import joblib


class geth_experiment:

    def __init__(self, model_module_file, process_data_fn, layer, layer_shape):
        self.model_module = importlib.import_module(model_module_file)
        self.process_data = process_data_fn
        self.layer = layer
        self.layer_shape = layer_shape

    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str)
        parser.add_argument("--gendim", type=int, default=100)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--n_samples_row", type=int, default=16)
        parser.add_argument("--n_iter", type=int, default=100000)
        parser.add_argument("--iter_save", type=int, default=5000)
        parser.add_argument("--img_size", type = int, default = 64)
        args = parser.parse_args()
        print args

        model_file = args.model
        gen_dim = args.gendim
        n_iter = args.n_iter
        batch_size = args.batch_size
        n_samples_row = args.n_samples_row
        #dataset = args.dataset
        iter_save = args.iter_save
        img_size = args.img_size

        data = dataset.stl10()
        model = self.model_module.GAN_model(img_shape=(img_size,img_size), gen_dim=gen_dim)
        model.load(model_file)

        desc = 'dcgan'
        model_dir = 'models/%s'%desc
        samples_dir = 'samples/%s'%desc
        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        reconstructor = partial_reconstructor(model, self.layer, self.layer_shape, batch_size)

        print "starting training"

        best_err = 1e6
        last_it = 0
        t = time()
        reconstructor.set_H_uniform()

        X_batch, _ = data.get_test_batch(0, batch_size)
        X_batch = self.process_data(X_batch)
        X_batch = data.scale_data(data.center_crop(X_batch, img_size))
        color_grid_vis(data.inv_scale_data(X_batch).transpose(0, 2, 3, 1), (batch_size/n_samples_row, n_samples_row),
                       'samples/%s/reconstruction_objective.png' % (desc))

        for it in xrange(n_iter):
            loss = reconstructor.train_h_on_batch(X_batch)

            if (it % iter_save == 0) or (it % 1000 == 0 and it < iter_save):

                samples = reconstructor.reconstruct()
                color_grid_vis(data.inv_scale_data(samples).transpose(0, 2, 3, 1), (batch_size/n_samples_row, n_samples_row),
                               'samples/%s/reconstruction_%d.png' % (desc, it))
                joblib.dump(reconstructor.get_H_value(), 'models/%s/H_%d.jl' % (desc, it))

                with open('rec_errors.log', 'a') as f:
                    f.write( " ".join(map(str, (it,it*batch_size) ))+" ")
                    f.write( " ".join(str(loss))+"\n")

                t2 = time()-t
                t += t2
                print "iter:%d/%d;    %f sec. per iteration"%(it,n_iter,t2/(1+it-last_it))
                last_it = it+1

        samples = reconstructor.reconstruct()
        color_grid_vis(data.inv_scale_data(samples).transpose(0, 2, 3, 1), (batch_size / n_samples_row, n_samples_row),
                       'samples/%s/reconstruction_last.png' % (desc))
        joblib.dump(reconstructor.get_H_value(), 'models/%s/H_last.jl' % (desc))
