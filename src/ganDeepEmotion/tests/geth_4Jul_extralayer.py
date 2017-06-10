from train_dcgan_stl10_geth_generic import geth_experiment

#def process_data(X_batch):
#    return X_batch

def process_data(X_batch):
    for row in range(32):
        X_batch[row] = X_batch[26]
        X_batch[row + 32] = X_batch[34]
        X_batch[row + 64] = X_batch[74]
        X_batch[row + 96] = X_batch[115]
    return X_batch

ngf = 128
img_size = 96
visible_subsamp = 3

layer = 2
layer_shape = (ngf*2*2*2, img_size/(2*2*2*visible_subsamp), img_size/(2*2*2*visible_subsamp))

experiment = geth_experiment('src.model_32x_sub_extralayer', process_data, layer, layer_shape)
experiment.run()
