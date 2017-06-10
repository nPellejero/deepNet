from train_dcgan_stl10_getz_generic import getz_experiment

def process_data(X_batch):
    for row in range(32):
        X_batch[row] = X_batch[26]
        X_batch[row + 32] = X_batch[34]
        X_batch[row + 64] = X_batch[74]
        X_batch[row + 96] = X_batch[115]
    return X_batch

experiment = getz_experiment('src.model_32x_sub', process_data)
experiment.run()
