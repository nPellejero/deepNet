import numpy as np

from data import *
 
class faces_semi(Dataset):

    def __init__(self, train_fold=0,
             n_valid_folds=5,
             valid_fold=0,
             train = True,
             valid = True,
             test = True,
             unlab = True,
             data_dir='/share/datasets/stl10',
             ):

        height = 96
        width = 96
        depth = 3

        # size of a single image in bytes
        dim = height * width * depth

        self.__dict__.update(locals())
        del self.self # We don't need this either

        x_min= 0
        x_max = 255

        if train:
            X_train, y_train = self.load_data(data_dir,'train')
            ifold = self.load_fold_indices(data_dir,train_fold)
            X_train = X_train[ifold]
            y_train = y_train[ifold]
            self.train_size = X_train.shape[0] # 1000
        else:
            X_train = None
            y_train = None

        if test:
            X_test, y_test = self.load_data(data_dir,'test')
            self.test_size = X_test.shape[0]
        else:
            X_test = None
            y_test = None

        if unlab:
            X_unlab, y_unlab = self.load_data(data_dir,'unlabeled')
            self.unlab_size = X_unlab.shape[0] # 100000
        else:
            X_unlab = None

        if valid:
            if train:
                self.valid_size = self.train_size / n_valid_folds
                valid_mask = np.zeros(self.train_size, dtype=bool)
                valid_mask[valid_fold*self.valid_size:(valid_fold+1)*self.valid_size] = True
                X_valid = X_train[valid_mask]
                y_valid = y_train[valid_mask]
                X_train = X_train[-valid_mask]
                y_train = y_train[-valid_mask]
                self.train_size = X_train.shape[0]
            elif unlab:
                self.valid_size = self.unlab_size / n_valid_folds
                valid_mask = np.zeros(self.unlab_size, dtype=bool)
                valid_mask[valid_fold*self.valid_size:(valid_fold+1)*self.valid_size] = True
                X_valid = X_unlab[valid_mask]
                y_valid = None
                X_unlab = X_unlab[-valid_mask]
                self.unlab_size = X_unlab.shape[0]
        else:
            X_valid = None
            y_valid = None

        super(stl10, self).__init__(X_train=X_train, y_train=y_train,
                                    X_valid=X_valid, y_valid=y_valid,
                                    X_test =X_test, y_test=y_test,
                                    X_unlab=X_unlab,x_min=x_min,x_max=x_max)
