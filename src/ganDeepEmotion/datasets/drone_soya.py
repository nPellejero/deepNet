import os
import numpy as np
from scipy import misc

# path to the directory with the data
DATA_DIR = '/share/datasets/drone_soya/'

def img2array(path):
    return misc.imread(path)


def make_npy(which_set='unlabeled'):
    y = None

    if which_set == 'unlabeled':
        path_to_data = DATA_DIR + which_set + '/data'
    else:
        raise Exception('no existe ' + which_set)

    print("Traversing images dir...")
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path_to_data):
        files.extend(os.path.join(dirpath, x) for x in filenames if x.endswith(".png"))

    print(files[0])
    print("Loading images...")
    X = np.array([img2array(path) for path in files])

    X = np.transpose(X, (0, 3, 1, 2))

    print("Done!")
    print("SHAPE:" + str(X.shape))

    return X, y


def save_npy(npy, out_file):
    print("Saving file...")
    np.save(out_file, X)
    print("Done!")
    return


def load_data(which_set='unlabeled'):
    filename = which_set + ".npy"

    try:
        X = np.load(filename)
    except IOError:
        print("Failed to load %s. Building it!"%filename)
        X, y = make_npy(which_set)

    return X, y


if __name__ == "__main__":
    X, _ = make_npy('unlabeled')
    save_npy(X, 'unlabeled.npy')
