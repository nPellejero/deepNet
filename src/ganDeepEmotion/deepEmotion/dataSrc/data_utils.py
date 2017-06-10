import numpy as np
from sklearn import utils as skutils
from scipy import misc

from rng import np_rng, py_rng
from theano_utils import floatX

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import svd

def img2arrayGS(path):
    return misc.imread(path, flatten=True)

def img2arrayRGB(path):
    return misc.imread(path)

def center_crop(x, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = x.shape[:2]
    j = int(round((h - ph)/2.))
    i = int(round((w - pw)/2.))
    return x[j:j+ph, i:i+pw]

def patch(x, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = x.shape[:2]
    j = py_rng.randint(0, h-ph)
    i = py_rng.randint(0, w-pw)
    x = x[j:j+ph, i:i+pw]
    return x

def list_shuffle(*data):
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]

def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], basestring):
        return list_shuffle(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)

def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

def transform(X):
    """
    shift data from [0,255] to [-1, 1]
    """
    return floatX(X)/127.5 - 1.

def inverse_transform(X, nc, npx, expand=False):
    """
    transpose shape and shift from [-1,1] to [0,1]
    """
    X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+1.)/2.

    if expand:
        X = X - X.min()
        X = X/X.max()

    return X

def get_batch(X, index, batch_size):
    """
    iterate through data set
    """
    size = X.shape[0]
    n1 = (index*batch_size)%size
    n2 = ((index+1)*batch_size)%size
    if n1>n2:
        return floatX(np.concatenate((X[n1:], X[:n2])))
    else:
        return floatX(X[n1:n2])

def flatten_tail(X):
    """
    flatten all but first dimensions of numpy ndarray X
    """
    return X.reshape(X.shape[0], -1)


def patch_work(image, min_height, step, patch_size):
    """
    INPUT: [h, w ,nc]
    OUTPUT: [n, nc, patch_size, patch_size]
    """
    sh = image.shape
    h, w, nc = sh
    cols = ((w              - patch_size) / step) + 1
    rows = ((h - min_height - patch_size) / step) + 1
    n = cols*rows

    X = np.zeros((n, patch_size, patch_size, nc))

    c=0
    for i in range(rows):
        for j in range(cols):
            vl = i*step + min_height
            vh = vl + patch_size
            hl = j*step
            hh = hl + patch_size
            X[c] = image[vl:vh, hl:hh, :]
            c = c + 1

    return np.transpose(X, (0, 3, 1, 2)), cols, rows

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n / size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])

def make_paths(n_code, n_paths, n_steps=480):
    """
    create a random path through code space by interpolating between points
    """
    paths = []
    p_starts = np.random.randn(n_paths, n_code)
    for i in range(n_steps/48):
        p_ends = np.random.randn(n_paths, n_code)
        for weight in np.linspace(0., 1., 48):
            paths.append(p_starts*(1-weight) + p_ends*weight)
        p_starts = np.copy(p_ends)

    paths = np.asarray(paths)
    return paths


class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, bias=.1, scale_by=1., copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy
        self.scale_by = float(scale_by)

    def fit(self, X, y=None):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X /= self.scale_by
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, VT = svd(np.dot(X.T, X) / n_samples, full_matrices=False)
        components = np.dot(VT.T * np.sqrt(1.0 / (S + self.bias)), VT)
        self.covar_ = np.dot(X.T, X)
        self.components_ = components[:self.n_components]
        return self

    def transform(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X /= self.scale_by
        X -= self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed
