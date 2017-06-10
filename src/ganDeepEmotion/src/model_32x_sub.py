import theano.tensor as T

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, running_mapnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.ops import running_l1_batchnorm as running_batchnorm
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch
from theano.sandbox.cuda.dnn import dnn_conv
from theano.tensor.signal.pool import pool_2d
from model import Model

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
softmax = activations.Softmax()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()

class GAN_model(Model):

    def __init__(self,
                 img_shape = (64,64), # tienen que ser multiplos de 32
                 gen_dim = 100, ngf = 128, ndf = 128, nchannels = 3, init_scale=0.02, bn_momentum=0.9):

        self.gen_dim = gen_dim
        self.ngf = ngf
        self.ndf = ndf
        if img_shape[0]%32 is not 0 or img_shape[1]%32 is not 0:
            raise 1
        self.last_shape = (4,4)
        self.w_visible_shape = (nchannels, img_shape[0]/16+1, img_shape[1]/16+1)
        self.visible_subsamp = (img_shape[0]/32,img_shape[1]/32)
        self.visible_bmode = (img_shape[0]/32,img_shape[1]/32)
        self.bn_momentum = bn_momentum

        h = self.last_shape[0]
        w = self.last_shape[1]
        gifn = inits.Normal(scale=init_scale)
        difn = inits.Normal(scale=init_scale)
        gain_ifn = inits.Normal(loc=1., scale=init_scale)
        bias_ifn = inits.Constant(c=0.)
        bnmean_fn = inits.Constant(c=0.)
        bnstd_fn = inits.Constant(c=1.)

        gw  = gifn((gen_dim, ngf*8*h*w), 'gw')
        gg = gain_ifn((ngf*8*h*w), 'gg')
        gb = bias_ifn((ngf*8*h*w), 'gb')
        gu = bnmean_fn((ngf*8*h*w), 'gu')
        gs = bnstd_fn((ngf*8*h*w), 'gs')
        gw2 = gifn((ngf*8, ngf*4, 5, 5), 'gw2')
        gg2 = gain_ifn((ngf*4), 'gg2')
        gb2 = bias_ifn((ngf*4), 'gb2')
        gu2 = bnmean_fn((ngf*4), 'gu2')
        gs2 = bnstd_fn((ngf*4), 'gs2')
        gw3 = gifn((ngf*4, ngf*2, 5, 5), 'gw3')
        gg3 = gain_ifn((ngf*2), 'gg3')
        gb3 = bias_ifn((ngf*2), 'gb3')
        gu3 = bnmean_fn((ngf*2), 'gu3')
        gs3 = bnstd_fn((ngf*2), 'gs3')
        gw4 = gifn((ngf*2, ngf, 5, 5), 'gw4')
        gg4 = gain_ifn((ngf), 'gg4')
        gb4 = bias_ifn((ngf), 'gb4')
        gu4 = bnmean_fn((ngf), 'gu4')
        gs4 = bnstd_fn((ngf), 'gs4')
        gwx = gifn((ngf,)+self.w_visible_shape, 'gwx')

        dw  = difn((ndf,)+self.w_visible_shape, 'dw')
        dw2 = difn((ndf*2, ndf, 5, 5), 'dw2')
        dg2 = gain_ifn((ndf*2), 'dg2')
        db2 = bias_ifn((ndf*2), 'db2')
        du2 = bnmean_fn((ndf*2), 'du2')
        ds2 = bnstd_fn((ndf*2), 'ds2')
        dw3 = difn((ndf*4, ndf*2, 5, 5), 'dw3')
        dg3 = gain_ifn((ndf*4), 'dg3')
        db3 = bias_ifn((ndf*4), 'db3')
        du3 = bnmean_fn((ndf*4), 'du3')
        ds3 = bnstd_fn((ndf*4), 'ds3')
        dw4 = difn((ndf*8, ndf*4, 5, 5), 'dw4')
        dg4 = gain_ifn((ndf*8), 'dg4')
        db4 = bias_ifn((ndf*8), 'db4')
        du4 = bnmean_fn((ndf*8), 'du4')
        ds4 = bnstd_fn((ndf*8), 'ds4')
        dg5 = gain_ifn((h,w), 'dg5')
        db5 = bias_ifn((h,w), 'db5')
        du5 = bnmean_fn((h,w), 'du5')
        ds5 = bnstd_fn((h,w), 'ds5')
        dwy = difn((ndf*8*h*w, 1), 'dwy')
        dwc = difn((ndf*8, 10), 'dwc')
        #dbc = bias_ifn((10), 'dbc')

        self.trainable_gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
        self.trainable_discrim_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy]
        self.trainable_classif_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dg5, db5, dwc]

        self.gen_params = [gw, gg, gb, gu, gs, gw2, gg2, gb2, gu2, gs2,
                           gw3, gg3, gb3, gu3, gs3, gw4, gg4, gb4, gu4, gs4, gwx]
        self.disclass_params = [dw, dw2, dg2, db2, du2, ds2, dw3, dg3, db3, du3, ds3, dw4, dg4, db4, du4, ds4, dg5, db5, du5, ds5, dwy, dwc]
        self.discrim_params = [dw, dw2, dg2, db2, du2, ds2, dw3, dg3, db3, du3, ds3, dw4, dg4, db4, du4, ds4, dwy]
        self.classif_params = [dw, dw2, dg2, db2, du2, ds2, dw3, dg3, db3, du3, ds3, dw4, dg4, db4, du4, ds4, dg5, db5, du5, ds5, dwc]
        self.params = self.gen_params + self.disclass_params
        
        self.X = T.tensor4()
        self.Z = T.matrix()
        self.genX, self.other_gen_updates = self.gen_function(False, self.Z, *self.gen_params)

        self.disX, self.classX, self.other_discrim_updates = self.discrim_function(False, self.X, *self.disclass_params)
        self.other_classif_updates = self.other_discrim_updates
        self.disgenX, self.classgenX, _ = self.discrim_function(False, self.genX, *self.disclass_params)

        self.genXTest, _ = self.gen_function(True, self.Z, *self.gen_params)
        self.disXTest, self.classXTest, _ = self.discrim_function(True, self.X, *self.disclass_params)
        self.disgenXTest, _, _ = self.discrim_function(True, self.genXTest, *self.disclass_params)

    def gen_function(self, test_mode, Z, w, g, b, u, s, w2, g2, b2, u2, s2, w3, g3, b3, u3, s3, w4, g4, b4, u4, s4, wx):
        x1, updates1 = running_batchnorm(T.dot(Z, w), g=g, b=b, running_u=u, running_s=s, momentum=self.bn_momentum,
                                         test_mode=test_mode)
        h = relu(x1)
        h = h.reshape((h.shape[0], self.ngf*8, self.last_shape[0], self.last_shape[1]))
        x2, updates2 = running_batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2,
                                         running_u=u2, running_s=s2, momentum=self.bn_momentum, test_mode=test_mode)
        h2 = relu(x2)
        x3, updates3 = running_batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3,
                                         running_u=u3, running_s=s3, momentum=self.bn_momentum, test_mode=test_mode)
        h3 = relu(x3)
        x4, updates4 = running_batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4,
                                         running_u=u4, running_s=s4, momentum=self.bn_momentum, test_mode=test_mode)
        h4 = relu(x4)
        x = tanh(deconv(h4, wx, subsample=self.visible_subsamp, border_mode=self.visible_bmode))
        return x, updates1 + updates2 + updates3 + updates4

    def discrim_function(self, test_mode, X, w, w2, g2, b2, u2, s2, w3, g3, b3, u3, s3, w4, g4, b4, u4, s4, g5, b5, u5, s5, wy, wc):
        h = lrelu(dnn_conv(X, w, subsample=self.visible_subsamp, border_mode=self.visible_bmode))
        x2, updates2 = running_batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2,
                                         running_u=u2, running_s=s2, momentum=self.bn_momentum, test_mode=test_mode)
        h2 = lrelu(x2)
        x3, updates3 = running_batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3,
                                         running_u=u3, running_s=s3, momentum=self.bn_momentum, test_mode=test_mode)
        h3 = lrelu(x3)
        x4, updates4 = running_batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4,
                                         running_u=u4, running_s=s4, momentum=self.bn_momentum, test_mode=test_mode)
        h4 = lrelu(x4)
        h4 = T.flatten(h4, 2)
        x4, updates5 = running_mapnorm(x4, g=g5, b=b5,running_u=u5, running_s=s5, test_mode=test_mode)
        h5 = pool_2d(x4, (self.last_shape[0],self.last_shape[1]), ignore_border=True, st=None, padding=(0, 0), mode='max')
        #h5 = h5.reshape((h5.shape[0],self.ndf,8))
        #h5 = T.max(h5, axis=2, keepdims=False)
        h5 = T.dot(T.flatten(h5,2), wc)
        c = softmax(h5) # + bc.dimshuffle('x', 0))
        y = sigmoid(T.dot(h4, wy))
        return y, c, updates2 + updates3 + updates4 + updates5
