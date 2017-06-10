import theano.tensor as T

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, running_batchnorm, running_mapnorm, conv_cond_concat, deconv, dropout, l2normalize
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
                 img_shape=(64, 64),  # tienen que ser multiplos de 32
                 gen_dim=100, ngf=128, ndf=128, nchannels=3, init_scale=0.02, bn_momentum=0.9):
        self.gen_dim = gen_dim
        self.ngf = ngf
        self.ndf = ndf
        if img_shape[0] % 32 is not 0 or img_shape[1] % 32 is not 0:
            raise 1
        self.last_shape = (4, 4)
        self.w_visible_shape = (nchannels, img_shape[0] / 16 + 1, img_shape[1] / 16 + 1)
        self.visible_subsamp = (img_shape[0] / 32, img_shape[1] / 32)
        self.visible_bmode = (img_shape[0] / 32, img_shape[1] / 32)
        self.bn_momentum = bn_momentum

        h = self.last_shape[0]
        w = self.last_shape[1]
        gifn = inits.Normal(scale=init_scale)
        difn = inits.Normal(scale=init_scale)
        gain_ifn = inits.Normal(loc=1., scale=init_scale)
        bias_ifn = inits.Constant(c=0.)
        bnmean_fn = inits.Constant(c=0.)
        bnstd_fn = inits.Constant(c=1.)

        # Generador
        gw = gifn((gen_dim, ngf * 8 * h * w), 'gw')
        gg = gain_ifn((ngf * 8 * h * w), 'gg')
        gb = bias_ifn((ngf * 8 * h * w), 'gb')
        gu = bnmean_fn((ngf * 8 * h * w), 'gu')
        gs = bnstd_fn((ngf * 8 * h * w), 'gs')

        gw2_0 = gifn((ngf * 8, ngf * 4, 3, 3), 'gw2_0')
        gg2_0 = gain_ifn((ngf * 4), 'gg2_0')
        gb2_0 = bias_ifn((ngf * 4), 'gb2_0')
        gu2_0 = bnmean_fn((ngf * 4), 'gu2_0')
        gs2_0 = bnstd_fn((ngf * 4), 'gs2_0')

        gw2 = gifn((ngf * 4, ngf * 4, 3, 3), 'gw2')
        gg2 = gain_ifn((ngf * 4), 'gg2')
        gb2 = bias_ifn((ngf * 4), 'gb2')
        gu2 = bnmean_fn((ngf * 4), 'gu2')
        gs2 = bnstd_fn((ngf * 4), 'gs2')

        gw3_0 = gifn((ngf * 4, ngf * 2, 3, 3), 'gw3_0')
        gg3_0 = gain_ifn((ngf * 2), 'gg3_0')
        gb3_0 = bias_ifn((ngf * 2), 'gb3_0')
        gu3_0 = bnmean_fn((ngf * 2), 'gu3_0')
        gs3_0 = bnstd_fn((ngf * 2), 'gs3_0')

        gw3 = gifn((ngf * 2, ngf * 2, 3, 3), 'gw3')
        gg3 = gain_ifn((ngf * 2), 'gg3')
        gb3 = bias_ifn((ngf * 2), 'gb3')
        gu3 = bnmean_fn((ngf * 2), 'gu3')
        gs3 = bnstd_fn((ngf * 2), 'gs3')

        gw4_0 = gifn((ngf * 2, ngf, 3, 3), 'gw4_0')
        gg4_0 = gain_ifn((ngf), 'gg4_0')
        gb4_0 = bias_ifn((ngf), 'gb4_0')
        gu4_0 = bnmean_fn((ngf), 'gu4_0')
        gs4_0 = bnstd_fn((ngf), 'gs4_0')

        gw4 = gifn((ngf, ngf, 3, 3), 'gw4')
        gg4 = gain_ifn((ngf), 'gg4')
        gb4 = bias_ifn((ngf), 'gb4')
        gu4 = bnmean_fn((ngf), 'gu4')
        gs4 = bnstd_fn((ngf), 'gs4')
        gwx = gifn((ngf,) + self.w_visible_shape, 'gwx')

        # Discriminador
        dw = difn((ndf,) + self.w_visible_shape, 'dw')

        dw2_0 = difn((ndf * 2, ndf, 3, 3), 'dw2_0')
        dg2_0 = gain_ifn((ndf * 2), 'dg2_0')
        db2_0 = bias_ifn((ndf * 2), 'db2_0')
        du2_0 = bnmean_fn((ndf * 2), 'du2_0')
        ds2_0 = bnstd_fn((ndf * 2), 'ds2_0')

        dw2 = difn((ndf * 2, ndf * 2, 3, 3), 'dw2')
        dg2 = gain_ifn((ndf * 2), 'dg2')
        db2 = bias_ifn((ndf * 2), 'db2')
        du2 = bnmean_fn((ndf * 2), 'du2')
        ds2 = bnstd_fn((ndf * 2), 'ds2')

        dw3_0 = difn((ndf * 4, ndf * 2, 3, 3), 'dw3_0')
        dg3_0 = gain_ifn((ndf * 4), 'dg3_0')
        db3_0 = bias_ifn((ndf * 4), 'db3_0')
        du3_0 = bnmean_fn((ndf * 4), 'du3_0')
        ds3_0 = bnstd_fn((ndf * 4), 'ds3_0')

        dw3 = difn((ndf * 4, ndf * 4, 3, 3), 'dw3')
        dg3 = gain_ifn((ndf * 4), 'dg3')
        db3 = bias_ifn((ndf * 4), 'db3')
        du3 = bnmean_fn((ndf * 4), 'du3')
        ds3 = bnstd_fn((ndf * 4), 'ds3')

        dw4_0 = difn((ndf * 8, ndf * 4, 3, 3), 'dw4_0')
        dg4_0 = gain_ifn((ndf * 8), 'dg4_0')
        db4_0 = bias_ifn((ndf * 8), 'db4_0')
        du4_0 = bnmean_fn((ndf * 8), 'du4_0')
        ds4_0 = bnstd_fn((ndf * 8), 'ds4_0')

        dw4 = difn((ndf * 8, ndf * 8, 3, 3), 'dw4')
        dg4 = gain_ifn((ndf * 8), 'dg4')
        db4 = bias_ifn((ndf * 8), 'db4')
        du4 = bnmean_fn((ndf * 8), 'du4')
        ds4 = bnstd_fn((ndf * 8), 'ds4')
        dg5 = gain_ifn((h, w), 'dg5')
        db5 = bias_ifn((h, w), 'db5')
        du5 = bnmean_fn((h, w), 'du5')
        ds5 = bnstd_fn((h, w), 'ds5')
        dwy = difn((ndf * 8, 1), 'dwy')
        dwc = difn((ndf, 10), 'dwc')
        dbc = bias_ifn((10), 'dbc')

        self.trainable_gen_params = [gw, gg, gb, gw2_0, gg2_0, gb2_0, gw2, gg2, gb2,
                                     gw3_0, gg3_0, gb3_0, gw3, gg3, gb3,
                                     gw4_0, gg4_0, gb4_0, gw4, gg4, gb4, gwx]
        self.trainable_discrim_params = [dw, dw2_0, dg2_0, db2_0, dw2, dg2, db2,
                                         dw3_0, dg3_0, db3_0, dw3, dg3, db3,
                                         dw4_0, dg4_0, db4_0, dw4, dg4, db4,
                                         dg5, db5, dwy]
        self.trainable_classif_params = [dw, dw2_0, dg2_0, db2_0, dw2, dg2, db2,
                                         dw3_0, dg3_0, db3_0, dw3, dg3, db3,
                                         dw4_0, dg4_0, db4_0, dw4, dg4, db4,
                                         dg5, db5, dwc, dbc]

        self.gen_params = [gw, gg, gb, gu, gs, gw2_0, gg2_0, gb2_0, gu2_0, gs2_0, gw2, gg2, gb2, gu2, gs2,
                           gw3_0, gg3_0, gb3_0, gu3_0, gs3_0, gw3, gg3, gb3, gu3, gs3,
                           gw4_0, gg4_0, gb4_0, gu4_0, gs4_0, gw4, gg4, gb4, gu4, gs4, gwx]
        self.disclass_params = [dw, dw2_0, dg2_0, db2_0, du2_0, ds2_0, dw2, dg2, db2, du2, ds2,
                                dw3_0, dg3_0, db3_0, du3_0, ds3_0, dw3, dg3, db3, du3, ds3,
                                dw4_0, dg4_0, db4_0, du4_0, ds4_0, dw4, dg4, db4, du4, ds4,
                                dg5, db5, du5, ds5, dwy, dwc, dbc]
        self.discrim_params = [dw, dw2_0, dg2_0, db2_0, du2_0, ds2_0, dw2, dg2, db2, du2, ds2,
                               dw3_0, dg3_0, db3_0, du3_0, ds3_0, dw3, dg3, db3, du3, ds3,
                               dw4_0, dg4_0, db4_0, du4_0, ds4_0, dw4, dg4, db4, du4, ds4,
                               dg5, db5, du5, ds5, dwy]
        self.classif_params = [dw, dw2_0, dg2_0, db2_0, du2_0, ds2_0, dw2, dg2, db2, du2, ds2,
                               dw3_0, dg3_0, db3_0, du3_0, ds3_0, dw3, dg3, db3, du3, ds3,
                               dw4_0, dg4_0, db4_0, du4_0, ds4_0, dw4, dg4, db4, du4, ds4,
                               dg5, db5, du5, ds5, dwc, dbc]

        self.X = T.tensor4()
        self.Z = T.matrix()
        self.genX, self.other_gen_updates = self.gen_function(False, self.Z, *self.gen_params)

        self.disX, self.classX, self.other_discrim_updates = self.discrim_function(False, self.X, *self.disclass_params)
        self.other_classif_updates = self.other_discrim_updates
        self.disgenX, self.classgenX, _ = self.discrim_function(False, self.genX, *self.disclass_params)

        self.genXTest, _ = self.gen_function(True, self.Z, *self.gen_params)
        self.disXTest, self.classXTest, _ = self.discrim_function(True, self.X, *self.disclass_params)
        self.disgenXTest, _, _ = self.discrim_function(True, self.genXTest, *self.disclass_params)

    def gen_function(self, test_mode, Z, w, g, b, u, s, w2_0, g2_0, b2_0, u2_0, s2_0, w2, g2, b2, u2, s2,
                     w3_0, g3_0, b3_0, u3_0, s3_0, w3, g3, b3, u3, s3,
                     w4_0, g4_0, b4_0, u4_0, s4_0, w4, g4, b4, u4, s4, wx):
        x1, updates1 = running_batchnorm(T.dot(Z, w), g=g, b=b, running_u=u, running_s=s, momentum=self.bn_momentum,
                                         test_mode=test_mode)
        h = relu(x1)
        h = h.reshape((h.shape[0], self.ngf * 8, self.last_shape[0], self.last_shape[1]))
        x2_0, updates2_0 = running_batchnorm(deconv(h, w2_0, subsample=(1, 1), border_mode=(1, 1)), g=g2_0, b=b2_0,
                                             running_u=u2_0, running_s=s2_0, momentum=self.bn_momentum, test_mode=test_mode)
        h2_0 = relu(x2_0)
        x2, updates2 = running_batchnorm(deconv(h2_0, w2, subsample=(2, 2), border_mode=(1, 1)), g=g2, b=b2,
                                         running_u=u2, running_s=s2, momentum=self.bn_momentum, test_mode=test_mode)
        h2 = relu(x2)
        x3_0, updates3_0 = running_batchnorm(deconv(h2, w3_0, subsample=(1, 1), border_mode=(1, 1)), g=g3_0, b=b3_0,
                                             running_u=u3_0, running_s=s3_0, momentum=self.bn_momentum, test_mode=test_mode)
        h3_0 = relu(x3_0)
        x3, updates3 = running_batchnorm(deconv(h3_0, w3, subsample=(2, 2), border_mode=(1, 1)), g=g3, b=b3,
                                         running_u=u3, running_s=s3, momentum=self.bn_momentum, test_mode=test_mode)
        h3 = relu(x3)
        x4_0, updates4_0 = running_batchnorm(deconv(h3, w4_0, subsample=(1, 1), border_mode=(1, 1)), g=g4_0, b=b4_0,
                                             running_u=u4_0, running_s=s4_0, momentum=self.bn_momentum, test_mode=test_mode)
        h4_0 = relu(x4_0)
        x4, updates4 = running_batchnorm(deconv(h4_0, w4, subsample=(2, 2), border_mode=(1, 1)), g=g4, b=b4,
                                         running_u=u4, running_s=s4, momentum=self.bn_momentum, test_mode=test_mode)
        h4 = relu(x4)
        x = tanh(deconv(h4, wx, subsample=self.visible_subsamp, border_mode=self.visible_bmode))
        return x, updates1 + updates2_0 + updates2 + updates3_0 + updates3 + updates4_0 + updates4

    def discrim_function(self, test_mode, X, w, w2_0, g2_0, b2_0, u2_0, s2_0, w2, g2, b2, u2, s2,
                         w3_0, g3_0, b3_0, u3_0, s3_0, w3, g3, b3, u3, s3,
                         w4_0, g4_0, b4_0, u4_0, s4_0, w4, g4, b4, u4, s4, g5, b5, u5, s5, wy, wc, bc):
        h = lrelu(dnn_conv(X, w, subsample=self.visible_subsamp, border_mode=self.visible_bmode))
        x2_0, updates2_0 = running_batchnorm(dnn_conv(h, w2_0, subsample=(1, 1), border_mode=(1, 1)), g=g2_0, b=b2_0,
                                         running_u=u2_0, running_s=s2_0, momentum=self.bn_momentum, test_mode=test_mode)
        h2_0 = lrelu(x2_0)
        x2, updates2 = running_batchnorm(dnn_conv(h2_0, w2, subsample=(2, 2), border_mode=(1, 1)), g=g2, b=b2,
                                         running_u=u2, running_s=s2, momentum=self.bn_momentum, test_mode=test_mode)
        h2 = lrelu(x2)
        x3_0, updates3_0 = running_batchnorm(dnn_conv(h2, w3_0, subsample=(1, 1), border_mode=(1, 1)), g=g3_0, b=b3_0,
                                             running_u=u3_0, running_s=s3_0, momentum=self.bn_momentum, test_mode=test_mode)
        h3_0 = lrelu(x3_0)
        x3, updates3 = running_batchnorm(dnn_conv(h3_0, w3, subsample=(2, 2), border_mode=(1, 1)), g=g3, b=b3,
                                         running_u=u3, running_s=s3, momentum=self.bn_momentum, test_mode=test_mode)
        h3 = lrelu(x3)
        x4_0, updates4_0 = running_batchnorm(dnn_conv(h3, w4_0, subsample=(1, 1), border_mode=(1, 1)), g=g4_0, b=b4_0,
                                             running_u=u4_0, running_s=s4_0, momentum=self.bn_momentum, test_mode=test_mode)
        h4_0 = lrelu(x4_0)
        x4, updates4 = running_batchnorm(dnn_conv(h4_0, w4, subsample=(2, 2), border_mode=(1, 1)), g=g4, b=b4,
                                         running_u=u4, running_s=s4, momentum=self.bn_momentum, test_mode=test_mode)
        h4 = lrelu(x4)
        h4 = T.flatten(h4, 2)
        x4, updates5 = running_mapnorm(x4, g=g5, b=b5, running_u=u5, running_s=s5, test_mode=test_mode)
        h5 = pool_2d(x4, (self.last_shape[0], self.last_shape[1]), ignore_border=True, st=None, padding=(0, 0),
                     mode='max')
        y = sigmoid(T.dot(T.flatten(h5, 2), wy))
        h5 = h5.reshape((h5.shape[0], self.ndf, 8))
        h5 = T.max(h5, axis=2, keepdims=False)
        c = softmax(T.dot(h5, wc) + bc.dimshuffle('x', 0))
        return y, c, updates2_0 + updates2 + updates3_0 + updates3 + updates4_0 + updates4 + updates5
