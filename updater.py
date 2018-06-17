#!/usr/bin/env python
import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np



def v(x):
    return Variable(np.asarray(x, dtype=np.float32))
def vi(x):
    return Variable(np.asarray(x, dtype=np.int32))


class CrossGradUpdater(chainer.training.StandardUpdater):

    def __init__(self, alpha=0, eps=0.5, *args, **kwargs):
        self.lenet, self.style = kwargs.pop('models')
        self.f = self.lenet
        self.alpha = alpha
        self.eps = eps
        super(CrossGradUpdater, self).__init__(*args, **kwargs)

    def loss_l(self, y, y_pred, y_pred_perturb):
        l = F.softmax_cross_entropy(y_pred, y)
        lp = F.softmax_cross_entropy(y_pred_perturb, y)
        losses = (1 - self.alpha) * l + self.alpha * lp
        chainer.report({'loss': losses}, self.f)
        chainer.report({'acc': F.accuracy(y_pred, y)}, self.f)
        return losses

    def loss_d(self, d, d_pred, d_pred_perturb):
        l = F.softmax_cross_entropy(d_pred, d)
        lp = F.softmax_cross_entropy(d_pred_perturb, d)
        losses = (1 - self.alpha) * l + self.alpha * lp
        chainer.report({'loss': losses}, self.style)
        chainer.report({'acc': F.accuracy(d_pred, d)}, self.style)
        return losses

    def update_core(self):
        """
        enc, dec, dis

        :return:
        """
        f_optimizer = self.get_optimizer('f')
        style_optimizer = self.get_optimizer('style')

        f, style = self.lenet, self.style
        xp = f.xp

        batch = self.get_iterator('main').next()

        batchsize = len(batch)
        in_ch = 1
        w_in = 28

        x = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        y = xp.zeros(batchsize).astype("i")
        d = xp.zeros(batchsize).astype("i")

        for i in range(batchsize):
            x[i] = xp.asarray(batch[i][0])
            y[i] = xp.asarray(batch[i][1])
            d[i] = xp.asarray(batch[i][2])

        xl = Variable(x)
        xd = Variable(x)

        # with chainer.using_config('train', False):
        g = style.feature(xl)
        g.unchain_backward()
        y_pred = f.call(xl, g)
        d_pred = style(xd)

        loss_l = F.softmax_cross_entropy(y_pred, y)
        loss_l.backward()
        loss_d = F.softmax_cross_entropy(d_pred, d)
        loss_d.backward()

        xl_perturb = xl + xd._grad_var * self.eps
        xl_perturb.zerograd()
        xd_perturb = xd + xl._grad_var * self.eps
        xd_perturb.zerograd()

        # with chainer.using_config('train', False):
        g = style.feature(xl_perturb)
        g.unchain_backward()
        y_pred_perturb = f.call(xl_perturb, g)
        d_pred_perturb = style(xd_perturb)

        # xl.cleargrad()
        # xd.cleargrad()
        # with chainer.using_config('train', False):
        #     g = style.feature(xl)
        #     g.unchain_backward()
        #     y_pred = f(xl, g)
        #     d_pred = style(xd)

        # y_pred_perturb = y_pred
        # d_pred_perturb = d_pred

        # update g and h
        f_optimizer.update(self.loss_l, y, y_pred, y_pred_perturb)
        style_optimizer.update(self.loss_d, d, d_pred, d_pred_perturb)