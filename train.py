import os
import glob
import argparse
import random

dataset_path = os.getenv('DATASET_PATH')
output_path = os.getenv('OUTPUT_PATH')


import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
from dataset import *
from updater import *

from model import *
from utils import notify


def v(x):
    return Variable(np.asarray(x, dtype=np.float32))

def vi(x):
    return Variable(np.asarray(x, dtype=np.int32))


def main(args):
    print('GPU: {}'.format(args.gpu))
    print('# Train')
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # dataset:
    if args.dataset == 'rot':
        train_set = RotateMnistDataset(src='train', rotate=[0, 15, 30, 45, 60])
        test_set = RotateMnistDataset(src='test', rotate=[75], return_domain=False)
    elif args.dataset == 'digits':
        mnist_train, test = get_mnist(ndim=3)
        mnist_m = MNIST_MDataset()
        usps = USPSDataset()
        svhn = SVHNDataset(k=5)
        train_set = CrossDomainDigitDataset(datasets=[mnist_train, mnist_m, usps, svhn])
        test_set = SVHNDataset(src='test')

    style = StyleNet(out_channels=train_set.n_domain)
    lenet = LeNet(out_channels=train_set.n_classes, style=style)
    lenet_cl = L.Classifier(lenet)

    opt_f = optimizers.SGD(lr=0.02)
    # opt_f = optimizers.RMSprop(lr=0.02)
    opt_f.setup(lenet)
    opt_f.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    opt_style = optimizers.SGD(lr=0.02)
    # opt_style = optimizers.RMSprop(lr=0.02)
    opt_style.setup(style)
    opt_style.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    train_iter = chainer.iterators.SerialIterator(train_set, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_set, args.batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CrossGradUpdater(
        alpha=args.alpha,
        eps=args.eps,
        models=(lenet, style),
        iterator={
            'main': train_iter,
            # 'test': test_iter,
        },
        optimizer={
            'f': opt_f,
            'style': opt_style},
        device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.output_path)

    snapshot_interval = (10, 'epoch')

    trainer.extend(extensions.snapshot(
        filename='snapshot'),
                   trigger=snapshot_interval)
    trainer.extend(extensions.Evaluator(test_iter, lenet_cl, device=args.gpu))
    trainer.extend(extensions.snapshot_object(
        lenet, 'f_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        style, 'style_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'f/loss', 'style/loss', 'f/acc', 'style/acc', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time'
    ]))
    trainer.extend(extensions.PlotReport(['f/acc', 'style/acc'], file_name='acc.png'))
    trainer.extend(extensions.PlotReport(['f/loss', 'style/loss'], file_name='loss.png'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(os.path.join(args.output_path, 'snapshot'), trainer)

    # Run the training
    trainer.run()

    # target_rotate = [[0, 15, 30, 45, 60], [75]]
    # notify(args.__repr__())
    # for r in target_rotate:
    #     test_set = RotateMnistDataset(src='test', rotate=r, return_domain=False)
    #     test_iter = chainer.iterators.SerialIterator(test_set, args.batchsize, repeat=False, shuffle=False)
    #
    #     e = extensions.Evaluator(test_iter, lenet_cl, device=args.gpu)
    #     res = e()
    #     print(res)
    #     notify('rotate:{}'.format(r), res.__repr__())


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # dataset io
    p.add_argument('-o', '--output_path', metavar='PATH', type=str, default='test',
                   help='output_path (default: ./output)')
    p.add_argument('-d', '--dataset', type=str, default='rot',
                   help='rot or digits')
    # p.add_argument('-m', '--model', type=str, default='U',
    #                help='U or Refine')

    # train
    p.add_argument('-b', '--batchsize', metavar='N', type=int, default=16,
                   help='batch size (default: 128)')
    p.add_argument('-w', '--weight_decay', metavar='N', type=float, default=0.0001,
                   help='weight decay coefficient (default 0.00001)')
    p.add_argument('--alpha', metavar='N', type=float, default=0.7,
                   help='perturb term weight')
    p.add_argument('--eps', metavar='N', type=float, default=1.5,
                   help='perturb norm')
    p.add_argument('-e', '--epoch', metavar='N', type=int, default=100,
                   help='number of epochs (default: 100)')
    p.add_argument('-g', '--gpu', metavar='N', type=int, default=0,
                   help='gpu id (-1 if use cpu)')
    p.add_argument('-r', '--resume', dest='resume', action='store_true')

    args = p.parse_args()
    print('dataset_path: ', dataset_path)
    print('output_path: ', output_path)
    print(args)
    args.output_path = os.path.join(output_path, args.output_path)
    os.makedirs(args.output_path, exist_ok=True)

    main(args)