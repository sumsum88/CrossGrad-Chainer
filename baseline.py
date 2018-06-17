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

    # train_set = RotateMnistDataset(src='train', rotate=[0, 15, 30, 45, 60], return_domain=False)
    # test_set = RotateMnistDataset(src='test', rotate=[0, 15, 30, 45, 60], return_domain=False)
    # dataset:
    if args.dataset == 'rot':
        train_set = RotateMnistDataset(src='train', rotate=[0, 15, 30, 45, 60], return_domain=False)
        test_set = RotateMnistDataset(src='test', rotate=[75], return_domain=False)

    elif args.dataset == 'digits':
        mnist_train, test = get_mnist(ndim=3)
        mnist_m = MNIST_MDataset()
        svhn = SVHNDataset()
        usps = USPSDataset()
        train_set = CrossDomainDigitDataset(datasets=[mnist_train, mnist_m, usps], return_domain=False)
        test_set = svhn

    lenet = LeNetBase(out_channels=train_set.n_classes)
    lenet_cl = L.Classifier(lenet)

    opt_f = optimizers.SGD(lr=0.02)
    opt_f.setup(lenet_cl)
    opt_f.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    train_iter = chainer.iterators.SerialIterator(train_set, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_set, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, opt_f, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.output_path)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_iter, lenet_cl, device=args.gpu))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'main/accuracy','validation/main/loss', 'validation/main/accuracy', 'elapsed_time'
    ]))
    trainer.extend(
        extensions.snapshot(
        filename='snapshot'),
        trigger=(100, 'epoch')
    )
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    target_rotate = [75]
    notify(args.__repr__())
    for r in target_rotate:
        test_set = RotateMnistDataset(src='test', rotate=[r], return_domain=False)
        test_iter = chainer.iterators.SerialIterator(test_set, args.batchsize, repeat=False, shuffle=False)
        e = extensions.Evaluator(test_iter, lenet_cl, device=args.gpu)
        res = e()
        print(res)
        notify('rotate:{}'.format(r), res.__repr__())


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # dataset io
    p.add_argument('-o', '--output_path', metavar='PATH', type=str, default='test',
                   help='output_path (default: ./output)')
    # p.add_argument('-d', '--dataset', type=str, default='CS',
    #                help='MSRC or DSD or CS')
    # p.add_argument('-m', '--model', type=str, default='U',
    #                help='U or Refine')

    # train
    p.add_argument('-b', '--batchsize', metavar='N', type=int, default=16,
                   help='batch size (default: 128)')
    p.add_argument('-w', '--weight_decay', metavar='N', type=float, default=0.0001,
                   help='weight decay coefficient (default 0.00001)')
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