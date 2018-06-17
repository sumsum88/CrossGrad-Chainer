import os
import numpy as np
from chainer.dataset import dataset_mixin
from scipy import io
from chainer.datasets import get_mnist
from scipy.misc import imresize
from scipy.misc import imrotate, imsave


class RotateMnistDataset(dataset_mixin.DatasetMixin):
    img_size = (28, 28)
    n_classes = 10

    def __init__(self, src='train', rotate=(0, 15, 30, 45, 60, 75), return_domain=True):
        if src == 'train':
            data, _ = get_mnist(ndim=3)
        elif src == 'test':
            _, data = get_mnist(ndim=3)
        else:
            raise ValueError

        self.data = data
        self.n_domain = len(rotate)
        self.rotate = rotate
        self.return_domain = return_domain
        self.src = src

    def __len__(self):
        return len(self.data)

    def get_example(self, i, r=None):
        x, y = self.data[i]

        if r is None:
            r = np.random.randint(self.n_domain)
        x = imrotate(np.tile(x, (3, 1, 1)), self.rotate[r]).transpose(2, 0, 1)[[0]]

        if self.return_domain:
            return x, y, r
        else:
            return x.astype(np.float32), y.astype(np.int8)


if __name__ == '__main__':
    m = RotateMnistDataset()
    x, y, r = m.get_example(0, r=0)
    imsave('./png/0.png', np.tile(x.transpose(1, 2, 0), (1, 1, 3)))
    x, y, r = m.get_example(0, r=1)
    imsave('./png/1.png', np.tile(x.transpose(1, 2, 0), (1, 1, 3)))
    x, y, r = m.get_example(0, r=4)
    imsave('./png/4.png', np.tile(x.transpose(1, 2, 0), (1, 1, 3)))
