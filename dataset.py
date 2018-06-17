import os
import numpy as np
from chainer.dataset import dataset_mixin
from scipy import io
from chainer.datasets import get_mnist
from scipy.misc import imresize
from scipy.misc import imrotate, imsave
from collections import defaultdict
from PIL import Image

dataset_path = os.getenv('DATASET_PATH')


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


class SVHNDataset(dataset_mixin.DatasetMixin):
    img_size = (28, 28)

    def __init__(self, root=os.path.join(dataset_path, 'SVHN_MNIST'), src='train', size=999999999, k=None):
        if src == 'train':
            mat = io.loadmat(os.path.join(root, 'train_32x32.mat'))
        elif src == 'test':
            mat = io.loadmat(os.path.join(root, 'test_32x32.mat'))
        else:
            raise ValueError

        matx = mat['X'].transpose(2, 3, 0, 1).mean(axis=0)
        maty = mat['y'][:, 0].astype(np.int8)
        if k is None:
            self.x = []
            for x in matx[:size]:
                self.x.append(imresize(x, self.img_size)[np.newaxis, ...])

            self.x = np.array(self.x, dtype=np.float32)
            self.y = maty[:size]

        else:
            self.x, self.y = [], []
            counter = defaultdict(int)

            n, i = 0, 0
            while n < k * 10:
                x = imresize(matx[n], self.img_size)[np.newaxis, ...]
                y = maty[n]
                if counter[y] < k:
                    self.x.append(x)
                    self.y.append(y)
                    n += 1
                i += 1
            self.x = np.array(self.x, dtype=np.float32)

    def __len__(self):
        return 11000#len(self.x)

    def get_example(self, i):
        i = i % len(self.x)
        return self.x[i], self.y[i]


class USPSDataset(dataset_mixin.DatasetMixin):
    img_size = (28, 28)
    n_classes = 10

    def __init__(self):
        mat = io.loadmat('./datasets/usps_all.mat')
        self.data = mat['data']

    def __len__(self):
        return 1100 * 10

    def get_example(self, i):
        ix = i // 10
        y = i % 10
        x = imresize(self.data[:, ix, y].reshape(16, 16), self.img_size)[np.newaxis, ...]
        return x, y


class MNIST_MDataset(dataset_mixin.DatasetMixin):
    img_size = (28, 28)
    n_classes = 10

    def __init__(self, data_root=os.path.join(dataset_path, 'mnist_m'), src='train', transform=None):
        self.transform = transform

        if src == 'train':
            data_list = os.path.join(data_root, 'mnist_m_train_labels.txt')
            self.root = os.path.join(data_root, 'mnist_m_train')
        elif src == 'test':
            data_list = os.path.join(data_root, 'mnist_m_test_labels.txt')
            self.root = os.path.join(data_root, 'mnist_m_test')
        else:
            raise ValueError

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def get_example(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')
        img = imresize(np.array(imgs), self.img_size).transpose(2, 0, 1).astype(np.float32)
        # nch 3 -> 1
        img = img.mean(axis=0)[np.newaxis, ...]

        if self.transform is not None:
            img = self.transform(img)

        labels = int(labels)

        return img, labels

    def __len__(self):
        return self.n_data


class CrossDomainDigitDataset(dataset_mixin.DatasetMixin):
    img_size = (28, 28)
    n_classes = 10

    def __init__(self, datasets, return_domain=True):
        self.n_domain = len(datasets)
        self.datasets = datasets
        self.return_domain = return_domain

    def get_example(self, i, r=None):
        if r is None:
            r = np.random.randint(self.n_domain)
        x, y = self.datasets[r][i]

        if self.return_domain:
            return x.astype(np.float32), y, r
        else:
            return x.astype(np.float32), y

    def __len__(self):
        return min(map(len, self.datasets))


if __name__ == '__main__':
    # m = RotateMnistDataset()
    # x, y, r = m.get_example(0, r=0)
    # imsave('./png/0.png', np.tile(x.transpose(1, 2, 0), (1, 1, 3)))
    # x, y, r = m.get_example(0, r=1)
    # imsave('./png/1.png', np.tile(x.transpose(1, 2, 0), (1, 1, 3)))
    # x, y, r = m.get_example(0, r=4)
    # imsave('./png/4.png', np.tile(x.transpose(1, 2, 0), (1, 1, 3)))
    mnist_train, test = get_mnist(ndim=3)
    mnist_m = MNIST_MDataset()
    svhn = SVHNDataset()
    usps = USPSDataset()

    digits = CrossDomainDigitDataset(datasets=[mnist_train, mnist_m, svhn, usps])
    print(len(digits))
    x, y, r = digits.get_example(0, r=0)
    print(x.shape, y, r)
    x, y, r = digits.get_example(1, r=1)
    print(x.shape, y, r)
    x, y, r = digits.get_example(2, r=2)
    print(x.shape, y, r)
    x, y, r = digits.get_example(3, r=3)
    print(x.shape, y, r)

