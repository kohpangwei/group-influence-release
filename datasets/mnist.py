from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import gzip
import numpy as np

from datasets.common import get_dataset_dir, maybe_download, DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
    """Extract the images into a 4D uint8 np array [index, y, x, depth].
    Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D unit8 np array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
    return data

def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 np array [index].
    Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D unit8 np array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
        return dense_to_one_hot(labels, num_classes)
    return labels

def load_mnist(validation_size=5000, data_dir=None):
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    dataset_dir = get_dataset_dir('mnist', data_dir=data_dir)

    local_files = [maybe_download(SOURCE_URL + image_file, image_file, dataset_dir)
                   for image_file in (TRAIN_IMAGES, TRAIN_LABELS,
                                      TEST_IMAGES, TEST_LABELS)]

    with open(local_files[0], 'rb') as f:
        train_images = extract_images(f)

    with open(local_files[1], 'rb') as f:
        train_labels = extract_labels(f)

    with open(local_files[2], 'rb') as f:
        test_images = extract_images(f)

    with open(local_files[3], 'rb') as f:
        test_labels = extract_labels(f)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train_images = train_images.astype(np.float32) / 255
    validation_images = validation_images.astype(np.float32) / 255
    test_images = test_images.astype(np.float32) / 255

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)

def load_small_mnist(validation_size=5000, random_seed=0, data_dir=None):
    dataset_dir = get_dataset_dir('mnist', data_dir=data_dir)
    mnist_small_file = 'mnist_small_val-{}_seed-{}.npz'.format(
        validation_size, random_seed)
    mnist_small_path = os.path.join(dataset_dir, mnist_small_file)

    if not os.path.exists(mnist_small_path):
        rng = np.random.RandomState(seed=random_seed)
        data_sets = load_mnist(validation_size, data_dir=data_dir)

        train_images = data_sets.train.x
        train_labels = data_sets.train.labels
        perm = np.arange(len(train_labels))
        rng.shuffle(perm)
        num_to_keep = int(len(train_labels) / 10)
        perm = perm[:num_to_keep]
        train_images = train_images[perm, :]
        train_labels = train_labels[perm]

        validation_images = data_sets.validation.x
        validation_labels = data_sets.validation.labels
        # perm = np.arange(len(validation_labels))
        # rng.shuffle(perm)
        # num_to_keep = int(len(validation_labels) / 10)
        # perm = perm[:num_to_keep]
        # validation_images = validation_images[perm, :]
        # validation_labels = validation_labels[perm]

        test_images = data_sets.test.x
        test_labels = data_sets.test.labels
        # perm = np.arange(len(test_labels))
        # rng.shuffle(perm)
        # num_to_keep = int(len(test_labels) / 10)
        # perm = perm[:num_to_keep]
        # test_images = test_images[perm, :]
        # test_labels = test_labels[perm]

        np.savez(mnist_small_path,
                 train_images=train_images,
                 train_labels=train_labels,
                 validation_images=validation_images,
                 validation_labels=validation_labels,
                 test_images=test_images,
                 test_labels=test_labels)
    else:
        data = np.load(mnist_small_path)
        train_images = data['train_images']
        train_labels = data['train_labels']
        validation_images = data['validation_images']
        validation_labels = data['validation_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)
