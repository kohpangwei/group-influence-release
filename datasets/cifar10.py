#adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/cifar10.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import pickle
import urllib
import zipfile, tarfile

from datasets.common import get_dataset_dir, maybe_download, DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

def _one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers)+1
    return np.eye(num_classes, dtype=float)[class_numbers]

def load_cifar10(validation_size=1000, data_dir=None):
    dataset_dir = get_dataset_dir('cifar10', data_dir=data_dir)
    cifar10_path = os.path.join(dataset_dir, 'cifar10.npz')

    img_size = 32
    num_channels = 3
    img_size_flat = img_size * img_size * num_channels
    num_classes = 10

    _num_files_train = 5
    _images_per_file = 10000
    _num_images_train = _num_files_train * _images_per_file

    if not os.path.exists(cifar10_path):
        def _unpickle(f):
            f = os.path.join(dataset_dir, 'cifar-10-batches-py', f)
            print("Loading data: " + f)
            with open(f, 'rb') as fo:
                data_dict = pickle.load(fo)
            return data_dict

        def _convert_images(raw):
            raw_float = np.array(raw, dtype=float) / 255.0
            images = raw_float.reshape([-1, num_channels, img_size, img_size])
            images = images.transpose([0,2,3,1])
            return images

        def _load_data(f):
            data = _unpickle(f)
            raw = data[b'data']
            labels = np.array(data[b'labels'])
            images = _convert_images(raw)
            return images, labels

        def _load_class_names():
            raw = _unpickle(f='batches.meta')[b'label_names']
            return [x.decode('utf-8') for x in raw]

        def _load_training_data():
            images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
            labels = np.zeros(shape=[_num_images_train], dtype=int)

            begin = 0
            for i in range(_num_files_train):
                images_batch, labels_batch = _load_data(f='data_batch_'+str(i+1))
                end = begin + len(images_batch)
                images[begin:end,:] = images_batch
                labels[begin:end] = labels_batch
                begin = end
            return images, labels, _one_hot_encoded(class_numbers=labels, num_classes=num_classes)

        def _load_test_data():
            images, labels = _load_data(f='test_batch')
            return images, labels, _one_hot_encoded(class_numbers=labels, num_classes=num_classes)

        SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        raw_cifar10_path = maybe_download(SOURCE_URL, 'cifar-10-python.tar.gz', dataset_dir)

        print('Extracting {}.'.format(raw_cifar10_path))
        with tarfile.open(raw_cifar10_path, 'r:gz') as tarf:
            tarf.extractall(path=dataset_dir)

        # no one-hot encoding of labels
        train_images, train_labels, _ = _load_training_data()
        test_images, test_labels, _ = _load_test_data()
        names = _load_class_names()

        validation_images = train_images[:validation_size]
        validation_labels = train_labels[:validation_size]
        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]

        np.savez(cifar10_path,
                 train_images=train_images,
                 train_labels=train_labels,
                 validation_images=validation_images,
                 validation_labels=validation_labels,
                 test_images=test_images,
                 test_labels=test_labels)
    else:
        data = np.load(cifar10_path)
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

def load_small_cifar10(validation_size=1000, random_seed=0, data_dir=None):
    dataset_dir = get_dataset_dir('cifar10', data_dir=data_dir)

    data_sets = load_cifar10(validation_size, data_dir=data_dir)
    rng = np.random.RandomState(random_seed)

    train_images = data_sets.train.x
    train_labels = data_sets.train.labels
    perm = np.arange(len(train_labels))
    rng.shuffle(perm)
    num_to_keep = int(len(train_labels)/10)
    perm = perm[:num_to_keep]
    train_images = train_images[perm,:]
    train_labels = train_labels[perm]

    validation_images = data_sets.validation.x
    validation_labels = data_sets.validation.labels
    test_images = data_sets.test.x
    test_labels = data_sets.test.labels

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)
    return base.Datasets(train=train, validation=validation, test=test)
