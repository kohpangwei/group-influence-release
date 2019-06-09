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

def load_dogfish(data_dir=None):
    dataset_dir = get_dataset_dir('processed_animals', data_dir=data_dir)

    BASE_URL = "http://mitra.stanford.edu/kundaje/pangwei/"
    TRAIN_FILE_NAME = "dogfish_900_300_inception_features_train.npz"
    TEST_FILE_NAME = "dogfish_900_300_inception_features_test.npz"
    train_path = maybe_download(BASE_URL + TRAIN_FILE_NAME, TRAIN_FILE_NAME, dataset_dir)
    test_path = maybe_download(BASE_URL + TEST_FILE_NAME, TEST_FILE_NAME, dataset_dir)

    data_train = np.load(train_path)
    data_test = np.load(test_path)

    X_train = data_train['inception_features_val']
    Y_train = data_train['labels'].astype(np.uint8)
    X_test = data_test['inception_features_val']
    Y_test = data_test['labels'].astype(np.uint8)

    train = DataSet(X_train, Y_train)
    test = DataSet(X_test, Y_test) 
    return base.Datasets(train=train, validation=None, test=test)

def load_animals(data_dir=None):
    dataset_dir = get_dataset_dir('processed_animals', data_dir=data_dir)

    BASE_URL = "http://mitra.stanford.edu/kundaje/pangwei/"
    TRAIN_FILE_NAME = "animals_900_300_inception_features_train.npz"
    TEST_FILE_NAME = "animals_900_300_inception_features_test.npz"
    train_path = maybe_download(BASE_URL + TRAIN_FILE_NAME, TRAIN_FILE_NAME, dataset_dir)
    test_path = maybe_download(BASE_URL + TEST_FILE_NAME, TEST_FILE_NAME, dataset_dir)

    data_train = np.load(train_path)
    data_test = np.load(test_path)

    X_train = data_train['inception_features_val']
    Y_train = data_train['labels'].astype(np.uint8)
    X_test = data_test['inception_features_val']
    Y_test = data_test['labels'].astype(np.uint8)

    train = DataSet(X_train, Y_train)
    test = DataSet(X_test, Y_test) 
    return base.Datasets(train=train, validation=None, test=test)
