from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import sys
import urllib
import numpy as np
from scipy.sparse import dok_matrix
import copy, warnings
import tarfile

from tensorflow.contrib.learn.python.learn.datasets import base

"""
The fixed directory (../../data/) to save datasets into.
"""
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def get_dataset_dir(dataset_name, data_dir=None):
    """
    Returns the absolute path to the canonical base directory for a dataset,
    creating it if it does not already exist.

    :param dataset_name: The name of the dataset
    :param data_dir: The path to the base dataset directory. If None, defaults to
                     a subdirectory of the influence project root.
    :return: The absolute path to the dataset's base directory
    """

    data_dir = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    dataset_dir = os.path.join(data_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise Exception('The directory ({}) for dataset {} already exists but is not a directory.'.format(dataset_dir), dataset_name)
    return dataset_dir

def maybe_download(url, filename, download_dir):
    """
    Downloads a file into the specified download directory if it does
    not already exist.

    :param url: the web URL to the file
    :param filename: the filename to save it as
    :param download_dir: the directory to download into
    :return: the absolute path to the downloaded file
    """

    save_path = os.path.abspath(os.path.join(download_dir, filename))

    if not os.path.exists(save_path):
        print("Downloading {} into {}".format(filename, download_dir))

        def _print_download_progress(count, blockSize, totalSize):
            percent = int(count * blockSize * 100.0 / totalSize)
            sys.stdout.write("\rDownloading {}: {}%".format(filename, percent))
            sys.stdout.flush()

        file_path, _ = urllib.urlretrieve(url=url,
                                          filename=save_path,
                                          reporthook=_print_download_progress)

        print("\nDownload complete.")

    return save_path

class DataSet(object):
    """
    Generic DataSet class containing batching functionality.

    Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
    """
    def __init__(self, x, labels, rng=None, seed=0):
        """
        Initializes the dataset.

        :param x: A numpy array representing the features, where the first dimension
                  is the number of examples. The data will be flattened to a 2-D array
                  so each example is a row vector.
        :param labels: A numpy int array representing the labels. The labels will be
                       flattened to a 1-D array.
        :param rng: The numpy.random.RandomState to initialize the dataset with. If
                    None, a new RandomState will be created from the supplied seed.
        :param seed: The seed to create a new RandomState with, if rng is None.
        """

        # Flatten x to a 2D array
        x = x.astype(np.float32)
        if len(x.shape) > 2:
            x = np.reshape(x, (x.shape[0], -1))

        # Flatten the labels to a 1D array
        labels = labels.reshape(-1)

        # Ensure they represent the same number of examples
        assert(x.shape[0] == labels.shape[0])

        self._x = x
        self._labels = labels
        self.initialize_rng(rng=rng, seed=seed)

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._x.shape[0]

    def clone(self):
        """
        Return an independent copy of this Dataset with the same underlying
        data and batching state, but keep the new batching state completely
        independent from the original.

        :return: An independent clone of the DataSet.
        """
        copy = DataSet(self._x, self._labels)
        copy.set_state(self.get_state())
        return copy

    def subset(self, indices, rng=None, seed=0):
        """
        Return a subset of this Dataset.

        :param indices: The example indices to keep, in the order they will
                        be in the new dataset.
        :return: A new dataset
        :param rng: The numpy.random.RandomState to initialize the dataset with. If
                    None, a new RandomState will be created from the supplied seed.
        :param seed: The seed to create a new RandomState with, if rng is None.
        """
        return DataSet(self._x[indices], self._labels[indices], rng=rng, seed=seed)

    def initialize_indices(self):
        """
        Initialize/reinitialize the batching order.
        """
        self._index_in_epoch = self.num_examples
        self._epoch_indices = np.arange(self.num_examples)

    def initialize_rng(self, rng=None, seed=0):
        """
        Initialize/reinitialize the random number generator and also
        by extension, the batching order.

        :param rng: The numpy.random.RandomState to initialize the dataset with. If
                    None, a new RandomState will be created from the supplied seed.
        :param seed: The seed to create a new RandomState with, if rng is None.
        """
        if rng is not None:
            self._rng = rng
        else:
            self._rng = np.random.RandomState(seed)
        self.initialize_indices()

    def _get_indices(self, max_indices):
        """
        Private helper function to get up to max_indices new randomized examples
        by their indices. Will return at least one index.

        :param max_indices: The maximum number of indices to return
        :return: A numpy array of indices. Contains at least 1 index.
        """

        # Ensure there is at least one more index in the current epoch
        if self._index_in_epoch == self.num_examples:
            self._index_in_epoch = 0
            self._epoch_indices = self._rng.permutation(self.num_examples)

        # Consume as many examples from the current epoch as possible
        start_index = self._index_in_epoch
        end_index = min(start_index + max_indices, self.num_examples)
        self._index_in_epoch = end_index
        return self._epoch_indices[start_index:end_index]

    def next_batch_indices(self, batch_size):
        """
        Get the indices for the next batch of examples.

        :param batch_size: The size of the next batch of examples.
        :return: A (batch_size,) numpy array containing the indices of the next batch.
        """
        assert batch_size > 0

        indices = []
        while len(indices) < batch_size:
            indices.extend(self._get_indices(batch_size - len(indices)))
        return np.array(indices)

    def next_batch(self, batch_size):
        """
        Get the next batch of examples.

        :param batch_size: The size of the next batch of examples.
        :return: A (batch_size, input dimension) numpy array containing the batch of examples,
                 and a (batch_size,) numpy array containing the labels for these examples.
        """
        indices = self.next_batch_indices(batch_size)
        return self._x[indices], self._labels[indices]

    def next_batch_with_indices(self, batch_size):
        """
        Get the next batch of examples along with their indices.

        :param batch_size: The size of the next batch of examples.
        :return: A (batch_size, input dimension) numpy array containing the batch of examples,
                 a (batch_size,) numpy array containing the labels for these examples, and
                 a (batch_size,) numpy array containing the indices of these examples.
        """
        indices = self.next_batch_indices(batch_size)
        return self._x[indices], self._labels[indices], indices

    def get_state(self):
        """
        Return the full state of this dataset batcher, without the underlying data..
        :return: A tuple containing the full state of the dataset batcher.
        """
        return self._rng.get_state(), self._index_in_epoch, list(self._epoch_indices)

    def set_state(self, state):
        """
        Sets the full state of this dataset batcher, without the underlying data.
        :param state: A tuple containing the full state of the dataset batcher.
        """
        rng_state, index_in_epoch, epoch_indices = state
        assert len(epoch_indices) == self.num_examples
        self._rng.set_state(rng_state)
        self._index_in_epoch = index_in_epoch
        self._epoch_indices = np.array(epoch_indices)

    def save_state(self, state_path):
        """
        Saves the full state of this dataset batcher, without the underlying data,
        into the given path. Uses pickle.

        :param state_path: The path to where the state should be saved.
        """
        state = self.get_state()
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, state_path):
        """
        Loads the full state of this dataset batcher, without the underlying data,
        from the given path. Uses pickle.

        :param state_path: The path to the state.
        """
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        self.set_state(state)

def filter_dataset(X, Y, pos_class, neg_class):
    """
    Filters out elements of X and Y that aren't one of pos_class or neg_class
    then transforms labels of Y so that +1 = pos_class, -1 = neg_class.
    """
    assert(X.shape[0] == Y.shape[0])
    assert(len(Y.shape) == 1)

    Y = Y.astype(int)

    pos_idx = Y == pos_class
    neg_idx = Y == neg_class
    Y[pos_idx] = 1
    Y[neg_idx] = -1
    idx_to_keep = pos_idx | neg_idx
    X = X[idx_to_keep, ...]
    Y = Y[idx_to_keep]
    return (X, Y)

def find_distances(target, X, theta=None):
    assert len(X.shape) == 2, "X must be 2D, but it is currently %s" % len(X.shape)
    target = np.reshape(target, -1)
    assert X.shape[1] == len(target), \
        "X (%s) and target (%s) must have same feature dimension" % (X.shape[1], len(target))

    if theta is None:
        return np.linalg.norm(X - target, axis=1)
    else:
        theta = np.reshape(theta, -1)

        # Project onto theta
        return np.abs((X - target).dot(theta))

def center_data(datasets):
    avg = np.mean(datasets.train.x, axis=0)
    train, validation, test = [
        DataSet(dataset.x - avg, dataset.labels) if dataset is not None else None
        for dataset in (datasets.train, datasets.validation, datasets.test)
    ]
    return base.Datasets(train=train, validation=validation, test=test)

def append_bias(datasets):
    def append_bias_x(A):
        return np.hstack((A, np.ones((A.shape[0], 1))))
    train, validation, test = [
        DataSet(append_bias_x(dataset.x), dataset.labels) if dataset is not None else None
        for dataset in (datasets.train, datasets.validation, datasets.test)
    ]
    return base.Datasets(train=train, validation=validation, test=test)
