from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import urllib
import sqlite3
import pandas as pd
import json

from tqdm import tqdm
from experiments.benchmark import benchmark

from datasets.common import get_dataset_dir, maybe_download, DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

#TODO: remove XXWO (351)

def load_mnli_ids(data_dir=None, force_refresh=False):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli_annotators.npz')

    if not os.path.exists(path) or force_refresh:
        orig_names = ['multinli_1.0_train_with_annotators.jsonl',
                      'multinli_1.0_dev_matched_with_annotators.jsonl',
                      'multinli_1.0_dev_mismatched_with_annotators.jsonl']
        all_ids = []
        for name in orig_names:
            ids = []
            with open(os.path.join(dataset_dir, name)) as f:
                for line in f:
                    ids.append(json.loads(line)['sentence2_author'])
            all_ids.append(np.array(ids))
        np.savez(os.path.join(dataset_dir, 'mnli_nonfires_annotators'), ids=all_ids[2])
        all_ids = [all_ids[0], [], all_ids[1]]
        np.savez(path, ids=all_ids)
    else:
        all_ids = np.load(path)['ids']
    return np.array(all_ids)

def load_reduced_mnli_ids(data_dir=None):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'reduced_mnli_annotators.npz')

    print('Loading reduced mnli annotators from {}'.format(path))
    indices = np.load(os.path.join(dataset_dir, 'reduced_mnli_indices.npz'))['indices']
    ids = load_mnli_ids(data_dir)
    ids[0] = ids[0][indices]
    np.savez(path, ids=ids)

    print('Loaded reduced mnli annotators for {} train examples'.format(len(ids[0])))
    return ids

def load_no_XXWO_mnli_ids(data_dir=None):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'reduced_mnli_annotators.npz')

    print('Loading no XXWO mnli annotators from {}'.format(path))
    indices = np.load(os.path.join(dataset_dir, 'no_XXWO_mnli_indices.npz'))['indices']
    ids = load_mnli_ids(data_dir)
    ids[0] = ids[0][indices]
    np.savez(path, ids=ids)

    print('Loaded no XXWO mnli annotators for {} train examples'.format(len(ids[0])))
    return ids

def load_mnli_nonfires_ids(data_dir=None, force_refresh=False):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli_nonfires_annotators.npz')

    if not os.path.exists(path) or force_refresh:
        ids = []
        with open(os.path.join(dataset_dir, 'multinli_1.0_dev_mismatched_with_annotators.jsonl')) as f:
            for line in f:
                ids.append(json.loads(line)['sentence2_author'])
                if ids[-1] == 'XXWO':
                    assert False #TODO
        ids = np.array(ids)
        np.savez(path, ids=ids)
    else:
        ids = np.load(path)['ids']
    return np.array(ids)

def load_mnli(data_dir=None, non_tf=False, mnli_num=4):
    num = mnli_num
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli_2.npz')

    print('Loading mnli from {}.'.format(path))
    data = np.load(path)
    labels = [data['lab0'],[],data['lab2']]
    x = [data['x0'],[],data['x2']]

    if num == 4:
        mask = range(1200)
    elif num == 3:
        mask = range(600) + range(900, 1200)
    elif num == 2:
        mask = range(600)
    x[0] = np.array(x[0])[:, mask]
    x[1] = np.array([],dtype=np.float32)
    labels[1] = np.array([],dtype=np.float32)
    x[2] = np.array(x[2])[:, mask]

    train = DataSet(x[0], labels[0])
    validation = DataSet(x[1], labels[1])
    test = DataSet(x[2], labels[2])

    print('Loaded mnli.')

    if non_tf: return (train, validation, test)
    return base.Datasets(train=train, validation=validation, test=test)

def load_reduced_mnli(data_dir=None, size=None, proportion=None, force_refresh=False, mnli_num=4): #TODO: don't force_refresh; there are problems with indices for different ways of reducing
    num = mnli_num
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'reduced_mnli_size-{}_proportion-{}.npz'.format(size, proportion))

    print('Loading reduced mnli from {}.'.format(path))

    if not os.path.exists(path) or force_refresh:
        ds = load_mnli(data_dir, non_tf=True)

        if size is None:
            ids = load_mnli_ids(data_dir)[0]
            uniq, inds = np.unique(ids, return_inverse=True)
            indices = []
            rng = np.random.RandomState(0)
            for i, turker in enumerate(uniq):
                valid_indices = np.where(inds==i)[0]
                if len(valid_indices) == 0: print(turker)
                indices.extend(rng.choice(valid_indices, int(valid_indices.shape[0] * proportion)))
            indices = np.array(indices)

        elif proportion is None:
            indices = np.random.RandomState(0).choice(ds[0].num_examples, size)

        np.savez(os.path.join(dataset_dir, 'reduced_mnli_indices'), indices=indices) #TODO: this is bad!! Need to support 400k
        train = ds[0].subset(indices)
        np.savez(path, x=[train.x, ds[1].x, ds[2].x], labels=[train.labels, ds[1].labels, ds[2].labels])

        x = [train.x, ds[1].x, ds[2].x]
        labels = [train.labels, ds[1].labels, ds[2].labels]

    else:
        data = np.load(path)
        x = np.array([ex for ex in data['x']])
        labels = np.array([lab for lab in data['labels']])

    if num == 4:
        mask = range(1200)
    elif num == 3:
        mask = range(600) + range(900, 1200)
    elif num == 2:
        mask = range(600)
    x[0] = np.array(x[0])[:, mask]
    x[1] = np.array([],dtype=np.float32)
    labels[1] = np.array([],dtype=np.float32)
    x[2] = np.array(x[2])[:, mask]

    train = DataSet(x[0], labels[0])
    validation = DataSet(x[1], labels[1])
    test = DataSet(x[2], labels[2])

    print('Loaded reduced mnli with {} training ex'.format(size))
    return base.Datasets(train=train, validation=validation, test=test)

def load_mnli_nonfires(data_dir=None, mnli_num=4):
    num = mnli_num
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli_nonfires.npz')

    print('Loading nonfires from {}.'.format(path))
    data = np.load(path)
    x = np.array(data['x'])
    labels = np.array(data['labels'])

    if num == 4:
        mask = range(1200)
    elif num == 3:
        mask = range(600) + range(900, 1200)
    elif num == 2:
        mask = range(600)

    x = x[:, mask]

    nonfires = DataSet(x, labels)

    print('Loaded mnli nonfires.')
    return nonfires

def load_mnli_genres(data_dir=None):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli_genres.npz')

    print('Loading mnli genres from {}.'.format(path))
    genres = np.load(path)['genres']

    print('Loaded mnli genres.')
    return np.array(genres)

def load_reduced_mnli_genres(data_dir=None):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'reduced_mnli_genres.npz')

    print('Loading reduced mnli genres from {}'.format(path))

    indices = np.load(os.path.join(dataset_dir, 'reduced_mnli_indices.npz'))['indices']
    genres = load_mnli_genres(data_dir)
    genres[0] = genres[0][indices]
    np.savez(path, genres=genres)

    print('Loaded reduced mnli genres for {} train examples'.format(len(genres[0])))
    return genres

def load_mnli_nonfires_genres(data_dir=None):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli_nonfires_genres.npz')

    print('Loading mnli nonfires genres from {}.'.format(path))
    genres = np.load(path)['genres']

    print('Loaded mnli nonfires genres.')
    return np.array(genres)
