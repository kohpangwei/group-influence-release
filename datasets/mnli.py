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
        all_ids = np.load(path, allow_pickle=True)['ids']
    return np.array(all_ids)

def load_mnli_nonfires_ids(data_dir=None, force_refresh=False):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli_nonfires_annotators.npz')

    if not os.path.exists(path) or force_refresh:
        ids = []
        with open(os.path.join(dataset_dir, 'multinli_1.0_dev_mismatched_with_annotators.jsonl')) as f:
            for line in f:
                ids.append(json.loads(line)['sentence2_author'])
        ids = np.array(ids)
        np.savez(path, ids=ids)
    else:
        ids = np.load(path, allow_pickle=True)['ids']
    return np.array(ids)

def load_mnli(data_dir=None, non_tf=False):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli.npz')

    print('Loading mnli from {}.'.format(path))
    data = np.load(path, allow_pickle=True)
    labels = [data['lab0'],[],data['lab2']]
    x = [data['x0'],[],data['x2']]

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

def load_mnli_nonfires(data_dir=None):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli_nonfires.npz')

    print('Loading nonfires from {}.'.format(path))
    data = np.load(path, allow_pickle=True)
    x = np.array(data['x'])
    labels = np.array(data['labels'])

    mask = range(600)
    x = x[:, mask]

    nonfires = DataSet(x, labels)

    print('Loaded mnli nonfires.')
    return nonfires

def load_mnli_genres(data_dir=None):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli_genres.npz')

    print('Loading mnli genres from {}.'.format(path))
    genres = np.load(path, allow_pickle=True)['genres']

    print('Loaded mnli genres.')
    return np.array(genres)

def load_mnli_nonfires_genres(data_dir=None):
    dataset_dir = get_dataset_dir('multinli_1.0', data_dir=data_dir)
    path = os.path.join(dataset_dir, 'mnli_nonfires_genres.npz')

    print('Loading mnli nonfires genres from {}.'.format(path))
    genres = np.load(path, allow_pickle=True)['genres']

    print('Loaded mnli nonfires genres.')
    return np.array(genres)
