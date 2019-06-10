from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import urllib
import sqlite3
import pandas as pd
import cPickle as pickle

from tqdm import tqdm
from scipy.sparse import dok_matrix, csr_matrix, save_npz, load_npz, coo_matrix
from experiments.benchmark import benchmark

from datasets.common import get_dataset_dir, maybe_download, DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

SPOUSE_SOURCE_URL="https://worksheets.codalab.org/rest/bundles/0x32a8394d80064f028859c9e57310a7f7/contents/blob/"
CDR_SOURCE_URL="https://worksheets.codalab.org/rest/bundles/0x38e021ee339a4f6aa21d9c4a16c1b7ee/contents/blob/"

def load_spouse_LFs(validation_size=None, data_dir=None, force_refresh=False):
    return load_LFs('spouse', SPOUSE_SOURCE_URL, validation_size, data_dir, force_refresh)

def load_cdr_LFs(validation_size=None, data_dir=None, force_refresh=False):
    return load_LFs('cdr', CDR_SOURCE_URL, validation_size, data_dir, force_refresh)

def load_LFs(ds_name, source_url, validation_size=None, data_dir=None, force_refresh=False):
    dataset_dir = get_dataset_dir('babble', data_dir=data_dir)
    path = os.path.join(dataset_dir, '{}_LF_ids.npz'.format(ds_name))

    if not os.path.exists(path) or force_refresh:
        
        raw_path = maybe_download(source_url, '{}.db'.format(ds_name), dataset_dir)

        print('Extracting {}.'.format(raw_path))
        conn = sqlite3.connect(raw_path)
        # ex_id = candidate_id
        LF_labels = pd.read_sql_query("select * from label;", conn) # label, ex_id, LF_id
        features = pd.read_sql_query("select * from feature;", conn) # feature value, ex_id, feature_id
        # all values are probably 1.0
        splits = pd.read_sql_query("select id, split from candidate;", conn) #ex_id, train/dev/test split (0-2)
        conn.close()

        split_ids = [splits['id'][splits['split'] == i] for i in range(3)]  # ex_ids in each split
        ids_dups = np.array(LF_labels['candidate_id'])                      # ex_id for each LF_ex
        which_split = [np.isin(ids_dups, split_ids[i]) for i in range(3)]   # which LF_ex in each split
        LF_ids = [np.array(LF_labels['key_id'][which_split[i]]) for i in range(3)]    # LF for each LF_ex

        np.savez(path,
                 LF_ids=LF_ids)
        print('Saved {} LF_ids to {}.'.format(ds_name, path))
    else:
        print('Loading {} LF_ids from {}.'.format(ds_name, path))
        data = np.load(path)
        LF_ids = data['LF_ids']

    print('Loaded {} LF_ids.'.format(ds_name))
    return LF_ids

def load_spouse_gold(validation_size=None, data_dir=None, force_refresh=False):
    return load_gold('spouse', SPOUSE_SOURCE_URL, validation_size, data_dir, force_refresh)

def load_cdr_gold(validation_size=None, data_dir=None, force_refresh=False):
    return load_gold('cdr', CDR_SOURCE_URL, validation_size, data_dir, force_refresh)

def load_gold(ds_name, source_url, validation_size=None, data_dir=None, force_refresh=False):
    raise NotImplementedError('NOT DONE DONT USE')

def load_spouse(validation_size=None, data_dir=None, force_refresh=False):
    return load('spouse', SPOUSE_SOURCE_URL, validation_size, data_dir, force_refresh)

def load_cdr(validation_size=None, data_dir=None, force_refresh=False):
    return load('cdr', CDR_SOURCE_URL, validation_size, data_dir, force_refresh)

def load(ds_name, source_url, validation_size=None, data_dir=None, force_refresh=False):
    dataset_dir = get_dataset_dir('babble', data_dir=data_dir)
    path = os.path.join(dataset_dir, '{}'.format(ds_name))

    if not os.path.exists(path+'.npz') or force_refresh:
        
        raw_path = maybe_download(source_url, '{}.db'.format(ds_name), dataset_dir)

        print('Extracting {}.'.format(raw_path))
        conn = sqlite3.connect(raw_path)
        # ex_id = candidate_id
        LF_labels = pd.read_sql_query("select * from label;", conn) # label, ex_id, LF_id
        features = pd.read_sql_query("select * from feature;", conn) # feature value, ex_id, feature_id
        # all values are probably 1.0
        splits = pd.read_sql_query("select id, split from candidate;", conn) #ex_id, train/dev/test split (0-2)
        start_test_ind = np.min(splits['id'][splits['split'] == 2]) # not the 1-indexing
        test_gold_labels = pd.read_sql_query("select value, candidate_id from gold_label where candidate_id>{} order by candidate_id asc;".format(start_test_ind-1), conn) # gold, ex_id
        conn.close()

        split_ids = [splits['id'][splits['split'] == i] for i in range(3)]  # ex_ids in each split
        ids_dups = np.array(LF_labels['candidate_id'])                      # ex_id for each LF_ex
        which_split = [np.isin(ids_dups, split_ids[i]) for i in range(3)]   # which LF_ex in each split
        labels = [np.array(LF_labels['value'][which_split[i]]) for i in range(3)]     # label for each LF_ex
        print('Extracted labels.')

        _num_features = max(features['key_id'])
        _num_ex = splits.shape[0]
        _num_LF_ex = [labels[i].shape[0] for i in range(3)]

        print('Creating map from examples to sparse features.')
        last_seen = features['candidate_id'][0]
        count = 0
        ex_id_to_ind = {last_seen: count}                       # in case the ex_id aren't consecutive
        ind_to_features = [dict() for i in range(_num_ex+1)]
        for val, ex_id, key_id in tqdm(np.array(features)):
            if ex_id > last_seen:
                count += 1
                last_seen = ex_id
                ex_id_to_ind[last_seen] = count
            ind_to_features[count][key_id] = val

        print('Creating sparse feature matrices for LF examples.')
        x = [dok_matrix((_num_LF_ex[i], _num_features)) for i in range(3)]
        counts = [0 for i in range(3)]
        for _, ex_id, _ in tqdm(np.array(LF_labels)):
            split = splits['split'][ex_id_to_ind[ex_id]]
            for key_id, val in ind_to_features[ex_id_to_ind[ex_id]].iteritems():
                x[split][counts[split], key_id-1] = val
            counts[split] += 1
        print('Extracted feature matrices.')

        print('Reverting test things to gold.')
        _num_test = sum(splits['split'] == 2)
        x[2] = dok_matrix((_num_test, _num_features))
        labels[2] = np.array(test_gold_labels['value'])
        count = 0
        for ex_id, split in tqdm(np.array(splits)):
            if split == 2:
                for key_id, val in ind_to_features[ex_id_to_ind[ex_id]].iteritems():
                    x[2][count, key_id-1] = val
                count += 1
        
        labels = [(labels[i]+1)/2 for i in range(3)] # convert (-1,1) to (0,1)

        for i in range(3):
            save_npz(path+'x{}.npz'.format(i),coo_matrix(x[i]))
        np.savez(path+'labels.npz', labels=labels)
    else:
        print('Loading {}.'.format(ds_name))
        data = np.load(path+'labels.npz')
        labels = data['labels']
        x = []
        for i in range(3):
            x.append(load_npz(path+'x{}.npz'.format(i)))

    train = DataSet(x[0], labels[0])
    validation = DataSet(x[1], labels[1])
    test = DataSet(x[2], labels[2])

    print('Loaded {}.'.format(ds_name))
    return base.Datasets(train=train, validation=validation, test=test)

def load_reduced_spouse(validation=None, data_dir=None, force_refresh=False):
    return load_reduced('spouse', validation, data_dir, force_refresh)

def load_reduced_cdr(validation=None, data_dir=None, force_refresh=False):
    return load_reduced('cdr', validation, data_dir, force_refresh)

def load_reduced(ds_name, validation=None, data_dir=None, force_refresh=False):
    # The correct way to do this would be to call the ds_reduce experiment
    # and take its output here. For now, we assume that's done.
    dataset_dir = get_dataset_dir('babble', data_dir=data_dir)
    red_path = os.path.join(dataset_dir, 'reduced_{}.npz'.format(ds_name))

    if not os.path.exists(red_path) or force_refresh:
        print('Building reduced {}.'.format(ds_name))
        x_path = os.path.join(dataset_dir, 'reduced_{}_x.npz'.format(ds_name))
        assert os.path.exists(x_path)

        x = np.load(x_path, allow_pickle=True)['reduced_x']
        x = [np.array(mat.todense()) for mat in x]
        data = np.load(os.path.join(dataset_dir, 'cdrlabels.npz'))
        labels = data['labels']
        np.savez(red_path,
                labels=labels,
                x=x)
    else:
        print('Retrieving reduced {}.'.format(ds_name))
        data = np.load(red_path, allow_pickle=True)
        x = data['x']
        labels = data['labels']

    train = DataSet(x[0], labels[0])
    validation = DataSet(x[1], labels[1])
    test = DataSet(x[2], labels[2])

    print('Loaded reduced {}.'.format(ds_name))
    return base.Datasets(train=train, validation=validation, test=test)

def load_spouse_weights(data_dir=None, force_refresh=False):
    return load_weights('spouse', SPOUSE_SOURCE_URL, data_dir=data_dir, force_refresh=force_refresh)

def load_cdr_weights(data_dir=None, force_refresh=False):
    return load_weights('cdr', CDR_SOURCE_URL, data_dir=data_dir, force_refresh=force_refresh)

def load_weights(ds_name, source_url, data_dir=None, force_refresh=False):
    dataset_dir = get_dataset_dir('babble', data_dir=data_dir)
    path = os.path.join(dataset_dir, '{}_weights.npz'.format(ds_name))

    if not os.path.exists(path) or force_refresh:
        
        raw_path = maybe_download(source_url, '{}.db'.format(ds_name), dataset_dir)

        print('Extracting {}.'.format(raw_path))
        conn = sqlite3.connect(raw_path)
        # ex_id = candidate_id
        LF_labels = pd.read_sql_query("select * from label;", conn) # label, ex_id, LF_id
        splits = pd.read_sql_query("select id, split from candidate;", conn) #ex_id, train/dev/test split (0-2)
        conn.close()

        split_ids = [splits['id'][splits['split'] == i] for i in range(3)]  # ex_ids in each split
        ids_dups = np.array(LF_labels['candidate_id'])                      # ex_id for each LF_ex
        which_split = [np.isin(ids_dups, split_ids[i]) for i in range(3)]   # which LF_ex in each split
        labels = [LF_labels['value'][which_split[i]] for i in range(3)]     # label for each LF_ex
        print('Extracted labels.')

        _num_ex = [split_ids[i].shape[0] for i in range(3)]
        _num_LF_ex = [labels[i].shape[0] for i in range(3)]

        print('Tracking occurrences of each candidate_id.')
        num_of_ex_ids = np.unique(LF_labels['candidate_id'], return_counts=True)[1]

        weights = [np.zeros(_num_LF_ex[0]), np.zeros(_num_LF_ex[1]), np.ones(_num_ex[2])]
        i = 0
        count = 0
        for num in num_of_ex_ids:                       # assumes all ex_id are adjacent, train then valid
            if count == _num_LF_ex[i]:
                i += 1
                if i == 2: break
                count = 0
            weights[i][count:count+num] = 1.0/num*np.ones(num)
            count += num

        print('Saved weights to {}'.format(path))
        np.savez(path,
                weights=weights)
    else:
        print('Loading {} weights from {}.'.format(ds_name, path))
        data = np.load(path)
        weights = data['weights']

    print('Loaded {} weights.'.format(ds_name))
    return weights

def load_spouse_nonfires(data_dir=None, force_refresh=False):
    return load_nonfires('spouse', SPOUSE_SOURCE_URL, data_dir=data_dir, force_refresh=force_refresh)

def load_cdr_nonfires(data_dir=None, force_refresh=False):
    return load_nonfires('cdr', CDR_SOURCE_URL, data_dir=data_dir, force_refresh=force_refresh)

def load_nonfires(ds_name, source_url, data_dir=None, force_refresh=False):
    dataset_dir = get_dataset_dir('babble', data_dir=data_dir)
    path = os.path.join(dataset_dir, '{}_nonfires'.format(ds_name))

    if not os.path.exists(path+'.npz') or force_refresh:
        
        raw_path = maybe_download(source_url, '{}.db'.format(ds_name), dataset_dir)

        print('Extracting {}.'.format(raw_path))
        conn = sqlite3.connect(raw_path)
        # ex_id = candidate_id
        LF_labels = pd.read_sql_query("select * from label;", conn) # label, ex_id, LF_id
        splits = pd.read_sql_query("select id, split from candidate;", conn) #ex_id, train/dev/test split (0-2)
        features = pd.read_sql_query("select * from feature;", conn) # feature value, ex_id, feature_id
        gold_labels = pd.read_sql_query("select value, candidate_id from gold_label;", conn) # gold, ex_id

        # all values are probably 1.0

        conn.close()

        split_ids = [splits['id'][splits['split'] == i] for i in range(3)]  # ex_ids in each split
        ids_dups = np.array(LF_labels['candidate_id'])                      # ex_id for each LF_ex
        which_split = [np.isin(ids_dups, split_ids[i]) for i in range(3)]   # which LF_ex in each split
        print('Extracted labels.')

        # Look through all LF labels and record seen candidate ids
        # Look through all candidates and find the non-seen ones
        # Collect the features of those

        _num_features = max(features['key_id'])

        print('Finding nonfires.')
        LF_labeled = set()
        for _, ex_id, _ in tqdm(np.array(LF_labels)):
            LF_labeled.add(ex_id)

        print('Collecting labels.')
        labels = []
        ex_id_to_ind = {}
        count = 0
        for lab, ex_id in tqdm(np.array(gold_labels)):
            if ex_id not in LF_labeled:
                labels.append((lab+1)/2)
                ex_id_to_ind[ex_id] = count
                count += 1
        labels = np.array(labels)
        _num_ex = count

        print('Collecting features.')
        x = dok_matrix((_num_ex, _num_features))
        for val, ex_id, f_id in tqdm(np.array(features)):
            if ex_id not in LF_labeled:
                x[ex_id_to_ind[ex_id], f_id-1] = val
        x = x.astype(np.float32)

        save_npz(path+'x.npz', coo_matrix(x))
        np.savez(path+'labels.npz', labels=[labels]) 
    else:
        print('Loading {} nonfires.'.format(ds_name))
        data = np.load(path+'labels.npz')
        labels = data['labels'][0]
        x = load_npz(path+'x.npz')

    nonfires = DataSet(x, labels)

    print('Loaded {} nonfires.'.format(ds_name))
    return nonfires

def load_reduced_spouse_nonfires(data_dir=None, force_refresh=False):
    return load_reduced_nonfires('spouse', SPOUSE_SOURCE_URL, data_dir=data_dir, force_refresh=force_refresh)

def load_reduced_cdr_nonfires(data_dir=None, force_refresh=False):
    return load_reduced_nonfires('cdr', CDR_SOURCE_URL, data_dir=data_dir, force_refresh=force_refresh)

def load_reduced_nonfires(ds_name, source_url, data_dir=None, force_refresh=False):
    # The correct way to do this would be to call the ds_reduce experiment
    # and take its output here. For now, we assume that's done.
    dataset_dir = get_dataset_dir('babble', data_dir=data_dir)
    red_path = os.path.join(dataset_dir, 'reduced_{}_nonfires.npz'.format(ds_name))

    if not os.path.exists(red_path) or force_refresh:
        print('Building reduced {} nonfires.'.format(ds_name))
        x_path = os.path.join(dataset_dir, 'reduced_{}_nonfires_x.npz'.format(ds_name))
        assert os.path.exists(x_path)

        x = np.load(x_path)['reduced_x'][0].todense()
        data = np.load(os.path.join(dataset_dir,'cdr_nonfireslabels.npz'))
        labels = data['labels'][0]

        np.savez(red_path,
                labels=labels,
                x=np.array(x))
    else:
        print('Retrieving reduced {} nonfires.'.format(ds_name))
        data = np.load(red_path)
        x = data['x']
        labels = data['labels']

    red_nonfires = DataSet(x, labels)

    print('Loaded reduced {} nonfires.'.format(ds_name))
    return red_nonfires

def load_spouse_labeling_info(data_dir=None, force_refresh=False):
    return load_labeling_info('spouse', SPOUSE_SOURCE_URL, data_dir=data_dir, force_refresh=force_refresh)

def load_cdr_labeling_info(data_dir=None, force_refresh=False):
    return load_labeling_info('cdr', CDR_SOURCE_URL, data_dir=data_dir, force_refresh=force_refresh)

def load_labeling_info(ds_name, source_url, data_dir=None, force_refresh=False):
    dataset_dir = get_dataset_dir('babble', data_dir=data_dir)
    LF_path = os.path.join(dataset_dir, '{}_LF_labels.pkl'.format(ds_name))
    gold_path = os.path.join(dataset_dir, '{}_gold_labels.pkl'.format(ds_name))

    if not os.path.exists(LF_path) or not os.path.exists(gold_path) or force_refresh:
        
        raw_path = maybe_download(source_url, '{}.db'.format(ds_name), dataset_dir)

        print('Extracting {} labeling info.'.format(ds_name))
        conn = sqlite3.connect(raw_path)
        # ex_id = candidate_id
        LF_labels = pd.read_sql_query("select * from label;", conn) # label, ex_id, LF_id
        gold_labels = pd.read_sql_query("select value, candidate_id from gold_label order by candidate_id asc;", conn) # gold, ex_id
        conn.close()
        
        LF_labels.to_pickle(LF_path)
        gold_labels.to_pickle(gold_path)
    else:
        print('Retrieving {} labeling info.'.format(ds_name))
        LF_labels = pd.read_pickle(LF_path)
        gold_labels = pd.read_pickle(gold_path)

    LF_labels = LF_labels.values
    gold_labels = gold_labels.values

    return LF_labels, gold_labels
        
