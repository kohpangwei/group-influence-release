from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os
import numpy as np
import tarfile

from datasets.nlprocessor import NLProcessor
from datasets.common import get_dataset_dir, maybe_download, DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

from scipy.io import savemat

def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        with open(os.path.join(folder, a_file), 'r') as f:
            a_list.append(f.read().decode("latin-1"))
    return a_list

def process_spam(dataset_dir, truncate=None):
    rng = np.random.RandomState(0)
    nlprocessor = NLProcessor(rng)

    spam = init_lists(os.path.join(dataset_dir, 'enron1', 'spam'))
    ham = init_lists(os.path.join(dataset_dir, 'enron1', 'ham'))

    docs, Y = nlprocessor.process_spam(spam[:truncate], ham[:truncate])
    num_examples = len(Y)

    train_fraction = 0.8
    valid_fraction = 0.0
    num_train_examples = int(train_fraction * num_examples)
    num_valid_examples = int(valid_fraction * num_examples)
    num_test_examples = num_examples - num_train_examples - num_valid_examples

    docs_train = docs[:num_train_examples]
    Y_train = Y[:num_train_examples]

    docs_valid = docs[num_train_examples : num_train_examples+num_valid_examples]
    Y_valid = Y[num_train_examples : num_train_examples+num_valid_examples]

    docs_test = docs[-num_test_examples:]
    Y_test = Y[-num_test_examples:]

    assert(len(docs_train) == len(Y_train))
    assert(len(docs_valid) == len(Y_valid))
    assert(len(docs_test) == len(Y_test))
    assert(len(Y_train) + len(Y_valid) + len(Y_test) == num_examples)
    
    nlprocessor.learn_vocab(docs_train)
    X_train = nlprocessor.get_bag_of_words(docs_train)
    X_valid = nlprocessor.get_bag_of_words(docs_valid)
    X_test = nlprocessor.get_bag_of_words(docs_test)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def load_spam(truncate=None, data_dir=None):
    dataset_dir = get_dataset_dir('spam', data_dir=data_dir)
    spam_path = os.path.join(dataset_dir, 'spam_truncate-{}.npz'.format(truncate))

    if not os.path.exists(spam_path):
        SPAM_URL = "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz"
        raw_spam_path = maybe_download(SPAM_URL, 'enron1.tar.gz', dataset_dir)

        print("Extracting {}".format(raw_spam_path))
        with tarfile.open(raw_spam_path, 'r:gz') as tarf:
            tarf.extractall(path=dataset_dir)

        print("Processing spam")
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = process_spam(dataset_dir, truncate)

        # Convert them to dense matrices
        X_train = X_train.toarray()
        X_valid = X_valid.toarray()
        X_test = X_test.toarray()

        np.savez(spam_path,
                 X_train=X_train,
                 Y_train=Y_train,
                 X_test=X_test,
                 Y_test=Y_test,
                 X_valid=X_valid,
                 Y_valid=Y_valid)
    else:
        data = np.load(spam_path)
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
        X_valid = data['X_valid']
        Y_valid = data['Y_valid']

    train = DataSet(X_train, Y_train)
    validation = DataSet(X_valid, Y_valid)
    test = DataSet(X_test, Y_test)

    return base.Datasets(train=train, validation=validation, test=test)
