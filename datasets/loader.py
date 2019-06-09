from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import datasets as ds
import datasets.mnist
import datasets.spam
import datasets.hospital
import datasets.cifar10
import datasets.common
import datasets.babble
import datasets.mnli
import datasets.processed_animals

DATASETS = {
    'mnist': ds.mnist.load_mnist,
    'mnist_small': ds.mnist.load_small_mnist,
    'spam': ds.spam.load_spam,
    'hospital': ds.hospital.load_hospital,
    'cifar10': ds.cifar10.load_cifar10,
    'cifar10_small': ds.cifar10.load_small_cifar10,
    'cdr': ds.babble.load_cdr,
    'reduced_cdr': ds.babble.load_reduced_cdr,
    'mnli': ds.mnli.load_mnli,
    'dogfish': ds.processed_animals.load_dogfish,
    'animals': ds.processed_animals.load_animals,
}

SUPPLEMENTS = {
    'cdr_LFs': ds.babble.load_cdr_LFs,
    'reduced_cdr_LFs': ds.babble.load_cdr_LFs,
    'cdr_nonfires': ds.babble.load_cdr_nonfires,
    'reduced_cdr_nonfires': ds.babble.load_reduced_cdr_nonfires,
    'cdr_weights': ds.babble.load_cdr_weights,
    'reduced_cdr_weights': ds.babble.load_cdr_weights,
    'cdr_labeling_info': ds.babble.load_cdr_labeling_info,
    'reduced_cdr_labeling_info': ds.babble.load_cdr_labeling_info,

    'mnli_nonfires': ds.mnli.load_mnli_nonfires,
    'mnli_nonfires_genres': ds.mnli.load_mnli_nonfires_genres,
    'mnli_genres': ds.mnli.load_mnli_genres,
    'mnli_ids': ds.mnli.load_mnli_ids,
    'mnli_nonfires_ids': ds.mnli.load_mnli_nonfires_ids,
}

def load_supplemental_info(description, data_dir=None, **kwargs):
    if description not in SUPPLEMENTS:
        raise ValueError('Unknown supplementary info description {}'.format(description))

    return SUPPLEMENTS[description](data_dir=data_dir, **kwargs)

def load_dataset(dataset_id,
                 center_data=False,
                 append_bias=False,
                 data_dir=None, **kwargs):
    if dataset_id not in DATASETS:
        raise ValueError('Unknown dataset_id {}'.format(dataset_id))

    datasets = DATASETS[dataset_id](data_dir=data_dir, **kwargs)

    if center_data:
        datasets = ds.common.center_data(datasets)

    if append_bias:
        datasets = ds.common.append_bias(datasets)

    return datasets
