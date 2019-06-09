from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import datasets.mnist
import datasets.spam
import datasets.hospital
import datasets.cifar10
import datasets.loader

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute subset self and test influences')

    # Dataset args
    parser.add_argument('--data-dir', default=None, type=str,
                        help="The base dataset directory")
    parser.add_argument('--center-data', dest='center_data', action='store_true',
                        help="Center the dataset")
    parser.add_argument('--append-bias', dest='append_bias', action='store_true',
                        help="Append a bias component to the dataset")
    parser.set_defaults(center_data=False, append_bias=False)

    args = parser.parse_args()

    dataset_ids = ['mnist', 'mnist_small', 'spam', 'hospital', 'cifar10', 'cifar10_small', 'dogfish', 'animals']
    for dataset_id in dataset_ids:
        print("Loading {}".format(dataset_id))
        dataset = datasets.loader.load_dataset(dataset_id=dataset_id,
                                               center_data=args.center_data,
                                               append_bias=args.append_bias,
                                               data_dir=args.data_dir)

        print("{}: train={}, val={}, test={}".format(
            dataset_id,
            None if dataset.train is None else "{}".format(dataset.train.x.shape),
            None if dataset.validation is None else "{}".format(dataset.validation.x.shape),
            None if dataset.test is None else "{}".format(dataset.test.x.shape)))
