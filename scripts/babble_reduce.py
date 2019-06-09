from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse

from experiments.babble_reduce import BabbleReduce

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute subset self and test influences')
    # Environment args
    parser.add_argument('--data-dir', default=None, type=str,
                        help="The base dataset directory")
    parser.add_argument('--out-dir', default=None, type=str,
                        help="The experiment output directory")
    parser.add_argument('--max-memory', default=1e9, type=int,
                        help="A rule-of-thumb estimate of the GPU memory capacity")

    # Execution args
    parser.add_argument('--force-refresh', dest='force_refresh', action='store_true',
                        help="Ignore previously saved results")
    parser.add_argument('--invalidate', default=None, type=int,
                        help="Invalidate phases starting from this phase index")
    parser.set_defaults(force_refresh=False)

    # Experiment args
    parser.add_argument('--dataset-id', default="spouse", type=str,
                        help="The dataset to use")
    parser.add_argument('--target-num-features', default=4000, type=int,
                        help="Target number of features to reduce to")
    parser.add_argument('--fit-intercept', dest='fit_intercept', action='store_true',
                        help="Whether the logistic regression should fit intercept")
    parser.add_argument('--regularization-type', default="l1", type=str,
                        help="Using l1 or l2 regularization to sparsify.")
    parser.add_argument('--sample-weights', dest='sample_weights', action='store_true',
                        help="Whether to weight examples according to # of LF contributions")
    parser.set_defaults(fit_intercept=True, sample_weights=True)
    args = parser.parse_args()

    dataset_config = {
        'dataset_id': args.dataset_id,
        'center_data': False,
        'append_bias': False,
        'data_dir': args.data_dir,
    }
    config = {
        'dataset_config': dataset_config,
        'normalized_cross_validation_range': {
            'spouse': (1e-7, 1, 10),
            'cdr': (1e-4, 1e-1, 10),
        }[args.dataset_id],
        'initial_reg_range': {
            'spouse': (1e-13, 1e3, 10),
            'cdr': (1e-13,1e3,17),
        }[args.dataset_id],
        'max_memory': int(args.max_memory),
        'target_num_features': args.target_num_features,
        'fit_intercept': args.fit_intercept,
        'regularization_type': args.regularization_type,
        'sample_weights': args.sample_weights,
    }

    exp = BabbleReduce(config, out_dir=args.out_dir)
    exp.run(force_refresh=args.force_refresh,
            invalidate_phase=args.invalidate)
