from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse

from experiments.credit_assignment import CreditAssignment

# Reserving 10GB should be enough memory
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
    parser.add_argument('--worker', dest='worker', action='store_true',
                        help="Behave only as a worker")
    parser.set_defaults(force_refresh=False, worker=False)

    # Experiment args
    parser.add_argument('--dataset-id', default="mnli", type=str,
                        help="The dataset to use")
    parser.add_argument('--subset-seed', default=0, type=int,
                        help="The seed to use for subset selection")
    parser.add_argument('--subset-rel-size', default=0.1, type=float,
                        help="The size of the subset relative to the dataset")
    parser.add_argument('--num-subsets', default=5, type=int,
                        help="The number of subsets per random choice type")
    parser.add_argument('--balance-nonfires', dest='balance_nonfires', action='store_true',
                        help="Force class balance for nonfires")
    parser.add_argument('--balance-test', dest='balance_test', action='store_true',
                        help="Force class balance for test set")
    parser.add_argument('--inverse-hvp-method', default=None, type=str,
                        help="The default inverse HVP method to use")
    parser.add_argument('--inverse-vp-method', default=None, type=str,
                        help="The default inverse VP method to use")
    parser.add_argument('--grad-batch-size', default=400, type=int,
                        help="Batch size when computing gradients, None for heuristic")
    parser.add_argument('--hessian-batch-size', default=30, type=int,
                        help="Batch size when computing hessian, None for heuristic")
    parser.set_defaults(balance_nonfires=False, balance_test=False)

    args = parser.parse_args()

    dataset_config = {
        'dataset_id': args.dataset_id,
        'center_data': False,
        'append_bias': False,
        'data_dir': args.data_dir,
    }

    config = {
        'dataset_config': dataset_config,
        'subset_seed': args.subset_seed,
        'subset_rel_size': args.subset_rel_size,
        'num_subsets': args.num_subsets,
        'cross_validation_folds': 5,
        'normalized_cross_validation_range': {
            'mnli': (1e-4, 1e-1, 10),
         }[args.dataset_id],
        'inverse_hvp_method': {
            'mnli': 'explicit',
        }[args.dataset_id],
        'inverse_vp_method': 'cholesky_lu',
        'max_memory': int(args.max_memory),
        'sample_weights': None,
        'balance_nonfires': args.balance_nonfires,
        'balance_test': args.balance_test,
        'grad_batch_size': args.grad_batch_size,
        'hessian_batch_size': args.hessian_batch_size,
     }

    if args.inverse_hvp_method is not None:
        config['inverse_hvp_method'] = args.inverse_hvp_method

    if args.inverse_vp_method is not None:
        config['inverse_vp_method'] = args.inverse_vp_method

    exp = CreditAssignment(config, out_dir=args.out_dir)
    if args.worker:
        exp.task_queue.run_worker()
    else:
        exp.run(force_refresh=args.force_refresh,
                invalidate_phase=args.invalidate)
