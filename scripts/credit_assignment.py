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
    parser.add_argument('--sample-weights', dest='sample_weights', action='store_true',
                        help="Weight examples according to number of contributing sources")
    parser.add_argument('--balance-nonfires', dest='balance_nonfires', action='store_true',
                        help="Force class balance for nonfires")
    parser.add_argument('--balance-test', dest='balance_test', action='store_true',
                        help="Force class balance for test set")
    parser.add_argument('--inverse-hvp-method', default=None, type=str,
                        help="The default inverse HVP method to use")
    parser.add_argument('--inverse-vp-method', default=None, type=str,
                        help="The default inverse VP method to use")
    parser.add_argument('--tf-training', dest='tf_training', action='store_true',
                        help="Whether we use TF to do training instead of sklearn")
    parser.add_argument('--grad-batch-size', default=400, type=int,
                        help="Batch size when computing gradients, None for heuristic")
    parser.add_argument('--hessian-batch-size', default=30, type=int,
                        help="Batch size when computing hessian, None for heuristic")
    parser.add_argument('--reduced-size', default=None, type=int,
                        help="Size of reduced mnli")
    parser.add_argument('--reduced-proportion', default=None, type=float,
                        help="Proportion of turkers in reduced mnli")
    parser.add_argument('--mnli-num', default=4, type=int,
                        help="Version of mnli to use")
    parser.set_defaults(sample_weights=False, balance_nonfires=False, balance_test=False, tf_training=False)

    # SGD config
    parser.add_argument('--batch-size', default=4000, type=int,
                        help="The size of each mini-batch")
    parser.add_argument('--max-batch-size', default=400000, type=int,
                        help="Max size when trying to do full batch gradient descent")
    parser.add_argument('--initial-learning-rate', default=0.001, type=float,
                        help="Initial SGD learning rate")
    parser.add_argument('--num-epochs', default=650, type=int,
                        help="Number of passes over dataset")
    parser.add_argument('--max-iter', default=1e15, type=int,
                        help="Maximum total number of steps during SGD")
    parser.add_argument('--save-epochs', default=100, type=int,
                        help="Every x epochs save the model")
    parser.add_argument('--display-epochs', default=10, type=int,
                        help="Display info every x epochs (negative for never)")
    parser.add_argument('--display-steps', default=-1, type=int,
                        help="Display info every x steps (negative for never)")
    parser.add_argument('--more-epochs', default=500, type=int,
                        help="Number of epochs during warm retraining")
    parser.add_argument('--full-batch-start-epoch', default=380, type=int,
                        help="Epoch num to start doing full/max batches to improve convergence")
    parser.add_argument('--switch-to-nonadam-epoch', default=500, type=int,
                        help="Epoch num to switch from AdamOptimizer to GradientDescentOptimizer")

    args = parser.parse_args()

    dataset_config = {
        'dataset_id': args.dataset_id,
        'center_data': False,
        'append_bias': False,
        'data_dir': args.data_dir,
        'mnli_num': args.mnli_num,
    }
    if args.dataset_id == 'reduced_mnli':
        dataset_config['size'] = args.reduced_size
        dataset_config['proportion'] = args.reduced_proportion

    if args.tf_training:
        sgd_config = {
                'batch_size': args.batch_size,
                'max_batch_size': args.max_batch_size,
                'initial_learning_rate': args.initial_learning_rate,
                'num_epochs': args.num_epochs,
                'max_iter': args.max_iter,
                'decay_epochs': {
                    'mnli': [10, 30, 60, 120, 180, 270, 500],
                    'reduced_mnli': [2000, 4000, 6000, 8000, 12000, 16000], #[60, 180, 400, 800, 1200, 1500],
                }[args.dataset_id],
                'save_epochs': args.save_epochs,
                'display_epochs': args.display_epochs,
                'display_steps': args.display_steps,
                'more_epochs': args.more_epochs,
                'full_batch_start_epoch': args.full_batch_start_epoch,
                'switch_to_nonadam_epoch': args.switch_to_nonadam_epoch,
        }
    else:
        sgd_config = {}
    config = {
        'dataset_config': dataset_config,
        'subset_seed': args.subset_seed,
        'subset_rel_size': args.subset_rel_size,
        'num_subsets': args.num_subsets,
        'cross_validation_folds': 5,
        'normalized_cross_validation_range': {
            'mnli': (1e-4, 1e-1, 10),
            'reduced_mnli': (1e-4, 1, 10),
        }[args.dataset_id],
        'inverse_hvp_method': {
            'mnli': 'explicit',
            'reduced_mnli': 'explicit',
        }[args.dataset_id],
        'inverse_vp_method': 'cholesky_lu',
        'max_memory': int(args.max_memory),
        'sample_weights': args.sample_weights,
        'balance_nonfires': args.balance_nonfires,
        'balance_test': args.balance_test,
        'tf_training': args.tf_training,
        'sgd_config': sgd_config,
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
