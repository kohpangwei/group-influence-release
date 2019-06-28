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
    parser.add_argument('--no-intermediate-results', dest='no_intermediate_results', action='store_true',
                        help="Do not save any intermediate results, and run only a single master without parallelization")
    parser.set_defaults(force_refresh=False, worker=False, no_intermediate_results=False)

    # Experiment args
    parser.add_argument('--dataset-id', default="mnli", type=str,
                        help="The dataset to use")
    parser.add_argument('--inverse-hvp-method', default=None, type=str,
                        help="The default inverse HVP method to use")
    parser.add_argument('--inverse-vp-method', default=None, type=str,
                        help="The default inverse VP method to use")
    parser.add_argument('--grad-batch-size', default=400, type=int,
                        help="Batch size when computing gradients, None for heuristic")
    parser.add_argument('--hessian-batch-size', default=30, type=int,
                        help="Batch size when computing hessian, None for heuristic")

    args = parser.parse_args()

    dataset_config = {
        'dataset_id': args.dataset_id,
        'center_data': False,
        'append_bias': False,
        'data_dir': args.data_dir,
    }

    config = {
        'dataset_config': dataset_config,
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
        'grad_batch_size': args.grad_batch_size,
        'hessian_batch_size': args.hessian_batch_size,
        'master_only': args.no_intermediate_results,
     }

    if args.inverse_hvp_method is not None:
        config['inverse_hvp_method'] = args.inverse_hvp_method

    if args.inverse_vp_method is not None:
        config['inverse_vp_method'] = args.inverse_vp_method

    exp = CreditAssignment(config, out_dir=args.out_dir)
    if args.worker:
        exp.task_queue.run_worker()
    else:
        if args.no_intermediate_results:
            exp.run(force_refresh=args.force_refresh,
                    invalidate_phase=args.invalidate,
                    save_phase_results=False)

            summary_keys = [
                'cv_l2_reg',
                'initial_test_losses',
                'subset_tags',
                'test_genres',
                'nonfires_genres',
                'fiction_test_actl_infl',
                'fiction_test_pred_infl',
                'government_test_actl_infl',
                'government_test_pred_infl',
                'slate_test_actl_infl',
                'slate_test_pred_infl',
                'telephone_test_actl_infl',
                'telephone_test_pred_infl',
                'travel_test_pred_infl',
                'travel_test_actl_infl',
                'facetoface_nonfires_actl_infl',
                'facetoface_nonfires_pred_infl',
                'letters_nonfires_actl_infl',
                'letters_nonfires_pred_infl',
                'nineeleven_nonfires_actl_infl',
                'nineeleven_nonfires_pred_infl',
                'oup_nonfires_actl_infl',
                'oup_nonfires_pred_infl',
                'verbatim_nonfires_actl_infl',
                'verbatim_nonfires_pred_infl',
                'subset_indices',
                'all_test_pred_infl',
            ]
            exp.save_summary(summary_keys)
        else:
            exp.run(force_refresh=args.force_refresh,
                    invalidate_phase=args.invalidate)
