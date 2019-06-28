from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse

import matplotlib as mpl
mpl.use('Agg')

from experiments.counterexamples import Counterexamples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute subset self and test influences')
    # Environment args
    parser.add_argument('--out-dir', default=None, type=str,
                        help="The experiment output directory")
    parser.add_argument('--max-memory', default=1e9, type=int,
                        help="A rule-of-thumb estimate of the GPU memory capacity")

    # Execution args
    parser.add_argument('--force-refresh', dest='force_refresh', action='store_true',
                        help="Ignore previously saved results")
    parser.set_defaults(force_refresh=False)
    parser.add_argument('--no-intermediate-results', dest='no_intermediate_results', action='store_true',
                        help="Do not save any intermediate results, and run only a single master without parallelization")
    parser.set_defaults(no_intermediate_results=False)
    parser.add_argument('--invalidate', default=None, type=int,
                        help="Invalidate phases starting from this phase index")
    parser.add_argument('--worker', dest='worker', action='store_true',
                        help="Behave only as a worker")
    parser.set_defaults(worker=False)

    # Experiment args
    parser.add_argument('--dataset-id', default="hospital", type=str,
                        help="The dataset to use")
    parser.add_argument('--subset-seed', default=0, type=int,
                        help="The seed to use for subset selection")
    args = parser.parse_args()

    config = {
        'dataset_id': args.dataset_id,
        'seed': args.subset_seed,
        'max_memory': int(args.max_memory),
        'master_only': args.no_intermediate_results,
    }
    
    exp = Counterexamples(config, out_dir=args.out_dir)
    if args.worker:
        exp.task_queue.run_worker()
    else:
        if args.no_intermediate_results:
            exp.run(force_refresh=args.force_refresh,
                    invalidate_phase=args.invalidate,
                    save_phase_results=False)

            summary_keys = [
                'cex_tags',
                'cex_lstsq_uv_K-120_subsets',
                'cex_subset_test_pred_margin_infl',
                'cex_subset_test_newton_pred_margin_infl',
            ]
            exp.save_summary(summary_keys)
        else:
            exp.run(force_refresh=args.force_refresh,
                    invalidate_phase=args.invalidate)

