from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import datasets as ds
import datasets.loader
import datasets.mnist
from datasets.common import DataSet
from experiments.common import Experiment, collect_phases, phase
from experiments.benchmark import benchmark
from experiments.distribute import TaskQueue
from experiments.plot import *
from influence.logistic_regression import LogisticRegression

import os
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.linalg

from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as hcluster

@collect_phases
class SubsetInfluenceLogreg(Experiment):
    """
    Compute various types of influence on subsets of the dataset
    """
    def __init__(self, config, out_dir=None):
        super(SubsetInfluenceLogreg, self).__init__(config, out_dir)
        self.datasets = ds.loader.load_dataset(**self.config['dataset_config'])
        self.train = self.datasets.train
        self.test = self.datasets.test
        self.validation = self.datasets.validation

        model_dir = os.path.join(self.base_dir, 'models')
        model_config = LogisticRegression.default_config()
        model_config['arch'] = LogisticRegression.infer_arch(self.datasets.train)
        model_config['arch']['fit_intercept'] = True

        # Heuristic for determining maximum batch evaluation sizes without OOM
        D = model_config['arch']['input_dim'] * model_config['arch']['num_classes']
        model_config['grad_batch_size'] =  max(1, self.config['max_memory'] // D)
        model_config['hessian_batch_size'] = max(1, self.config['max_memory'] // (D * D))

        # Set the method for computing inverse HVP
        model_config['inverse_hvp_method'] = self.config['inverse_hvp_method']

        self.model_dir = model_dir
        self.model_config = model_config

        # Convenience member variables
        self.dataset_id = self.config['dataset_config']['dataset_id']
        self.num_train = self.datasets.train.num_examples
        self.num_classes = self.model_config['arch']['num_classes']
        self.num_subsets = self.config['num_subsets']
        if self.subset_choice_type == "types":
            self.subset_size = int(self.num_train * self.config['subset_rel_size'])
        elif self.subset_choice_type == "range":
            self.subset_min_size = int(self.num_train * self.config['subset_min_rel_size'])
            self.subset_max_size = int(self.num_train * self.config['subset_max_rel_size'])

        tasks_dir = os.path.join(self.base_dir, 'tasks')
        self.task_queue = TaskQueue(tasks_dir)
        self.task_queue.define_task('retrain_subsets', self.retrain_subsets)
        self.task_queue.define_task('self_pred_infl', self.self_pred_infl)
        self.task_queue.define_task('newton_batch', self.newton_batch)

    experiment_id = "ss_logreg"

    @property
    def subset_choice_type(self):
        return self.config.get('subset_choice_type', 'types')

    @property
    def run_id(self):
        if self.subset_choice_type == "types":
            run_id = "{}_ihvp-{}_seed-{}_size-{}_num-{}".format(
                self.config['dataset_config']['dataset_id'],
                self.config['inverse_hvp_method'],
                self.config['subset_seed'],
                self.config['subset_rel_size'],
                self.config['num_subsets'])
        elif self.subset_choice_type == "range":
            run_id = "{}_ihvp-{}_seed-{}_sizes-{}-{}_num-{}".format(
                self.config['dataset_config']['dataset_id'],
                self.config['inverse_hvp_method'],
                self.config['subset_seed'],
                self.config['subset_min_rel_size'],
                self.config['subset_max_rel_size'],
                self.config['num_subsets'])
        if self.config.get('tag', None) is not None:
            run_id = "{}_{}".format(run_id, self.config['tag'])
        return run_id

    def get_model(self):
        if not hasattr(self, 'model'):
            self.model = LogisticRegression(self.model_config, self.model_dir, random_state=np.random.RandomState(2))
        return self.model

    @phase(0)
    def cross_validation(self):
        model = self.get_model()
        res = dict()

        reg_min, reg_max, reg_samples = self.config['normalized_cross_validation_range']
        reg_min *= self.num_train
        reg_max *= self.num_train

        num_folds = self.config['cross_validation_folds']

        regs = np.logspace(np.log10(reg_min), np.log10(reg_max), reg_samples)
        cv_errors = np.zeros_like(regs)
        cv_accs = np.zeros_like(regs)
        fold_size = (self.num_train + num_folds - 1) // num_folds
        folds = [(k * fold_size, min((k + 1) * fold_size, self.num_train)) for k in range(num_folds)]

        for i, reg in enumerate(regs):
            with benchmark("Evaluating CV error for reg={}".format(reg)):
                cv_error = 0.0
                cv_acc = 0.0
                for k, fold in enumerate(folds):
                    fold_begin, fold_end = fold
                    train_indices = np.concatenate((np.arange(0, fold_begin), np.arange(fold_end, self.num_train)))
                    val_indices = np.arange(fold_begin, fold_end)

                    model.fit(self.train.subset(train_indices), l2_reg=reg)
                    fold_loss = model.get_total_loss(self.train.subset(val_indices), l2_reg=0)
                    acc = model.get_accuracy(self.train.subset(val_indices))
                    cv_error += fold_loss
                    cv_acc += acc
                    print('Acc: {}, loss: {}'.format(acc, fold_loss))

            cv_errors[i] = cv_error
            cv_accs[i] = cv_acc / num_folds
            print('Cross-validation acc {}, error {} for reg={}.'.format(cv_accs[i], cv_errors[i], reg))

        best_i = np.argmax(cv_accs)
        best_reg = regs[best_i]
        print('Cross-validation errors: {}'.format(cv_errors))
        print('Cross-validation accs: {}'.format(cv_accs))
        print('Selecting weight_decay {}, with acc {}, error {}.'.format(\
                best_reg, cv_accs[best_i], cv_errors[best_i]))

        res['cv_regs'] = regs
        res['cv_errors'] = cv_errors
        res['cv_accs'] = cv_accs
        res['cv_l2_reg'] = best_reg
        return res

    @phase(1)
    def initial_training(self):
        model = self.get_model()
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        with benchmark("Training original model"):
            model.fit(self.train, l2_reg=l2_reg)
            model.print_model_eval(self.datasets, l2_reg=l2_reg)
            model.save('initial')

        res['initial_train_losses'] = model.get_indiv_loss(self.train)
        res['initial_train_accuracy'] = model.get_accuracy(self.train)
        res['initial_test_losses'] = model.get_indiv_loss(self.test)
        res['initial_test_accuracy'] = model.get_accuracy(self.test)
        if self.num_classes == 2:
            res['initial_train_margins'] = model.get_indiv_margin(self.train)
            res['initial_test_margins'] = model.get_indiv_margin(self.test)

        with benchmark("Computing gradients"):
            res['train_grad_loss'] = model.get_indiv_grad_loss(self.train)

        return res

    @phase(2)
    def pick_test_points(self):
        dataset_id = self.config['dataset_config']['dataset_id']

        # Freeze each set after the first run
        if dataset_id == "hospital":
            fixed_test = [2267, 54826, 66678, 41567, 485, 25286]
        elif dataset_id == "spam":
            fixed_test = [92, 441, 593, 275, 267, 415]
        elif dataset_id == "mnist_small":
            fixed_test = [6172, 2044, 2293, 5305, 324, 3761]
        elif dataset_id == "mnist":
            fixed_test = [9009, 1790, 2293, 5844, 8977, 9433]
        elif dataset_id == "dogfish":
            fixed_test = [300, 339, 222, 520, 323, 182]
        elif dataset_id == "animals":
            fixed_test = [684,  850, 1492, 2380, 1539, 1267]
        elif dataset_id == "cifar10":
            fixed_test = [3629, 1019, 5259, 1082, 4237, 6811]
        else:
            test_losses = self.R['initial_test_losses']
            argsort = np.argsort(test_losses)
            high_loss = argsort[-3:] # Pick 3 high loss points
            random_loss = np.random.choice(argsort[:-3], 3, replace=False) # Pick 3 random points

            fixed_test = list(high_loss) + list(random_loss)

        print("Fixed test points: {}".format(fixed_test))
        return { 'fixed_test': fixed_test }

    @phase(3)
    def hessian(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        if self.config['inverse_hvp_method'] == 'explicit':
            with benchmark("Computing hessian"):
                res['hessian'] = hessian = model.get_hessian(self.train, l2_reg=l2_reg)
        elif self.config['inverse_hvp_method'] == 'cg':
            print("Not computing explicit hessian.")
            res['hessian'] = None

        return res

    @phase(4)
    def fixed_test_influence(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        hessian = self.R['hessian']
        inverse_hvp_args = {
            'hessian_reg': hessian,
            'dataset': self.train,
            'l2_reg': l2_reg,
            'verbose': False,
            'verbose_cg': True,
            'inverse_vp_method': self.config['inverse_vp_method'],
        }

        fixed_test = self.R['fixed_test']
        fixed_test_grad_loss = []
        fixed_test_pred_infl = []
        fixed_test_pred_margin_infl = []
        for test_idx in fixed_test:
            single_test_point = self.test.subset([test_idx])

            with benchmark('Scalar infl for all training points on test_idx {}.'.format(test_idx)):
                test_grad_loss = model.get_indiv_grad_loss(single_test_point).reshape(-1, 1)
                test_grad_loss_H_inv = model.get_inverse_hvp(test_grad_loss, **inverse_hvp_args).reshape(-1)
                pred_infl = np.dot(self.R['train_grad_loss'], test_grad_loss_H_inv)
                fixed_test_grad_loss.append(test_grad_loss)
                fixed_test_pred_infl.append(pred_infl)

            if self.num_classes == 2:
                with benchmark('Scalar margin infl for all training points on test_idx {}.'.format(test_idx)):
                    test_grad_margin = model.get_indiv_grad_margin(single_test_point).reshape(-1, 1)
                    test_grad_margin_H_inv = model.get_inverse_hvp(test_grad_margin, **inverse_hvp_args).reshape(-1)
                    pred_margin_infl = np.dot(self.R['train_grad_loss'], test_grad_margin_H_inv)
                    fixed_test_pred_margin_infl.append(pred_margin_infl)

        res['fixed_test_pred_infl'] = np.array(fixed_test_pred_infl)
        if self.num_classes == 2:
            res['fixed_test_pred_margin_infl'] = np.array(fixed_test_pred_margin_infl)

        return res

    def get_random_subsets(self, rng, subset_sizes):
        subsets = []
        for i, subset_size in enumerate(subset_sizes):
            subsets.append(rng.choice(self.num_train, subset_size, replace=False))
        return np.array(subsets)

    def get_same_class_subsets(self, rng, labels, subset_sizes):
        label_vals, label_counts = np.unique(labels, return_counts=True)
        label_indices = [ np.nonzero(labels == label_val)[0] for label_val in label_vals ]

        subsets = []
        for i, subset_size in enumerate(subset_sizes):
            valid_label_indices = np.nonzero(label_counts >= subset_size)[0]

            if len(valid_label_indices) == 0: continue
            valid_label_idx = rng.choice(valid_label_indices)
            subset = rng.choice(label_indices[valid_label_idx], subset_size, replace=False)
            subsets.append(subset)
        return np.array(subsets)

    def get_scalar_infl_tails(self, rng, pred_infl, subset_sizes):
        window = int(1.5 * np.max(subset_sizes))
        assert window <= self.num_train

        scalar_infl_indices = np.argsort(pred_infl).reshape(-1)
        pos_subsets, neg_subsets = [], []
        for i, subset_size in enumerate(subset_sizes):
            neg_subsets.append(rng.choice(scalar_infl_indices[:window], subset_size, replace=False))
            pos_subsets.append(rng.choice(scalar_infl_indices[-window:], subset_size, replace=False))
        return np.array(neg_subsets), np.array(pos_subsets)

    def get_clusters(self, X, n_clusters=None):
        """
        Clusters a set of points and returns the indices of the points
        within each cluster.
        :param X: An (N, D) tensor representing N points in D dimensions
        :param n_clusters: The number of clusters to use for KMeans, or None to use hierarchical
                           clustering and automatically determine the number of clusters.
        :returns: cluster_indices, a list of lists of indices
        """
        if n_clusters is None:
            cluster_labels = hcluster.fclusterdata(X, 1)
            print("Hierarchical clustering returned {} clusters".format(len(set(cluster_labels))))
        else:
            km = KMeans(n_clusters=n_clusters)
            km.fit(X)
            cluster_labels = km.labels_
        cluster_indices = [ np.nonzero(cluster_labels == label)[0] for label in set(cluster_labels) ]
        return cluster_indices

    def get_subsets_by_clustering(self, rng, X, subset_sizes):
        cluster_indices = []
        for n_clusters in (None, 4, 8, 16, 32, 64, 128):
            with benchmark("Clustering with k={}".format(n_clusters)):
                clusters = self.get_clusters(X, n_clusters=n_clusters)
                print("Cluster sizes:", [len(cluster) for cluster in clusters])
                cluster_indices.extend(clusters)

        cluster_sizes = np.array([len(indices) for indices in cluster_indices])

        subsets = []
        for i, subset_size in enumerate(subset_sizes):
            valid_clusters = np.nonzero(cluster_sizes >= subset_size)[0]
            if len(valid_clusters) == 0: continue

            cluster_idx = rng.choice(valid_clusters)
            subset = rng.choice(cluster_indices[cluster_idx], subset_size, replace=False)
            subsets.append(subset)
        return np.array(subsets)

    def get_subsets_by_projection(self, rng, X, subset_sizes):
        subsets = []
        for subset_size in subset_sizes:
            dim = rng.choice(X.shape[1])
            indices = np.argsort(np.array(list(X[:, dim])).reshape(-1))
            print(indices.shape)
            middle = rng.choice(X.shape[0])
            st = max(middle - subset_size // 2, 0)
            en = min(st + subset_size, self.num_train)
            st = en - subset_size
            subsets.append(indices[st:en])
        print(subsets)
        return subsets

    @phase(5)
    def pick_subsets(self):
        rng = np.random.RandomState(self.config['subset_seed'])

        tagged_subsets = []
        if self.subset_choice_type == "types":
            subset_sizes = np.ones(self.num_subsets).astype(np.int) * self.subset_size
        elif self.subset_choice_type == "range":
            subset_sizes = np.linspace(self.subset_min_size,
                                       self.subset_max_size,
                                       self.num_subsets).astype(np.int)

        with benchmark("Random subsets"):
            random_subsets = self.get_random_subsets(rng, subset_sizes)
            tagged_subsets += [('random', s) for s in random_subsets]

        with benchmark("Same class subsets"):
            same_class_subsets = self.get_same_class_subsets(rng, self.train.labels, subset_sizes)
            same_class_subset_labels = [self.train.labels[s[0]] for s in same_class_subsets]
            tagged_subsets += [('random_same_class-{}'.format(label), s) for s, label in zip(same_class_subsets, same_class_subset_labels)]

        with benchmark("Scalar infl tail subsets"):
            # 1) pick x*N out of the top 1.5 * 0.025 * N where x in (0.0025 - 0.025)
            # 2) pick x*N out of the top 1.5 * 0.1 * N where x in (0.0025 - 0.1)
            # 3) pick x*N out of the top 1.5 * 0.25 * N where x in (0.0025 - 0.25)
            size_1, size_2, size_3, size_4 = list(int(self.num_train * x) for x in (0.0025, 0.025, 0.1, 0.25))
            subsets_per_phase = self.num_subsets // 3
            subset_size_phases = [ np.linspace(size_1, size_2, subsets_per_phase).astype(int),
                                   np.linspace(size_1, size_3, subsets_per_phase).astype(int),
                                   np.linspace(size_1, size_4, self.num_subsets - 2 * subsets_per_phase).astype(int) ]
            for pred_infl, test_idx in zip(self.R['fixed_test_pred_infl'], self.R['fixed_test']):
                for phase, subset_sizes in enumerate(subset_size_phases, 1):
                    neg_tail_subsets, pos_tail_subsets = self.get_scalar_infl_tails(rng, pred_infl, subset_sizes)
                    tagged_subsets += [('neg_tail_test-{}-{}'.format(phase, test_idx), s) for s in neg_tail_subsets]
                    tagged_subsets += [('pos_tail_test-{}-{}'.format(phase, test_idx), s) for s in pos_tail_subsets]
                print('Found scalar infl tail subsets for test idx {}.'.format(test_idx))

        with benchmark("Same features subsets"):
            same_features_subsets = self.get_subsets_by_clustering(rng, self.train.x, subset_sizes)
            tagged_subsets += [('same_features', s) for s in same_features_subsets]

        with benchmark("Same gradient subsets"):
            same_grad_subsets = self.get_subsets_by_clustering(rng, self.R['train_grad_loss'], subset_sizes)
            tagged_subsets += [('same_grad', s) for s in same_grad_subsets]

        with benchmark("Same feature subsets by windowing"):
            feature_window_subsets = self.get_subsets_by_projection(rng, self.train.x, subset_sizes)
            tagged_subsets += [('feature_window', s) for s in feature_window_subsets]

        subset_tags = [tag for tag, subset in tagged_subsets]
        subset_indices = [subset for tag, subset in tagged_subsets]

        return { 'subset_tags': subset_tags, 'subset_indices': subset_indices }

    def retrain_subsets(self, subset_start, subset_end):
        print("Retraining subsets {}-{}".format(subset_start, subset_end))
        # Workers might need to reload results
        self.load_phases([0, 5], verbose=False)

        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']

        start_time = time.time()
        train_losses, test_losses = [], []
        train_margins, test_margins = [], []
        for i, remove_indices in enumerate(subset_indices[subset_start:subset_end], subset_start):
            print('Retraining model for subset {} out of {} (tag={})'.format(i, len(subset_indices), subset_tags[i]))

            s = np.ones(self.num_train)
            s[remove_indices] = 0

            model.warm_fit(self.train, s, l2_reg=l2_reg)
            model.save('subset_{}'.format(i))
            train_losses.append(model.get_indiv_loss(self.train, verbose=False))
            test_losses.append(model.get_indiv_loss(self.test, verbose=False))
            if model.num_classes == 2:
                train_margins.append(model.get_indiv_margin(self.train, verbose=False))
                test_margins.append(model.get_indiv_margin(self.test, verbose=False))

            cur_time = time.time()
            time_per_retrain = (cur_time - start_time) / ((i + 1) - subset_start)
            remaining_time = time_per_retrain * (len(subset_indices) - (i + 1))
            print('Each retraining takes {} s, {} s remaining'.format(time_per_retrain, remaining_time))

        res['subset_train_losses'] = np.array(train_losses)
        res['subset_test_losses'] = np.array(test_losses)

        if self.num_classes == 2:
            res['subset_train_margins'] = np.array(train_margins)
            res['subset_test_margins'] = np.array(test_margins)

        return res

    @phase(6)
    def retrain(self):
        num_subsets = len(self.R['subset_indices'])
        subsets_per_batch = 32
        results = self.task_queue.execute('retrain_subsets', [
            (i, min(i + subsets_per_batch, num_subsets))
            for i in range(0, num_subsets, subsets_per_batch)])

        return self.task_queue.collate_results(results)

    def self_pred_infl(self, subset_start, subset_end):
        self.load_phases([0, 1, 3, 5], verbose=False)

        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']

        hessian = self.R['hessian']
        inverse_hvp_args = {
            'hessian_reg': hessian,
            'dataset': self.train,
            'l2_reg': l2_reg,
            'verbose': False,
            'verbose_cg': True,
            'inverse_vp_method': self.config['inverse_vp_method'],
        }
        train_grad_loss = self.R['train_grad_loss']

        # It is important that the influence gets calculated before the model is retrained,
        # so that the parameters are the original parameters
        start_time = time.time()
        subset_pred_dparam = []
        self_pred_infls = []
        self_pred_margin_infls = []
        for i, remove_indices in enumerate(subset_indices[subset_start:subset_end], subset_start):
            print('Computing self-influences for subset {} out of {} (tag={})'.format(i, len(subset_indices), subset_tags[i]))

            grad_loss = np.sum(train_grad_loss[remove_indices, :], axis=0)
            H_inv_grad_loss = model.get_inverse_hvp(grad_loss.reshape(-1, 1), **inverse_hvp_args).reshape(-1)
            pred_infl = np.dot(grad_loss, H_inv_grad_loss)
            subset_pred_dparam.append(H_inv_grad_loss)
            self_pred_infls.append(pred_infl)

            if model.num_classes == 2:
                grad_margin = model.get_total_grad_margin(self.train.subset(remove_indices))
                pred_margin_infl = np.dot(grad_margin, H_inv_grad_loss)
                self_pred_margin_infls.append(pred_margin_infl)

            cur_time = time.time()
            time_per_retrain = (cur_time - start_time) / ((i + 1) - subset_start)
            remaining_time = time_per_retrain * (len(subset_indices) - (i + 1))
            print('Each self-influence calculation takes {} s, {} s remaining'.format(time_per_retrain, remaining_time))

        res['subset_pred_dparam'] = np.array(subset_pred_dparam)
        res['subset_self_pred_infl'] = np.array(self_pred_infls)
        if self.num_classes == 2:
            res['subset_self_pred_margin_infl'] = np.array(self_pred_margin_infls)

        return res

    @phase(7)
    def compute_self_pred_infl(self):
        num_subsets = len(self.R['subset_indices'])
        subsets_per_batch = 32
        results = self.task_queue.execute('self_pred_infl', [
            (i, min(i + subsets_per_batch, num_subsets))
            for i in range(0, num_subsets, subsets_per_batch)])

        return self.task_queue.collate_results(results)

    @phase(8)
    def compute_actl_infl(self):
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']

        # Helper to collate fixed test infl and subset self infl on a quantity q
        def compute_collate_infl(fixed_test, fixed_test_pred_infl_q,
                                 initial_train_q, initial_test_q,
                                 subset_train_q, subset_test_q):
            subset_fixed_test_actl_infl = subset_test_q[:, fixed_test] - initial_test_q[fixed_test]
            subset_fixed_test_pred_infl = np.array([
                np.sum(fixed_test_pred_infl_q[:, remove_indices], axis=1).reshape(-1)
                for remove_indices in subset_indices])
            subset_self_actl_infl = np.array([
                np.sum(subset_train_q[i][remove_indices]) - np.sum(initial_train_q[remove_indices])
                for i, remove_indices in enumerate(subset_indices)])
            return subset_fixed_test_actl_infl, subset_fixed_test_pred_infl, subset_self_actl_infl

        # Compute influences on loss
        res['subset_fixed_test_actl_infl'], \
        res['subset_fixed_test_pred_infl'], \
        res['subset_self_actl_infl'] = compute_collate_infl(
            *[self.R[key] for key in ["fixed_test", "fixed_test_pred_infl",
                                      "initial_train_losses", "initial_test_losses",
                                      "subset_train_losses", "subset_test_losses"]])

        if self.num_classes == 2:
            # Compute influences on margin
            res['subset_fixed_test_actl_margin_infl'], \
            res['subset_fixed_test_pred_margin_infl'], \
            res['subset_self_actl_margin_infl'] = compute_collate_infl(
                *[self.R[key] for key in ["fixed_test", "fixed_test_pred_margin_infl",
                                          "initial_train_margins", "initial_test_margins",
                                          "subset_train_margins", "subset_test_margins"]])

        return res

    def newton_batch(self, subset_start, subset_end):
        self.load_phases([0, 1, 2, 3, 5], verbose=False)

        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        # The Newton approximation is obtained by evaluating
        # -g(|w|, theta_0)^T H(s+w, theta_0)^{-1} g(w, theta_0)
        # where w is the difference in weights. Since we already have the full
        # hessian H_reg(s), we can compute H(w) (with no regularization) and
        # use it to update H_reg(s+w) = H_reg(s) + H(w) instead.

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']

        hessian = self.R['hessian']
        train_grad_loss = self.R['train_grad_loss']

        test_grad_loss = model.get_indiv_grad_loss(self.test.subset(self.R['fixed_test']))
        if self.num_classes == 2:
            test_grad_margin = model.get_indiv_grad_margin(self.test.subset(self.R['fixed_test']))

        # It is important that the gradients get calculated on the original model
        # so that the parameters are the original parameters
        start_time = time.time()
        subset_newton_dparam = []
        self_newton_infls = []
        self_newton_margin_infls = []
        fixed_test_newton_infls = []
        fixed_test_newton_margin_infls = []
        subset_hessian_spectrum = []
        for i, remove_indices in enumerate(subset_indices[subset_start:subset_end], subset_start):
            print('Computing Newton influences for subset {} out of {} (tag={})'.format(i, len(subset_indices), subset_tags[i]))

            grad_loss = np.sum(train_grad_loss[remove_indices, :], axis=0).reshape(-1, 1)
            if self.config['inverse_hvp_method'] == 'explicit':
                hessian_w = model.get_hessian(self.train.subset(remove_indices),
                                              -np.ones(len(remove_indices)), l2_reg=0, verbose=False)
                H_inv_grad_loss = model.get_inverse_vp(hessian + hessian_w, grad_loss,
                                                       inverse_vp_method=self.config['inverse_vp_method']).reshape(-1)

                if not self.config['skip_hessian_spectrum']:
                    H_inv_H_w = model.get_inverse_vp(hessian, hessian_w,
                                                     inverse_vp_method=self.config['inverse_vp_method'])
                    hessian_spectrum = scipy.linalg.eigvals(H_inv_H_w)
                    subset_hessian_spectrum.append(hessian_spectrum)
            elif self.config['inverse_hvp_method'] == 'cg':
                sample_weights = np.ones(self.num_train)
                sample_weights[remove_indices] = 0
                inverse_hvp_args = {
                    'dataset': self.train,
                    'sample_weights': sample_weights,
                    'l2_reg': l2_reg,
                    'verbose': False,
                    'verbose_cg': True,
                }
                H_inv_grad_loss = model.get_inverse_hvp(grad_loss, **inverse_hvp_args).reshape(-1)

            self_newton_infl = np.dot(grad_loss.reshape(-1), H_inv_grad_loss)
            subset_newton_dparam.append(H_inv_grad_loss)
            self_newton_infls.append(self_newton_infl)

            fixed_test_newton_infl = np.dot(test_grad_loss, H_inv_grad_loss)
            fixed_test_newton_infls.append(fixed_test_newton_infl)

            if model.num_classes == 2:
                s = np.zeros(self.num_train)
                s[remove_indices] = 1
                grad_margin = model.get_total_grad_margin(self.train, s)
                self_newton_margin_infl = np.dot(grad_margin, H_inv_grad_loss)
                self_newton_margin_infls.append(self_newton_margin_infl)
                fixed_test_newton_margin_infl = np.dot(test_grad_margin, H_inv_grad_loss)
                fixed_test_newton_margin_infls.append(fixed_test_newton_margin_infl)

            cur_time = time.time()
            time_per_retrain = (cur_time - start_time) / ((i + 1) - subset_start)
            remaining_time = time_per_retrain * (len(subset_indices) - (i + 1))
            print('Each Newton influence calculation takes {} s, {} s remaining'.format(time_per_retrain, remaining_time))

        res['subset_newton_dparam'] = np.array(subset_newton_dparam)
        res['subset_self_newton_infl'] = np.array(self_newton_infls)
        res['subset_fixed_test_newton_infl'] = np.array(fixed_test_newton_infls)
        if self.num_classes == 2:
            res['subset_self_newton_margin_infl'] = np.array(self_newton_margin_infls)
            res['subset_fixed_test_newton_margin_infl'] = np.array(fixed_test_newton_margin_infls)

        if self.config['inverse_hvp_method'] == 'explicit' and not self.config['skip_hessian_spectrum']:
            res['subset_hessian_spectrum'] = np.array(subset_hessian_spectrum)

        return res

    @phase(9)
    def newton(self):
        if self.config['skip_newton']:
            return dict()

        num_subsets = len(self.R['subset_indices'])
        subsets_per_batch = 32
        results = self.task_queue.execute('newton_batch', [
            (i, min(i + subsets_per_batch, num_subsets))
            for i in range(0, num_subsets, subsets_per_batch)])

        return self.task_queue.collate_results(results)

    @phase(10)
    def fixed_test_newton(self):
        # Merged into the newton phase above to avoid duplicating work
        return dict()

    @phase(11)
    def param_changes(self):
        # The later phases do not need task workers
        self.task_queue.notify_exit()

        model = self.get_model()
        res = dict()

        model.load('initial')
        initial_param = model.get_params_flat()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        n, n_report = len(subset_indices), max(len(subset_indices) // 100, 1)

        # Calculate actual changes in parameters
        subset_dparam = []
        subset_train_acc, subset_test_acc = [], []
        for i, remove_indices in enumerate(subset_indices):
            model.load('subset_{}'.format(i))
            param = model.get_params_flat()
            subset_dparam.append(param - initial_param)
            subset_train_acc.append(model.get_accuracy(self.train))
            subset_test_acc.append(model.get_accuracy(self.test))
        res['subset_dparam'] = np.array(subset_dparam)
        res['subset_train_accuracy'] = np.array(subset_train_acc)
        res['subset_test_accuracy'] = np.array(subset_test_acc)

        return res

    @phase(12)
    def param_change_norms(self):
        if self.config['skip_param_change_norms']:
            return dict()

        res = dict()
        model = self.get_model()
        l2_reg = self.R['cv_l2_reg']
        model.load('initial')

        # Compute l2 norm of gradient
        train_grad_loss = self.R['train_grad_loss']
        res['subset_grad_loss_l2_norm'] = np.array([
            np.linalg.norm(np.sum(train_grad_loss[remove_indices, :], axis=0))
            for remove_indices in self.R['subset_indices']])

        # Compute l2 norms and norms under the Hessian metric of parameter changes
        l2_reg = self.R['cv_l2_reg']
        hessian = self.R['hessian']
        for dparam_type in ('subset_dparam', 'subset_pred_dparam', 'subset_newton_dparam'):
            dparam = self.R[dparam_type]
            res[dparam_type + '_l2_norm'] = np.linalg.norm(dparam, axis=1)
            if self.config['inverse_hvp_method'] == 'explicit':
                hvp = np.dot(dparam, hessian)
            else:
                hvp = model.get_hvp(dparam.T, self.train, l2_reg=l2_reg)
            res[dparam_type + '_hessian_norm'] = np.sqrt(np.sum(dparam * hvp, axis=1))

        return res

    @phase(13)
    def z_norms(self):
        if self.config['skip_z_norms'] or self.num_classes != 2:
            return dict()

        res = dict()
        model = self.get_model()
        l2_reg = self.R['cv_l2_reg']
        model.load('initial')

        inverse_hvp_args = {
            'hessian_reg': self.R['hessian'],
            'dataset': self.train,
            'l2_reg': l2_reg,
            'verbose': False,
            'verbose_cg': True,
            'inverse_vp_method': self.config['inverse_vp_method'],
        }

        # z_i = sqrt(sigma''_i) x_i so that H = ZZ^T
        res['zs'] = zs = model.get_zs(self.train)
        ihvp_zs = model.get_inverse_hvp(zs.T, **inverse_hvp_args).T
        res['z_norms'] = np.linalg.norm(zs, axis=1)
        res['z_hessian_norms'] = np.sqrt(np.sum(zs * ihvp_zs, axis=1))

        return res

    @phase(14)
    def compute_pparam_infl(self):
        model = self.get_model()
        res = dict()

        model.load('initial')
        initial_param = model.get_params_flat()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']

        fixed_test = self.R['fixed_test']
        fixed_test_ds = self.test.subset(fixed_test)

        initial_fixed_test_loss = self.R['initial_test_losses'][fixed_test]
        if self.num_classes == 2:
            initial_fixed_test_margin = self.R['initial_test_margins'][fixed_test]

        def compute_pparam_influences(pred_dparam, pparam_type='pparam'):
            # Calculate change in loss/margin at predicted parameters
            subset_fixed_test_pparam_infl = []
            subset_self_pparam_infl = []
            subset_fixed_test_pparam_margin_infl = []
            for i, remove_indices in enumerate(subset_indices):

                subset_ds = self.train.subset(remove_indices)
                pred_param = initial_param + pred_dparam[i, :]
                model.set_params_flat(pred_param)

                pparam_fixed_test_loss = model.get_indiv_loss(fixed_test_ds, verbose=False)
                pparam_self_loss = model.get_total_loss(subset_ds, l2_reg=0, verbose=False)
                initial_self_loss = np.sum(self.R['initial_train_losses'][remove_indices])
                subset_fixed_test_pparam_infl.append(pparam_fixed_test_loss - initial_fixed_test_loss)
                subset_self_pparam_infl.append(pparam_self_loss - initial_self_loss)

                if self.num_classes == 2:
                    pparam_fixed_test_margin = model.get_indiv_margin(fixed_test_ds, verbose=False)
                    subset_fixed_test_pparam_margin_infl.append(pparam_fixed_test_margin - initial_fixed_test_margin)

            res['subset_fixed_test_{}_infl'.format(pparam_type)] = np.array(subset_fixed_test_pparam_infl)
            res['subset_self_{}_infl'.format(pparam_type)] = np.array(subset_self_pparam_infl)
            if self.num_classes == 2:
                res['subset_fixed_test_{}_margin_infl'.format(pparam_type)] = np.array(subset_fixed_test_pparam_margin_infl)

        compute_pparam_influences(self.R['subset_pred_dparam'], 'pparam')
        if not self.config['skip_newton']:
            compute_pparam_influences(self.R['subset_newton_dparam'], 'nparam')

        return res

    def plot_z_norms(self, save_and_close=False):
        if 'z_norms' not in self.R: return

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
        plot_distribution(ax[0][0], self.R['z_norms'],
                          title='Z-norms', xlabel='Z-norm',
                          subtitle=self.get_subtitle())
        if save_and_close:
            fig.savefig(os.path.join(self.plot_dir, 'z-norms.png'), bbox_inches='tight')
            plt.close(fig)

    def get_simple_subset_tags(self):
        def simplify_tag(tag):
            if '-' in tag: return tag.split('-')[0]
            return tag
        return map(simplify_tag, self.R['subset_tags'])

    def get_subtitle(self):
        if self.subset_choice_type == "types":
            subtitle='{}, {} subsets per type, proportion {}'.format(
                self.dataset_id, self.num_subsets, self.config['subset_rel_size'])
        elif self.subset_choice_type == "range":
            subtitle='{}, {} subsets per type, proportion {}-{}'.format(
                self.dataset_id, self.num_subsets,
                self.config['subset_min_rel_size'],
                self.config['subset_max_rel_size'])
        return subtitle

    def plot_group_influence(self,
                             influence_type, # 'self' or 'fixed-test-{:test_idx}'
                             quantity, # 'loss' or 'margin'
                             x, y,
                             x_approx_type, y_approx_type, # 'actl', 'pred', 'newton', 'pparam', 'nparam'
                             save_and_close=False):
        subset_tags = self.get_simple_subset_tags()
        subset_sizes = np.array([len(indices) for indices in self.R['subset_indices']])

        if influence_type.find('self') == 0:
            title = 'Group self-influence on '
        elif influence_type.find('fixed-test-') == 0:
            test_idx = influence_type.rsplit('-',  1)[-1]
            title = "Group influence on test pt {}'s ".format(test_idx)
        title += quantity

        filename = '{}_{}_{}-{}.png'.format(
            influence_type, quantity, x_approx_type, y_approx_type)

        approx_type_to_label = { 'actl': 'Actual influence',
                                 'pred': 'First-order influence',
                                 'newton': 'Newton influence',
                                 'pparam': 'Predicted parameter influence',
                                 'nparam': 'Newton predicted parameter influence' }
        xlabel = approx_type_to_label[x_approx_type]
        ylabel = approx_type_to_label[y_approx_type]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
        plot_influence_correlation(ax[0][0], x, y,
                                   label=subset_tags,
                                   title=title,
                                   subtitle=self.get_subtitle(),
                                   xlabel=xlabel,
                                   ylabel=ylabel)
        if save_and_close:
            fig.savefig(os.path.join(self.plot_dir, filename), bbox_inches='tight')
            plt.close(fig)

        range_x, range_y = np.max(x) - np.min(x), np.max(y) - np.min(y)
        imbalanced = (np.abs(range_x) < 1e-9 or np.abs(range_y) < 1e-9 or
                      range_x / range_y < 1e-1 or range_y / range_x < 1e-1)
        if imbalanced:
            filename = '{}_{}_{}-{}_imbalanced.png'.format(
                influence_type, quantity, x_approx_type, y_approx_type)
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
            plot_influence_correlation(ax[0][0], x, y,
                                       label=subset_tags,
                                       title=title,
                                       subtitle=self.get_subtitle(),
                                       xlabel=xlabel,
                                       ylabel=ylabel,
                                       equal=False)
            if save_and_close:
                fig.savefig(os.path.join(self.plot_dir, filename), bbox_inches='tight')
                plt.close(fig)

    def plot_self_influence(self, save_and_close=False):
        if 'subset_self_actl_infl' not in self.R: return
        if 'subset_self_pred_infl' not in self.R: return

        self.plot_group_influence('self', 'loss',
                                  self.R['subset_self_actl_infl'],
                                  self.R['subset_self_pred_infl'],
                                  'actl', 'pred',
                                  save_and_close=save_and_close)

        if self.num_classes == 2:
            self.plot_group_influence('self', 'margin',
                                      self.R['subset_self_actl_margin_infl'],
                                      self.R['subset_self_pred_margin_infl'],
                                      'actl', 'pred',
                                      save_and_close=save_and_close)

    def plot_fixed_test_influence(self, save_and_close=False):
        if 'subset_fixed_test_actl_infl' not in self.R: return
        if 'subset_fixed_test_pred_infl' not in self.R: return

        subset_tags = self.get_simple_subset_tags()

        for i, test_idx in enumerate(self.R['fixed_test']):
            self.plot_group_influence('fixed-test-{}'.format(test_idx), 'loss',
                                      self.R['subset_fixed_test_actl_infl'][:, i],
                                      self.R['subset_fixed_test_pred_infl'][:, i],
                                      'actl', 'pred',
                                      save_and_close=save_and_close)

            if self.num_classes == 2:
                self.plot_group_influence('fixed-test-{}'.format(test_idx), 'margin',
                                          self.R['subset_fixed_test_actl_margin_infl'][:, i],
                                          self.R['subset_fixed_test_pred_margin_infl'][:, i],
                                          'actl', 'pred',
                                          save_and_close=save_and_close)

    def plot_newton_influence(self, save_and_close=False):
        if 'subset_self_newton_infl' not in self.R: return
        if 'subset_fixed_test_newton_infl' not in self.R: return

        def compare_newton(influence_type, quantity, actl, pred, newton):
            self.plot_group_influence(influence_type, quantity, actl, newton, 'actl', 'newton',
                                      save_and_close=save_and_close)
            self.plot_group_influence(influence_type, quantity, pred, newton, 'pred', 'newton',
                                      save_and_close=save_and_close)

        compare_newton('self', 'loss',
                       self.R['subset_self_actl_infl'],
                       self.R['subset_self_pred_infl'],
                       self.R['subset_self_newton_infl'])

        for i, test_idx in enumerate(self.R['fixed_test']):
            compare_newton('fixed-test-{}'.format(test_idx), 'loss',
                           self.R['subset_fixed_test_actl_infl'][:, i],
                           self.R['subset_fixed_test_pred_infl'][:, i],
                           self.R['subset_fixed_test_newton_infl'][:, i])

            if self.num_classes == 2:
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import datasets as ds
import datasets.loader
import datasets.mnist
from datasets.common import DataSet
from experiments.common import Experiment, collect_phases, phase
from experiments.benchmark import benchmark
from experiments.distribute import TaskQueue
from experiments.plot import *
from influence.logistic_regression import LogisticRegression

import os
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.linalg

from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as hcluster

@collect_phases
class SubsetInfluenceLogreg(Experiment):
    """
    Compute various types of influence on subsets of the dataset
    """
    def __init__(self, config, out_dir=None):
        super(SubsetInfluenceLogreg, self).__init__(config, out_dir)
        self.datasets = ds.loader.load_dataset(**self.config['dataset_config'])
        self.train = self.datasets.train
        self.test = self.datasets.test
        self.validation = self.datasets.validation

        model_dir = os.path.join(self.base_dir, 'models')
        model_config = LogisticRegression.default_config()
        model_config['arch'] = LogisticRegression.infer_arch(self.datasets.train)
        model_config['arch']['fit_intercept'] = True

        # Heuristic for determining maximum batch evaluation sizes without OOM
        D = model_config['arch']['input_dim'] * model_config['arch']['num_classes']
        model_config['grad_batch_size'] =  max(1, self.config['max_memory'] // D)
        model_config['hessian_batch_size'] = max(1, self.config['max_memory'] // (D * D))

        # Set the method for computing inverse HVP
        model_config['inverse_hvp_method'] = self.config['inverse_hvp_method']

        self.model_dir = model_dir
        self.model_config = model_config

        # Convenience member variables
        self.dataset_id = self.config['dataset_config']['dataset_id']
        self.num_train = self.datasets.train.num_examples
        self.num_classes = self.model_config['arch']['num_classes']
        self.num_subsets = self.config['num_subsets']
        if self.subset_choice_type == "types":
            self.subset_size = int(self.num_train * self.config['subset_rel_size'])
        elif self.subset_choice_type == "range":
            self.subset_min_size = int(self.num_train * self.config['subset_min_rel_size'])
            self.subset_max_size = int(self.num_train * self.config['subset_max_rel_size'])

        tasks_dir = os.path.join(self.base_dir, 'tasks')
        self.task_queue = TaskQueue(tasks_dir)
        self.task_queue.define_task('retrain_subsets', self.retrain_subsets)
        self.task_queue.define_task('self_pred_infl', self.self_pred_infl)
        self.task_queue.define_task('newton_batch', self.newton_batch)

    experiment_id = "ss_logreg"

    @property
    def subset_choice_type(self):
        return self.config.get('subset_choice_type', 'types')

    @property
    def run_id(self):
        if self.subset_choice_type == "types":
            run_id = "{}_ihvp-{}_seed-{}_size-{}_num-{}".format(
                self.config['dataset_config']['dataset_id'],
                self.config['inverse_hvp_method'],
                self.config['subset_seed'],
                self.config['subset_rel_size'],
                self.config['num_subsets'])
        elif self.subset_choice_type == "range":
            run_id = "{}_ihvp-{}_seed-{}_sizes-{}-{}_num-{}".format(
                self.config['dataset_config']['dataset_id'],
                self.config['inverse_hvp_method'],
                self.config['subset_seed'],
                self.config['subset_min_rel_size'],
                self.config['subset_max_rel_size'],
                self.config['num_subsets'])
        if self.config.get('tag', None) is not None:
            run_id = "{}_{}".format(run_id, self.config['tag'])
        return run_id

    def get_model(self):
        if not hasattr(self, 'model'):
            self.model = LogisticRegression(self.model_config, self.model_dir, random_state=np.random.RandomState(2))
        return self.model

    @phase(0)
    def cross_validation(self):
        model = self.get_model()
        res = dict()

        reg_min, reg_max, reg_samples = self.config['normalized_cross_validation_range']
        reg_min *= self.num_train
        reg_max *= self.num_train
