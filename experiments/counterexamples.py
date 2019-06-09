from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import datasets as ds
import datasets.loader
import datasets.mnist
from datasets.common import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

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
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg
import itertools

@collect_phases
class Counterexamples(Experiment):
    """
    Synthesize toy datasets and find counterexamples to possible
    properties of influence approximations
    """
    def __init__(self, config, out_dir=None):
        super(Counterexamples, self).__init__(config, out_dir)
        self.dataset_id = config['dataset_id']
        tasks_dir = os.path.join(self.base_dir, 'tasks')
        self.task_queue = TaskQueue(tasks_dir)
        self.task_queue.define_task('retrain_and_newton_batch', self.retrain_and_newton_batch)
        self.task_queue.define_task('compute_cex_test_infl_batch', self.compute_cex_test_infl_batch)

    experiment_id = "counterexamples"

    @property
    def run_id(self):
        return "{}".format(self.dataset_id)

    @phase(0)
    def generate_datasets(self):
        res = dict()

        rng = np.random.RandomState(self.config['seed'])

        # Separated Gaussian mixture
        def generate_gaussian_mixture(N_per_class, D, axis):
            X_pos = rng.normal(0, 1, size=(N_per_class, D)) + axis / 2
            X_neg = rng.normal(0, 1, size=(N_per_class, D)) - axis / 2
            X = np.vstack([X_pos, X_neg])
            Y = np.hstack([np.zeros(N_per_class), np.ones(N_per_class)])
            indices = np.arange(Y.shape[0])
            rng.shuffle(indices)
            return X[indices, :], Y[indices]

        N_per_class, D = 20, 5
        separator = rng.normal(0, 1, size=(D,))
        separator = separator / np.linalg.norm(separator) * 1
        res['gauss_train_X'], res['gauss_train_Y'] = generate_gaussian_mixture(N_per_class, D, separator)
        res['gauss_test_X'], res['gauss_test_Y'] = generate_gaussian_mixture(N_per_class, D, separator)

        # Fixed dataset
        X_fixed = np.array([[1, 0], [1, 0], [1, 0],
                            [0, 1], [0, 1], [0, 1]])
        Y_fixed = np.array([0, 0, 0, 1, 1, 1])
        X_confuse = np.array([[0.75, 0.25], [0.25, 0.75]])
        Y_confuse = np.array([1, 0])
        X = np.vstack([X_fixed, X_confuse])
        Y = np.hstack([Y_fixed, Y_confuse])
        indices = np.arange(Y.shape[0])
        rng.shuffle(indices)
        X, Y = X[indices, :], Y[indices]

        res['fixed_train_X'], res['fixed_train_Y'] = X, Y
        res['fixed_test_X'], res['fixed_test_Y'] = X, Y

        # Repeats dataset
        N_random, N_unique, D = 40, 20, 20
        X_unique = rng.normal(0, 1, (1, D))
        X_unique /= np.linalg.norm(X_unique)
        while X_unique.shape[0] < N_unique:
            X_new = rng.normal(0, 1, (1, D))
            new_rank = np.linalg.matrix_rank(np.vstack([X_unique, X_new]))
            if new_rank == X_unique.shape[0]: continue
            X_new -= np.dot(np.dot(X_unique.T, X_unique), X_new.T).T
            if np.linalg.norm(X_new) < 1e-3: continue
            X_new /= np.linalg.norm(X_new)
            X_unique = np.vstack([X_unique, X_new])
        axis = rng.normal(0, 1, (D,))
        Y_unique = rng.randint(0, 2, (N_unique,))

        X, Y = np.zeros((0, D)), np.zeros(0)
        for i in range(N_unique):
            X = np.vstack([X, np.repeat(X_unique[np.newaxis, i, :], i + 1, axis=0)])
            Y = np.hstack([Y, np.repeat(Y_unique[i], i + 1)])

        X_random = rng.normal(0, 0.1, (N_random, D))
        Y_random = rng.randint(0, 2, (N_random,))
        X = np.vstack([X, X_random])
        Y = np.hstack([Y, Y_random])

        res['repeats_train_X'], res['repeats_train_Y'] = X, Y
        res['repeats_test_X'], res['repeats_test_Y'] = X, Y
        res['repeats_N_unique'] = N_unique

        # Separated Gaussian mixture, high dimension
        N_per_class, D = 20, 10
        separator = rng.normal(0, 1, size=(D,))
        separator = separator / np.linalg.norm(separator) * 0
        res['gauss2_train_X'], res['gauss2_train_Y'] = generate_gaussian_mixture(N_per_class, D, separator)
        res['gauss2_test_X'], res['gauss2_test_Y'] = generate_gaussian_mixture(N_per_class, D, separator)

        # 2N < D
        N_per_class, D = 40, 100
        separator = rng.normal(0, 1, size=(D,))
        separator = separator / np.linalg.norm(separator) * 0.1
        res['gauss3_train_X'], res['gauss3_train_Y'] = generate_gaussian_mixture(N_per_class, D, separator)
        res['gauss3_test_X'], res['gauss3_test_Y'] = generate_gaussian_mixture(N_per_class, D, separator)

        # N = D
        N_per_class, D = 60, 60
        separator = rng.normal(0, 1, size=(D,))
        separator = separator / np.linalg.norm(separator) * 0.5
        res['gauss4_train_X'], res['gauss4_train_Y'] = generate_gaussian_mixture(N_per_class, D, separator)
        res['gauss4_test_X'], res['gauss4_test_Y'] = generate_gaussian_mixture(N_per_class, D, separator)

        # Orthogonal with different distances from origin
        D = 2
        X = np.zeros((0, D))
        Y = np.zeros((0,))
        Xa = np.eye(D)
        unique_ids = []
        for i in range(D):
            repeats_r = (40, 40)[i]
            repeats_s = (20, 20)[i]
            r = (0.25 * 0.3, 1)[i]
            s = (0.5 * 0.3, 0.1)[i]

            X = np.vstack([X,
                           np.repeat(Xa[i, :][np.newaxis, :] * r, repeats_r, axis=0),
                           np.repeat(Xa[i, :][np.newaxis, :] * s, repeats_s, axis=0)])
            Y = np.hstack([Y, np.full(repeats_r, 0)])
            Y = np.hstack([Y, np.full(repeats_s, 1)])
            unique_ids.extend([2 * i] * repeats_r)
            unique_ids.extend([2 * i + 1] * repeats_s)
        res['ortho2_train_X'], res['ortho2_train_Y'] = X, Y
        res['ortho2_test_X'], res['ortho2_test_Y'] = X, Y
        res['ortho2_ids'] = np.array(unique_ids)

        # Gaussian on one plane and orthogonal elsewhere
        N_random, N_per_class, D = 20, 40, 10
        separator = rng.normal(0, 1, size=(D,))
        separator = separator / np.linalg.norm(separator) * 1
        X, Y = generate_gaussian_mixture(N_per_class, D, separator)
        X = np.hstack([X, np.full((X.shape[0], 1), 0)])
        X_random = rng.normal(0, 1, (N_random, D + 1))
        Y_random = rng.randint(0, 2, (N_random,))
        X = np.vstack([X, X_random])
        Y = np.hstack([Y, Y_random])
        res['gauss5_train_X'], res['gauss5_train_Y'] = X, Y
        res['gauss5_test_X'], res['gauss5_test_Y'] = X, Y

        return res

    def get_dataset(self, dataset_id=None):
        dataset_id = dataset_id if dataset_id is not None else self.dataset_id
        if not hasattr(self, 'datasets'):
            self.datasets = dict()
        if not dataset_id in self.datasets:
            ds_keys = ['{}_{}'.format(dataset_id, key) for key in
                         ('train_X', 'train_Y', 'test_X', 'test_Y')]
            if any(ds_key not in self.R for ds_key in ds_keys):
                raise ValueError('Dataset gauss has not been generated')
            train_X, train_Y, test_X, test_Y = [self.R[ds_key] for ds_key in ds_keys]
            train = DataSet(train_X, train_Y)
            test = DataSet(test_X, test_Y)
            self.datasets[dataset_id] = base.Datasets(train=train, test=test, validation=None)
        return self.datasets[dataset_id]

    def get_model(self, dataset_id=None):
        if not hasattr(self, 'model'):
            dataset = self.get_dataset(dataset_id)
            model_config = LogisticRegression.default_config()
            model_config['arch'] = LogisticRegression.infer_arch(dataset.train)
            model_dir = os.path.join(self.base_dir, 'models')
            self.model = LogisticRegression(model_config, model_dir)
        return self.model

    @phase(1)
    def training(self):
        res = dict()

        ds = self.get_dataset()
        model = self.get_model()

        res['l2_reg'] = l2_reg = ds.train.num_examples * 1e-3

        with benchmark("Training original model"):
            model.fit(ds.train, l2_reg=l2_reg)
            model.print_model_eval(ds, l2_reg=l2_reg)
            model.save('initial')

        res['train_losses'] = model.get_indiv_loss(ds.train)
        res['train_margins'] = model.get_indiv_margin(ds.train)
        res['train_accuracy'] = model.get_accuracy(ds.train)
        res['test_losses'] = model.get_indiv_loss(ds.test)
        res['test_margins'] = model.get_indiv_margin(ds.test)
        res['test_accuracy'] = model.get_accuracy(ds.test)

        with benchmark("Computing gradients"):
            res['train_grad_losses'] = model.get_indiv_grad_loss(ds.train)
            res['train_grad_margins'] = model.get_indiv_grad_margin(ds.train)
            res['test_grad_losses'] = model.get_indiv_grad_loss(ds.test)
            res['test_grad_margins'] = model.get_indiv_grad_margin(ds.test)

        res['hessian'] = model.get_hessian(ds.train, l2_reg=l2_reg)

        return res

    @phase(2)
    def pick_subsets(self):
        ds = self.get_dataset()

        if self.dataset_id == "repeats":
            N_unique = self.R['repeats_N_unique']
            subset_indices = [
                list(range(i * (i + 1) // 2, i * (i + 1) // 2 + size))
                for i in range(N_unique)
                for size in range(1, (i + 1) + 1)
            ]
        elif self.dataset_id == "ortho2":
            unique_ids = np.unique(self.R['ortho2_ids'])
            subset_indices = []
            for id in unique_ids:
                repeats = np.nonzero(self.R['ortho2_ids'] == id)[0]
                for size in range(1, len(repeats)):
                    subset_indices.append(repeats[:size])
        else:
            if self.dataset_id == "gauss2" or self.dataset_id == "gauss3" or self.dataset_id =="gauss4":
                size_min, size_max = 2, 2
            elif self.dataset_id == "gauss":
                size_min, size_max = 1, 3
            elif self.dataset_id == "gauss5":
                size_min, size_max = 1, 1
            else:
                size_min, size_max = 2, 3
            subset_indices = list(list(subset)
                for r in range(size_min, size_max + 1)
                for subset in itertools.combinations(range(ds.train.num_examples), r))

        return {'subset_indices': subset_indices}

    def retrain_and_newton_batch(self, subset_start, subset_end):
        res = dict()
        self.load_phases([0, 1, 2], verbose=False)

        ds = self.get_dataset()
        model = self.get_model()
        model.load('initial')
        initial_params = model.get_params_flat()
        l2_reg = self.R['l2_reg']
        hessian = self.R['hessian']

        subsets = self.R['subset_indices'][subset_start:subset_end]
        num_subsets = len(subsets)
        train_grad_losses = self.R['train_grad_losses']
        subset_grad_losses = np.array([np.sum(train_grad_losses[subset, :], axis=0) for subset in subsets])

        start_time = time.time()

        with benchmark('Computing first-order predicted parameters for subsets {}-{}'.format(subset_start, subset_end)):
            inverse_hvp_args = {
                'hessian_reg': hessian,
                'verbose': False,
                'inverse_hvp_method': 'explicit',
                'inverse_vp_method': 'cholesky',
            }
            res['subset_pred_dparam'] = model.get_inverse_hvp(subset_grad_losses.T, **inverse_hvp_args).T

        with benchmark('Computing Newton predicted parameters for subsets {}-{}'.format(subset_start, subset_end)):
            newton_pred_dparam = np.zeros((num_subsets, model.params_dim))
            for i, subset in enumerate(subsets):
                hessian_w = model.get_hessian(ds.train.subset(subset), l2_reg=0, verbose=False)
                inverse_hvp_args = {
                    'hessian_reg': hessian - hessian_w,
                    'verbose': False,
                    'inverse_hvp_method': 'explicit',
                    'inverse_vp_method': 'cholesky',
                }
                subset_grad_loss = subset_grad_losses[i, :].reshape(-1, 1)
                pred_dparam = model.get_inverse_hvp(subset_grad_loss, **inverse_hvp_args).reshape(-1)
                newton_pred_dparam[i, :] = pred_dparam
            res['subset_newton_pred_dparam'] = newton_pred_dparam

        with benchmark('Computing actual parameters for subsets {}-{}'.format(subset_start, subset_end)):
            actl_dparam = np.zeros((num_subsets, model.params_dim))
            for i, subset in enumerate(subsets):
                s = np.ones(ds.train.num_examples)
                s[subset] = 0
                model.warm_fit(ds.train, s, l2_reg=l2_reg)
                model.save('subset_{}'.format(i + subset_start))
                actl_dparam[i, :] = model.get_params_flat() - initial_params

            res['subset_dparam'] = actl_dparam

        end_time = time.time()
        time_per_subset = (end_time - start_time) / num_subsets
        remaining_time = (len(self.R['subset_indices']) - subset_end) * time_per_subset
        print('Each retraining and iHVP takes {} s, {} s remaining'.format(time_per_subset, remaining_time))

        return res

    @phase(3)
    def retrain_and_newton(self):
        num_subsets = len(self.R['subset_indices'])
        subsets_per_batch = 64
        results = self.task_queue.execute('retrain_and_newton_batch', [
            (i, min(i + subsets_per_batch, num_subsets))
            for i in range(0, num_subsets, subsets_per_batch)],
            force_refresh=True)

        return self.task_queue.collate_results(results)

    def compute_test_influence(self, subset_start, subset_end, ds_test):
        res = dict()

        model = self.get_model()
        model.load('initial')
        initial_params = model.get_params_flat()
        actl_dparam = self.R['subset_dparam'][subset_start:subset_end, :]
        pred_dparam = self.R['subset_pred_dparam'][subset_start:subset_end, :]
        newton_pred_dparam = self.R['subset_newton_pred_dparam'][subset_start:subset_end, :]

        test_losses = model.get_indiv_loss(ds_test, verbose=False)
        test_margins = model.get_indiv_margin(ds_test, verbose=False)

        subsets = self.R['subset_indices'][subset_start:subset_end]
        num_subsets = len(subsets)

        with benchmark('Computing actual parameters and influence for subsets {}-{}'.format(subset_start, subset_end)):
            subset_test_actl_infl = np.zeros((num_subsets, ds_test.num_examples))
            subset_test_actl_margin_infl = np.zeros((num_subsets, ds_test.num_examples))
            for i, subset in enumerate(subsets):
                actl_param = initial_params + actl_dparam[i, :]

                model.set_params_flat(actl_param)
                actl_losses = model.get_indiv_loss(ds_test, verbose=False)
                actl_margins = model.get_indiv_margin(ds_test, verbose=False)

                subset_test_actl_infl[i, :] = actl_losses - test_losses
                subset_test_actl_margin_infl[i, :] = actl_margins - test_margins

            res['subset_test_actl_infl'] = subset_test_actl_infl
            res['subset_test_actl_margin_infl'] = subset_test_actl_margin_infl

        with benchmark('Computing influence approximates for subsets {}-{}'.format(subset_start, subset_end)):
            subset_test_pparam_infl = np.zeros((num_subsets, ds_test.num_examples))
            subset_test_pparam_margin_infl = np.zeros((num_subsets, ds_test.num_examples))
            subset_test_nparam_infl = np.zeros((num_subsets, ds_test.num_examples))
            subset_test_nparam_margin_infl = np.zeros((num_subsets, ds_test.num_examples))
            for i, subset in enumerate(subsets):
                pparam = initial_params + pred_dparam[i, :]
                nparam = initial_params + newton_pred_dparam[i, :]

                model.set_params_flat(pparam)
                pparam_losses = model.get_indiv_loss(ds_test, verbose=False)
                pparam_margins = model.get_indiv_margin(ds_test, verbose=False)

                model.set_params_flat(nparam)
                nparam_losses = model.get_indiv_loss(ds_test, verbose=False)
                nparam_margins = model.get_indiv_margin(ds_test, verbose=False)

                subset_test_pparam_infl[i, :] = pparam_losses - test_losses
                subset_test_pparam_margin_infl[i, :] = pparam_margins - test_margins
                subset_test_nparam_infl[i, :] = nparam_losses - test_losses
                subset_test_nparam_margin_infl[i, :] = nparam_margins - test_margins

            res['subset_test_pparam_infl'] = subset_test_pparam_infl
            res['subset_test_pparam_margin_infl'] = subset_test_pparam_margin_infl
            res['subset_test_nparam_infl'] = subset_test_nparam_infl
            res['subset_test_nparam_margin_infl'] = subset_test_nparam_margin_infl

        return res

    @phase(4)
    def find_adversarial_test(self):
        res = dict()

        ds = self.get_dataset()
        model = self.get_model()
        model.load('initial')

        rng = np.random.RandomState(self.config['seed'])

        subset_pred_dparam = self.R['subset_pred_dparam']
        subset_newton_pred_dparam = self.R['subset_newton_pred_dparam']
        norm = np.linalg.norm(subset_pred_dparam, axis=1) * np.linalg.norm(subset_newton_pred_dparam, axis=1)
        subset_cos_dparam = np.sum(subset_pred_dparam * subset_newton_pred_dparam, axis=1) / (norm + 1e-4)
        res['subset_cos_dparam'] = subset_cos_dparam

        # For K subsets, find a distribution of test points such that
        # (pred_margin, newton_pred_margin) ~ gaussian
        D = self.R['subset_pred_dparam'].shape[1]
        cex_X = np.zeros((0, D))
        cex_Y = np.zeros((0,))
        cex_tags = []
        cos_indices = np.argsort(subset_cos_dparam)
        for K in (D // 2, D, 2 * D, 3 * D):
            if K > len(cos_indices): continue
            # Number of test points to try this method for
            N_test = 100

            # Subsets with a lower cosine similarity between dparams are easier to find
            # counterexample test points for
            easiest_subsets = cos_indices[:K]
            res['cex_lstsq_uv_K-{}_subsets'.format(K)] = easiest_subsets

            X = []
            A = np.vstack([subset_pred_dparam[easiest_subsets, :],
                           subset_newton_pred_dparam[easiest_subsets, :]])
            for u, v in rng.normal(0, 1, (N_test, 2)):
                B = np.hstack([np.full(K, u), np.full(K, v)])
                x = np.linalg.lstsq(A, B, rcond=None)[0]
                x /= np.linalg.norm(x)
                X.append(x)
            X = np.array(X)
            Y = np.ones(X.shape[0])

            cex_X = np.vstack([cex_X, X])
            cex_Y = np.hstack([cex_Y, Y])
            cex_tags.extend(['lstsq_uv_K-{}'.format(K)] * N_test)

            for y in rng.normal(0, 1, (N_test, A.shape[0])):
                x = np.linalg.lstsq(A, y, rcond=None)[0]
                cex_X = np.vstack([cex_X, x])
                cex_Y = np.hstack([cex_Y, 1])
                cex_tags.extend(['lstsq_uv_K-{}_gauss'.format(K)])

        # Pick the top K subsets and find a bad test point based on the average pred and newton dparams
        for K in [1] + list(range(10, 201, 10)):
            easiest_subsets = cos_indices[:K]
            res['cex_avg_K-{}'.format(K)] = easiest_subsets

            a = np.mean(subset_pred_dparam[easiest_subsets, :], axis=0)
            b = np.mean(subset_newton_pred_dparam[easiest_subsets, :], axis=0)

            # Minimizing a^T x_test - C b^T x_test subject to norm(x_test) = 1
            # this is just x_test = normalize(-a + C b)
            for C in (1e-1, 1e-5, 1, 2, 10):
                x = -a + C * b
                x /= np.linalg.norm(x)
                cex_X = np.vstack([cex_X, x])
                cex_Y = np.hstack([cex_Y, 1])
                cex_tags.append('cex_avg_K-{}_C-{}'.format(K, C))

            # Find x_test such that a^T x_test = 0 and b^T x_test is big
            x = b - a * np.dot(a, b) / np.dot(a, a)
            x /= np.linalg.norm(x)
            cex_X = np.vstack([cex_X, x])
            cex_Y = np.hstack([cex_Y, 1])
            cex_tags.append('cex_avg_K-{}_01'.format(K, C))

        # Sort subsets by b^T b - (a^T b)^2 / (a^T a) and use those to find a^T x = 0, b^T x = big
        aTa = np.sum(subset_pred_dparam ** 2, axis=1)
        aTb = np.sum(subset_pred_dparam * subset_newton_pred_dparam, axis=1)
        bTb = np.sum(subset_newton_pred_dparam ** 2, axis=1)
        factor = bTb - aTb ** 2 / aTa
        factor_indices = np.argsort(factor)
        for s in factor_indices[::-1][:50]:
            a = subset_pred_dparam[s, :]
            b = subset_newton_pred_dparam[s, :]

            # Find x_test such that a^T x_test = 0 and b^T x_test is big
            x = b - a * np.dot(a, b) / np.dot(a, a)
            x /= np.linalg.norm(x)
            cex_X = np.vstack([cex_X, x])
            cex_Y = np.hstack([cex_Y, 1])
            cex_tags.append('cex_factor')

        # Pick random small points
        N_random = 50
        X = rng.normal(0, 1, (N_random, cex_X.shape[1]))
        Y = rng.randint(0, 2, (N_random,))
        cex_X = np.vstack([cex_X, X])
        cex_Y = np.hstack([cex_Y, Y])
        cex_tags.extend(['cex_random'] * N_random)

        cex_X = np.vstack([cex_X, np.full((1, cex_X.shape[1]), 1)])
        cex_Y = np.hstack([cex_Y, 1])
        cex_tags.extend(['cex_one'])

        res['cex_X'] = cex_X
        res['cex_Y'] = cex_Y
        res['cex_tags'] = cex_tags

        return res

    def compute_cex_test_infl_batch(self, subset_start, subset_end):
        self.load_phases([0, 1, 2, 3, 4], verbose=False)

        start_time = time.time()

        ds_test = DataSet(self.R['cex_X'], self.R['cex_Y'])
        res = dict(('cex_' + key, value)
                   for key, value in self.compute_test_influence(subset_start, subset_end, ds_test).items())

        end_time = time.time()
        time_per_subset = (end_time - start_time) / (subset_end - subset_start)
        remaining_time = (len(self.R['subset_indices']) - subset_end) * time_per_subset
        print('Each subset takes {} s, {} s remaining'.format(time_per_subset, remaining_time))

        return res

    @phase(5)
    def compute_test_infl(self):
        num_subsets = len(self.R['subset_indices'])
        subsets_per_batch = 256
        results = self.task_queue.execute('compute_cex_test_infl_batch', [
            (i, min(i + subsets_per_batch, num_subsets))
            for i in range(0, num_subsets, subsets_per_batch)],
            force_refresh=True)

        res = self.task_queue.collate_results(results)

        ds_test = DataSet(self.R['cex_X'], self.R['cex_Y'])
        model = self.get_model()
        model.load('initial')
        test_grad_losses = model.get_indiv_grad_loss(ds_test, verbose=False)
        test_grad_margins = model.get_indiv_grad_margin(ds_test, verbose=False)

        pred_dparam = self.R['subset_pred_dparam']
        newton_pred_dparam = self.R['subset_newton_pred_dparam']
        res['cex_subset_test_pred_infl'] = np.dot(pred_dparam, test_grad_losses.T)
        res['cex_subset_test_pred_margin_infl'] = np.dot(pred_dparam, test_grad_margins.T)
        res['cex_subset_test_newton_pred_infl'] = np.dot(newton_pred_dparam, test_grad_losses.T)
        res['cex_subset_test_newton_pred_margin_infl'] = np.dot(newton_pred_dparam, test_grad_margins.T)

        return res

    def plot_overestimates(self, save_and_close=False):
        ds = self.get_dataset()
        pred = self.R['subset_pred_margin_infl'].reshape(-1)
        newton = self.R['subset_newton_pred_margin_infl'].reshape(-1)
        actl = self.R['subset_actl_margin_infl'].reshape(-1)
        overestimates = self.R['overestimates'].reshape(-1)
        tags = np.array(['first-order < sign(first-order) * newton'] * len(pred))
        tags[overestimates] = 'first-order > sign(first-order) * newton'
        if self.dataset_id != "repeats":
            subset_sizes = np.repeat([len(subset) for subset in self.R['subset_indices']], ds.test.num_examples).reshape(-1)
            tags = ['{} (size {})'.format(tag, size) for tag, size in zip(tags, subset_sizes)]

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        plot_influence_correlation(ax, pred, newton,
                                   label=tags,
                                   xlabel='First-order influence',
                                   ylabel='Newton influence',
                                   title='Influence on margin, for all combinations of test points and subsets',
                                   subtitle=self.dataset_id,
                                   size=1)
        if save_and_close:
            fig.savefig(os.path.join(self.plot_dir, 'pred_over_newton.png'), bbox_inches='tight')
            plt.close(fig)

    def plot_repeats(self, save_and_close=False):
        if self.dataset_id != "repeats": return
        ds = self.get_dataset()
        pred = self.R['subset_pred_margin_infl'].reshape(-1)
        newton = self.R['subset_newton_pred_margin_infl'].reshape(-1)

        sizes = np.array([len(subset) for subset in self.R['subset_indices']])
        norm = mpl.colors.Normalize(vmin=1, vmax=np.max(sizes))
        cmap = plt.get_cmap('plasma')
        color_by_size = np.repeat(cmap(norm(sizes)), ds.test.num_examples, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        plot_influence_correlation(ax, pred, newton,
                                   colors=color_by_size,
                                   xlabel='First-order influence',
                                   ylabel='Newton influence',
                                   title='Influence on margin, for all combinations of test points and subsets',
                                   subtitle=self.dataset_id,
                                   size=1,
                                   balanced=True,
                                   equal=False)
        ax.set_xlim([x * 0.5 for x in ax.get_xlim()])
        ax.set_ylim([x * 0.5 for x in ax.get_ylim()])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.set_ylabel('number of repeats removed', rotation=90)

        if save_and_close:
            fig.savefig(os.path.join(self.plot_dir, 'pred_over_newton_size.png'), bbox_inches='tight')
            plt.close(fig)

        N_unique = self.R['repeats_N_unique']
        repeat_ids = np.repeat(np.array([i for i in range(N_unique) for _ in range(i + 1)]), ds.test.num_examples)
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(repeat_ids))
        cmap = plt.get_cmap('rainbow', N_unique)
        color_by_id = cmap(repeat_ids)
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        plot_influence_correlation(ax, pred, newton,
                                   colors=color_by_id,
                                   xlabel='First-order influence',
                                   ylabel='Newton influence',
                                   title='Influence on margin, for all combinations of test points and subsets',
                                   subtitle=self.dataset_id,
                                   size=3,
                                   balanced=True,
                                   equal=False)
        ax.set_xlim([x * 0.5 for x in ax.get_xlim()])
        ax.set_ylim([x * 0.5 for x in ax.get_ylim()])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.set_ylabel('repeated point id', rotation=90)

        if save_and_close:
            fig.savefig(os.path.join(self.plot_dir, 'pred_over_newton_id.png'), bbox_inches='tight')
            plt.close(fig)

    def plot_counterex_distribution(self, save_and_close=False):
        overestimates = np.mean(self.R['overestimates'], axis=1)
        std = np.std(overestimates)
        if std < 1e-8: return
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_distribution(ax, overestimates,
                          title='Distribution of counterexamples',
                          xlabel='Fraction of test points with first-order > sign(first-order) * newton',
                          ylabel='Number of subsets',
                          subtitle=self.dataset_id)
        if save_and_close:
            fig.savefig(os.path.join(self.plot_dir, 'pred_over_newton_dist.png'), bbox_inches='tight')
            plt.close(fig)

    def plot_dist_infl(self, save_and_close=False):
        K = self.R['dist_subset_pred_margin_infl'].shape[0]

        pred_margin = self.R['dist_subset_pred_margin_infl']
        newton_margin = self.R['dist_subset_newton_pred_margin_infl']
        actl_margin = self.R['dist_subset_actl_margin_infl']

        pred = self.R['dist_subset_pred_infl']
        newton = self.R['dist_subset_newton_pred_infl']
        actl = self.R['dist_subset_actl_infl']

        def compare_influences(x, y, x_approx_type, y_approx_type, infl_type):
            approx_type_to_label = { 'pred': 'First-order influence',
                                     'newton': 'Newton influence',
                                     'actl': 'Actual influence' }
            xlabel = approx_type_to_label[x_approx_type]
            ylabel = approx_type_to_label[y_approx_type]
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            plot_influence_correlation(ax, x.reshape(-1), y.reshape(-1),
                                       xlabel=xlabel,
                                       ylabel=ylabel,
                                       title='Influence on {}, for {} subsets and a constructed test set'.format(infl_type, K),
                                       subtitle=self.dataset_id,
                                       size=3,
                                       equal=False)
            if save_and_close:
                fig.savefig(os.path.join(self.plot_dir, 'dist_{}-{}_{}.png'.format(
                    x_approx_type, y_approx_type, "infl" if infl_type == "loss" else "margin_infl")), bbox_inches='tight')
                plt.close(fig)

        compare_influences(pred_margin, newton_margin, 'pred', 'newton', 'margin')
        compare_influences(actl_margin, pred_margin, 'actl', 'pred', 'margin')
        compare_influences(pred, newton, 'pred', 'newton', 'loss')
        compare_influences(actl, pred, 'actl', 'pred', 'loss')


    def plot_all(self, save_and_close=False):
        self.plot_overestimates(save_and_close)
        self.plot_repeats(save_and_close)
        self.plot_counterex_distribution(save_and_close)
        self.plot_dist_infl(save_and_close)
