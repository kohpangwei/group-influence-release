from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import datasets as ds
import datasets.loader
from datasets.common import DataSet
from experiments.common import Experiment, collect_phases, phase
from experiments.benchmark import benchmark
from influence.logistic_regression import LogisticRegression

import os
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets import base

from experiments.plot import *

from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import f1_score

@collect_phases
class DataValuation(Experiment):
    def __init__(self, config, out_dir=None):
        super(DataValuation, self).__init__(config, out_dir)
        self.datasets = ds.loader.load_dataset(**self.config['dataset_config'])
        self.dataset_id = self.config['dataset_config']['dataset_id']
        self.data_dir = self.config['dataset_config']['data_dir']
        self.train = self.datasets.train
        self.test = self.datasets.test
        self.validation = self.datasets.validation
        self.sample_weights = ds.loader.load_supplemental_info(self.dataset_id + '_weights',
                data_dir=self.data_dir)\
                if self.config['sample_weights'] else [np.ones(self.train.x.shape[0]),
                        np.ones(self.validation.x.shape[0]),
                        np.ones(self.test.x.shape[0])]
        self.nonfires = ds.loader.load_supplemental_info(self.dataset_id + '_nonfires',
                data_dir=self.data_dir)
        def balance(ds, rngNum, weights=None):
            pos_inds = np.where(ds.labels == 1)[0]
            neg_inds = np.where(ds.labels == 0)[0]
            num_per_class = min(len(pos_inds), len(neg_inds))
            pos_inds = np.random.RandomState(rngNum).choice(pos_inds, num_per_class)
            neg_inds = np.random.RandomState(rngNum).choice(neg_inds, num_per_class)
            inds = np.concatenate((pos_inds, neg_inds))
            if weights is not None:
                return ds.subset(inds), weights[inds]
            return ds.subset(inds)
        
        if 'balance_nonfires' in self.config and self.config['balance_nonfires']:
            self.nonfires = balance(self.nonfires, 0)
        if 'balance_test' in self.config and self.config['balance_test']:
            self.test, self.sample_weights[2] = balance(self.test, 1, self.sample_weights[2])
            self.datasets = base.Datasets(train=self.train, validation=self.validation, test=self.test)

        model_dir = os.path.join(self.base_dir, 'models')
        model_config = LogisticRegression.default_config()
        model_config['arch'] = LogisticRegression.infer_arch(self.train)
        model_config['arch']['fit_intercept'] = True

        # Heuristic for determining maximum batch evaluation sizes without OOM
        D = model_config['arch']['input_dim'] * model_config['arch']['num_classes']
        model_config['grad_batch_size'] =  max(1, self.config['max_memory'] // D)
        model_config['hessian_batch_size'] = max(1, self.config['max_memory'] // (D * D))

        self.model_dir = model_dir
        self.model_config = model_config

        # Convenience member variables
        self.num_train = self.train.num_examples
        self.num_classes = self.model_config['arch']['num_classes']
        self.num_subsets = self.config['num_subsets']
        self.subset_size = int(self.num_train * self.config['subset_rel_size'])

    experiment_id = "data_valuation"

    @property
    def run_id(self):
        return "{}_sample_weights-{}{}{}".format(
            self.dataset_id,
            self.config['sample_weights'],
            '-bal-nonfires' if 'balance_nonfires' in self.config and self.config['balance_nonfires'] else '',
            '-bal-test' if 'balance_test' in self.config and self.config['balance_test'] else '')

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
        cv_f1s = np.zeros_like(regs)
        fold_size = (self.num_train + num_folds - 1) // num_folds
        folds = [(k * fold_size, min((k + 1) * fold_size, self.num_train)) for k in range(num_folds)]

        for i, reg in enumerate(regs):
            with benchmark("Evaluating CV error for reg={}".format(reg)):
                cv_error = 0.0
                cv_acc = 0.0
                cv_f1 = 0.0
                for k, fold in enumerate(folds):
                    print('Beginning fold {}'.format(k))
                    fold_begin, fold_end = fold
                    train_indices = np.concatenate((np.arange(0, fold_begin), np.arange(fold_end, self.num_train)))
                    val_indices = np.arange(fold_begin, fold_end)

                    print('Fitting model.')
                    model.fit(self.train.subset(train_indices), l2_reg=reg, sample_weights=self.sample_weights[0][train_indices])
                    fold_loss = model.get_total_loss(self.train.subset(val_indices), reg=False, sample_weights=self.sample_weights[0][val_indices])
                    acc = model.get_accuracy(self.train.subset(val_indices))
                    cv_error += fold_loss
                    cv_acc += acc
                    score = f1_score(self.test.labels, np.array(model.get_predictions(self.test.x)[:,1]>0.5).astype(np.int), sample_weight=self.sample_weights[2])
                    cv_f1 += score
                    print('F1: {}, Acc: {}, loss: {}'.format(score, acc, fold_loss))

            cv_errors[i] = cv_error
            cv_accs[i] = cv_acc / num_folds
            cv_f1s[i] = cv_f1 / num_folds
            print('Cross-validation f1 {}, acc {}, error {} for reg={}.'.format(cv_f1s[i], cv_accs[i], cv_errors[i], reg))

        best_i = np.argmax(cv_f1s)
        best_reg = regs[best_i]
        print('Cross-validation errors: {}'.format(cv_errors))
        print('Cross-validation accs: {}'.format(cv_accs))
        print('Cross-validation F1s: {}'.format(cv_f1s))
        print('Selecting weight_decay {}, with f1 {}, acc {}, error {}.'.format(\
                best_reg, cv_f1s[best_i], cv_accs[best_i], cv_errors[best_i]))

        res['cv_regs'] = regs
        res['cv_errors'] = cv_errors
        res['cv_accs'] = cv_accs
        res['cv_l2_reg'] = best_reg
        res['cv_f1s'] = cv_f1s
        return res

    @phase(1)
    def initial_training(self):
        model = self.get_model()
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        with benchmark("Training original model"):
            model.fit(self.train, l2_reg=l2_reg, sample_weights=self.sample_weights[0])
            model.print_model_eval(self.datasets, sample_weights=self.sample_weights)
            model.save('initial')

        res['initial_train_losses'] = model.get_indiv_loss(self.train)
        res['initial_test_losses'] = model.get_indiv_loss(self.test)
        res['initial_nonfires_losses'] = model.get_indiv_loss(self.nonfires)
        if self.num_classes == 2:
            res['initial_train_margins'] = model.get_indiv_margin(self.train)
            res['initial_test_margins'] = model.get_indiv_margin(self.test)
            res['initial_nonfires_margins'] = model.get_indiv_margin(self.nonfires)

        print('F1 test score: {}'.format(f1_score(self.test.labels, np.array(model.get_predictions(self.test.x)[:,1]>0.5).astype(np.int), sample_weight=self.sample_weights[2])))

        with benchmark("Computing gradients"):
            res['train_grad_loss'] = model.get_indiv_grad_loss(self.train)

        return res

    @phase(2)
    def pick_test_points(self):

        # Freeze each set after the first run
        if self.dataset_id in ['cdr', 'reduced_cdr']:
            if 'balance_test' in self.config and self.config['balance_test']:
                fixed_test = [898, 293, 14, 1139, 1783, 1100]
            else:
                fixed_test = [1502, 1021, 3778, 830, 3894, 3149]
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

        with benchmark("Computing hessian"):
            res['hessian'] = hessian = model.get_hessian(self.train, l2_reg=l2_reg, sample_weights=self.sample_weights[0])

        return res

    @phase(4)
    def all_and_fixed_test_and_nonfire_influence(self):
        model = self.get_model()
        model.load('initial')
        res = dict()

        res['nonfires_predictions'] = model.get_predictions(self.nonfires.x)
        hessian = self.R['hessian']

        def compute_test_like_infl(points, grad_fn, **kwargs):
            test_grad = grad_fn(points, **kwargs).reshape(-1,1)
            test_grad_H_inv = model.get_inverse_vp(hessian, test_grad).reshape(-1)
            pred_infl = np.dot(self.R['train_grad_loss'], test_grad_H_inv)
            return pred_infl

        fixed_test = self.R['fixed_test']
        fixed_test_pred_infl = []
        fixed_test_pred_margin_infl = []
        for test_idx in fixed_test:
            single_test_point = self.test.subset([test_idx])
            with benchmark('Scalar infl for all training points on test_idx {}.'.format(test_idx)):
                fixed_test_pred_infl.append(compute_test_like_infl(single_test_point,\
                        model.get_indiv_grad_loss))
            if self.num_classes == 2:
                with benchmark('Scalar margin infl for all training points on test_idx {}.'.format(test_idx)):
                    fixed_test_pred_margin_infl.append(compute_test_like_infl(single_test_point,\
                            model.get_indiv_grad_margin))

        # Compute influence on the entire test set
        with benchmark('Scalar infl for all training points on entire test set.'):
            res['all_test_pred_infl'] = np.array(compute_test_like_infl(self.test,\
                    model.get_total_grad_loss, sample_weights=self.sample_weights[2]))
        if self.num_classes == 2:
            with benchmark('Scalar margin infl for all training points on entire test set.'):
                res['all_test_pred_margin_infl'] = np.array(compute_test_like_infl(self.test,\
                        model.get_total_grad_margin, sample_weights=self.sample_weights[2]))

        # Compute influence on the positive and negative parts of the test set
        pos_inds = np.where(self.test.labels == 1)[0]
        neg_inds = np.where(self.test.labels == 0)[0]
        with benchmark('Scalar infl for all training points on positive test set.'):
            res['pos_test_pred_infl'] = np.array(compute_test_like_infl(self.test.subset(pos_inds),\
                    model.get_total_grad_loss, sample_weights=self.sample_weights[2][pos_inds]))
        if self.num_classes == 2:
            with benchmark('Scalar margin infl for all training points on positive test set.'):
                res['pos_test_pred_margin_infl'] = np.array(compute_test_like_infl(self.test.subset(pos_inds),\
                        model.get_total_grad_margin, sample_weights=self.sample_weights[2][pos_inds]))
        with benchmark('Scalar infl for all training points on negative test set.'):
            res['neg_test_pred_infl'] = np.array(compute_test_like_infl(self.test.subset(neg_inds),\
                    model.get_total_grad_loss, sample_weights=self.sample_weights[2][neg_inds]))
        if self.num_classes == 2:
            with benchmark('Scalar margin infl for all training points on negative test set.'):
                res['neg_test_pred_margin_infl'] = np.array(compute_test_like_infl(self.test.subset(neg_inds),\
                        model.get_total_grad_margin, sample_weights=self.sample_weights[2][neg_inds]))


        # Compute influence on the entire train set
        with benchmark('Scalar infl for all training points on entire training set.'):
            res['all_train_pred_infl'] = np.array(compute_test_like_infl(self.train,\
                    model.get_total_grad_loss, sample_weights=self.sample_weights[0]))
        if self.num_classes == 2:
            with benchmark('Scalar margin infl for all training points on entire training set.'):
                res['all_train_pred_margin_infl'] = np.array(compute_test_like_infl(self.train,\
                        model.get_total_grad_margin, sample_weights=self.sample_weights[0]))

        # Do this for the nonfires
        nonfires_pred_infl = []
        nonfires_pred_margin_infl = []
        for idx in range(self.nonfires.x.shape[0]):
            single_point = self.nonfires.subset([idx])
            with benchmark('Scalar infl for all training points on nonfire idx {}.'.format(idx)):
                nonfires_pred_infl.append(compute_test_like_infl(single_point,\
                        model.get_indiv_grad_loss))
            if self.num_classes == 2:
                with benchmark('Scalar margin infl for all training points on nonfire idx {}.'.format(idx)):
                    nonfires_pred_margin_infl.append(compute_test_like_infl(single_point,\
                            model.get_indiv_grad_margin))

        res['fixed_test_pred_infl'] = np.array(fixed_test_pred_infl)
        if self.num_classes == 2:
            res['fixed_test_pred_margin_infl'] = np.array(fixed_test_pred_margin_infl)
        res['nonfires_pred_infl'] = np.array(nonfires_pred_infl)
        if self.num_classes == 2:
            res['nonfires_pred_margin_infl'] = np.array(nonfires_pred_margin_infl)

        return res

    def get_random_subsets(self, rng):
        subsets = []
        for i in range(self.num_subsets):
            subsets.append(rng.choice(self.num_train, self.subset_size, replace=False))
        return np.array(subsets)

    def get_scalar_infl_tails(self, rng, pred_infl):
        window = 2 * self.subset_size
        assert window < self.num_train
        scalar_infl_indices = np.argsort(pred_infl).reshape(-1)
        pos_subsets, neg_subsets = [], []
        for i in range(self.num_subsets):
            neg_subsets.append(rng.choice(scalar_infl_indices[:window], self.subset_size, replace=False))
            pos_subsets.append(rng.choice(scalar_infl_indices[-window:], self.subset_size, replace=False))
        return np.array(neg_subsets), np.array(pos_subsets)

    def get_same_grad_dir(self, rng, train_grad_loss):
        # Using Pigeonhole to guarantee we get a sufficiently large cluster
        n_clusters = int(math.floor(1 / self.config['subset_rel_size']))
        km = KMeans(n_clusters=n_clusters)
        km.fit(train_grad_loss)
        labels, centroids = km.labels_, km.cluster_centers_
        _, counts = np.unique(labels, return_counts=True)

        best = max([(count, i) for i, count in enumerate(counts) if count >= self.subset_size])[1]
        cluster_indices = np.where(labels == best)[0]
        subsets = []
        for i in range(self.num_subsets):
            subsets.append(rng.choice(cluster_indices, self.subset_size, replace=False))
        return np.array(subsets), best, labels

    def get_same_features_subsets(self, rng, features, labels):
        center_data = self.config['dataset_config']['center_data']
        if self.dataset_id in ['cdr', 'reduced_cdr'] and not center_data:
            from datasets.loader import load_supplemental_info
            LF_ids = load_supplemental_info(self.dataset_id + '_LFs', data_dir=self.data_dir)
            subsets = []
            for LF_id in range(1,np.max(LF_ids[0])+1):
                covered = np.where(LF_ids[0] == LF_id)[0]
                if (len(covered) > 0):
                    subsets.append(covered)
                print('LF {} covers {} examples'.format(LF_id, len(covered)))
            return subsets
        else:
            print("Warning: unimplemented method to get subsets with the same features")
            return []

    @phase(5)
    def pick_subsets(self):
        rng = np.random.RandomState(self.config['subset_seed'])
        tagged_subsets = []

        with benchmark("Random subsets"):
            random_subsets = self.get_random_subsets(rng)
            tagged_subsets += [('random', s) for s in random_subsets]

        with benchmark("Scalar infl tail subsets"):
            for pred_infl, test_idx in zip(self.R['fixed_test_pred_infl'], self.R['fixed_test']):
                neg_tail_subsets, pos_tail_subsets = self.get_scalar_infl_tails(rng, pred_infl)
                tagged_subsets += [('neg_tail_test-{}'.format(test_idx), s) for s in neg_tail_subsets]
                tagged_subsets += [('pos_tail_test-{}'.format(test_idx), s) for s in pos_tail_subsets]
                print('Found scalar infl tail subsets for test idx {}.'.format(test_idx))

        with benchmark("Same features subsets"):
            same_features_subsets = self.get_same_features_subsets(rng, self.train.x, self.train.labels)
            tagged_subsets += [('same_features', s) for s in same_features_subsets]

        with benchmark("Same gradient subsets"):
            same_grad_subsets, cluster_label, cluster_labels = self.get_same_grad_dir(rng, self.R['train_grad_loss'])
            tagged_subsets += [('same_grad', s) for s in same_grad_subsets]

        subset_tags = [tag for tag, subset in tagged_subsets]
        subset_indices = [subset for tag, subset in tagged_subsets]

        subset_sizes = np.unique([len(subset) for tag, subset in tagged_subsets])

        return { 'subset_tags': subset_tags, 'subset_indices': subset_indices }
    
    @phase(6)
    def retrain(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        n, n_report = len(subset_indices), max(len(subset_indices) // 100, 1)

        start_time = time.time()
        train_losses, test_losses, nonfires_losses = [], [], []
        train_margins, test_margins, nonfires_margins = [], [], []
        for i, remove_indices in enumerate(subset_indices):
            if (i % n_report == 0):
                print('Retraining model {} out of {} (tag={}, size={})'.format(
                    i, n, subset_tags[i], len(remove_indices)))

            s = np.array(self.sample_weights[0])
            s[remove_indices] = 0

            model.warm_fit(self.train, s, l2_reg=l2_reg)
            model.save('subset_{}'.format(i))
            train_losses.append(model.get_indiv_loss(self.train))
            test_losses.append(model.get_indiv_loss(self.test))
            nonfires_losses.append(model.get_indiv_loss(self.nonfires))
            if model.num_classes == 2:
                train_margins.append(model.get_indiv_margin(self.train))
                test_margins.append(model.get_indiv_margin(self.test))
                nonfires_margins.append(model.get_indiv_margin(self.nonfires))

            if (i % n_report == 0):
                cur_time = time.time()
                time_per_retrain = (cur_time - start_time) / (i + 1)
                remaining_time = time_per_retrain * (n - i - 1)
                print('Each retraining takes {} s, {} s remaining'.format(time_per_retrain, remaining_time))

        res['subset_train_losses'] = np.array(train_losses)
        res['subset_test_losses'] = np.array(test_losses)
        res['subset_nonfires_losses'] = np.array(nonfires_losses)

        if self.num_classes == 2:
            res['subset_train_margins'] = np.array(train_margins)
            res['subset_test_margins'] = np.array(test_margins)
            res['subset_nonfires_margins'] = np.array(nonfires_margins)

        return res

    @phase(7)
    def compute_self_pred_infl(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        n, n_report = len(subset_indices), max(len(subset_indices) // 100, 1)

        hessian = self.R['hessian']
        train_grad_loss = self.R['train_grad_loss']

        # It is important that the influence gets calculated before the model is retrained,
        # so that the parameters are the original parameters
        start_time = time.time()
        subset_pred_dparam = []
        self_pred_infls = []
        self_pred_margin_infls = []
        for i, remove_indices in enumerate(subset_indices):
            if (i % n_report == 0):
                print('Computing self-influences for subset {} out of {} (tag={})'.format(i, n, subset_tags[i]))

            grad_loss = np.einsum('ai,a->i', train_grad_loss[remove_indices,:], self.sample_weights[0][remove_indices])
            H_inv_grad_loss = model.get_inverse_vp(hessian, grad_loss.reshape(1, -1).T).reshape(-1)
            pred_infl = np.dot(grad_loss, H_inv_grad_loss)
            subset_pred_dparam.append(H_inv_grad_loss)
            self_pred_infls.append(pred_infl)

            if model.num_classes == 2:
                s = np.zeros(self.num_train)
                s[remove_indices] = self.sample_weights[0][remove_indices]
                grad_margin = model.get_total_grad_margin(self.train, s)
                pred_margin_infl = np.dot(grad_margin, H_inv_grad_loss)
                self_pred_margin_infls.append(pred_margin_infl)

            if (i % n_report == 0):
                cur_time = time.time()
                time_per_vp = (cur_time - start_time) / (i + 1)
                remaining_time = time_per_vp * (n - i - 1)
                print('Each self-influence calculation takes {} s, {} s remaining'.format(time_per_vp, remaining_time))

        res['subset_pred_dparam'] = np.array(subset_pred_dparam)
        res['subset_self_pred_infl'] = np.array(self_pred_infls)
        if self.num_classes == 2:
            res['subset_self_pred_margin_infl'] = np.array(self_pred_margin_infls)

        return res

    @phase(8)
    def compute_actl_infl(self):
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']

        # Helper to collate all/fixed test infl and subset self infl on a quantity q
        def compute_collate_infl(fixed_test, fixed_test_pred_infl_q,
                                 pos_test_pred_infl_q, neg_test_pred_infl_q,
                                 all_test_pred_infl_q, all_train_pred_infl_q, nonfires_pred_infl_q,
                                 initial_train_q, initial_test_q, initial_nonfires_q,
                                 subset_train_q, subset_test_q, subset_nonfires_q):
            subset_fixed_test_actl_infl = subset_test_q[:, fixed_test] - initial_test_q[fixed_test]
            subset_fixed_test_pred_infl = np.array([
                np.einsum('ai,i->a',fixed_test_pred_infl_q[:, remove_indices],
                    self.sample_weights[0][remove_indices]).reshape(-1)
                for remove_indices in subset_indices])
            pos_inds = np.where(self.test.labels == 1)[0]
            neg_inds = np.where(self.test.labels == 0)[0]
            subset_pos_test_actl_infl = np.einsum('ai,i->a', subset_test_q[:,pos_inds] - initial_test_q[pos_inds],
                    self.sample_weights[2][pos_inds])
            subset_pos_test_pred_infl = np.array([
                np.dot((pos_test_pred_infl_q[remove_indices]).reshape(-1), self.sample_weights[0][remove_indices])
                for remove_indices in subset_indices])
            subset_neg_test_actl_infl = np.einsum('ai,i->a', subset_test_q[:,neg_inds] - initial_test_q[neg_inds],
                    self.sample_weights[2][neg_inds])
            subset_neg_test_pred_infl = np.array([
                np.dot((neg_test_pred_infl_q[remove_indices]).reshape(-1), self.sample_weights[0][remove_indices])
                for remove_indices in subset_indices])
            subset_all_test_actl_infl = np.einsum('ai,i->a', subset_test_q - initial_test_q,
                    self.sample_weights[2])
            subset_all_test_pred_infl = np.array([
                np.dot((all_test_pred_infl_q[remove_indices]).reshape(-1), self.sample_weights[0][remove_indices])
                for remove_indices in subset_indices])
            subset_all_train_actl_infl = np.einsum('ai,i->a', subset_train_q - initial_train_q,
                    self.sample_weights[0])
            subset_all_train_pred_infl = np.array([
                np.dot((all_train_pred_infl_q[remove_indices]).reshape(-1), self.sample_weights[0][remove_indices])
                for remove_indices in subset_indices])
            subset_nonfires_actl_infl = subset_nonfires_q - initial_nonfires_q
            subset_nonfires_pred_infl = np.array([
                np.einsum('ai,i->a',nonfires_pred_infl_q[:, remove_indices],
                    self.sample_weights[0][remove_indices]).reshape(-1)
                for remove_indices in subset_indices])
            subset_self_actl_infl = np.array([
                np.dot(subset_train_q[i][remove_indices]-initial_train_q[remove_indices],
                    self.sample_weights[0][remove_indices])
                for i, remove_indices in enumerate(subset_indices)])
            return subset_fixed_test_actl_infl, subset_fixed_test_pred_infl,\
                    subset_pos_test_actl_infl, subset_pos_test_pred_infl,\
                    subset_neg_test_actl_infl, subset_neg_test_pred_infl,\
                    subset_all_test_actl_infl, subset_all_test_pred_infl,\
                    subset_all_train_actl_infl, subset_all_train_pred_infl,\
                    subset_nonfires_actl_infl, subset_nonfires_pred_infl,\
                    subset_self_actl_infl

        # Compute influences on loss
        res['subset_fixed_test_actl_infl'], \
        res['subset_fixed_test_pred_infl'], \
        res['subset_pos_test_actl_infl'], \
        res['subset_pos_test_pred_infl'], \
        res['subset_neg_test_actl_infl'], \
        res['subset_neg_test_pred_infl'], \
        res['subset_all_test_actl_infl'], \
        res['subset_all_test_pred_infl'], \
        res['subset_all_train_actl_infl'], \
        res['subset_all_train_pred_infl'], \
        res['subset_nonfires_actl_infl'], \
        res['subset_nonfires_pred_infl'], \
        res['subset_self_actl_infl'] = compute_collate_infl(
            *[self.R[key] for key in ["fixed_test", "fixed_test_pred_infl",
                                      "pos_test_pred_infl", "neg_test_pred_infl",
                                      "all_test_pred_infl", "all_train_pred_infl", "nonfires_pred_infl",
                                      "initial_train_losses", "initial_test_losses", "initial_nonfires_losses",
                                      "subset_train_losses", "subset_test_losses", "subset_nonfires_losses"]])

        if self.num_classes == 2:
            # Compute influences on margin
            res['subset_fixed_test_actl_margin_infl'], \
            res['subset_fixed_test_pred_margin_infl'], \
            res['subset_pos_test_actl_margin_infl'], \
            res['subset_pos_test_pred_margin_infl'], \
            res['subset_neg_test_actl_margin_infl'], \
            res['subset_neg_test_pred_margin_infl'], \
            res['subset_all_test_actl_margin_infl'], \
            res['subset_all_test_pred_margin_infl'], \
            res['subset_all_train_actl_margin_infl'], \
            res['subset_all_train_pred_margin_infl'], \
            res['subset_nonfires_actl_margin_infl'], \
            res['subset_nonfires_pred_margin_infl'], \
            res['subset_self_actl_margin_infl'] = compute_collate_infl(
                *[self.R[key] for key in ["fixed_test", "fixed_test_pred_margin_infl",
                                          "pos_test_pred_margin_infl", "neg_test_pred_margin_infl",
                                          "all_test_pred_margin_infl", "all_train_pred_margin_infl", "nonfires_pred_margin_infl",
                                          "initial_train_margins", "initial_test_margins", "initial_nonfires_margins",
                                          "subset_train_margins", "subset_test_margins", "subset_nonfires_margins"]])

        return res

    def get_simple_subset_tags(self):
        def simplify_tag(tag):
            if 'pos_tail_test' in tag: return 'pos_tail_test'
            elif 'neg_tail_test' in tag: return 'neg_tail_test'
            return tag
        return map(simplify_tag, self.R['subset_tags'])

    def get_subtitle(self):
        subtitle='{}'.format(self.dataset_id)#, {} subsets per type, proportion {}'.format(
                #self.dataset_id, self.num_subsets, self.config['subset_rel_size'])
        return subtitle

    def get_same_features_indices(self):
        inds = []
        for i, tag in enumerate(self.R['subset_tags']):
            if 'same_features' in tag: inds.append(i)
        return inds

    def plot_influence(self, title, figname, actl_loss, pred_loss, actl_margin=None, pred_margin=None,
            same_features_only=False, verbose=True):
        inds = range(len(self.R['subset_tags'])) if not same_features_only\
                else self.get_same_features_indices()

        title_add = '; same features only' if same_features_only else ''
        figname_add = '-same-features-only' if same_features_only else ''

        if same_features_only: subset_tags = ['Data source {}'.format(i+1) for i in range(len(inds))]
        else: subset_tags = self.get_simple_subset_tags()

        fig, ax = plt.subplots(1,1,figsize=(8,8),squeeze=False)
        plot_influence_correlation(ax[0][0],
                                   actl_loss[inds],
                                   pred_loss[inds],
                                   label=subset_tags,
                                   title='Group influence on '+title+title_add,
                                   subtitle=self.get_subtitle(),
                                   sorted_labels=same_features_only)
        fig.savefig(os.path.join(self.plot_dir, figname+'_loss'+figname_add+'.png'), bbox_inches='tight')
        plt.close(fig)

        if self.num_classes == 2:
            fig, ax = plt.subplots(1,1,figsize=(8,8),squeeze=False)
            plot_influence_correlation(ax[0][0],
                                       actl_margin[inds],
                                       pred_margin[inds],
                                       label=subset_tags,
                                       title='Group margin influence on '+title+title_add,
                                       subtitle=self.get_subtitle(),
                                       sorted_labels=same_features_only)
            fig.savefig(os.path.join(self.plot_dir, figname+'_margin'+figname_add+'.png'), bbox_inches='tight')
            plt.close(fig)

        if verbose: print('Finished plotting {} influence.'.format(title))

    def print_info(self, phrase, data):
        for group, ind in enumerate(self.get_same_features_indices()):
            print('{} {}: {}'.format(phrase, group+1, data[ind]))

    def plot_all(self, num_nonfires=15):
        nonfire_labels = np.random.choice(range(len(self.nonfires.labels)), num_nonfires)
        print('Will save plots to {}'.format(self.plot_dir))
        for boolval in [True, False]:
            self.plot_influence('self', 'self-influence',
                                self.R['subset_self_actl_infl'],
                                self.R['subset_self_pred_infl'],
                                self.R['subset_self_actl_margin_infl'],
                                self.R['subset_self_pred_margin_infl'],
                                boolval)
            for i, test_idx in enumerate(self.R['fixed_test']):
                self.plot_influence('fixed test {}'.format(test_idx), 'fixed-test-{}'.format(test_idx),
                                    self.R['subset_fixed_test_actl_infl'][:, i],
                                    self.R['subset_fixed_test_pred_infl'][:, i],
                                    self.R['subset_fixed_test_actl_margin_infl'][:, i],
                                    self.R['subset_fixed_test_pred_margin_infl'][:, i],
                                    boolval)
            for i in nonfire_labels:
                self.plot_influence('nonfire {}'.format(i), 'nonfires-{}'.format(i),
                                    self.R['subset_nonfires_actl_infl'][:,i],
                                    self.R['subset_nonfires_pred_infl'][:,i],
                                    self.R['subset_nonfires_actl_margin_infl'][:,i],
                                    self.R['subset_nonfires_pred_margin_infl'][:,i],
                                    boolval)
            self.plot_influence('pos test set', 'pos-test',
                                self.R['subset_pos_test_actl_infl'],
                                self.R['subset_pos_test_pred_infl'],
                                self.R['subset_pos_test_actl_margin_infl'],
                                self.R['subset_pos_test_pred_margin_infl'],
                                boolval)
            self.plot_influence('neg test set', 'neg-test',
                                self.R['subset_neg_test_actl_infl'],
                                self.R['subset_neg_test_pred_infl'],
                                self.R['subset_neg_test_actl_margin_infl'],
                                self.R['subset_neg_test_pred_margin_infl'],
                                boolval)
            self.plot_influence('test set', 'all-test',
                                self.R['subset_all_test_actl_infl'],
                                self.R['subset_all_test_pred_infl'],
                                self.R['subset_all_test_actl_margin_infl'],
                                self.R['subset_all_test_pred_margin_infl'],
                                boolval)
            self.plot_influence('train set', 'all-train',
                                self.R['subset_all_train_actl_infl'],
                                self.R['subset_all_train_pred_infl'],
                                self.R['subset_all_train_actl_margin_infl'],
                                self.R['subset_all_train_pred_margin_infl'],
                                boolval)

            nonfires_pos = np.where(self.nonfires.labels == 1)[0]
            nonfires_neg = np.where(self.nonfires.labels == 0)[0]

            self.plot_influence('avg pos nonfires', 'avg-pos-nonfires',
                                np.mean(self.R['subset_nonfires_actl_infl'][:, nonfires_pos], axis=1),
                                np.mean(self.R['subset_nonfires_pred_infl'][:, nonfires_pos], axis=1),
                                np.mean(self.R['subset_nonfires_actl_margin_infl'][:, nonfires_pos], axis=1),
                                np.mean(self.R['subset_nonfires_pred_margin_infl'][:, nonfires_pos], axis=1),
                                boolval)

            self.plot_influence('avg neg nonfires', 'avg-neg-nonfires',
                                np.mean(self.R['subset_nonfires_actl_infl'][:, nonfires_neg], axis=1),
                                np.mean(self.R['subset_nonfires_pred_infl'][:, nonfires_neg], axis=1),
                                np.mean(self.R['subset_nonfires_actl_margin_infl'][:, nonfires_neg], axis=1),
                                np.mean(self.R['subset_nonfires_pred_margin_infl'][:, nonfires_neg], axis=1),
                                boolval)

            self.plot_influence('avg nonfires', 'avg-nonfires',
                                np.mean(self.R['subset_nonfires_actl_infl'], axis=1),
                                np.mean(self.R['subset_nonfires_pred_infl'], axis=1),
                                np.mean(self.R['subset_nonfires_actl_margin_infl'], axis=1),
                                np.mean(self.R['subset_nonfires_pred_margin_infl'], axis=1),
                                boolval)


    def print_all(self): #TODO: not updated
        print('Info about predictions:')
        self.print_info('Self-infl of group', self.R['subset_self_pred_infl'])
        self.print_info('Test-set infl of group', self.R['subset_all_test_pred_infl'])
        self.print_info('Train-set infl of group', self.R['subset_all_train_pred_infl'])
        for j, test_idx in enumerate(self.R['fixed_test']):
            self.print_info('Fixed-test infl on test_idx {} of group'.format(test_idx),
                    self.R['subset_fixed_test_pred_infl'][:, j])
        self.print_info('Avg nonfire infl of group', np.mean(self.R['subset_nonfires_pred_infl'], axis=1))
