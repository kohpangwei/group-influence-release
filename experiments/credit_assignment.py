from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import datasets as ds
import datasets.loader
from datasets.common import DataSet
from experiments.common import Experiment, collect_phases, phase, add
from experiments.benchmark import benchmark
from experiments.distribute import TaskQueue
from influence.logistic_regression import LogisticRegression
from tensorflow.contrib.learn.python.learn.datasets import base

import os
import time
import math
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from experiments.plot import *

from sklearn.cluster import KMeans
import sklearn
from tqdm import tqdm

@collect_phases
class CreditAssignment(Experiment):
    def __init__(self, config, out_dir=None):
        super(CreditAssignment, self).__init__(config, out_dir)
        self.datasets = ds.loader.load_dataset(**self.config['dataset_config'])
        self.dataset_id = self.config['dataset_config']['dataset_id']
        self.data_dir = self.config['dataset_config']['data_dir']

        self.train = self.datasets.train
        print("Shape of training set: {}".format(self.train.x.shape))
        self.test = self.datasets.test
        self.validation = self.datasets.validation
        self.sample_weights = ds.loader.load_supplemental_info(self.dataset_id + '_weights',
                data_dir=self.data_dir)\
                if self.config['sample_weights'] else [np.ones(self.train.x.shape[0]),
                        np.ones(self.validation.x.shape[0]),
                        np.ones(self.test.x.shape[0])]

        self.num_train = self.train.num_examples

        model_dir = os.path.join(self.base_dir, 'models')
        model_config = LogisticRegression.default_config()
        model_config['arch'] = LogisticRegression.infer_arch(self.train)
        model_config['arch']['fit_intercept'] = True 

        # Heuristic for determining maximum batch evaluation sizes without OOM
        D = model_config['arch']['input_dim'] * model_config['arch']['num_classes']
        if 'grad_batch_size' in self.config and self.config['grad_batch_size'] is not None:
            model_config['grad_batch_size'] = self.config['grad_batch_size']
        else:
            model_config['grad_batch_size'] =  max(1, self.config['max_memory'] // D)
        if 'hessian_batch_size' in self.config and self.config['hessian_batch_size'] is not None:
            model_config['hessian_batch_size'] = self.config['hessian_batch_size']
        else:
            model_config['hessian_batch_size'] = max(1, self.config['max_memory'] // (D * D))

        self.model_dir = model_dir
        self.model_config = model_config

        # Convenience member variables
        self.num_classes = self.model_config['arch']['num_classes']
        self.nonfires = ds.loader.load_supplemental_info(self.dataset_id + '_nonfires',
                data_dir=self.data_dir)

        def print_class_balance(ds, name):
            print("Dataset {}:".format(name))
            for i, val in enumerate(np.bincount(ds.labels)/ds.labels.shape[0]):
                print("Class {} is {} of the dataset.".format(i, val))

        print_class_balance(self.train, 'train')
        print_class_balance(self.test, 'test')
        print_class_balance(self.nonfires, 'nonfires')

        self.task_queue = TaskQueue(os.path.join(self.base_dir, 'tasks'))
        self.task_queue.define_task('compute_all_and_fixed_test_and_nonfire_influence',\
                self.compute_all_and_fixed_test_and_nonfire_influence)
        self.task_queue.define_task('retrain_subsets', self.retrain_subsets)
        self.task_queue.define_task('self_pred_infl', self.self_pred_infl)

    experiment_id = "credit_assignment"

    @property
    def run_id(self):
        return "{}_sample_weights-{}".format(
            self.dataset_id,
            self.config['sample_weights'])

    def get_model(self):
        if not hasattr(self, 'model'):
            self.model = LogisticRegression(self.model_config, self.model_dir)
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
        res = dict()
        l2_reg = self.R['cv_l2_reg']

        with benchmark("Training original model"):
            model.fit(self.train, l2_reg=l2_reg, sample_weights=self.sample_weights[0])
            model.print_model_eval(self.datasets, train_sample_weights=self.sample_weights[0], test_sample_weights=self.sample_weights[2], l2_reg=l2_reg)
            model.save('initial')

        res['initial_train_losses'] = model.get_indiv_loss(self.train)
        res['initial_train_accuracy'] = model.get_accuracy(self.train)
        res['initial_test_losses'] = model.get_indiv_loss(self.test)
        res['initial_test_accuracy'] = model.get_accuracy(self.test)
        res['initial_nonfires_losses'] = model.get_indiv_loss(self.nonfires)
        res['initial_nonfires_accuracy'] = model.get_accuracy(self.nonfires)
        if self.num_classes == 2:
            res['initial_train_margins'] = model.get_indiv_margin(self.train)
            res['initial_test_margins'] = model.get_indiv_margin(self.test)
            res['initial_nonfires_margins'] = model.get_indiv_margin(self.nonfires)

        return res

    @phase(2)
    def pick_test_points(self):
        res = dict()

        # Freeze each set after the first run
        if self.dataset_id == "mnli":
            fixed_test = [8487, 3448, 3156, 1127, 4218, 6907]
        else:
            test_losses = self.R['initial_test_losses']
            argsort = np.argsort(test_losses)
            high_loss = argsort[-3:] # Pick 3 high loss points
            random_loss = np.random.choice(argsort[:-3], 3, replace=False) # Pick 3 random points

            fixed_test = list(high_loss) + list(random_loss)

        if self.dataset_id in ["mnli"]:
            res['fixed_nonfires'] = [1722, 2734, 9467, 7378, 9448, 2838]
            print("Fixed nonfires points: {}".format(res['fixed_nonfires']))

        print("Fixed test points: {}".format(fixed_test))
        res['fixed_test'] = fixed_test
        return res

    @phase(3)
    def hessian(self):
        res = dict()

        if self.config['inverse_hvp_method'] == 'explicit':
            model = self.get_model()
            model.load('initial')
            l2_reg = self.R['cv_l2_reg']
            with benchmark("Computing hessian"):
                res['hessian'] = hessian = model.get_hessian(self.train, l2_reg=l2_reg, sample_weights=self.sample_weights[0])
        elif self.config['inverse_hvp_method'] == 'cg':
            print("Not computing explicit hessian.")
            res['hessian'] = None

        return res

    def get_turker_subsets(self):
        if self.dataset_id in ['mnli']:
            from datasets.loader import load_supplemental_info
            turk_IDs = load_supplemental_info(self.dataset_id + '_ids', data_dir=self.data_dir)
            subsets = []
            uniq, inds = np.unique(turk_IDs[0], return_inverse=True)
            for i in range(uniq.shape[0]):
                subsets.append(np.where(inds == i)[0])
            return uniq, subsets
        return [], []

    def get_genre_subsets(self, split=0):
        if self.dataset_id in ['mnli']:
            from datasets.loader import load_supplemental_info
            genre_IDs = load_supplemental_info(self.dataset_id + '_genres', data_dir=self.data_dir)
            subsets = []
            uniq, inds = np.unique(genre_IDs[split], return_inverse=True)
            for i in range(uniq.shape[0]):
                subsets.append(np.where(inds == i)[0])
            return uniq, subsets
        return [], []

    def get_nonfires_genre_subsets(self):
        if self.dataset_id in ['mnli']:
            from datasets.loader import load_supplemental_info
            genre_IDs = load_supplemental_info(self.dataset_id + '_nonfires_genres', data_dir=self.data_dir)
            subsets = []
            uniq, inds = np.unique(genre_IDs, return_inverse=True)
            for i in range(uniq.shape[0]):
                subsets.append(np.where(inds == i)[0])
            return uniq, subsets
        return [], []

    @phase(4)
    def pick_subsets(self):
        tagged_subsets = []
        res = dict()

        with benchmark("Turker ID subsets"):
            names, inds = self.get_turker_subsets()
            for name, ind in zip(names, inds):
                tagged_subsets.append(('same_turker-{}'.format(name), ind))

        with benchmark("Genre subset"):
            names, inds = self.get_genre_subsets()
            for name, ind in zip(names, inds):
                tagged_subsets.append(('same_genre-{}'.format(name), ind))

        subset_tags = [tag for tag, subset in tagged_subsets]
        subset_indices = [subset for tag, subset in tagged_subsets]

        res['test_genres'], res['test_genre_inds'] = self.get_genre_subsets(split=2)
        res['nonfires_genres'], res['nonfires_genre_inds'] = self.get_nonfires_genre_subsets()

        res['subset_tags'], res['subset_indices'] = subset_tags, subset_indices

        return res

    def compute_all_and_fixed_test_and_nonfire_influence(self, subset_start, subset_end):
        print("Computing pred infl for subsets {}-{}".format(subset_start, subset_end))
        self.load_phases([1,2,3,4])

        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        res['nonfires_predictions'] = model.get_predictions(self.nonfires.x)
        hessian = self.R['hessian']
        inverse_hvp_args = {
                'hessian_reg': hessian,
                'dataset': self.train,
                'l2_reg': l2_reg,
                'verbose': False,
                'verbose_cg': True,
                'inverse_vp_method': self.config['inverse_vp_method'],
                'inverse_hvp_method': self.config['inverse_hvp_method'],
        }
                
        def compute_test_like_infl(points, grad_fn, train_grad, **kwargs):
            test_grad = grad_fn(points, **kwargs).reshape(-1,1)
            test_grad_H_inv = model.get_inverse_hvp(test_grad, **inverse_hvp_args).reshape(-1)
            pred_infl = np.dot(train_grad, test_grad_H_inv)
            return np.array(pred_infl)

        fixed_test = self.R['fixed_test']
        fixed_nonfires = self.R['fixed_nonfires']

        subset_indices = self.R['subset_indices']
        subset_tags = self.R['subset_tags']

        test_genres, test_genre_inds = self.R['test_genres'], self.R['test_genre_inds']
        nonfires_genres, nonfires_genre_inds = self.R['nonfires_genres'], self.R['nonfires_genre_inds']

        def compute_infls(infl_name, infl_total_fn, infl_indiv_fn):
            for i, remove_indices in enumerate(subset_indices[subset_start:subset_end], subset_start):
                print('Computing influences on model for subset {} out of {} (tag={})'.format(i, len(subset_indices), subset_tags[i]))
                tag = subset_tags[i]
                inds = remove_indices
                with benchmark('Computing {} for subset {}'.format(infl_name, tag)):
                    grad = infl_total_fn(self.train.subset(inds),
                            sample_weights=self.sample_weights[0][inds], l2_reg=0)
                    add(res, 'subset_train_grad_for_{}'.format(infl_name), grad)

                    # For big parts of the datasets
                    datasets = [self.train, self.test, self.nonfires]
                    weights = [self.sample_weights[0], self.sample_weights[2],\
                            np.ones(self.nonfires.num_examples)]
                    dataset_names = ['train', 'test', 'nonfires']
                    class_names = ['class_{}'.format(i) for i in range(self.num_classes)]
                    for ds, ds_name, weight in zip(datasets, dataset_names, weights):
                        # all
                        name = 'all_{}_{}'.format(ds_name, infl_name)
                        with benchmark('Computing {}'.format(name)):
                            infl = compute_test_like_infl(ds, infl_total_fn, grad, sample_weights=weight)
                            add(res, name, infl)

                        # class-specific
                        for i, class_name in enumerate(class_names):
                            class_inds = np.where(ds.labels == i)[0]
                            name = '{}_{}_{}'.format(class_name, ds_name, infl_name)
                            with benchmark('Computing {}'.format(name)):
                                infl = compute_test_like_infl(ds.subset(class_inds), infl_total_fn,\
                                        grad, sample_weights=weight[class_inds])
                                add(res, name, infl)

                    # test/nonfires genres
                    for ds, ds_name, weight, genre_names, genre_inds in\
                            zip(datasets[1:], dataset_names[1:], weights[1:],\
                                [test_genres, nonfires_genres], [test_genre_inds, nonfires_genre_inds]):
                        for genre_name, genre_ind in zip(genre_names, genre_inds):
                            name = '{}_{}_{}'.format(genre_name, ds_name, infl_name)
                            with benchmark('Computing {}'.format(name)):
                                infl = compute_test_like_infl(ds.subset(genre_ind), infl_total_fn,\
                                        grad, sample_weights=weight[genre_ind])
                                add(res, name, infl)
                
                    # For a few specific points
                    specific_names = ['fixed_test', 'fixed_nonfires']
                    for name, fixed_inds, ds in zip(specific_names, [fixed_test, fixed_nonfires],\
                            [self.test, self.nonfires]):
                        data = []
                        for ind in fixed_inds:
                            with benchmark('Computing {} {}'.format(name, ind)):
                                data.append(compute_test_like_infl(ds.subset([ind]), infl_indiv_fn, grad))
                        add(res, '{}_{}'.format(name, infl_name), data)

        compute_infls('pred_infl', model.get_total_grad_loss, model.get_indiv_grad_loss)
        if self.num_classes == 2:
            compute_infls('pred_margin_infl', mode.get_total_grad_margin, model.get_indiv_grad_margin)

        for key, val in res.items():
            res[key] = np.array(val)

        return res

    @phase(5)
    def all_and_fixed_test_and_nonfire_influence(self):
        num_subsets = len(self.R['subset_indices'])
        subsets_per_batch = 24
        results = self.task_queue.execute('compute_all_and_fixed_test_and_nonfire_influence', [
            (i, min(i + subsets_per_batch, num_subsets))
            for i in range(0, num_subsets, subsets_per_batch)])

        return self.task_queue.collate_results(results)

    
    def retrain_subsets(self, subset_start, subset_end):
        print("Retraining subsets {}-{}".format(subset_start, subset_end))
        self.load_phases([1, 4])

        model = self.get_model()
        model.load('initial')
        l2_reg = self.R['cv_l2_reg']
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']

        start_time = time.time()
        train_losses, test_losses, nonfires_losses = [], [], []
        train_margins, test_margins, nonfires_margins = [], [], []
        for i, remove_indices in enumerate(subset_indices[subset_start:subset_end], subset_start):
            print('Retraining model for subset {} out of {} (tag={})'.format(i, len(subset_indices), subset_tags[i]))
            inds = [j for j in range(self.num_train) if j not in remove_indices]

            model.warm_fit(self.train.subset(inds), l2_reg=l2_reg)
            model.save('subset_{}'.format(i))
            train_losses.append(model.get_indiv_loss(self.train))
            test_losses.append(model.get_indiv_loss(self.test))
            nonfires_losses.append(model.get_indiv_loss(self.nonfires))
            if model.num_classes == 2:
                train_margins.append(model.get_indiv_margin(self.train))
                test_margins.append(model.get_indiv_margin(self.test))
                nonfires_margins.append(model.get_indiv_margin(self.nonfires))

        cur_time = time.time()
        time_per_retrain = (cur_time - start_time) / (subset_end - subset_start)
        remaining_time = time_per_retrain * (len(subset_indices) - subset_end)
        print('Each retraining takes {} s, {} s remaining'.format(time_per_retrain, remaining_time))

        res['subset_train_losses'] = np.array(train_losses)
        res['subset_test_losses'] = np.array(test_losses)
        res['subset_nonfires_losses'] = np.array(nonfires_losses)

        if self.num_classes == 2:
            res['subset_train_margins'] = np.array(train_margins)
            res['subset_test_margins'] = np.array(test_margins)
            res['subset_nonfires_margins'] = np.array(nonfires_margins)

        return res

    @phase(6)
    def retrain(self):
        num_subsets = len(self.R['subset_indices'])
        subsets_per_batch = 24
        results = self.task_queue.execute('retrain_subsets', [
            (i, min(i + subsets_per_batch, num_subsets))
            for i in range(0, num_subsets, subsets_per_batch)])

        return self.task_queue.collate_results(results)

    def self_pred_infl(self, subset_start, subset_end):
        self.load_phases([1, 3, 4, 5])

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

        subset_train_grads = self.R['subset_train_grad_for_pred_infl']
        if self.num_classes==2:
            subset_train_margin_grads = self.R['subset_train_grad_for_pred_margin_infl']

        start_time = time.time()
        subset_pred_dparam = []
        self_pred_infls = []
        self_pred_margin_infls = []
        for i, remove_indices in enumerate(subset_indices[subset_start:subset_end], subset_start):
            print('Computing self-influences for subset {} out of {} (tag={})'.format(i, len(subset_indices), subset_tags[i]))
            grad_loss = subset_train_grads[i]
            H_inv_grad_loss = model.get_inverse_hvp(grad_loss.reshape(1, -1).T, **inverse_hvp_args).reshape(-1)
            pred_infl = np.dot(grad_loss, H_inv_grad_loss)
            subset_pred_dparam.append(H_inv_grad_loss)
            self_pred_infls.append(pred_infl)

            if model.num_classes == 2:
                grad_margin = subset_train_margin_grads[i]
                pred_margin_infl = np.dot(grad_margin, H_inv_grad_loss)
                self_pred_margin_infls.append(pred_margin_infl)

        cur_time = time.time()
        time_per_vp = (cur_time - start_time) / (subset_end - subset_start)
        remaining_time = time_per_vp * (len(subset_indices) - subset_end)
        print('Each self-influence calculation takes {} s, {} s remaining'.format(time_per_vp, remaining_time))

        res['subset_pred_dparam'] = np.array(subset_pred_dparam)
        res['self_pred_infl'] = np.array(self_pred_infls)
        if self.num_classes == 2:
            res['self_pred_margin_infl'] = np.array(self_pred_margin_infls)

        return res

    @phase(7)
    def compute_self_pred_infl(self):
        num_subsets = len(self.R['subset_indices'])
        subsets_per_batch = 24
        results = self.task_queue.execute('self_pred_infl', [
            (i, min(i + subsets_per_batch, num_subsets))
            for i in range(0, num_subsets, subsets_per_batch)])

        return self.task_queue.collate_results(results)

    @phase(8)
    def compute_actl_infl(self):
        self.task_queue.notify_exit()
        res = dict()

        subset_tags, subset_indices = self.R['subset_tags'], self.R['subset_indices']
        fixed_test, fixed_nonfires = self.R['fixed_test'], self.R['fixed_nonfires']
        test_genres, test_genre_inds = self.R['test_genres'], self.R['test_genre_inds']
        nonfires_genres, nonfires_genre_inds = self.R['nonfires_genres'], self.R['nonfires_genre_inds']

        def compute_infls(infl_name, initial_vals, subset_vals):
            datasets = [self.train, self.test, self.nonfires]
            weights = [self.sample_weights[0], self.sample_weights[2],\
                    np.ones(self.nonfires.num_examples)]
            dataset_names = ['train', 'test', 'nonfires']
            class_names = ['class_{}'.format(i) for i in range(self.num_classes)]
            for ds, ds_name, weight, initial_val, subset_val in\
                    zip(datasets, dataset_names, weights, initial_vals, subset_vals):
                # all
                name = 'all_{}_{}'.format(ds_name, infl_name)
                with benchmark('Computing {}'.format(name)):
                    infl = np.einsum('ai,i->a', subset_val-initial_val, weight)
                    res[name] = infl

                # class-specific
                for i, class_name in enumerate(class_names):
                    class_inds = np.where(ds.labels == i)[0]
                    name = '{}_{}_{}'.format(class_name, ds_name, infl_name)
                    with benchmark('Computing {}'.format(name)):
                        infl = np.einsum('ai,i->a', subset_val[:,class_inds]-initial_val[class_inds],\
                                weight[class_inds])
                        res[name] = infl

            # test/nonfires genres
            for ds, ds_name, weight, initial_val, subset_val, genre_names, genre_inds in\
                    zip(datasets[1:], dataset_names[1:], weights[1:],\
                        initial_vals[1:], subset_vals[1:],\
                        [test_genres, nonfires_genres], [test_genre_inds, nonfires_genre_inds]):
                for genre_name, genre_ind in zip(genre_names, genre_inds):
                    name = '{}_{}_{}'.format(genre_name, ds_name, infl_name)
                    with benchmark('Computing {}'.format(name)):
                        infl = np.einsum('ai,i->a', subset_val[:,genre_ind]-initial_val[genre_ind],\
                                weight[genre_ind])
                        res[name] = infl

            # For a few specific points
            specific_names = ['fixed_test', 'fixed_nonfires']
            for name, fixed_inds, initial_val, subset_val in\
                    zip(specific_names, [fixed_test, fixed_nonfires],\
                    initial_vals[1:], subset_vals[1:]):
                res_name = '{}_{}'.format(name, infl_name)
                for ind in fixed_inds:
                    with benchmark('Computing {} {}'.format(name, ind)):
                        infl = subset_val[:,ind] - initial_val[ind]
                        add(res, res_name, infl)
                res[res_name] = np.transpose(res[res_name])

            # self influence
            for subset_val, remove_indices in zip(subset_vals[0], subset_indices):
                infl = np.dot(subset_val[remove_indices] - initial_vals[0][remove_indices],\
                        weights[0][remove_indices])
                add(res, 'self_{}'.format(infl_name), infl)
            res['self_{}'.format(infl_name)] = np.transpose(res['self_{}'.format(infl_name)])

        dataset_names = ['train', 'test', 'nonfires']
        initial_losses = [self.R['initial_{}_losses'.format(ds_name)] for ds_name in dataset_names]
        subset_losses = [self.R['subset_{}_losses'.format(ds_name)] for ds_name in dataset_names]
        compute_infls('actl_infl', initial_losses, subset_losses)
        if self.num_classes == 2:
            initial_margins = [self.R['initial_{}_margins'.format(ds_name)] for ds_name in dataset_names]
            subset_margins = [self.R['subset_{}_margins'.format(ds_name)] for ds_name in dataset_names]
            compute_infls('actl_margin_infl', initial_margins, subset_margins)

        for key, val in res.items():
            res[key] = np.array(val)

        return res

    def get_simple_subset_tags(self):
        def simplify_tag(tag):
            if 'same_turker' in tag: return 'same_turker'
            return tag
        return map(simplify_tag, self.R['subset_tags'])

    def get_subtitle(self):
        subtitle='{}'.format(self.dataset_id)
        return subtitle

    def plot_influence(self, title, figname, actl_loss, pred_loss, actl_margin=None, pred_margin=None,
            verbose=True):
        subset_tags = self.get_simple_subset_tags()

        fig, ax = plt.subplots(1,1,figsize=(8,8),squeeze=False)
        plot_influence_correlation(ax[0][0],
                                   actl_loss,
                                   pred_loss,
                                   label=subset_tags,
                                   title='Group influence on '+title,
                                   subtitle=self.get_subtitle())
        fig.savefig(os.path.join(self.plot_dir, figname+'_loss.png'), bbox_inches='tight')
        plt.close(fig)

        if self.num_classes == 2:
            fig, ax = plt.subplots(1,1,figsize=(8,8),squeeze=False)
            plot_influence_correlation(ax[0][0],
                                       actl_margin,
                                       pred_margin,
                                       label=subset_tags,
                                       title='Group margin influence on '+title,
                                       subtitle=self.get_subtitle())
            fig.savefig(os.path.join(self.plot_dir, figname+'_margin.png'), bbox_inches='tight')
            plt.close(fig)

        if verbose: print('Finished plotting {} influence.'.format(title))

    @phase(9)
    def plot_all(self):
        print('Will save plots to {}'.format(self.plot_dir))
        mask = range(len(self.R['subset_tags'])-5)
        print(self.R['subset_tags'][mask])
        add = ''
        self.plot_influence('self', 'self-influence{}'.format(add),
                            self.R['self_actl_infl'][mask],
                            self.R['self_pred_infl'][mask])
        for i, test_idx in enumerate(self.R['fixed_test']):
            self.plot_influence('fixed test {}'.format(test_idx), 'fixed-test-{}{}'.format(test_idx, add),
                                self.R['fixed_test_actl_infl'][mask, i],
                                self.R['fixed_test_pred_infl'][mask, i])
        for i, nonfire_idx in enumerate(self.R['fixed_nonfires']):
            self.plot_influence('fixed nonfire {}'.format(i), 'fixed-nonfires-{}{}'.format(i, add),
                                self.R['fixed_nonfires_actl_infl'][mask,i],
                                self.R['fixed_nonfires_pred_infl'][mask,i])
        for i, genre_inds in enumerate(self.R['test_genre_inds']):
            name = self.R['test_genres'][i]
            self.plot_influence('test genre {}'.format(name), 'test-genre-{}{}'.format(name, add),
                                self.R['{}_test_actl_infl'.format(name)][mask],
                                self.R['{}_test_pred_infl'.format(name)][mask])
        for i, genre_inds in enumerate(self.R['nonfires_genre_inds']):
            name = self.R['nonfires_genres'][i]
            self.plot_influence('nonfire genre {}'.format(name), 'nonfires-genre-{}{}'.format(name, add),
                                self.R['{}_nonfires_actl_infl'.format(name)][mask],
                                self.R['{}_nonfires_pred_infl'.format(name)][mask])
        prefixes = ['all'] + ['class_{}'.format(i) for i in range(self.num_classes)]
        for prefix in prefixes:
            for ds_name in ['train', 'test', 'nonfires']:
                self.plot_influence('{} {} set'.format(prefix, ds_name), '{}-{}{}'.format(prefix, ds_name, add),
                        self.R['{}_{}_actl_infl'.format(prefix, ds_name)][mask],
                        self.R['{}_{}_pred_infl'.format(prefix, ds_name)][mask])

        return dict()
