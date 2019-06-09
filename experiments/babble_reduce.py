from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import datasets as ds
import datasets.loader
from datasets.common import DataSet
from experiments.common import Experiment, collect_phases, phase
from experiments.benchmark import benchmark
import influence.logistic_regression

import os
import time
import math
import numpy as np
import tensorflow as tf
from copy import copy

from sklearn.cluster import KMeans
import sklearn.linear_model
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix

@collect_phases
class BabbleReduce(Experiment):
    def __init__(self, config, out_dir=None):
        np.random.seed(0)
        super(BabbleReduce, self).__init__(config, out_dir)
        self.datasets = ds.loader.load_dataset(**self.config['dataset_config'])
        self.sample_weights = ds.loader.load_supplemental_info(self.config['dataset_config']['dataset_id'] + '_weights', data_dir=self.config['dataset_config']['data_dir']) if self.config['sample_weights'] else None
        self.nonfires = ds.loader.load_supplemental_info(self.config['dataset_config']['dataset_id'] + '_nonfires', data_dir=self.config['dataset_config']['data_dir'])
        self.train = self.datasets.train
        self.test = self.datasets.test
        self.validation = self.datasets.validation
        self.num_train = self.datasets.train.num_examples
        self.target_num_features = self.config['target_num_features'] # not actually used
        self.fit_intercept = self.config['fit_intercept']
        self.regularization_type = self.config['regularization_type']
        assert self.regularization_type in ['l1','l2']

    experiment_id = "babble_reduce_reruns" #TODO

    @property
    def run_id(self):
        return "{}_target_num_features-{}_regularization_type-{}_sample_weights-{}".format(
            self.config['dataset_config']['dataset_id'],
            self.config['target_num_features'],
            self.config['regularization_type'],
            self.config['sample_weights'])

    def get_model(self):
        if not hasattr(self, 'model'):
            self.model = sklearn.linear_model.LogisticRegression(penalty=self.regularization_type, fit_intercept=self.fit_intercept, random_state=np.random.RandomState(1))
        return self.model

    @phase(0)
    def try_regs(self):
        model = self.get_model()
        res = dict()

        reg_min, reg_max, reg_samples = self.config['initial_reg_range']
        reg_min *= self.num_train
        reg_max *= self.num_train

        regs = np.logspace(np.log10(reg_min), np.log10(reg_max), reg_samples)
        accs = [np.zeros_like(regs) for i in range(3)]
        coefs = [None for i in range(len(regs))]
        intercepts = np.zeros_like(regs)
        nonzeros = np.zeros_like(regs)
        f1s = [np.zeros_like(regs) for i in range(3)]

        for i, reg in enumerate(regs):
            with benchmark("Training model for reg={}".format(reg)):
                model.set_params(C=1.0/reg)
                model.fit(self.train.x, self.train.labels,
                        sample_weight=self.sample_weights[0])

                coefs[i] = model.coef_[0]
                print('Coefs: {}'.format(coefs[i]))
                intercepts[i] = model.intercept_ if self.fit_intercept else 0
                print('Intercept: {}'.format(intercepts[i]))
                nonzeros[i] = np.sum(coefs[i] != 0) + (intercepts[i] != 0)
                print('Nonzeros {}/{}'.format(nonzeros[i], coefs[i].size+1))

                accs[0][i] = model.score(self.train.x, self.train.labels,
                        sample_weight=self.sample_weights[0])
                accs[1][i] = model.score(self.validation.x, self.validation.labels,
                        sample_weight=self.sample_weights[1])
                accs[2][i] = model.score(self.test.x, self.test.labels,
                        sample_weight=self.sample_weights[2])
                print('Train acc: {}'.format(accs[0][i]))
                print('Validation acc: {}'.format(accs[1][i]))
                print('Test acc: {}'.format(accs[2][i]))

                y_preds = [None for j in range(3)]
                y_preds[0] = model.predict(self.train.x)
                y_preds[1] = model.predict(self.validation.x)
                y_preds[2] = model.predict(self.test.x)

                f1s[0][i] = f1_score(self.train.labels, y_preds[0],
                        sample_weight=self.sample_weights[0])
                f1s[1][i] = f1_score(self.validation.labels, y_preds[1],
                        sample_weight=self.sample_weights[1])
                f1s[2][i] = f1_score(self.test.labels, y_preds[2],
                        sample_weight=self.sample_weights[2])
                print('Train f1: {}'.format(f1s[0][i]))
                print('Validation f1: {}'.format(f1s[1][i]))
                print('Test f1: {}'.format(f1s[2][i]))
        
        best_i = np.argmax(f1s[1])
        print('Using reg {}'.format(regs[best_i]))
        res['best_i'] = best_i
        res['feature_inds'] = np.where(coefs[best_i] != 0)[0]
        res['regs'] = regs
        res['accs'] = np.array(accs)
        res['coefs'] = np.array(coefs)
        res['intercepts'] = intercepts
        res['nonzeros'] = nonzeros
        res['f1s'] = np.array(f1s)
        return res

    @phase(1)
    def create_reduced_matrices(self):
        res = dict()
        if (self.regularization_type != 'l2'):
            features = [self.train.x, self.validation.x, self.test.x]
            features = [csr_matrix(mat) for mat in features]
            reduced_features = [np.array(features[i][:,self.R['feature_inds']]) for i in range(3)]
            name = 'reduced_{}_x'.format(self.config['dataset_config']['dataset_id'])
            if self.config['sample_weights']: name += '_via_sample_weights'
            np.savez(os.path.join(self.out_dir, name),
                    reduced_x=reduced_features)
            res['reduced_features'] = reduced_features
            print('Using {} features, shape {}.'.format(len(self.R['feature_inds']), np.array(reduced_features).shape))

            features_nonfires = csr_matrix(self.nonfires.x)
            reduced_nonfires = np.array(features_nonfires[:, self.R['feature_inds']])
            name = 'reduced_{}_nonfires_x'.format(self.config['dataset_config']['dataset_id'])
            if self.config['sample_weights']: name += '_via_sample_weights'
            res['reduced_features_nonfires'] = reduced_nonfires
            np.savez(os.path.join(self.out_dir, name),
                    reduced_x=[reduced_nonfires])
            print('Saved reduced nonfires to {}'.format(os.path.join(self.out_dir, name)))
        return res

    @phase(2)
    def compare_to_l2(self):
        res = dict()
        if (self.regularization_type != 'l2'):
            config_copy = copy(self.config['dataset_config'])
            config_copy['dataset_id'] = 'reduced_' + config_copy['dataset_id']
            reduced_ds = ds.loader.load_dataset(**config_copy)
            self.train = reduced_ds.train
            self.validation = reduced_ds.validation
            self.test = reduced_ds.test

            reg_min, reg_max, reg_samples = self.config['normalized_cross_validation_range']
            reg_min *= self.num_train
            reg_max *= self.num_train
            regs = np.logspace(np.log10(reg_min), np.log10(reg_max), reg_samples)

            # We don't do true cross validation; just use validation set.
            cv_accs = [np.zeros_like(regs) for i in range(3)]
            cv_coefs = [None for i in range(len(regs))]
            cv_intercepts = np.zeros_like(regs)
            cv_f1s = [np.zeros_like(regs) for i in range(3)]

            for i, reg in enumerate(regs):
                with benchmark("Evaluating sklearn l2 model for reg={}".format(reg)):
                    np.random.seed(0)
                    l2_model = sklearn.linear_model.LogisticRegression(penalty='l2', fit_intercept=self.fit_intercept, random_state=np.random.RandomState(2)) #TODO
                    l2_model.set_params(C=1.0/reg)

                    l2_model.fit(self.train.x, self.train.labels,
                            sample_weight=self.sample_weights[0])

                    cv_coefs[i] = l2_model.coef_[0]
                    #print('Coefs: {}'.format(cv_coefs[i]))
                    cv_intercepts[i] = l2_model.intercept_ if self.fit_intercept else 0
                    print('Intercept: {}'.format(cv_intercepts[i]))

                    cv_accs[0][i] = l2_model.score(self.train.x, self.train.labels,
                            sample_weight=self.sample_weights[0])
                    cv_accs[1][i] = l2_model.score(self.validation.x, self.validation.labels,
                            sample_weight=self.sample_weights[1])
                    cv_accs[2][i] = l2_model.score(self.test.x, self.test.labels,
                            sample_weight=self.sample_weights[2])
                    print('Train acc: {}'.format(cv_accs[0][i]))
                    print('Validation acc: {}'.format(cv_accs[1][i]))
                    print('Test acc: {}'.format(cv_accs[2][i]))

                    y_preds = [None for j in range(3)]
                    y_preds[0] = l2_model.predict(self.train.x)
                    y_preds[1] = l2_model.predict(self.validation.x)
                    y_preds[2] = l2_model.predict(self.test.x)

                    cv_f1s[0][i] = f1_score(self.train.labels, y_preds[0],
                            sample_weight=self.sample_weights[0])
                    cv_f1s[1][i] = f1_score(self.validation.labels, y_preds[1],
                            sample_weight=self.sample_weights[1])
                    cv_f1s[2][i] = f1_score(self.test.labels, y_preds[2],
                            sample_weight=self.sample_weights[2])
                    print('Train f1: {}'.format(cv_f1s[0][i]))
                    print('Validation f1: {}'.format(cv_f1s[1][i]))
                    print('Test f1: {}'.format(cv_f1s[2][i]))
                    print(f1_score(self.test.labels, np.array(l2_model.predict(self.test.x)>0.5).astype(np.int), sample_weight=self.sample_weights[2]))

                    if reg == 2.4177:
                        print(self.train.x.shape, reg, self.sample_weights[0])
                        print(l2_model.get_params()) #TODO
                        print(l2_model.coef_, l2_model.intercept_)
                        return



            best_CV_i = np.argmax(cv_f1s[1])
            best_l2_reg = regs[best_CV_i]
            cv_accs = np.array(cv_accs)
            cv_f1s = np.array(cv_f1s)
            print('Selecting l2 regularization {}, with accs {} and f1s {}.'.format(best_l2_reg, cv_accs[:, best_CV_i], cv_f1s[:,best_CV_i]))
            #print('L2 coefs {}'. format(cv_coefs[best_CV_i]))

            best_i = self.R['best_i']
            feature_indices = self.R['feature_inds']
            l1_coefs = self.R['coefs'][best_i][feature_indices]
            l1_accs = self.R['accs'][:,best_i]
            l1_f1s = self.R['f1s'][:,best_i]
            #print('Nonzero coeffs under l1 were {}.'.format(l1_coefs))
            print('L1 model had accs {} and f1s {}.'.format(l1_accs, l1_f1s))

            res['cv_best_i'] = best_CV_i
            res['cv_regs'] = regs
            res['cv_accs'] = np.array(cv_accs)
            res['cv_f1s'] = np.array(cv_f1s)
            res['cv_l2_reg'] = best_l2_reg
            res['cv_coefs'] = np.array(cv_coefs)
            res['cv_intercepts'] = cv_intercepts
    
        return res
