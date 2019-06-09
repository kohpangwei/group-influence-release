from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from influence.logistic_regression import LogisticRegression
import datasets as ds
import datasets.loader
import datasets.mnist
from datasets.common import DataSet
from experiments.common import Experiment, collect_phases, phase
from experiments.benchmark import benchmark

import os
import time
import numpy as np
import sklearn
import sklearn.linear_model

@collect_phases
class TestLogreg(Experiment):
    """
    Test the LogisticRegression model's functionality.
    """
    def __init__(self, config, out_dir=None):
        super(TestLogreg, self).__init__(config, out_dir)
        self.datasets = ds.loader.load_dataset(**self.config['dataset_config'])

        model_dir = os.path.join(self.base_dir, 'models')
        model_config = LogisticRegression.default_config()
        model_config['arch'] = LogisticRegression.infer_arch(self.datasets.train)
        model_config['arch']['fit_intercept'] = self.config['fit_intercept']
        self.model_dir = model_dir
        self.model_config = model_config

        MAX_MEMORY = int(1e7)
        D = model_config['arch']['input_dim'] * model_config['arch']['num_classes']
        self.eval_args = {
            'grad_batch_size': max(1, MAX_MEMORY // D),
            'hess_batch_size': max(1, MAX_MEMORY // (D * D)),
        }

    experiment_id = "test_logreg"

    @property
    def run_id(self):
        return "{}-{}".format(self.config['dataset_config']['dataset_id'],
                              self.config['fit_intercept'])

    def get_model(self):
        if not hasattr(self, 'model'):
            self.model = LogisticRegression(self.model_config, self.model_dir)
        return self.model

    @phase(0)
    def train_model(self):
        results = dict()

        model = self.get_model()
        l2_reg = self.config['l2_reg']

        with benchmark("Training the model"):
            model.fit(self.datasets.train, l2_reg=l2_reg)
            model.print_model_eval(self.datasets, l2_reg=l2_reg)

        with benchmark("Computing losses"):
            results['train_loss'] = model.get_total_loss(self.datasets.train, l2_reg=l2_reg)
            results['indiv_train_loss'] = model.get_indiv_loss(self.datasets.train)
            results['test_loss'] = model.get_total_loss(self.datasets.test, l2_reg=l2_reg)
            results['indiv_test_loss'] = model.get_indiv_loss(self.datasets.test)

        with benchmark("Saving model"):
            model.save('initial')

        return results

    @phase(1)
    def retrain_model(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.config['l2_reg']

        print("Sanity check: reloading the model gives same train and test losses")
        indiv_train_loss = model.get_indiv_loss(self.datasets.train)
        indiv_test_loss = model.get_indiv_loss(self.datasets.test)
        print("train loss l2 diff: {}, test loss l2 diff: {}".format(
            np.linalg.norm(indiv_train_loss - self.results['train_model']['indiv_train_loss']),
            np.linalg.norm(indiv_test_loss - self.results['train_model']['indiv_test_loss'])))

        print("Sanity check: warm fit is fast")
        with benchmark("Performing warm fit"):
            model.warm_fit(self.datasets.train, l2_reg=l2_reg)
            model.print_model_eval(self.datasets, l2_reg=l2_reg)

        print("Sanity check: force-recreate the model and load it")
        del self.model
        model = self.get_model()
        model.load('initial')

        print("Sanity check: losses should still be the same")
        indiv_train_loss = model.get_indiv_loss(self.datasets.train)
        indiv_test_loss = model.get_indiv_loss(self.datasets.test)
        print("train loss l2 diff: {}, test loss l2 diff: {}".format(
            np.linalg.norm(indiv_train_loss - self.results['train_model']['indiv_train_loss']),
            np.linalg.norm(indiv_test_loss - self.results['train_model']['indiv_test_loss'])))

        return {}

    @phase(2)
    def compute_grad_loss(self):
        model = self.get_model()
        model.load('initial')

        result = dict()

        with benchmark("Computing gradients individually"):
            result['indiv_grad_loss'] = model.get_indiv_grad_loss(self.datasets.train,
                                                                  method='from_total_grad')

        with benchmark("Computing gradients batched"):
            result['batch_indiv_grad_loss'] = model.get_indiv_grad_loss(self.datasets.train,
                                                                        method="batched")

        grad_loss_1 = result['indiv_grad_loss']
        grad_loss_2 = result['batch_indiv_grad_loss']
        print("l2 between gradients: {}".format(
            np.linalg.norm(grad_loss_1 - grad_loss_2)))

        return result

    @phase(3)
    def compare_sklearn(self):
        model = self.get_model()
        model.load('initial')
        l2_reg = self.config['l2_reg']

        with benchmark("Copying params to sklearn"):
            C = 1.0 / l2_reg
            multi_class = "ovr" if self.model_config['arch']['num_classes'] == 2 else "multinomial"
            fit_intercept = self.model_config['arch']['fit_intercept']
            sklearn_model = sklearn.linear_model.LogisticRegression(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                solver='lbfgs',
                multi_class=multi_class,
                warm_start=False,
                max_iter=2048)
            sklearn_model.intercept_ = 0
            model.copy_params_to_sklearn_model(sklearn_model)

        preds = model.get_predictions(self.datasets.train.x)
        preds_sk = sklearn_model.predict_proba(self.datasets.train.x)
        print("l2 between predictions: {}".format(
            np.linalg.norm(preds - preds_sk)))

        result = dict()
        result['preds'] = preds
        result['preds_sk'] = preds_sk
        return result

    @phase(4)
    def hess(self):
        result = dict()
        model = self.get_model()
        model.load('initial')
        l2_reg = self.config['l2_reg']

        with benchmark("Computing hessian"):
            result['hessian_reg'] = model.get_hessian(self.datasets.train,
                                                      l2_reg=l2_reg,
                                                      **self.eval_args)

        if result['hessian_reg'].shape[0] < 800:
            with benchmark("Finding hessian eigenvalues"):
                result['eigs'] = eigs = np.linalg.eigvalsh(result['hessian_reg'])

            print("Hessian eigenvalue range:", np.min(eigs), np.max(eigs))

        return result

    @phase(5)
    def hvp(self):
        model = self.get_model()
        model.load('initial')

        result = dict()
        indiv_grad_loss = self.results['compute_grad_loss']['indiv_grad_loss']
        some_indices = [1, 6, 2, 4, 3]
        vectors = indiv_grad_loss[some_indices, :].T

        hessian = self.results['hess']['hessian_reg']

        with benchmark("Inverse HVP"):
            result['inverse_hvp'] = model.get_inverse_vp(hessian, vectors,
                                                         **self.eval_args)

        return result

    @phase(6)
    def margins(self):
        result = dict()
        model = self.get_model()
        model.load('initial')
        l2_reg = self.config['l2_reg']

        if model.num_classes == 2:
            print("Model is binary, we can compute margins.")

            with benchmark("Computing margins"):
                indiv_margin = model.get_indiv_margin(self.datasets.train)

            s = np.zeros(self.datasets.train.num_examples)
            some_indices = [35, 6, 1, 8, 42]
            s[some_indices] = 1

            with benchmark("Computing margin gradients"):
                grad_margin = model.get_total_grad_margin(self.datasets.train, s, l2_reg=l2_reg)

            result['indiv_margin'] = indiv_margin
            result['grad_margin'] = grad_margin

        return result
