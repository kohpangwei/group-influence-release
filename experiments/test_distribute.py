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

import os
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.linalg

@collect_phases
class TestDistribute(Experiment):
    """
    Test the TaskQueue
    """
    def __init__(self, config, out_dir=None):
        super(TestDistribute, self).__init__(config, out_dir)
        task_dir = os.path.join(self.base_dir, 'tasks')
        self.task_queue = TaskQueue(task_dir)
        self.task_queue.define_task('is_prime', self.is_prime)
        self.task_queue.define_task('random_vector', self.random_vector)

    experiment_id = "test_distribute"

    @property
    def run_id(self):
        return "test"

    @phase(0)
    def initialize(self):
        return { 'max_prime': 200 }

    def is_prime(self, n):
        for i in range(2, n):
            if i * i > n: return True
            if n % i == 0: return False
        return True

    @phase(1)
    def count_primes(self):
        results = self.task_queue.execute('is_prime', [
            (n,) for n in range(1, self.R['max_prime'] + 1)])
        return { 'num_primes': sum(results) }

    def random_vector(self, seed_start, seed_end):
        vectors, scalars = [], []
        for seed in range(seed_start, seed_end):
            rng = np.random.RandomState(seed)
            vector = rng.normal(0, 1, (30,))
            scalar = rng.normal(0, 1)
            vectors.append(vector)
            scalars.append(scalar)

        res = dict()
        res['vector'] = np.array(vectors)
        res['scalar'] = np.array(scalars)
        import time
        time.sleep(1)
        return res

    @phase(2)
    def make_random_vectors(self):
        num_seeds = 200
        seeds_per_batch = 4
        results = self.task_queue.execute('random_vector', [
            (i, min(i + seeds_per_batch, num_seeds))
            for i in range(0, num_seeds, seeds_per_batch)])
        self.task_queue.notify_exit()

        return self.task_queue.collate_results(results)
