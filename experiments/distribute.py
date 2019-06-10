from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import time
import pickle
import numpy as np
import uuid

class TaskQueue(object):
    def __init__(self, task_dir,
                 claim_timeout=2 * 60 * 60,
                 master_only=False):
        self.task_dir = task_dir
        if not os.path.exists(task_dir):
            os.makedirs(self.task_dir)
        self.task_by_id = dict()

        self.tasks = []
        self.num_tasks_by_id = dict()
        self.claim_timeout = claim_timeout
        self.master_only = master_only

        self.uuid = str(uuid.uuid4())

    def define_task(self, task_id, task_func):
        if task_id in self.task_by_id:
            raise ValueError('Task {} has already been defined'.format(task_id))
        self.task_by_id[task_id] = task_func
        self.num_tasks_by_id[task_id] = 0

    @property
    def tasks_path(self):
        return os.path.join(self.task_dir, 'all_tasks.pickle')

    @property
    def exit_path(self):
        return os.path.join(self.task_dir, 'exit')

    def save_all_tasks(self):
        data = { 'tasks': self.tasks, 'num_tasks_by_id': self.num_tasks_by_id }
        with open(self.tasks_path, 'wb') as f:
            pickle.dump(data, f)

    def load_all_tasks(self):
        if not os.path.exists(self.tasks_path):
            self.tasks = []
            self.num_tasks_by_id = dict()
            return

        with open(self.tasks_path, 'rb') as f:
            data = pickle.load(f)
        self.tasks = data['tasks']
        self.num_tasks_by_id = data['num_tasks_by_id']

    def purge_claims(self, tasks, force_refresh=False):
        # Purge all claims with no results, or purge all claims
        # if doing a refresh
        for task_id, index, args in tasks:
            claim_path = self.task_claim_path(task_id, index)
            result_path = self.task_result_path(task_id, index)
            if force_refresh and os.path.exists(result_path):
                os.remove(result_path)
            if os.path.exists(claim_path) and not os.path.exists(result_path):
                os.remove(claim_path)

    def execute_eager(self, task_id, task_args):
        results = []
        for args in task_args:
            index = self.num_tasks_by_id[task_id]
            self.num_tasks_by_id[task_id] += 1

            print("Claimed task {}_{}.".format(task_id, index))
            task_func = self.task_by_id[task_id]
            result = task_func(*args)
            print("Completed task {}_{}.".format(task_id, index))
            results.append(result)

        return results

    def execute(self, task_id, task_args, force_refresh=False):
        if self.master_only:
            return self.execute_eager(task_id, task_args)

        current_tasks = []
        for args in task_args:
            index = self.num_tasks_by_id[task_id]
            self.tasks.append((task_id, index, args))
            current_tasks.append((task_id, index, args))
            self.num_tasks_by_id[task_id] += 1
        self.save_all_tasks()
        self.purge_claims(current_tasks, force_refresh=force_refresh)

        while True:
            # Do some work as the master too
            self.work()

            # Wait for all tasks to be done
            all_done = all(self.task_is_complete(task_id, index)
                           for task_id, index, args in current_tasks)
            if all_done:
                break

            # Time out tasks if necessary, then wait
            self.timeout_tasks()
            time.sleep(5)

        results = [self.get_task_result(task_id, index)
                   for task_id, index, args in current_tasks]
        return results

    def work(self):
        self.load_all_tasks()

        while True:
            unclaimed_task = self.find_unclaimed_task()
            if unclaimed_task is None:
                break
            task_id, index, args = unclaimed_task
            if not self.claim_task(task_id, index):
                continue

            print("Claimed task {}_{}.".format(task_id, index))
            task_func = self.task_by_id[task_id]
            result = task_func(*args)
            self.complete_task(task_id, index, result)
            print("Completed task {}_{}.".format(task_id, index))

    def task_claim_path(self, task_id, index):
        return os.path.join(self.task_dir, '{}_{}_claim'.format(task_id, index))

    def task_result_path(self, task_id, index):
        return os.path.join(self.task_dir, '{}_{}_result'.format(task_id, index))

    def find_unclaimed_task(self):
        for task_id, index, args in self.tasks:
            path = self.task_claim_path(task_id, index)
            if not os.path.exists(path):
                return task_id, index, args
        return None

    def timeout_tasks(self):
        # Check if any tasks have timed out
        for task_id, index, args in self.tasks:
            claim_path = self.task_claim_path(task_id, index)
            result_path = self.task_result_path(task_id, index)
            if os.path.exists(claim_path) and not os.path.exists(result_path):
                with open(claim_path, 'r') as f:
                    data = f.read().split()
                old_time = float(data[1])
                cur_time = time.time()
                if cur_time - old_time > self.claim_timeout:
                    print('Task {}_{} timed out. Removing claim.'.format(task_id, index))
                    os.remove(claim_path)

    def claim_task(self, task_id, index):
        path = self.task_claim_path(task_id, index)
        if os.path.exists(path):
            return False
        with open(path, 'w') as f:
            f.write("{} {}\n".format(self.uuid, time.time()))
        time.sleep(2)
        with open(path, 'r') as f:
            data = f.read().split()[0]
        return data == self.uuid

    def complete_task(self, task_id, index, result):
        path = self.task_result_path(task_id, index)
        with open(path, 'wb') as f:
            pickle.dump(result, f)

    def task_is_complete(self, task_id, index):
        path = self.task_result_path(task_id, index)
        return os.path.exists(path)

    def get_task_result(self, task_id, index):
        path = self.task_result_path(task_id, index)
        if not os.path.exists(path):
            raise ValueError('Task {}_{} is not complete yet'.format(task_id, index))
        with open(path, 'rb') as f:
            result = pickle.load(f)
        return result

    def run_worker(self):
        while True:
            time.sleep(1)
            if os.path.exists(self.exit_path):
                break
            self.work()

    def notify_exit(self):
        with open(self.exit_path, 'w') as f:
            f.write('done')

    def collate_results(self, results):
        keys = set(results[0].keys())
        all_keys_same = all([set(result.keys()) == keys for result in results])
        if not all_keys_same:
            raise ValueError("Could not collate results, not all keys are equal.")

        collated_results = dict()
        for key in keys:
            shape = results[0][key].shape
            if len(shape) == 1:
                collated_results[key] = np.hstack([result[key] for result in results])
            else:
                collated_results[key] = np.vstack([result[key] for result in results])

        return collated_results
