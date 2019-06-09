from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import time

class Benchmark(object):
    """
    Helper ContextManager that records the time between
    the beginning and end of the context
    """
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        print("{}...".format(self.task))
        self.start = time.time()

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.end = time.time()
        if ex_type is not None:
            return False
        print("{} took {} seconds.".format(self.task, self.end - self.start))

def benchmark(task):
    return Benchmark(task)
