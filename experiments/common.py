from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import time
import numpy as np
import pickle
from collections import namedtuple

"""
The default output directory (../../output/)
"""
DEFAULT_OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '..', 'output'))

def add(d, key, val):
    if key in d:
        d[key].append(val)
    else:
        d[key] = [val]

"""
Metadata class for a phase of an experiment.
"""
ExperimentPhase = namedtuple('Phase', ['name', 'index', 'deps', 'instancemethod'])

def phase(index, dep_names=None):
    """
    Decorates an instancemethod of an Experiment class to denote
    that it is an experiment phase. The tags will be used by the class
    decorator `collect_phases' to register phases as a class variable.

    :param index: The index used to order phases. Phases will be ordered
                  by index. Indices must be unique across phases, but indices
                  need not be contiguous.
    :param dep_names: A list of the names of the phases that this phase
                      depends on. This is used to figure out which phases
                      need re-evaluation. Each phase can only depend on
                      phases with a smaller index. If None, the default
                      is to consider it dependent on all phases before it.
    """
    def decorator(func):
        phase_name = func.__name__

        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        wrapper.is_phase = True
        wrapper.name = phase_name
        wrapper.index = index
        wrapper.dep_names = dep_names
        return wrapper
    return decorator

def collect_phases(cls):
    """
    Decorates a class inheriting Experiment. Compiles all instancemethods
    decorated by `phase' into a class variable PHASES containing
    `ExperimentPhase's in the correct order.
    """
    phase_attrs_by_index = dict()
    for attrname in dir(cls):
        attr = getattr(cls, attrname)
        if not getattr(attr, 'is_phase', False): continue
        if attr.index not in phase_attrs_by_index:
            phase_attrs_by_index[attr.index] = []
        phase_attrs_by_index[attr.index].append(attr)

    # Fail if multiple phases have the same index
    same_index = [
        ['{}-{}'.format(index, phase_attr.name)
         for phase_attr in phase_attrs]
        for index, phase_attrs in phase_attrs_by_index.items()
        if len(phase_attrs) > 1
    ]
    if len(same_index) > 0:
        error = ("Phases must have unique indices. "
                 "These phases have the same index: {}").format(same_index)
        raise ValueError(error)

    # Construct list of phases in order of execution
    cls.PHASES = []
    phase_indices = sorted(list(phase_attrs_by_index.keys()))
    for index in phase_indices:
        phase_attr = phase_attrs_by_index[index][0]

        deps = []
        if phase_attr.dep_names:
            for dep_name in phase_attr.dep_names:
                dep_attr = getattr(cls, dep_name, None)
                if dep_attr is None:
                    error = ("Phase {}-{} depends on phase {}, "
                             "which could not be found in the class.").format(
                        index, phase_attr.name, dep_name)
                    raise ValueError(error)
                elif dep_attr.index > index:
                    error = ("Phase {}-{} cannot depend on phase {}-{}, "
                             "because it has a higher index.").format(
                        index, phase_attr.name, dep_attr.index, dep_name)
                    raise ValueError(error)
                else:
                    deps.append(phase_attr.index)
        else:
            # By default, each phase depends on all the phases before it
            deps = [phase.index for phase in cls.PHASES]

        phase = ExperimentPhase(phase_attr.name, index, deps, phase_attr)
        cls.PHASES.append(phase)

    # Support non-contiguous indices
    cls.PHASE_BY_INDEX = { phase.index: phase for phase in cls.PHASES }

    return cls

def add(d, key, val):
    if key in d:
        d[key].append(val)
    else:
        d[key] = [val]

@collect_phases
class Experiment(object):
    """
    Base class to be inherited by any experiment.
    """
    def __init__(self, config, out_dir=None):
        """
        :param config: A dictionary containing the experiment's configuration.
        :param out_dir: The directory to save an experiment's results in. If None,
                                        defaults to DEFAULT_OUT_DIR.
        """
        self.config = config
        self.out_dir = out_dir if out_dir is not None else DEFAULT_OUT_DIR
        self.results = dict()
        self.R = dict()

    """
    A string uniquely identifying the series of experiments.
    """
    experiment_id = "exp"

    @property
    def run_id(self):
        """
        A string that uniquely identifies this particular run of the experiment
        among other runs, depending on the configuration.
        """
        raise NotImplementedError('Each experiment must generate a run_id that is unique across runs')

    @staticmethod
    def get_base_dir(out_dir, experiment_id, run_id):
        """
        Return the experiment base directory given the output directory,
        experiment id and run id. Also ensures that the directory exists.

        :param out_dir: The output directory for all experiments.
        :param experiment_id: The unique experiment id.
        :param run_id: The unique run id within this experiment.
        :return: The path to the base directory for this run.
        """
        base_dir = os.path.join(out_dir, experiment_id, run_id)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.isdir(base_dir):
            raise Exception('{} already exists but is not a directory.'.format(base_id))
        return base_dir

    @property
    def base_dir(self):
        """
        The path to the base directory of this run of the experiment.
        """
        return self.get_base_dir(self.out_dir, self.experiment_id, self.run_id)

    @property
    def plot_dir(self):
        """
        The path to the plots subdirectory of the experiment.
        """
        plot_dir = os.path.join(self.base_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        if not os.path.isdir(plot_dir):
            raise Exception('{} already exists but is not a directory.'.format(base_id))
        return plot_dir

    def get_result_path(self, phase_index):
        """
        Returns the path to save the result of a phase in.

        :param phase_index: The index of the phase in PHASES.
        :return: The path to the result.
        """
        phase = self.PHASE_BY_INDEX[phase_index]
        result_name = "result_{}-{}.npz".format(phase.index, phase.name)
        result_path = os.path.join(self.base_dir, result_name)
        return result_path

    @staticmethod
    def get_config_path(base_dir):
        """
        Returns the path to the saved config, given the base dir.

        :param base_dir: The base directory
        :return: The path to the saved config
        """
        return os.path.join(base_dir, 'config.pickle')

    @property
    def config_path(self):
        """
        The path to the config of this run of the experiment.
        """
        return self.get_config_path(self.base_dir)

    @staticmethod
    def load_phase_result(result_path):
        """
        Loads the result of a phase from the give result path. This is independent
        of the phase itself because results are constrained to be homogenous.

        :param result_path: The path to the result.
        :return: The result dictionary.
        """
        data = np.load(result_path, allow_pickle=True)
        result = dict(data)
        for key, value in result.items():
            if value.shape == tuple():
                # This value was previously a scalar, extract it
                result[key] = value.reshape(1)[0]
        return result

    @staticmethod
    def save_phase_result(result_path, result):
        """
        Saves the result of a phase into the given result path.

        :param result_path: The path to the result.
        :param result: A dictionary representing the result of a phase.
        """
        np.savez(result_path, allow_pickle=True, **result)

    @staticmethod
    def load_config(config_path):
        """
        Loads a config from the given path.

        :param config_path: The path to the config.
        :return: The saved config dict.
        """
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        return config

    @staticmethod
    def save_config(config_path, config):
        """
        Saves a config into the given path. Configs are saved using pickle.

        :param config_path: The path to the config.
        :param config: A dictionary representing all configuration for this experiment.
        """
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

    def run(self,
            force_refresh=False,
            invalidate_phase=None):
        """
        Runs all phases of the experiment, skipping phases that have already
        been run if possible, and if desired. Previous results will be overwritten.

        :param force_refresh: If False, phases with previously saved results will be loaded
                              and skipped. Otherwise, every phase will be re-run.
        :param invalidate_phase: If not None, every phase including and after this
                                 phase index will be re-run.
        """
        print("Experiment {}: running {}".format(self.experiment_id, self.run_id))
        print("Results will be stored in {}".format(self.base_dir))
        print()

        self.save_config(self.config_path, self.config)

        exp_start = time.time()

        for phase in self.PHASES:
            result_path = self.get_result_path(phase.index)

            invalidated = invalidate_phase is not None and phase.index >= invalidate_phase
            if not force_refresh and not invalidated and os.path.exists(result_path):
                print("Loading phase {}-{} from previous run:".format(phase.index, phase.name))
                print(result_path)
                result = self.load_phase_result(result_path)
            else:
                print("Running phase {}-{}...".format(phase.index, phase.name))

                phase_start = time.time()
                result = phase.instancemethod(self)
                phase_time = time.time() - phase_start
                print("Phase {}-{} took {} seconds".format(phase.index, phase.name, phase_time))

                if not isinstance(result, dict):
                    raise ValueError('Experiment phases should return dictionaries.')
                self.save_phase_result(result_path, result)

            self.results[phase.name] = result
            self.R.update(result)
            print()

        exp_time = time.time() - exp_start

        print("Experiment {}: run {} completed in {} seconds.".format(
            self.experiment_id, self.run_id, exp_time))
        print()

    def load_results(self, verbose=True):
        """
        Loads the results of a previously run experiment. Supports partial runs.
        Previously loaded/run results will be overwritten.
        """
        result_paths = [self.get_result_path(phase.index) for phase in self.PHASES]

        load_start = time.time()

        for phase, result_path in zip(self.PHASES, result_paths):
            if os.path.exists(result_path):
                if verbose:
                    print("Loading phase {}-{} from previous run:".format(phase.index, phase.name))
                    print(result_path)
                self.results[phase.name] = self.load_phase_result(result_path)
                self.R.update(self.results[phase.name])

        load_time = time.time() - load_start

        if verbose:
            print("Experiment {}: run {} results loaded in {} seconds.".format(
                self.experiment_id, self.run_id, load_time))

    def load_phases(self, phase_indices, verbose=True, overwrite=True):
        """
        Load specific phases from the results of a previously run experiment.
        If previously loaded, these specific phases will be overwritten, unless
        overwrite is False.
        """
        current_phases = [self.PHASES[phase_index] for phase_index in phase_indices]
        result_paths = [self.get_result_path(phase_index) for phase_index in phase_indices]

        load_start = time.time()

        for phase, result_path in zip(current_phases, result_paths):
            if not overwrite and phase.name in self.results: continue
            if os.path.exists(result_path):
                if verbose:
                    print("Loading phase {}-{} from previous run:".format(phase.index, phase.name))
                self.results[phase.name] = self.load_phase_result(result_path)
                self.R.update(self.results[phase.name])

        load_time = time.time() - load_start

        if verbose:
            print("Experiment {}: run {}, phase {} results loaded in {} seconds.".format(
                self.experiment_id, self.run_id, phase_indices, load_time))
            print()

    @classmethod
    def load_run(cls, run_id, out_dir=None, phase_indices=None, verbose=True):
        """
        Loads a previously run experiment.
        Fails if config is missing.
        Not yet configured to fail if any of the phases are missing,
        i.e. may contain only partial results.
        :param run_id: The unique run_id to load
        :param out_dir: The directory that the experiment's results are saved in.
                                        If None, defaults to DEFAULT_OUT_DIR.
        :return: An Experiment of this type with all previously saved data loaded.
        """
        out_dir = out_dir if out_dir is not None else DEFAULT_OUT_DIR
        config_path = cls.get_config_path(cls.get_base_dir(out_dir, cls.experiment_id, run_id))
        config = cls.load_config(config_path)
        exp = cls(config, out_dir=out_dir)
        if phase_indices is not None:
            exp.load_phases(phase_indices, verbose=verbose)
        else:
            exp.load_results(verbose=verbose)
        return exp

@collect_phases
class TestExperiment(Experiment):
    """
    Example experiment class demonstrating how to write experiment phases.
    """
    def __init__(self, config, out_dir=None):
        super(TestExperiment, self).__init__(config, out_dir)

    experiment_id = "test_exp"

    @property
    def run_id(self):
        return "some_meaningful_id"

    # The final phase order is bar, then foo, then foo_bar.

    @phase(1)
    def foo(self):
        print("foo")
        return { 'foo_result': 3, 'foo_dict': { 'key': 'value' } }

    @phase(2)
    def foo_bar(self):
        print("foo_bar")
        # Phases can depend on the results of previous phases.
        return { 'foo_bar': self.results['foo']['foo_result'] * self.results['bar']['bar'] }

    @phase(0)
    def bar(self):
        # This phase is run before foo because of its index
        print("bar")
        return { 'bar': np.random.normal((5, 5)) }
