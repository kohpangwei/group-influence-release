from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import pickle
import numpy as np
import tensorflow as tf

from datasets.common import DataSet

DEFAULT_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 '..', 'output', 'models'))

class Model(object):
    """
    The base influence model class from which all other models should derive.

    A model should behave mostly deterministically once the weights are reset.
    However, this might not be fully possible due to Tensorflow's seeds and
    optimizer states. At the very least, a model should behave
    deterministically between saves and loads, which uses a Tensorflow Saver to
    checkpoint the entire model graph.

    To this end, the model should not store any information about the datasets
    it has been fed, save for what it receives from the config. Dataset
    iteration order and batching state is managed by the dataset object itself.
    """
    def __init__(self, config, model_dir=None):
        """
        Initialize a model with the given configuration.

        :param config: The model configuration dict.
        :param model_dir: The base directory to save the model into.
        """
        self.config = config
        self.model_dir = model_dir if model_dir is not None else DEFAULT_MODEL_DIR
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.isdir(self.model_dir):
            raise Exception('{} already exists but is not a directory.'.format(self.model_dir))
        Model.save_config(self.config_path, self.config)

        self.initialize_session()
        self.build_graph()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        self.sess.close()

    @property
    def config_path(self):
        return os.path.join(self.model_dir, 'model_config.pickle')

    def initialize_session(self):
        """
        Initializes the Tensorflow session.
        """
        tf.reset_default_graph()
        self.sess = tf.Session()
        tf.set_random_seed(self.config['tf_init_seed'])

    def build_graph(self):
        """
        Build the Tensorflow graph for the model.
        """
        raise NotImplementedError()

    def fit(self, dataset, sample_weights=None, **kwargs):
        """
        Resets the model's parameters and trains the model to fit the dataset.

        :param dataset: The dataset to fit the model to.
        :param sample_weights: The weight of each example in the dataset.
        """
        raise NotImplementedError()

    def warm_fit(self, dataset, sample_weights=None, **kwargs):
        """
        Trains the model to fit the dataset, using the previously stored
        parameters as a starting point.

        :param dataset: The dataset to fit the model to.
        :param sample_weights: The weight of each example in the dataset.
        """
        raise NotImplementedError()

    def save(self, state_id, global_step=None):
        """
        Saves the model state (parameters and other tensorflow state)
        to the subdirectory state_id in model_dir.

        :param state_id: A string uniquely addressing this model state.
        """
        save_path = os.path.join(self.model_dir, state_id)
        checkpoint_path = '{}.ckpt'.format(state_id)
        self.saver.save(self.sess,
                        save_path,
                        global_step=global_step,
                        latest_filename=checkpoint_path)

        # Empty out the old checkpoints list so the next checkpoint file
        # does not contain previous saves
        self.saver.set_last_checkpoints_with_time([])

    def load(self, state_id, global_step=None):
        """
        Loads the model state (parameters and other tensorflow state)
        from the subdirectory state_id in model_dir.

        :param state_id: A string uniquely addressing this model state.
        """
        save_path = os.path.join(self.model_dir, state_id)
        if global_step is not None:
            save_path = "{}-{}".format(save_path, global_step)
        self.saver.restore(self.sess, save_path)

    @staticmethod
    def load_config(config_path):
        """
        Loads a model_config from the given path.

        :param config_path: The path to the config.
        :return: The saved config dict.
        """
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        return config

    @staticmethod
    def save_config(config_path, model_config):
        """
        Saves a model_config into the given path. Configs are saved using pickle.

        :param config_path: The path to the config.
        :param model_config: A dictionary representing all configuration for this experiment.
        """
        with open(config_path, 'wb') as f:
            pickle.dump(model_config, f)

    def get_params(self):
        raise NotImplementedError()

    def set_params(self):
        raise NotImplementedError()

    def get_total_loss(self, dataset, sample_weights=None, reg=False, **kwargs):
        raise NotImplementedError()

    def get_indiv_loss(self, dataset, **kwargs):
        raise NotImplementedError()

    def get_total_grad_loss(self, dataset, sample_weights=None, reg=False, **kwargs):
        raise NotImplementedError()

    def get_indiv_grad_loss(self, dataset, **kwargs):
        raise NotImplementedError()

    def get_indiv_grad_loss_from_total_grad(self, dataset, **kwargs):
        indiv_grad_loss = self.batch_evaluate(
            lambda xs, labels: [self.get_total_grad_loss(DataSet(xs, labels), l2_reg=0, **kwargs)],
            lambda v1, v2: v1.extend(v2) or v1,
            1, dataset, value_name="Gradients")
        return np.array(indiv_grad_loss)

    @staticmethod
    def batch_evaluate(evaluate_fn, reduce_fn, batch_size,
                       dataset, sample_weights=None,
                       value_name="Batched values",
                       **kwargs):
        """
        Helper to evaluate a quantity over the entire dataset. The order of examples
        in the dataset is preserved.

        :param evaluate_fn: Function to evaluate the quantity given a batch
                            of examples from the dataset. If sample_weights is None,
                            evaluate_fn(xs, labels) will be called. If it is not None,
                            evaluate_fn(xs, labels, weights) will be called.
        :param reduce_fn: Function to accumulate values from two successive calles to
                          evaluate. If v1 is the value from the previous batch and v2
                          is the value from the second batch, reduce_fn(v1, v2) should
                          return the result of evaluating on both batches combined.
        :param batch_size: The maximum number of examples per batch.
        :param dataset: The dataset to evaluate the quantity over.
        :param sample_weights: The sample weights of the dataset examples, or if this
                               is not needed, None.
        :param value_name: The name of the value being computed for debugging and
                           feedback purposes.
        :return: The quantity, evaluated over the entire dataset.
        """
        value = None
        verbose = kwargs.get('verbose', True)
        for i in range(0, dataset.num_examples, batch_size):
            if verbose:
                print("\r{} computed: {}/{}".format(value_name, i, dataset.num_examples), end="")
            end = min(i + batch_size, dataset.num_examples)

            args = (dataset.x[i:end, :], dataset.labels[i:end])
            if sample_weights is not None:
                args = args + (sample_weights[i:end],)
            batch_value = evaluate_fn(*args)

            value = batch_value if value is None else reduce_fn(value, batch_value)
        if verbose:
            print("\r{} computed: {}/{}".format(value_name, dataset.num_examples, dataset.num_examples))
        return value

    @staticmethod
    def infer_arch(dataset):
        raise NotImplementedError()

    @staticmethod
    def default_config():
        raise NotImplementedError()

def variable_with_l2_reg(name, shape, stddev=None, l2_reg=None):
    """
    Helper to create an initialized Variable with l2 regularization if desired.
    Note that the Variable is initialized with a truncated Gaussian distribution
    if stddev is not None, and to 0 if it is None.

    :param name: The name of the variable
    :param shape: The shape of the variable
    :param stddev: The standard deviation of the truncated Gaussian used
                   for initialization
    :param l2_reg: If None, no regularization is added. Otherwise, l2
                   regularization is added for this Variable.
    :return: Tensor representing the variable.
    """

    dtype = tf.float32
    if stddev is not None:
        initializer = tf.truncated_normal_initializer(
            stddev=stddev,
            dtype=dtype)
    else:
        initializer = tf.constant_initializer(0.0)
    var = tf.get_variable(name,
                          shape,
                          initializer=initializer,
                          dtype=dtype)
    if l2_reg is not None:
      l2_reg_loss = tf.multiply(tf.nn.l2_loss(var), l2_reg,
                                name='{}_l2_reg_loss'.format(name))
      tf.add_to_collection('regularization', l2_reg_loss)
      tf.add_to_collection('losses', l2_reg_loss)

    return var

def get_accuracy(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label.

    :param logits: Logits tensor, float32 of shape (batch_size, num_classes).
    :param labels: Labels tensor, int32 of shape (batch_size,), with values in the
                   range [0, num_classes).
    :return: A scalar int32 tensor with the number of examples (out of batch_size)
             that were predicted correctly.
    """
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32)) / tf.shape(labels)[0]

def get_assigners(tensors):
    """
    Helper to generate placeholders and assignment operators for a list of
    tensors.

    :param tensors: A list of tensors to be assigned to.
    :return: A list of (assign_op, placeholder) tuples with which you can
             assign values to the tensors.
    """
    ops = []
    for tensor in tensors:
        placeholder = tf.placeholder(dtype=tensor.dtype, shape=tensor.shape)
        assign_op = tf.assign(tensor, placeholder, validate_shape=True)
        ops.append((assign_op, placeholder))
    return ops

def flatten(tensors):
    """
    Flatten and concatenate a list of tensors into a single 1-D tensor.

    :param tensors: The list of tensors to flatten.
    :return: A 1-D tensor.
    """
    return tf.concat([tf.reshape(tensor, (-1,)) for tensor in tensors], axis=0)

def split_like(tensors, flat_tensor):
    """
    Split a flattened (and possibly batched) tensor back into a list of tensors.
    If flat_tensor is not batched, then the result will be a list of tensors of the
    same shape as `tensors`. Otherwise, let [A_{i,0}, ..., A_{i,n_i-1}] be the
    shape of the ith tensor and D_i = A_{i,0} * ... * A_{i,n_i-1} be its size.
    Then flat_tensor must be of shape [?, sum_i D_i]. The result will be a
    list of tensors of shape [?, A_{i,0}, ..., A_{i,n_i-1}].

    :param tensors: A list of tensors of the desired shape to split flat_tensor into.
    :param flat_tensor: The tensor to split, possibly batched.
    :return: A list of tensors of the same shape as `tensors`, but possibly batched.
    """
    split_sizes, split_shapes = [], []
    for tensor in tensors:
        total_dim = 1
        for dim in tensor.shape:
            if dim is None:
                raise ValueError("Source tensors must have concrete shapes.")
            total_dim *= int(dim)
        split_sizes.append(int(total_dim))
        split_shapes.append([int(dim) for dim in tensor.shape])

    split_axis = len(flat_tensor.shape) - 1
    split_tensors = tf.split(flat_tensor, split_sizes, split_axis)
    batched_shapes = split_shapes
    if split_axis != 0:
        batched_shapes = [[-1] + shape for shape in batched_shapes]

    reshaped_tensors = [
        tf.reshape(split_tensor, batched_shape)
        for split_tensor, batched_shape in zip(split_tensors, batched_shapes)
    ]
    return reshaped_tensors

