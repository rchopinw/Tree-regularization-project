from __future__ import print_function
from __future__ import absolute_import

from copy import copy
from builtins import range

import pydotplus
from copy import deepcopy

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.optimizers import sgd, adam
from autograd import grad
import _pickle as cPickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from model import build_gru, build_mlp
from model import softplus
import autograd.numpy as np
from autograd.core import primitive
import _pickle as cPickle

with open('./data/train.pkl', 'rb') as fp:
    data_train = cPickle.load(fp)
    X_train = data_train['X']
    F_train = data_train['F']
    y_train = data_train['y']

IN_COUNT = 1
STATE_COUNT = 250
HIDDEN_SIZES = [25]
OUT_COUNT = 4

def build_gru(input_count, state_count, output_count):
    """Constructor for gated-recurrent unit.

    @param input_count: integer
                        number of input dimensions
    @param state_count: integer
                        number of hidden dimensions
    @param output_count: integer
                         number of binary outputs
                         (no continuous support at the moment)
    @return predict: function
                     used to predict y_hat
    @return log_likelihood: function
                            used to compute log likelihood
    @return parser: WeightsParser object
                    object to organize weights
    """
    parser = WeightsParser()
    parser.add_shape('init_hiddens', (1, state_count))
    parser.add_shape('update_x_weights', (input_count + 1, state_count))
    parser.add_shape('update_h_weights', (state_count, state_count))
    parser.add_shape('reset_x_weights', (input_count + 1, state_count))
    parser.add_shape('reset_h_weights', (state_count, state_count))
    parser.add_shape('thidden_x_weights', (input_count + 1, state_count))
    parser.add_shape('thidden_h_weights', (state_count, state_count))
    parser.add_shape('output_h_weights', (state_count, output_count))

    def update(curr_input, prev_hiddens, update_x_weights,
               update_h_weights, reset_x_weights, reset_h_weights,
               thidden_x_weights, thidden_h_weights):
        """Update function for GRU."""
        update = sigmoid(np.dot(curr_input, update_x_weights) +
                         np.dot(prev_hiddens, update_h_weights))
        reset = sigmoid(np.dot(curr_input, reset_x_weights) +
                        np.dot(prev_hiddens, reset_h_weights))
        thiddens = np.tanh(np.dot(curr_input, thidden_x_weights) +
                           np.dot(reset * prev_hiddens, thidden_h_weights))
        hiddens = (1 - update) * prev_hiddens + update * thiddens
        return hiddens

    def outputs(weights, input_set, fence_set, output_set=None, return_pred_set=False):
        update_x_weights = parser.get(weights, 'update_x_weights')
        update_h_weights = parser.get(weights, 'update_h_weights')
        reset_x_weights = parser.get(weights, 'reset_x_weights')
        reset_h_weights = parser.get(weights, 'reset_h_weights')
        thidden_x_weights = parser.get(weights, 'thidden_x_weights')
        thidden_h_weights = parser.get(weights, 'thidden_h_weights')
        output_h_weights = parser.get(weights, 'output_h_weights')
        data_count = len(fence_set) - 1
        feat_count = input_set.shape[0]

        ll = 0.0
        n_i_track = -1
        fence_base = fence_set[0]
        pred_set = None

        if return_pred_set:
            pred_set = np.zeros((output_count, int(input_set.shape[1]/250)))

        # loop through sequences and time steps
        for data_iter in range(data_count):

            # print('Executing iteration %d'%data_iter)

            hiddens = copy(parser.get(weights, 'init_hiddens'))

            fence_post_1 = fence_set[data_iter] - fence_base
            fence_post_2 = fence_set[data_iter + 1] - fence_base
            time_count = fence_post_2 - fence_post_1
            curr_input = input_set[:, fence_post_1:fence_post_2]

            for time_iter in range(time_count):
                hiddens = update(np.expand_dims(np.hstack((curr_input[:, time_iter], 1)), axis=0),
                                 hiddens, update_x_weights, update_h_weights, reset_x_weights,
                                 reset_h_weights, thidden_x_weights, thidden_h_weights)

            n_i_track += 1

            if output_set is not None:
                # subtract a small number so -1
                out_proba = softmax_sigmoid(np.dot(hiddens, output_h_weights))
                out_lproba = safe_log(out_proba)
                ll += np.sum(-1 * output_set[:, n_i_track] * out_lproba)
                print(ll)
            else:
                out_proba = softmax_sigmoid(np.dot(hiddens, output_h_weights))
                out_lproba = safe_log(out_proba)

            if return_pred_set:
                print('lproba of dataiteration: ', out_lproba)
                agm = np.argmax(out_lproba[0])
                pred_set[agm, n_i_track] = int(1)

        return ll, pred_set

    def predict(weights, input_set, fence_set):
        _, output_set = outputs(weights, input_set, fence_set, return_pred_set=True)
        return output_set

    def log_likelihood(weights, input_set, fence_set, output_set):
        ll, _ = outputs(weights, input_set, fence_set, output_set=output_set)
        return ll

    return predict, log_likelihood, parser


class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)


def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1)


def softmax_sigmoid(x):
    return np.exp(x) / np.sum(np.exp(x))


def safe_log(x, minval=1e-100):
    return np.log(np.maximum(x, minval))


pred_fun, loglike_fun, parser = build_gru(IN_COUNT, STATE_COUNT, OUT_COUNT)
