from scipy.io import loadmat
from part1 import *

import numpy as np
import matplotlib.pyplot as plt

# images, classes = load_mnist_mat('mnist_all.mat')
#
# x_train, t_train, x_validation, t_validation, x_test, t_test = \
#     make_sets(images, classes, train=40000, validation=10000)
#

def feed_forward(inputs, weights, biases, functions):
    if not (len(weights) == len(biases) and len(biases) == len(functions)):
        raise ValueError("Inconsistent Model. Different number of weights, biases and functions.")
    layers = len(weights)
    activations = []

    for l in range(layers):
        activation = functions[l](np.dot(inputs, weights[l]) + biases[l])
        inputs = activation
        activations.append(activation)

    return inputs, activations
