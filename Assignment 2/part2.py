from scipy.io import loadmat
from part1 import *

import numpy as np
import matplotlib.pyplot as plt


def feed_forward(inputs, weights, biases, functions):
    if not (len(weights) == len(biases) and len(biases) == len(functions)):
        raise ValueError("Inconsistent Model. Different number of weights, biases and functions.")
    layers = len(weights)
    activations = []

    for l in range(layers):
        activation = np.dot(inputs, weights[l]) + biases[l]
        activations.append(activation)
        inputs = functions[l](activation)
    # del activations[-1]
    return inputs, activations
