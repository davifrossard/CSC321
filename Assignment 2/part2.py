from scipy.io import loadmat
from part1 import *

import numpy as np
import matplotlib.pyplot as plt


def feed_forward(inputs, weights, biases, functions):
    if not (len(weights) == len(biases) and len(biases) == len(functions)):
        raise ValueError("Inconsistent Model. Different number of weights, biases and functions.")
    layers = len(weights)
    outputs = []
    activations = []

    for l in range(layers):
        output = np.dot(inputs, weights[l]) + biases[l]
        outputs.append(output)
        inputs = functions[l](output)
        activations.append(inputs)
    return inputs, outputs, activations
