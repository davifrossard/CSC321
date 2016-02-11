import numpy as np
from part2 import feed_forward

# --------------------------------------------
def softmax(x):
    return 1 / (1 + np.exp(-x))


def dsoftmax(x):
    return softmax(x) * (1 - softmax(x))


# --------------------------------------------
def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1 - np.square(np.tanh(x))


# --------------------------------------------
def relu(x):
    return np.maximum(0, x)


def drelu(x):
    return 0.5 * (np.sign(x) + 1)


# --------------------------------------------
def cross_entropy(x, w, b, funcs, y):
    activation = (feed_forward(x, w, b, funcs)[1])[-1]
    return np.sum(np.maximum(activation, 0) - y * activation + np.log(1 + np.exp(-np.abs(activation))),axis=0)/len(y)


