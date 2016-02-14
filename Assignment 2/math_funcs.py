import numpy as np
from part2 import feed_forward

# --------------------------------------------
def softmax(x):
    e = np.exp(x)
    sum = np.exp(x).sum(axis=1)
    return e / sum[:,None]


def dsoftmax(x):
    y = softmax(x)
    return y*(1-y)


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
    output = feed_forward(x, w, b, funcs)[0]
    xent = -np.sum(y*np.log(output))
    return (1.0/len(y)) * xent

