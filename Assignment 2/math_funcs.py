import numpy as np


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
def cross_entropy(x, w, b, y):
    logit = x.dot(w)+b
    return np.sum(np.maximum(logit, 0) - y * logit + np.log(1 + np.exp(-np.abs(logit))),axis=0)/len(y)


