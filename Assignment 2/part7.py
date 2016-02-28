from math_funcs import *

import numpy as np

def dCost_dWeight2(x, w, b, y):
    w2, b2 = w[1], b[1]
    w1, b1 = w[0], b[0]
    m = len(y)

    a1 = tanh(x.dot(w1) + b1)
    z2 = a1.dot(w2) + b2

    delta2 = (softmax(z2)-y)
    dcdw2 = (1.0/m) * a1.T.dot(delta2)
    return dcdw2

def dCost_dBias2(x, w, b, y):
    w2, b2 = w[1], b[1]
    w1, b1 = w[0], b[0]
    m = len(y)

    a1 = tanh(x.dot(w1) + b1)
    z2 = a1.dot(w2) + b2
    r = (1.0/m) * sum((softmax(z2)-y))
    return r

def dCost_dWeight1(x, w, b, y):
    w2, b2 = w[1], b[1]
    w1, b1 = w[0], b[0]
    m = len(y)

    z1 = x.dot(w1) + b1
    a1 = tanh(z1)
    z2 = a1.dot(w2) + b2
    a2 = softmax(z2)

    delta2 = (a2-y)

    delta1 = np.dot(delta2, w2.T) * dtanh(z1)

    grad_weight1 = (1.0/m) * np.dot(x.T, delta1)
    return grad_weight1


def dCost_dBias1(x, w, b, y):
    w2, b2 = w[1], b[1]
    w1, b1 = w[0], b[0]
    m = len(y)

    z1 = x.dot(w1) + b1
    a1 = tanh(z1)
    z2 = a1.dot(w2) + b2

    delta2 = (softmax(z2)-y)

    delta1 = np.dot(delta2, w2.T) * dtanh(z1)
    dcdb1 = np.sum(delta1, axis=0)
    return (1.0/m)*dcdb1