from math_funcs import *

import numpy as np

def dCost_dWeight(x, w, b, y):
    m = len(y)
    z = x.dot(w) + b
    r = (1.0/m) * x.T.dot(softmax(z)-y)
    return r

def dCost_dBias(x, w, b, y):
    m = len(y)
    z = x.dot(w) + b
    r = (1.0/m) * sum((softmax(z)-y))
    return r
