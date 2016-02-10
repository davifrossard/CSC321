from math_funcs import *

import numpy as np

def dCost_dWeight(x, w, b, y):
    m = len(y)
    z = x.dot(w) + b
    r = (1.0/m) * x.T.dot(softmax(z)-y)[0]
    return r

def dCost_dBias(x, w, b, y):
    m = len(y)
    z = x.dot(w) + b
    r = (1.0/m) * sum((softmax(z)-y))
    return r

def compare_gradient(x, w, b, y):
    gradw = dCost_dWeight(x, w, b, y)
    gradb = dCost_dBias(x, w, b, y)

    w1 = cross_entropy(x, w, b, y)
    w2 = cross_entropy(x, w+0.0001, b, y)
    gradw_approx = (w2-w1)/0.0001

    b1 = cross_entropy(x, w, b, y)
    b2 = cross_entropy(x, w, b+0.0001, y)
    gradb_approx = (b2-b1)/0.0001

    print "Gradient:",gradw, "Finite Differences:",gradw_approx
    print "Error:", gradw-gradw_approx
    print "Gradient:",gradb, "Finite Differences:",gradb_approx
    print "Error:", gradb-gradb_approx
