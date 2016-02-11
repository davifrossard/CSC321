from part3 import *
from math_funcs import *

def compare_gradient(x, w, b, funcs, y):
    gradw = np.sum(dCost_dWeight(x, w[0], b[0], y))
    gradb = np.sum(dCost_dBias(x, w[0], b[0], y))

    w1 = cross_entropy(x, w, b, funcs, y)
    w2 = cross_entropy(x, [w[0]+0.0001], b, funcs, y)
    gradw_approx = np.sum((w2-w1)/0.0001)

    b1 = cross_entropy(x, w, b, funcs, y)
    b2 = cross_entropy(x, w, [b[0]+0.0001], funcs, y)
    gradb_approx = np.sum((b2-b1)/0.0001)

    print "Weight Gradient:",gradw
    print "Finite Differences:",gradw_approx
    print "Error:", gradw-gradw_approx,"\n"
    print "Bias Gradient:",gradb
    print "Finite Differences:",gradb_approx
    print "Error:", gradb-gradb_approx