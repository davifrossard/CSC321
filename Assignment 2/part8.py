from part7 import *
from math_funcs import *

def compare_gradient2(x, w, b, funcs, y):
    gradw2 = dCost_dWeight2(x, w, b, y).sum(axis=0)
    gradb2 = dCost_dBias2(x, w, b, y)

    w21 = cross_entropy(x, w, b, funcs, y)
    w22 = cross_entropy(x, [w[0], w[1]+0.0001], b, funcs, y)
    gradw2_approx = (w22-w21)/0.0001

    b21 = cross_entropy(x, w, b, funcs, y)
    b22 = cross_entropy(x, w, [b[0], b[1]+0.0001], funcs, y)
    gradb2_approx = (b22-b21)/0.0001

    print "Weight Gradient:",gradw2
    print "Finite Differences:",gradw2_approx
    print "Error:", gradw2-gradw2_approx,"\n"
    print "Bias Gradient:",gradb2
    print "Finite Differences:",gradb2_approx
    print "Error:", gradb2-gradb2_approx