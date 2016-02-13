from copy import deepcopy

from part7 import *
from math_funcs import *

def compare_gradient2(x, w, b, y, delta=1e-10, numpoints=3):
    np.random.seed(0)
    print "----------------------------------"
    print "Evaluating Gradients of 2nd Layer:"
    print "----------------------------------"

    idw = zip(np.random.choice(len(w[1]), numpoints), np.random.choice(len(w[1][0]), numpoints))
    idb = np.random.choice(len(b[1]), numpoints)

    gradsw, gradsb = np.zeros(numpoints), np.zeros(numpoints)
    b2 = cross_entropy(x, w, b, [tanh, softmax], y)

    print "Evaluating gradient for weights", idw
    for k, index in enumerate(idw):
        wl = deepcopy(w)
        i,j = index

        wl[1][i][j] += delta
        b1 = cross_entropy(x, wl, b, [tanh, softmax], y)
        gradsw[k] = (b1-b2)/delta

    gradw = dCost_dWeight2(x, w, b, y)[zip(*idw)]
    print "Gradient:",gradw
    print "Finite Differences:",gradsw
    print "Error:", gradw-gradsw,"\n"


    print "Evaluating gradient for bias", idb
    for k, index in enumerate(idb):
        bl = deepcopy(b)

        bl[1][index] += delta
        b1 = cross_entropy(x, w, bl, [tanh, softmax], y)
        gradsb[k] = (b1-b2)/delta

    gradb = dCost_dBias2(x, w, b, y)[idb]
    print "Bias Gradient:",gradb
    print "Finite Differences:",gradsb
    print "Error:", gradb-gradsb

    print "----------------------------------"
    print "Evaluating Gradients of 1st Layer:"
    print "----------------------------------"

    idw = zip(np.random.choice(len(w[0]), numpoints), np.random.choice(len(w[0][0]), numpoints))
    idb = np.random.choice(len(b[0]), numpoints)

    gradsw, gradsb = np.zeros(numpoints), np.zeros(numpoints)

    print "Evaluating gradient for weights", idw
    for k, index in enumerate(idw):
        wl = deepcopy(w)
        i,j = 350,164

        wl[0][i][j] += delta
        b1 = cross_entropy(x, wl, b, [tanh, softmax], y)
        gradsw[k] = (b1-b2)/delta

    gradw = dCost_dWeight1(x, w, b, y)[zip(*idw)]
    print "Gradient:",gradw
    print "Finite Differences:",gradsw
    print "Error:", gradw-gradsw,"\n"

    for k, index in enumerate(idb):
        bl = deepcopy(b)

        bl[0][index] += delta
        b1 = cross_entropy(x, w, bl, [tanh, softmax], y)
        gradsb[k] = (b1-b2)/delta

    gradb = dCost_dBias1(x, w, b, y)[idb]

    print "Evaluating gradient for bias", idb
    print "Bias Gradient:",gradb
    print "Finite Differences:",gradsb
    print "Error:", gradb-gradsb

