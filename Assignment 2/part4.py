from part3 import *
from math_funcs import *

def compare_gradient(x, w, b, y, delta=1e-5):
    np.random.seed(0)
    idw = zip(np.random.choice(len(w), 3), np.random.choice(len(w[0]), 3))
    idb = np.random.choice(len(b), 3)

    wd = np.array([w,w,w])
    bd = np.array([b,b,b])

    for i, index in enumerate(idw):
        wd[i, index[0], index[1]] += delta

    for i, index in enumerate(idb):
        bd[i, index] += delta

    gradsw = np.zeros(3)
    for i in range(3):
        w1 = cross_entropy(x, [wd[i]], [b], [softmax], y)
        w2 = cross_entropy(x, [w], [b], [softmax], y)
        gradsw[i] = (w1-w2)/delta

    gradsb = np.zeros(3)
    for i in range(3):
        b1 = cross_entropy(x, [w], [bd[i]], [softmax], y)
        b2 = cross_entropy(x, [w], [b], [softmax], y)
        gradsb[i] = (b1-b2)/delta

    gradw = dCost_dWeight(x, w, b, y)[zip(*idw)]
    gradb = dCost_dBias(x, w, b, y)[idb]

    print "Evaluating gradient for weights", idw
    print "Gradient:",gradw
    print "Finite Differences:",gradsw
    print "Error:", gradw-gradsw,"\n"


    print "Evaluating gradient for bias", idb
    print "Bias Gradient:",gradb
    print "Finite Differences:",gradsb
    print "Error:", gradb-gradsb