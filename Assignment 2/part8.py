import csv
from copy import deepcopy
from part7 import *
from math_funcs import *

def compare_gradient2(x, w, b, y, delta=1e-10, numpoints=4):
    np.random.seed(10)
    print "----------------------------------"
    print "Evaluating Gradients of 2nd Layer:"
    print "----------------------------------"

    gradw = dCost_dWeight2(x, w, b, y)
    gradb = dCost_dBias2(x, w, b, y)
    idw = [(ix/10, ix%10) for ix in (-abs(gradw)).flatten().argsort()[:numpoints]]
    idb = (-abs(gradb)).flatten().argsort()[:4]
    gradb = gradb[idb]
    gradw = gradw[zip(*idw)]

    gradsw, gradsb = np.zeros(numpoints), np.zeros(numpoints)
    b2 = cross_entropy(x, w, b, [tanh, softmax], y)

    print "Evaluating gradient for weights", idw
    for k, index in enumerate(idw):
        wl = deepcopy(w)
        i,j = index

        wl[1][i][j] += delta
        b1 = cross_entropy(x, wl, b, [tanh, softmax], y)
        gradsw[k] = (b1-b2)/delta


    print "Gradient:",gradw
    print "Finite Differences:",gradsw
    print "Error:", gradw-gradsw,"\n"


    print "Evaluating gradient for bias", idb
    for k, index in enumerate(idb):
        bl = deepcopy(b)

        bl[1][index] += delta
        b1 = cross_entropy(x, w, bl, [tanh, softmax], y)
        gradsb[k] = (b1-b2)/delta

    print "Bias Gradient:",gradb
    print "Finite Differences:",gradsb
    print "Error:", gradb-gradsb

    with open('results/part8_weight_2nd.out', 'w') as fl:
        fl.write("Entries:            "+', '.join(format(str(x), "^10s") for x in idw)+"\n")
        fl.write("Gradient:           "+', '.join(format(x, "10.7f") for x in gradw)+"\n")
        fl.write("Finite Differences: "+', '.join(format(x, "10.7f") for x in gradsw)+"\n")
        fl.write("Error:              "+', '.join(format(x, "10.7f") for x in gradw-gradsw)+"\n")


    with open('results/part8_bias_2nd.out', 'w') as fl:
        fl.write("Entries:            "+', '.join(format(str(x), "^10s") for x in idb)+"\n")
        fl.write("Gradient:           "+', '.join(format(x, "10.7f") for x in gradb)+"\n")
        fl.write("Finite Differences: "+', '.join(format(x, "10.7f") for x in gradsb)+"\n")
        fl.write("Error:              "+', '.join(format(x, "10.7f") for x in gradb-gradsb)+"\n")

    print "----------------------------------"
    print "Evaluating Gradients of 1st Layer:"
    print "----------------------------------"

    gradw = dCost_dWeight1(x, w, b, y)
    gradb = dCost_dBias1(x, w, b, y)
    idw = [(ix/300, ix%300) for ix in (-abs(gradw)).flatten().argsort()[:numpoints]]
    idb = (-abs(gradb)).flatten().argsort()[:numpoints]
    gradb = gradb[idb]
    gradw = gradw[zip(*idw)]
    gradsw, gradsb = np.zeros(numpoints), np.zeros(numpoints)

    for k, index in enumerate(idw):
        wl = deepcopy(w)

        wl[0][index] += delta
        b1 = cross_entropy(x, wl, b, [tanh, softmax], y)
        gradsw[k] = (b1-b2)/delta

    print "Evaluating gradient for weights", idw
    print "Gradient:",gradw
    print "Finite Differences:",gradsw
    print "Error:", gradw-gradsw,"\n"

    for k, index in enumerate(idb):
        bl = deepcopy(b)

        bl[0][index] += delta
        b1 = cross_entropy(x, w, bl, [tanh, softmax], y)
        gradsb[k] = (b1-b2)/delta


    print "Evaluating gradient for bias", idb
    print "Bias Gradient:",gradb
    print "Finite Differences:",gradsb
    print "Error:", gradb-gradsb


    with open('results/part8_weight_1st.out', 'w') as fl:
        fl.write("Entries:            "+', '.join(format(str(x), "^10s") for x in idw)+"\n")
        fl.write("Gradient:           "+', '.join(format(x, "10.7f") for x in gradw)+"\n")
        fl.write("Finite Differences: "+', '.join(format(x, "10.7f") for x in gradsw)+"\n")
        fl.write("Error:              "+', '.join(format(x, "10.7f") for x in gradw-gradsw)+"\n")


    with open('results/part8_bias_1st.out', 'w') as fl:
        fl.write("Entries:            "+', '.join(format(str(x), "^10s") for x in idb)+"\n")
        fl.write("Gradient:           "+', '.join(format(x, "10.7f") for x in gradb)+"\n")
        fl.write("Finite Differences: "+', '.join(format(x, "10.7f") for x in gradsb)+"\n")
        fl.write("Error:              "+', '.join(format(x, "10.7f") for x in gradb-gradsb)+"\n")