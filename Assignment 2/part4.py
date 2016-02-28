from part3 import *
from math_funcs import *
import csv

def compare_gradient(x, w, b, y, delta=1e-5, numpoints=4):
    np.random.seed(0)
    gradw = dCost_dWeight(x, w, b, y)
    gradb = dCost_dBias(x, w, b, y)
    idw = [(ix/10, ix%10) for ix in (-abs(gradw)).flatten().argsort()[:numpoints]]
    idb = (-abs(gradb)).flatten().argsort()[:numpoints]
    gradb = gradb[idb]
    gradw = gradw[zip(*idw)]

    wd = np.array([w,w,w,w])
    bd = np.array([b,b,b,b])

    for i, index in enumerate(idw):
        wd[i, index[0], index[1]] += delta

    for i, index in enumerate(idb):
        bd[i, index] += delta

    gradsw = np.zeros(numpoints)
    for i in range(numpoints):
        w1 = cross_entropy(x, [wd[i]], [b], [softmax], y)
        w2 = cross_entropy(x, [w], [b], [softmax], y)
        gradsw[i] = (w1-w2)/delta

    gradsb = np.zeros(numpoints)
    for i in range(numpoints):
        b1 = cross_entropy(x, [w], [bd[i]], [softmax], y)
        b2 = cross_entropy(x, [w], [b], [softmax], y)
        gradsb[i] = (b1-b2)/delta

    errorw = gradw-gradsw
    errorb = gradb-gradsb
    print "Evaluating gradient for weights", idw
    print "Gradient:",gradw
    print "Finite Differences:",gradsw
    print "Error:", errorw,"\n"

    print "Evaluating gradient for bias", idb
    print "Bias Gradient:",gradb
    print "Finite Differences:",gradsb
    print "Error:", errorb


    with open('results/part4_weight.out', 'w') as fl:
        fl.write("Entries:            "+', '.join(format(str(x), "^10s") for x in idw)+"\n")
        fl.write("Gradient:           "+', '.join(format(x, "10.7f") for x in gradw)+"\n")
        fl.write("Finite Differences: "+', '.join(format(x, "10.7f") for x in gradsw)+"\n")
        fl.write("Error:              "+', '.join(format(x, "10.7f") for x in errorw)+"\n")


    with open('results/part4_bias.out', 'w') as fl:
        fl.write("Entries:            "+', '.join(format(str(x), "^10s") for x in idb)+"\n")
        fl.write("Gradient:           "+', '.join(format(x, "10.7f") for x in gradb)+"\n")
        fl.write("Finite Differences: "+', '.join(format(x, "10.7f") for x in gradsb)+"\n")
        fl.write("Error:              "+', '.join(format(x, "10.7f") for x in errorb)+"\n")