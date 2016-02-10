from numpy import *

def knn_classify(train_x, train_y, x, k, distfun, distances=None):

    if not distances:
        if distances is None:
            distances = []
        for xi in train_x:
            distances.append(distfun(x, xi))

    # Find nearest neighbors
    nn = sorted(enumerate(distances), key=lambda y: y[1])
    nn = [i[0] for i in nn]

    # Select k nearest neighbors
    nn = nn[:k]
    # Select class with most occurrences
    classes = train_y[nn].tolist()

    x_class = max(classes, key=lambda y: classes.count(y))

    return x_class, nn