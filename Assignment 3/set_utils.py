from random import shuffle
import numpy as np
from collections import OrderedDict

def make_sets(input, classes, train=85, validation=30):

    # Shuffle Sets
    np.random.seed(1)
    ids = np.random.permutation(len(input))

    input = np.array(input)[ids]
    classes = np.array(classes)[ids]

    x_train = input[:train]
    t_train = classes[:train]

    x_validation = input[train:validation+train]
    t_validation = classes[train:validation+train]

    x_test = input[validation+train:]
    t_test = classes[validation+train:]

    return x_train, t_train, x_validation, t_validation, x_test, t_test

def to_one_hot(input):
    classes = list(OrderedDict.fromkeys(input))

    one_hot = np.zeros([len(input), len(classes)])

    for i, c in enumerate(classes):
        ocurrences = np.where(np.array(input) == c)
        one_hot[ocurrences, i] = 1

    return one_hot