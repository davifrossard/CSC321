from random import shuffle
import numpy as np
import csv
from collections import OrderedDict

def get_data(fname):
    best_val_accuracy = 0.0
    last_i = 0
    layer = 4
    with open(fname, 'a+') as file:
        models = csv.reader(file, delimiter=',')
        for line in models:
            last_i = int(line[0]) or last_i
            val_accuracy = float(line[-2])
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                try:
                    layer = int(line[4])
                except:
                    pass
    return last_i, best_val_accuracy, layer

def make_sets(input, classes, train=85, validation=30):

    # Shuffle Sets
    np.random.seed(0)
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