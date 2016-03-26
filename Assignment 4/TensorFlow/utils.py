import numpy as np
from math import ceil

def one_hot(v, size):
    return np.eye(size)[v]

def round_up(val, base):
    return int(ceil(float(val)/base)*base)

