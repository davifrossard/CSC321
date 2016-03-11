from pylab import load
import os
net_data = load(os.path.dirname(os.path.realpath(__file__)) + "/bvlc_alexnet.npy").item()