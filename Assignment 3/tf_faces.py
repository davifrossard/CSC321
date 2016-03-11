import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import os
import sys
import cPickle as cp
from get_data_file import *
from set_utils import *
from neural_net import train_neural_net, visualize_weights
from AlexNet.myalexnet import alex_net


def get_data(fname):
    best_val_accuracy = 0.0
    last_i = 0
    with open(fname, 'a+') as file:
        models = csv.reader(file, delimiter=',')
        for line in models:
            last_i = int(line[0]) or last_i
            val_accuracy = float(line[-2])
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
    return last_i, best_val_accuracy

# Configurations
datafile = 'data.txt'
conv = False

# Create results dir
if not os.path.exists("results"):
    os.makedirs("results")

# Find IDs of trained networks and best accuracy so far


act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']

faces, classes = fetch_data(datafile, act, -1)

# if raw_input("Use raw images (0) or AlexNet output (1) ? [1]").lower() != '0':
if True:
    conv = True
    dim = (227,227)
    for i, face in enumerate(faces):
        imgr = imresize(face, dim)/255.0
        img = (np.random.random((1, 227, 227, 3)) / 255.).astype('float32')
        img[0, :, :, :] = imgr[:, :, :3]
        img = img - np.mean(img)
        faces[i] = img
    i, best_val_accuracy = get_data('results/models_conv.csv')


else:

    dim = (100,100)
    dimf = dim[0] * dim[1]
    for i, face in enumerate(faces):
        faces[i] = rgb2gray(imresize(face, dim)).reshape(dimf)/255.0

    i, best_val_accuracy = get_data('results/models.csv')
    set = make_sets(faces, classes, 85*len(act), 35*len(act))

''' Hyperparameter Search '''
# if raw_input("Search for hyper-parameters? [y/N] ").lower() == 'y':
if True:
    print("Searching hyper-parameters, use Ctrl+C to stop.")
    try:
        maxlayer = 1
        while True:
            try:
                i+=1
                np.random.seed(int(time.time()))
                nlayer = np.random.randint(1,maxlayer+1)
                funcs_n = np.array(['relu', 'tanh'])
                funcs = np.array([tf.nn.relu, tf.nn.tanh])
                idx = np.random.choice(2, nlayer)
                hu = np.random.randint(200, 850, nlayer)
                funs = funcs_n[idx]
                fun = funcs[idx]
                lmbda = np.random.randint(0,10) * 10 ** np.random.randint(-5,-3)
                convlayer = None
                if conv:
                    convlayer = np.random.randint(1,6)
                    set = make_sets(alex_net(faces, convlayer), classes, 85*len(act), 35*len(act))


                print "Model:"
                print "\t", hu, funs, lmbda, convlayer
                c,a = train_neural_net(set, hu, fun, dropout=0.5, batch_size=50, index=i, lmbda=lmbda, best=best_val_accuracy,
                                       max_iter=5000, conv_input=conv)

                if a[2] > best_val_accuracy:
                    best_val_accuracy = a[1]
                    print "Found new best model!"
                    print "\tCost (Accuracy):"
                    print "\t\t %f (%4.2f%%)" %(best_val_accuracy, a[1])
                    print "\tModel:"
                    print "\t\t", hu, funs, lmbda, "\n\n"

                if conv:
                    with open('results/models_conv.csv', 'a') as fl:
                        fl.write("%d, %s, %s, %s, %d, %5.4f, "
                                 "%5.4f, %5.4f, %5.2f, %5.2f, %5.2f\n" %(i, hu, funs, lmbda, convlayer, c[0], c[1], c[2],
                                                                         a[0], a[1], a[2]))

                else:
                    with open('results/models.csv', 'a') as fl:
                        fl.write("%d, %s, %s, %s, %5.4f, %5.4f,"
                                 " %5.4f, %5.2f, %5.2f, %5.2f\n" %(i, hu, funs, lmbda, c[0], c[1], c[2], a[0], a[1], a[2]))
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print "Error in iteration, restarting script..."
                os.execl(sys.executable, sys.executable, *sys.argv)

    except KeyboardInterrupt:
        print "Stopping hyper-parameter search"

with open('results/neural_network_ff.pickle', 'rb') as f:
    params = cp.load(f)
if len(params) > 2:
    [w0, w1, b0, b1] = params
else:
    [w0, b0] = params

visualize_weights(w0, 100, plot=False, title="FF")


with open('results/neural_network_conv.pickle', 'rb') as f:
    params = cp.load(f)
if len(params) > 2:
    [w0, w1, b0, b1] = params
else:
    [w0, b0] = params

visualize_weights(w0, 100, plot=False, title="Conv")