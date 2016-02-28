from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt
import shutil, os

def make_sets(data, classes, train=1500, validation=1000):

    np.random.seed(0)
    source = zip(data, classes)
    np.random.shuffle(source)
    data, classes = zip(*source)
    data = np.array(np.array(data).astype('float32')/np.max(data))
    classes = np.array(classes)

    x_train = data[:train]
    t_train = classes[:train]

    x_validation = data[train:validation + train]
    t_validation = classes[train:validation+train]

    x_test = data[validation + train:]
    t_test = classes[validation+train:]

    return x_train, t_train, x_validation, t_validation, x_test, t_test

def to_one_hot(data):
    num_classes = max(data) + 1

    one_hot = np.zeros([len(data), num_classes])

    for i, hot in enumerate(data):
        one_hot[i, hot] = 1

    return one_hot

def load_mnist_mat(file, plot=False, save_ext='pdf'):
    images = []
    classes = []
    data = loadmat(file)
    plt.gray()
    plt.suptitle('Digit Samples', size=20)
    for i in range(10):
        for j in range(10):        
            plt.subplot(10,10,i*10+j+1)
            plt.imshow(data['train%d'%(i)][j].reshape((28,28)))
            plt.axis('off')
        for image in data['train%d'%(i)]:
            images.append(image)
            classes.append(i)
    images = np.array(images)
    classes = to_one_hot(classes)
    plt.savefig('results/part1.%s' %(save_ext), bbox_inches='tight')
    plt.show() if plot else plt.close()
    return images, classes
