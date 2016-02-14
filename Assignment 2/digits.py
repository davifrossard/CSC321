from math_funcs import *
from part1 import *
from part2 import *
from part3 import *
from part4 import *
from part5 import *
from part6 import *
from part7 import *
from part8 import *
from part9 import *

import pickle
import os
import matplotlib.pyplot as plt
import time, datetime

np.random.seed(0)

plot = raw_input("Show plots? [y/N]").lower() == 'y'
ext = raw_input("Extension to save plots? [pdf]") or 'pdf'

print("\n--------------\nRUNNING PART 1\n--------------")

images, classes = load_mnist_mat('mnist_all.mat', plot, ext)

x_train, t_train, x_validation, t_validation, x_test, t_test = \
    make_sets(images, classes, train=40000, validation=10000)


print("\n--------------\nRUNNING PART 4\n--------------")
w, b = np.zeros((784,10)), np.zeros(10)
compare_gradient(x_test[0:50], w, b, t_test[0:50])

print("\n--------------\nRUNNING PART 5\n--------------")
epoch = 0

train_cost, train_accuracy = [], []
validation_cost, validation_accuracy = [], []
test_cost, test_accuracy = [], []
converged = False

if os.path.exists('logistic_regression.pickle'):
    load = raw_input('Found saved neural network on disk, load? [Y/n]').lower() != 'n'
    if load:
        with open('logistic_regression.pickle') as f:
            w, b, epoch, converged, train_cost, train_accuracy, validation_cost,\
            validation_accuracy, test_cost, test_accuracy = pickle.load(f)
            converged = False

while not converged and epoch < 500:
    epoch+=1
    w, b = backpropagate_momentum_epoch(x_train, w, b, t_train, 0.1, 0.07, 50)

    train_cost.append(evaluate_cost(x_train, w, b, t_train))
    train_accuracy.append(evaluate_accuracy(x_train, w, b, t_train)[0])

    validation_cost.append(evaluate_cost(x_validation, w, b, t_validation))
    validation_accuracy.append(evaluate_accuracy(x_validation, w, b, t_validation)[0])

    test_cost.append(evaluate_cost(x_test, w, b, t_test))
    test_accuracy.append(evaluate_accuracy(x_test, w, b, t_test)[0])
    #
    # if epoch > 2 and abs(train_cost[-1] - train_cost[-2]) < 1e-4:
    #     converged = True

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print "[%s] Epoch: %3d" % (st, epoch)
    print "\tTraining Set:   Cost: %8.3f Accuracy: %5.2f%%" %(train_cost[-1], train_accuracy[-1]*100)
    print "\tValidation Set: Cost: %8.3f Accuracy: %5.2f%%" %(validation_cost[-1], validation_accuracy[-1]*100)
    print "\tTest Set:       Cost: %8.3f Accuracy: %5.2f%%\n\n" %(test_cost[-1], test_accuracy[-1]*100)

    with open('logistic_regression.pickle', 'w') as f:
        pickle.dump([w, b, epoch, converged, train_cost, train_accuracy, validation_cost,\
                     validation_accuracy, test_cost, test_accuracy], f)

# Cost History Graph
plt.figure()
plt.plot(range(epoch), train_cost, label='Train Set')
plt.plot(range(epoch), validation_cost, label='Validation Set')
plt.plot(range(epoch), test_cost, label='Test Set')
plt.legend(loc='best')
plt.grid()
plt.title('Cost History')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.savefig('results/part5_cost.%s' %(ext), bbox_inches='tight')
plt.show() if plot else plt.close()

# Accuracy History Graph
plt.figure()
plt.plot(range(epoch), train_accuracy, label='Train Set')
plt.plot(range(epoch), validation_accuracy, label='Validation Set')
plt.plot(range(epoch), test_accuracy, label='Test Set')
plt.legend(loc='best')
plt.grid()
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('results/part5_accuracy.%s' %(ext), bbox_inches='tight')
plt.show() if plot else plt.close()

# Plot Successes and Failures
_, matches, outputs = evaluate_accuracy(x_test, w, b, t_test)
failures = [i for i in range(len(x_test)) if i not in matches]

fig, axes = plt.subplots(nrows=4, ncols=5)
fig.suptitle('Success Cases', size=20)
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[matches[i]].reshape((28,28)))
    ax.set_title(outputs[matches[i]])
    ax.set_axis_off()
plt.savefig('results/part5_successes.%s' %(ext), bbox_inches='tight')
plt.show() if plot else plt.close()

fig, axes = plt.subplots(nrows=2, ncols=5)
fig.suptitle('Failure Cases', size=20)
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[failures[i]].reshape((28,28)))
    ax.set_title(outputs[failures[i]])
    ax.set_axis_off()
plt.savefig('results/part5_failures.%s' %(ext), bbox_inches='tight')
plt.show() if plot else plt.close()


print("\n--------------\nRUNNING PART 6\n--------------")
visualize_weights(w, plot, ext)


print("\n--------------\nRUNNING PART 8\n--------------")
x, y, w, b = x_test, t_test, [np.zeros((784,300)), np.zeros((300,10))], [np.random.rand(300), np.random.rand(10)]
compare_gradient2(x[0:10], w, b, y[0:10])


print("\n--------------\nRUNNING PART 9\n--------------")
epoch = 0

train_cost, train_accuracy = [], []
validation_cost, validation_accuracy = [], []
test_cost, test_accuracy = [], []
converged = False

if os.path.exists('neural_network.pickle'):
    load = raw_input('Found saved neural network on disk, load? [Y/n]').lower() != 'n'
    if load:
        with open('neural_network.pickle') as f:
            w, b, epoch, converged, train_cost, train_accuracy, validation_cost,\
            validation_accuracy, test_cost, test_accuracy = pickle.load(f)

while not converged and epoch < 500:
    epoch+=1
    w, b = backpropagate_momentum_epoch2(x_train, w, b, t_train, 0.01, 0.005, 50)

    train_cost.append(evaluate_cost2(x_train, w, b, t_train))
    train_accuracy.append(evaluate_accuracy2(x_train, w, b, t_train)[0])

    validation_cost.append(evaluate_cost2(x_validation, w, b, t_validation))
    validation_accuracy.append(evaluate_accuracy2(x_validation, w, b, t_validation)[0])

    test_cost.append(evaluate_cost2(x_test, w, b, t_test))
    test_accuracy.append(evaluate_accuracy2(x_test, w, b, t_test)[0])

    # if epoch > 2 and abs(train_cost[-1] - train_cost[-2]) < 1e-4:
    #     converged = True

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print "[%s] Epoch: %3d" % (st, epoch)
    print "\tTraining Set:   Cost: %8.3f Accuracy: %5.2f%%" %(train_cost[-1], train_accuracy[-1]*100)
    print "\tValidation Set: Cost: %8.3f Accuracy: %5.2f%%" %(validation_cost[-1], validation_accuracy[-1]*100)
    print "\tTest Set:       Cost: %8.3f Accuracy: %5.2f%%\n\n" %(test_cost[-1], test_accuracy[-1]*100)

    with open('neural_network.pickle', 'w') as f:
        pickle.dump([w, b, epoch, converged, train_cost, train_accuracy, validation_cost,\
                     validation_accuracy, test_cost, test_accuracy], f)

# Cost History Graph
plt.figure()
plt.plot(range(epoch), train_cost, label='Train Set')
plt.plot(range(epoch), validation_cost, label='Validation Set')
plt.plot(range(epoch), test_cost, label='Test Set')
plt.legend(loc='best')
plt.grid()
plt.title('Cost History')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.savefig('results/part9_cost.%s' %(ext), bbox_inches='tight')
plt.show() if plot else plt.close()

# Accuracy History Graph
plt.figure()
plt.plot(range(epoch), train_accuracy, label='Train Set')
plt.plot(range(epoch), validation_accuracy, label='Validation Set')
plt.plot(range(epoch), test_accuracy, label='Test Set')
plt.legend(loc='best')
plt.grid()
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('results/part9_accuracy.%s' %(ext), bbox_inches='tight')
plt.show() if plot else plt.close()

# Plot Successes and Failures
_, matches, outputs = evaluate_accuracy2(x_test, w, b, t_test)
failures = [i for i in range(len(x_test)) if i not in matches]

fig, axes = plt.subplots(nrows=4, ncols=5)
fig.suptitle('Success Cases', size=20)
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[matches[i]].reshape((28,28)))
    ax.set_title(outputs[matches[i]])
    ax.set_axis_off()
plt.savefig('results/part9_successes.%s' %(ext), bbox_inches='tight')
plt.show() if plot else plt.close()

fig, axes = plt.subplots(nrows=2, ncols=5)
fig.suptitle('Failure Cases', size=20)
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[failures[i]].reshape((28,28)))
    ax.set_title(outputs[failures[i]])
    ax.set_axis_off()
plt.savefig('results/part9_failures.%s' %(ext), bbox_inches='tight')
plt.show() if plot else plt.close()

print("\n--------------\nRUNNING PART 10\n--------------")
wis = visualize_weights(w[0], plot, ext, 10)
for i, wi in enumerate(wis):
    print "Weights Associated with",i
    print w[1][wi]