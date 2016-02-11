from math_funcs import *
from part1 import *
from part2 import *
from part3 import *
from part4 import *
from part5 import *

plot = raw_input("Show plots? [y/N]").lower() == 'y'
ext = raw_input("Extension to save plots? [pdf]") or 'pdf'

print("RUNNING PART 1\n--------------")

images, classes = load_mnist_mat('mnist_all.mat', plot, ext)

x_train, t_train, x_validation, t_validation, x_test, t_test = \
    make_sets(images, classes, train=40000, validation=10000)


print("RUNNING PART 4\n--------------")

compare_gradient(x_train, [np.ones((784,10))], [np.ones(10)], [softmax], np.array(t_train))


print("RUNNING PART 5\n--------------")
w, b = np.zeros((784,10)), np.zeros(10)
epoch = 0
while True:
    epoch+=1
    w, b = backpropagate_momentum_step(x_train, w, b, t_train, 0.1, 0, 50)

    print "Epoch: %3d" % (epoch)
    train_cost, train_accuracy = \
        evaluate_cost(x_train, w, b, t_train), evaluate_accuracy(x_train, w, b, t_train)
    validation_cost, validation_accuracy = \
        evaluate_cost(x_validation, w, b, t_validation), evaluate_accuracy(x_validation, w, b, t_validation)
    test_cost, test_accuracy\
        = evaluate_cost(x_test, w, b, t_test), evaluate_accuracy(x_test, w, b, t_test)
    print "\tTraining Set:   Cost: %8.3f Accuracy: %5.2f%%" %(train_cost, train_accuracy*100)
    print "\tValidation Set: Cost: %8.3f Accuracy: %5.2f%%" %(validation_cost, validation_accuracy*100)
    print "\tTest Set:       Cost: %8.3f Accuracy: %5.2f%%\n\n" %(test_cost, test_accuracy*100)