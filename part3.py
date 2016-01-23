from part2 import fetch_sets
import numpy as np
from k_nearest_neighbors import knn_classify
from distance_functions import euclidean_distance
import matplotlib.pyplot as plt
from scipy.misc import imresize

x_train_f, t_train_f, x_validation_f, t_validation_f, x_test_f, t_test_f = fetch_sets("subset_actresses.txt", ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon'])
x_train_m, t_train_m, x_validation_m, t_validation_m, x_test_m, t_test_m = fetch_sets("subset_actors.txt", ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan'])

x_train = x_train_f + x_train_m
t_train = np.array(t_train_f + t_train_m)

x_validation = x_validation_f + x_validation_m
t_validation = np.array(t_validation_f + t_validation_m)

x_test = x_test_f + x_test_m
t_test = np.array(t_test_f + t_test_m)

min_dim = min(map(np.shape, x_train+x_validation+x_test))

x_train = np.array([np.hstack(imresize(x, min_dim)) for x in x_train])
x_validation = np.array([np.hstack(imresize(x, min_dim)) for x in x_validation])
x_test = np.array([np.hstack(imresize(x, min_dim)) for x in x_test])

# krange = range(1,len(x_train),2)
krange = [3, 9, 11, 15]

validation_errors = np.zeros(len(krange))
test_errors = np.zeros(len(krange))
for j, k in enumerate(krange):
    for i, xi in enumerate(x_validation):
        ti = knn_classify(x_train, t_train, xi, k, euclidean_distance)
        if ti != t_validation[i]:
            validation_errors[j] += 1
    for i, xi in enumerate(x_test):
        ti = knn_classify(x_train, t_train, xi, k, euclidean_distance)
        if ti != t_test[i]:
            test_errors[j] += 1
    print "K = %d - Validation Errors = %d - Test Errors = %d" %(k, validation_errors[j], test_errors[j])

plt.plot([i for i in krange], validation_errors/len(x_validation), label='Validation Errors')
plt.plot([i for i in krange], test_errors/len(x_test), label='Test Errors')
plt.xlabel('Neighbors Considered')
plt.ylabel('% of Validation Errors')
plt.title('K Sensitivity Test')
plt.legend(loc=0)
plt.grid()
plt.show()