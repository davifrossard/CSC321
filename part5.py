from part2 import fetch_sets
import numpy as np
from k_nearest_neighbors import knn_classify
from distance_functions import euclidean_distance
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imresize
import os
import shutil
import sys

if len(sys.argv) == 3:
    save_ext = sys.argv[1]
    plot_graphs = (sys.argv[2] == '1')
else:
    save_ext = 'eps'
    plot_graphs = False

plt.gray()
if os.path.exists("results/part 5/k sensitivity"):
    shutil.rmtree("results/part 5/k sensitivity")
os.makedirs("results/part 5/k sensitivity")
if os.path.exists("results/part 5/mislabels"):
    shutil.rmtree("results/part 5/mislabels")
os.makedirs("results/part 5/mislabels")

x_train_f, _, x_validation_f, _, x_test_f, _ = fetch_sets("subset_actresses.txt", ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon'], 100, 10, 10)
x_train_m, _, x_validation_m, _, x_test_m, _ = fetch_sets("subset_actors.txt", ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan'], 100, 10, 10)
genders = ['Male', 'Female']
x_train = x_train_f + x_train_m
t_train = np.hstack((np.ones(len(x_train_f)), np.zeros(len(x_train_m))))

x_validation = x_validation_f + x_validation_m
t_validation = np.hstack((np.ones(len(x_validation_f)), np.zeros(len(x_validation_m))))

x_test = x_test_f + x_test_m
t_test = np.hstack((np.ones(len(x_test_f)), np.zeros(len(x_test_m))))

min_dim = min(map(np.shape, x_train+x_validation+x_test))

xto = np.array(x_train)
xteo = np.array(x_test)

x_train = np.array([np.hstack(imresize(x, (32,32))) for x in x_train])
x_validation = np.array([np.hstack(imresize(x, (32,32))) for x in x_validation])
x_test = np.array([np.hstack(imresize(x, (32,32))) for x in x_test])

krange = range(1,len(x_train),2)
validation_errors = np.zeros(len(krange))
for j, k in enumerate(krange):
    for i, xi in enumerate(x_validation):
        ti, _ = knn_classify(x_train, t_train, xi, k, euclidean_distance)

        if ti != t_validation[i]:
            validation_errors[j] += 1
    print "K = %d - Validation Errors = %d" %(k, validation_errors[j])

best_k = np.where(validation_errors == validation_errors.min())[0]
best_k = np.array(krange)[best_k]
print "Best values for k: %s\n" %best_k

test_errors = np.zeros(len(best_k))
trigger = 0
for j, k in enumerate(best_k):
    for i, xi in enumerate(x_test):
        ti, _ = knn_classify(x_train, t_train, xi, k, euclidean_distance)
        if ti != t_test[i]:
            test_errors[j] += 1

            if trigger % 2 == 0:
                _, nn = knn_classify(x_train, t_train, xi, 5, euclidean_distance)
                plt.subplot(1,2,1)
                plt.imshow(xteo[i])
                plt.title(genders[int(t_test[i])])
                plt.axis('off')
                for n, m in enumerate([3,4,7,8,11]):
                    plt.subplot(3,4,m)
                    plt.imshow(xto[nn[n]])
                    plt.title(genders[int(t_train[nn[n]])])
                    plt.axis('off')
                plt.savefig('results/part 6/mislabels/%d_%d.%s' %(k, i, save_ext))
                plt.clf()
            trigger += 1
    print "K = %d - Test Errors = %d" %(k, test_errors[j])

best_k_t = np.where(test_errors == test_errors.min())[0]
best_k_t = best_k[best_k_t]
print "Best values for k: %s" %best_k_t

font = {'size' : 15}
matplotlib.rc('font', **font)
plt.axis('on')
plt.plot([i for i in krange], validation_errors/len(x_validation)*100, label='Validation Set', marker='o')
plt.scatter([i for i in best_k], test_errors/len(x_test)*100, label='Test Set', marker='*', s=150, color='r')
plt.xlabel('Neighbors Considered')
plt.ylabel('% of Classification Errors')
plt.title('K Sensitivity Test')
plt.axis([-10, len(x_train), 0, 100])
plt.legend(loc=0)
plt.grid()
plt.savefig('results/part 5/k sensitivity/k sensitivity.%s' %(save_ext))
plt.show() if plot_graphs else plt.clf()