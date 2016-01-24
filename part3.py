from part2 import fetch_sets
import numpy as np
from k_nearest_neighbors import knn_classify
from distance_functions import euclidean_distance
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imresize
from random import shuffle, seed, random
import os
import shutil

plt.gray()
seed(0)
if os.path.exists("results/k sensitivity"):
    shutil.rmtree("results/k sensitivity")
os.makedirs("results/k sensitivity")
if os.path.exists("results/mislabels"):
    shutil.rmtree("results/mislabels")
os.makedirs("results/mislabels")

x_train_f, t_train_f, x_validation_f, t_validation_f, x_test_f, t_test_f = fetch_sets("subset_actresses.txt", ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon'])
x_train_m, t_train_m, x_validation_m, t_validation_m, x_test_m, t_test_m = fetch_sets("subset_actors.txt", ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan'])

x_train = x_train_f + x_train_m
t_train = np.array(t_train_f + t_train_m)

x_validation = x_validation_f + x_validation_m
t_validation = np.array(t_validation_f + t_validation_m)

x_test = x_test_f + x_test_m
t_test = np.array(t_test_f + t_test_m)

# min_dim = min(map(np.shape, x_train+x_validation+x_test))
min_dim = (32,32)

xto = np.array(x_train)
xteo = np.array(x_test)

x_train = np.array([np.hstack(imresize(x, min_dim)) for x in x_train])
x_validation = np.array([np.hstack(imresize(x, min_dim)) for x in x_validation])
x_test = np.array([np.hstack(imresize(x, min_dim)) for x in x_test])

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
for j, k in enumerate(best_k):
    for i, xi in enumerate(x_test):
        ti, _ = knn_classify(x_train, t_train, xi, k, euclidean_distance)
        if ti != t_test[i]:
            test_errors[j] += 1

            if random() <= 0.5:
                _, nn = knn_classify(x_train, t_train, xi, 5, euclidean_distance)
                plt.subplot(1,2,1)
                plt.imshow(xteo[i])
                plt.title(t_test[i])
                plt.axis('off')
                for n, m in enumerate([3,4,7,8,11]):
                    plt.subplot(3,4,m)
                    plt.imshow(xto[nn[n]])
                    plt.title(t_train[nn[n]])
                    plt.axis('off')
                plt.savefig('results/mislabels/'+str(j)+'_'+str(i)+'.eps')
                plt.clf()
    print "K = %d - Test Errors = %d" %(k, test_errors[j])

best_k_t = np.where(test_errors == test_errors.min())[0]
best_k_t = best_k[best_k_t]
print "Best values for k: %s" %best_k_t

font = {'size' : 17}
matplotlib.rc('font', **font)
plt.axis('on')
plt.plot([i for i in krange], validation_errors/len(x_validation)*100, label='Validation Set', marker='o', markersize=5)
plt.scatter([i for i in best_k], test_errors/len(x_test)*100, label='Test Set', marker='*', s=200, color='r')
plt.xlabel('Neighbors Considered')
plt.ylabel('% of Classification Errors')
plt.title('K Sensitivity Test')
plt.axis([-10, len(x_train), 0, 100])
plt.legend(loc=0)
plt.grid()
plt.savefig('results/k sensitivity/k sensitivity.eps')
plt.show()