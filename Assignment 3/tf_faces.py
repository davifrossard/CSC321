import tensorflow as tf
import time
import cPickle as cp
import matplotlib.pyplot as plt
from copy import deepcopy

from tensorflow.python.ops import gradients

from get_data_file import *
from set_utils import *
import neural_net as nnt
from neural_net import train_neural_net, visualize_weights
from AlexNet.myalexnet import alex_net
from AlexNet.anet_actors import *

hp_search, conv_search = None, None
if len(sys.argv) == 3:
    hp_search = (sys.argv[1] == '1')
    conv_search = (sys.argv[2] == '1')

# Configurations
datafile = 'data.txt'
conv = False

# Create results dir
if not os.path.exists("results"):
    os.makedirs("results")

# Retrieve data from file
act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
faces_o, classes = fetch_data(datafile, act, -1)



''' PART 1 AND 2 '''
print "\n\nRUNNING PART 1/2"
''' Hyperparameter Search '''
if hp_search or raw_input("Search for hyper-parameters? [y/N]\n[Must be ran at least once for each network]").lower() == 'y':
    if conv_search or (conv_search is None and raw_input("Use raw images (0) or AlexNet output (1) ? [1] ").lower() != '0'):
        conv = True
        faces = convert_faces(deepcopy(faces_o))
        i, best_val_accuracy, _ = get_data('results/models_conv.csv')


    else:
        dim = (100,100)
        dimf = dim[0] * dim[1]
        faces = deepcopy(faces_o)
        for i, face in enumerate(faces_o):
            faces[i] = rgb2gray(imresize(face, dim)).reshape(dimf)/255.0

        i, best_val_accuracy, _ = get_data('results/models.csv')
        set = make_sets(faces, classes, 85*len(act), 27*len(act))

    print "Best accuracy so far: %.2f%%" %best_val_accuracy
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
                hu = np.random.randint(200, 850, nlayer)
                idx = np.random.choice(2, nlayer)
                funs = funcs_n[idx]
                fun = funcs[idx]
                lmbda = np.random.randint(0,10) * 10 ** np.random.randint(-5,-2)
                convlayer = None
                if conv:
                    convlayer = np.random.randint(1,6)
                    print "Preparing AlexNet outputs:"
                    set = make_sets(alex_net(faces, convlayer), classes, 85*len(act), 27*len(act))
                    print "\n"

                print "Model %d:" %i
                print "\t", hu, funs, lmbda, convlayer
                bench,cs,acs = train_neural_net(set, hu, fun, dropout=0.5, batch_size=50, index=i, lmbda=lmbda,
                                                best=best_val_accuracy, max_iter=1500, conv_input=conv)
                c,a = bench
                # Found new model according to validation accuracy
                if a[1] > best_val_accuracy:
                    best_val_accuracy = a[1]
                    print "Found new best model!"
                    print "\tCost (Accuracy):"
                    print "\t\t %f (%4.2f%%)" %(best_val_accuracy, a[1])
                    print "\tModel:"
                    print "\t\t", hu, funs, lmbda, "\n\n"
                    nnt.plot_curves(cs,acs,autotitle=True, conv=conv)
                    if conv:
                        with open('results/models_conv_best.csv', 'w+') as fl:
                            fl.write("%s, %s, %s, %d, %5.4f, "
                                     "%5.4f, %5.4f, %5.2f, %5.2f, %5.2f\n" %(hu, funs, lmbda, convlayer, c[0], c[1], c[2],
                                                                             a[0], a[1], a[2]))
                    else:
                        with open('results/models_best.csv', 'w+') as fl:
                            fl.write(" %s, %s, %s, %5.4f, %5.4f,"
                                     " %5.4f, %5.2f, %5.2f, %5.2f\n" %(hu, funs, lmbda, c[0], c[1], c[2], a[0], a[1], a[2]))
                # Save model to log
                if conv:
                    with open('results/models_conv.csv', 'a') as fl:
                        fl.write("%d, %s, %s, %s, %d, %5.4f, "
                                 "%5.4f, %5.4f, %5.2f, %5.2f, %5.2f\n" %(i, hu, funs, lmbda, convlayer, c[0], c[1], c[2],
                                                                         a[0], a[1], a[2]))

                else:
                    with open('results/models.csv', 'a') as fl:
                        fl.write("%d, %s, %s, %s, %5.4f, %5.4f,"
                                 " %5.4f, %5.2f, %5.2f, %5.2f\n" %(i, hu, funs, lmbda, c[0], c[1], c[2], a[0], a[1], a[2]))

                # For automation purposes, do not search mode than 50 models of each
                if i>=25:
                    if not conv:
                        os.execl(sys.executable, 'python', __file__, '1', '1')
                    else:
                        break
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print "Error in iteration, restarting script..."
                os.execl(sys.executable, 'python', __file__, '1', str(int(conv)))

    except KeyboardInterrupt:
        print "Stopping hyper-parameter search"

np.random.seed(0)

''' PART 3 '''
print "\n\nRUNNING PART 3"
# Model with 300 hidden units
if not os.path.exists('results/neural_network_ff_300.pickle'):
    dim = (100,100)
    dimf = dim[0] * dim[1]
    faces = deepcopy(faces_o)
    for i, face in enumerate(faces_o):
        faces[i] = rgb2gray(imresize(face, dim)).reshape(dimf)/255.0
    set = make_sets(faces, classes, 85*len(act), 35*len(act))
    train_neural_net(set, [300], [tf.nn.relu], dropout=0.5, batch_size=50, index=100, lmbda=0.001,
                                       max_iter=1000, save=True, title='300')
with open('results/neural_network_ff_300.pickle', 'rb') as f:
    params = cp.load(f)
    w0 = params[0]
visualize_weights(w0, 36, title="300")


# Model with 800 hidden units
if not os.path.exists('results/neural_network_ff_800.pickle'):
    dim = (100,100)
    dimf = dim[0] * dim[1]
    faces = deepcopy(faces_o)
    for i, face in enumerate(faces_o):
        faces[i] = rgb2gray(imresize(face, dim)).reshape(dimf)/255.0
    set = make_sets(faces, classes, 85*len(act), 35*len(act))
    train_neural_net(set, [800], [tf.nn.relu], dropout=0.5, batch_size=50, index=100, lmbda=0.001,
                                       max_iter=1000, save=True, title='800')
with open('results/neural_network_ff_800.pickle', 'rb') as f:
    params = cp.load(f)
    w0 = params[0]
visualize_weights(w0, 36, title="800")



''' PART 4 '''
print "\n\nRUNNING PART 4"
dim = (227,227)

with open('results/neural_network_conv.pickle', 'rb') as f:
    params = cp.load(f)
    _,_, layer = get_data('results/models_conv.csv')
    params.append(layer)

outputs, _ = eval_alex_net(deepcopy(faces_o), params)
ids = np.random.choice(len(faces_o), 16)
for i, id in enumerate(ids):
    plt.subplot(4,4,i+1)
    plt.tight_layout()
    plt.imshow(faces_o[id])
    outs = outputs[id][0]
    out_classes = np.argsort(outs)
    pclass, pclass2 = out_classes[-1], out_classes[-2]
    pcertainty, pcertainty2 = outs[pclass]*100, outs[pclass2]*100
    title = "%s (%.2f%%)\n%s (%.2f%%)" %(act[pclass], pcertainty, act[pclass2], pcertainty2)
    if pclass != np.argmax(classes[id]):
        plt.title(title, color='red', size=10)
    else:
        plt.title(title, color='green', size=10)
    plt.axis('off')
plt.suptitle('Neural Network Predictions', size=20)
plt.subplots_adjust(top=0.8)
plt.savefig('results/part4_predictions.pdf')
plt.close()


''' PART 5 '''
print "\n\nRUNNING PART 5"
with open('results/neural_network_conv.pickle', 'rb') as f:
    params = cp.load(f)
    _,_, layer = get_data('results/models_conv.csv')
    params.append(layer)
gradient_output(deepcopy(faces_o[720:]), params)
plt.savefig('results/part5_gradient.pdf')
plt.close()


''' PART 6 '''
print "\n\nRUNNING PART 6"
with open('results/neural_network_conv.pickle', 'rb') as f:
    params = cp.load(f)
    _,_, layer = get_data('results/models_conv.csv')
    params.append(layer)

facesi = convert_faces(deepcopy(faces_o))
set = make_sets(facesi, classes, 85*len(act), 35*len(act))

cost, accuracies, newbest = train_alex_net(set, params, lmbda=0.0001, keep_prob=0.7, max_iter=200)
if not os.path.exists("results/part6_cost_curve.pdf") or newbest:
    plot_curves(cost, accuracies)


''' PART 7 '''
print "\n\nRUNNING PART 7"
with open('results/neural_network_conv.pickle', 'rb') as f:
    params = cp.load(f)
    _,_, layer = get_data('results/models_conv.csv')
    params.append(layer)

plot_gradients(deepcopy(faces_o[720:]), params, act)
plt.savefig('results/part7_gradient.pdf')
plt.close()
