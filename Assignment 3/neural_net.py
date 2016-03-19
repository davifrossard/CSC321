import numpy as np
import tensorflow as tf
import time, datetime
import matplotlib.pyplot as plt
import cPickle as cp

from operator import mul
from functools import reduce

def train_neural_net(sets, hidden_units=[300], functions=[tf.nn.relu], batch_size=50, dropout=0.8, index=0, lmbda=0,
                     max_iter=3000, best=float('inf'), conv_input=False, save=False, title=None):

    x_train, t_train, x_validation, t_validation, x_test, t_test = sets

    in_size = reduce(mul, list(x_train[0].shape))
    in_dim = int(np.sqrt(in_size))
    nclass = 6
    EXP = 1e-5 # Convergence tolerance
    two_layer = len(hidden_units) > 1

    x = tf.placeholder(tf.float32, [None, in_size], name="Input")
    y = tf.placeholder(tf.float32, [None, nclass], name="Expected_Output")
    keep_prob = tf.placeholder(tf.float32, name="Keep_Probability")

    ''' NEURAL NETWORK TOPOLOGY '''
    # Hidden Layer
    with tf.name_scope("Hidden_Layer") as scope:
        w0 = tf.Variable(tf.truncated_normal([in_size, hidden_units[0]], mean=0.01, stddev=0.01), name="Weight")
        b0 = tf.Variable(tf.truncated_normal([hidden_units[0]], mean=0.01, stddev=0.01), name="Bias")
        a0 = functions[0](tf.matmul(x, w0) + b0)
        d0 = tf.nn.dropout(a0, keep_prob)
        out = d0

    # Second hidden layer
    if two_layer:
        with tf.name_scope("2nd_Hidden_Layer") as scope:
            w1 = tf.Variable(tf.truncated_normal([hidden_units[0], hidden_units[1]], mean=0.01, stddev=0.01), name="Weight")
            b1 = tf.Variable(tf.truncated_normal([hidden_units[1]], mean=0.01, stddev=0.01), name="Bias")
            a1 = functions[1](tf.matmul(out, w1) + b1)
            d1 = tf.nn.dropout(a1, keep_prob)
            out = d1

    # Output Layer
    with tf.name_scope("Output_Layer") as scope:
        wout = tf.Variable(tf.truncated_normal([hidden_units[-1], nclass], mean=0.01, stddev=0.01), name="Weight")
        bout = tf.Variable(tf.truncated_normal([nclass], mean=0.01, stddev=0.01), name="Bias")
        logits = tf.matmul(out, wout) + bout
        output = tf.nn.softmax(logits)
    ''' END NEURAL NETWORK TOPOLOGY '''


    ''' OBJECTIVE PARAMETERS '''
    #Training Specification
    with tf.name_scope("Training") as scope:
        iter_var = tf.Variable(0)
        with tf.name_scope("Regularization") as scope:
            regularizer = tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) if two_layer else tf.nn.l2_loss(w0)
            l1_regularizer = tf.reduce_mean(tf.abs(w0)) + tf.reduce_mean(tf.abs(w1)) if two_layer else tf.reduce_mean(tf.abs(w0))
        cost_batch = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        cost = tf.reduce_mean(cost_batch) + lmbda * regularizer
        optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=iter_var)

    # Test accuracy
    with tf.name_scope("Evaluation") as scope:
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100
    ''' END OBJECTIVE PARAMETERS '''

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    ''' TENSORBOARD CONFIGURATION '''
    writer = tf.train.SummaryWriter("tboard/%d" %(index), sess.graph_def)

    # Histograms
    w_hist = tf.histogram_summary("1st_Weights", w0)
    b_hist = tf.histogram_summary("1st_Biases", b0)
    y_hist = tf.histogram_summary("Output", output)
    hists = tf.merge_summary([w_hist, b_hist, y_hist])
    if two_layer:
        w1_hist = tf.histogram_summary("2nd_Weights", w1)
        b1_hist = tf.histogram_summary("2nd_Biases", b1)
        hists = tf.merge_summary([hists, w1_hist, b1_hist])
    # weights_image = tf.reshape(w0, [-1, in_dim, in_dim, 1])
    # w_image = tf.image_summary('Weights', weights_image, max_images=25)
    # hists = tf.merge_summary([hists, w_image])

    # Summaries
    train_accuracy_summary = tf.scalar_summary("Train_Accuracy", accuracy)
    validation_accuracy_summary = tf.scalar_summary("Validation_Accuracy", accuracy)
    test_accuracy_summary = tf.scalar_summary("Test_Accuracy", accuracy)

    train_cost_summary = tf.scalar_summary("Train_Cost", cost)
    validation_cost_summary = tf.scalar_summary("Validation_Cost", cost)
    test_cost_summary = tf.scalar_summary("Test_Cost", cost)

    train_ops = [tf.merge_summary([hists, train_accuracy_summary, train_cost_summary]), cost, accuracy]
    validation_ops = [tf.merge_summary([validation_accuracy_summary, validation_cost_summary]), cost, accuracy]
    test_ops = [tf.merge_summary([test_accuracy_summary, test_cost_summary]), cost, accuracy]
    '''END TENSORBOARD CONFIGURATION'''

    last_cost = 0
    last_val = 0

    with sess.as_default():
        # Training cycle
        total_batches = int(x_train.shape[0] / batch_size)
        while True:
            epoch = iter_var.eval()/total_batches+1

            #Shuffle training set
            ids = np.random.permutation(len(x_train))
            x_train = x_train[ids]
            t_train = t_train[ids]

            # Complete training epoch
            for i in range(total_batches):
                batch_xs = x_train[i*batch_size:(i+1)*batch_size]
                batch_ys = t_train[i*batch_size:(i+1)*batch_size]
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

            # Evaluate network
            if epoch % 5 == 0:
                train = sess.run(train_ops, feed_dict={x: x_train, y: t_train, keep_prob: 1})
                writer.add_summary(train[0], epoch)

                validation = sess.run(validation_ops, feed_dict={x: x_validation, y: t_validation, keep_prob: 1})
                writer.add_summary(validation[0], epoch)

                test = sess.run(test_ops, feed_dict={x: x_test, y: t_test, keep_prob: 1})
                writer.add_summary(test[0], epoch)

                if abs(train[1] - last_cost) < EXP:
                    print "Converged!"
                    break
                
                if validation[2] > last_val:
                    print "New best generation:"
                    print "\t Accuracy: %4.2f%%" %(validation[2])
                    bench_val = [train[1], validation[1], test[1]], [train[2], validation[2], test[2]]
                    last_val = validation[2]
                    if two_layer:
                        params = [w0.eval(), w1.eval(), b0.eval(), b1.eval(), wout.eval(), bout.eval()]
                    else:
                        params = [w0.eval(), b0.eval(), wout.eval(), bout.eval()]

                last_cost = train[1]

                if epoch % 50 == 0:

                    st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
                    print "[%s] Epoch: %3d" % (st, epoch)
                    print "\tTraining Set:   Cost: %8.3f Accuracy: %d%%" %(train[1], train[2])
                    print "\tValidation Set: Cost: %8.3f Accuracy: %d%%" %(validation[1], validation[2])
                    print "\tTest Set:       Cost: %8.3f Accuracy: %d%%\n\n" %(test[1], test[2])

                    if epoch % max_iter == 0:
                        print "Too many epochs!"
                        break

    if validation[2] > best or save:
        with open('results/neural_network_%s%s.pickle' %('conv' if conv_input else 'ff', '_'+str(title) if title else ''), 'wb') as f:
            cp.dump(params, f)

    sess.close()

    print "Optimization Finished!"

    return bench_val


def visualize_weights(w, num=-1, title=None):
    dim = w.shape[1]
    shp = [int(np.sqrt(w.shape[0]))] * 2
    dims = int(np.sqrt(dim)) if num == -1 else min(int(np.sqrt(num)),int(np.sqrt(dim)))
    ids = range(dim) if num == -1 else np.random.choice(dim, min(dim, num*2))
    fig, axes = plt.subplots(nrows=dims, ncols=dims)
    fig.suptitle('Weight Visualization', size=20)
    for i, ax in enumerate(axes.flat):
        heatmap = ax.imshow(w[:,ids[i]].reshape(shp), cmap = plt.cm.coolwarm)
        ax.set_axis_off()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(heatmap, cax=cbar_ax)
    plt.savefig('results/weights_%s.pdf' %(title), bbox_inches='tight')
    plt.close()
