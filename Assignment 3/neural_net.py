import numpy as np
import tensorflow as tf
import time, datetime

def train_neural_net(sets, hidden_units=[1500], functions=[tf.nn.relu], batch_size=50, training_epochs=5000, dropout=1):

    x_train, t_train, x_validation, t_validation, x_test, t_test = sets

    in_size = len(x_train[0])
    nclass = 6

    topology = {'topology': str("%d x %s x %d" %(in_size, " x ".join((str(hu) for hu in hidden_units)), nclass))}
    funcs = {'activation functions':functions}
    #NN Model
    x = tf.placeholder(tf.float32, [None, in_size], name="Inputs")
    y = tf.placeholder(tf.float32, [None, nclass], name="Expected_Ouputs")
    keep_prob = tf.placeholder("float", name="Dropout_Rate")

    layer_w, layer_b, layer_a, layer_drop = [], [], [], []

    #Input Layer
    with tf.name_scope("Input_Layer") as scope:
        layer_w.append(tf.Variable(tf.random_uniform([in_size, hidden_units[0]]), name="Input_Weight"))
        layer_b.append(tf.Variable(tf.random_uniform([hidden_units[0]]), name="Input_Bias"))
        layer_drop.append(tf.nn.dropout(layer_w[0], keep_prob))
        layer_a.append(functions[0](tf.matmul(x, layer_drop[0]) + layer_b[0]))
        #Prevent overfitting with dropout


    for i in range(1,len(hidden_units)):
        with tf.name_scope("Hidden_Layer_%d" %(i)) as scope:
            layer_w.append(tf.Variable(tf.random_uniform([hidden_units[i-1], hidden_units[i]]), name="Hidden_Weight_%d" %(i)))
            layer_b.append(tf.Variable(tf.random_uniform([hidden_units[i]]), name="Hidden_Bias_%d" %(i)))
            layer_drop.append(tf.nn.dropout(layer_w[i], keep_prob))
            layer_a.append(functions[i](tf.matmul(layer_a[-1], layer_drop[i]) + layer_b[i]))


    with tf.name_scope("Output_Layer") as scope:
        layer_w.append(tf.Variable(tf.random_uniform([hidden_units[-1], nclass]), name="Output_Weight"))
        layer_b.append(tf.Variable(tf.random_uniform([nclass]), name="Output_Bias"))
        layer_drop.append(tf.nn.dropout(layer_w[-1], keep_prob))
        logits = tf.add(tf.matmul(layer_a[-1], layer_drop[-1]), layer_b[-1])
        layer_a.append(tf.nn.softmax(logits))
    output_a = layer_a[-1]


    #Training Specification
    with tf.name_scope("Train") as scope:
        epoch = tf.Variable(0)
        cost_batch = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        regularizer = tf.reduce_mean(sum([tf.nn.l2_loss(w) for w in layer_w]))
        cost = tf.reduce_mean(cost_batch)
        optimizer = tf.train.AdagradOptimizer(1e-2).minimize(cost + 5e-4*regularizer, global_step=epoch)

    # Test accuracy
    with tf.name_scope("Output_Accuracy") as scope:
        correct_prediction = tf.equal(tf.argmax(output_a, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100

    sess = tf.Session()
    costs = []
    accuracies = []

    init = tf.initialize_all_variables()
    sess.run(init)

    print "------------------------------------"
    print topology
    print funcs
    print "------------------------------------"

    with sess.as_default():
        # Training cycle
        while epoch < training_epochs:
            epoch += 1

            total_batches = int(x_train.shape[0] / batch_size)

            combined = zip(x_train, t_train)
            np.random.shuffle(combined)
            x_train[:], t_train[:] = zip(*combined)
            # Loop over all batches
            for i in range(total_batches):
                batch_xs = x_train[i*batch_size:(i+1)*batch_size]
                batch_ys = t_train[i*batch_size:(i+1)*batch_size]
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

            if epoch.eval() % 10 == 0:
                train_cost = sess.run(cost, feed_dict={x: x_train, y: t_train, keep_prob: 1})
                train_accuracy = accuracy.eval({x: x_train, y: t_train, keep_prob: 1})

                validation_cost = sess.run(cost, feed_dict={x: x_validation, y: t_validation, keep_prob: 1})
                validation_accuracy = accuracy.eval({x: x_validation, y: t_validation, keep_prob: 1})

                test_cost = sess.run(cost, feed_dict={x: x_test, y: t_test, keep_prob: 1})
                test_accuracy = accuracy.eval({x: x_test, y: t_test, keep_prob: 1})

                costs.append([train_cost, validation_cost, test_cost])
                accuracies.append([train_accuracy, validation_accuracy, test_accuracy])

                st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
                print "[%s] Epoch: %3d" % (st, epoch.eval())
                print "\tTraining Set:   Cost: %8.3f Accuracy: %d%%" %(train_cost, train_accuracy)
                print "\tValidation Set: Cost: %8.3f Accuracy: %d%%" %(validation_cost, validation_accuracy)
                print "\tTest Set:       Cost: %8.3f Accuracy: %d%%\n\n" %(test_cost, test_accuracy)

                # if len(costs) > 3 and validation_cost > 1.01*costs[-3][1] and validation_cost > costs[-2][1]:
                #     print "Early stopping!"
                #     break

        print "Optimization Finished!"

    costs = np.array(costs)

    accuracies = np.array(accuracies)

    return costs, accuracies
