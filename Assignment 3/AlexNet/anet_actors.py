################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from pylab import *
from AlexNet import net_data
from scipy.misc import imresize
from get_data_file import rgb2gray
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import time
import os

def convert_faces(faces):
    '''
        Convert faces to the shape AlexNet uses, also normalizes.
    :param faces: List of face images
    :return: List of processed face images
    '''
    for i, face in enumerate(faces):
        imgr = imresize(face, (227,227))/255.0
        img = (np.random.random((1, 227, 227, 3)) / 255.).astype('float32')
        img[0, :, :, :] = imgr[:, :, :3]
        img = img - np.mean(img)
        faces[i] = img
    return faces

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())


def alex_net_actors(params):
    '''
        Create an AlexNet network given a set of parameters
    :param params: FC1 Weight, FC1 Bias, Output Weight, Output Bias, AlexNet Network Layer
    :return: Output layer of the network
    '''
    keep_prob = tf.placeholder(tf.float32)
    img_input = tf.placeholder(tf.float32, [1, 227, 227, 3])
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(img_input, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    conv1 = tf.nn.dropout(conv1, keep_prob)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    conv2 = tf.nn.dropout(conv2, keep_prob)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    conv3 = tf.nn.dropout(conv3, keep_prob)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    conv4 = tf.nn.dropout(conv4, keep_prob)


    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)
    conv5 = tf.nn.dropout(conv5, keep_prob)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    maxpool5 = tf.nn.dropout(maxpool5, keep_prob)

    layers = [maxpool1, maxpool2, conv3, conv4, maxpool5]
    layers = layers[:params[-1]]
    used = layers[-1]
    used = tf.reshape(used, [1, int(prod(used.get_shape()[1:]))])

    #densely connected
    # input layer
    dcW = tf.Variable(params[0])
    dcB = tf.Variable(params[1])
    dcZ = tf.add(tf.matmul(used, dcW), dcB)
    dcA_p = tf.nn.relu(dcZ)
    dcA = tf.nn.dropout(dcA_p, keep_prob)
    # output layer
    oW = tf.Variable(params[2])
    oB = tf.Variable(params[3])
    oZ = tf.add(tf.matmul(dcA, oW), oB)
    oA = tf.nn.softmax(oZ)

    weights = [dcW, oW, conv5W, conv4W, conv3W, conv2W, conv1W]
    activations = layers + [dcA]

    regularizer = 0
    for i in range(params[4]):
        regularizer += tf.nn.l2_loss(weights[i])

    return img_input, oA, oZ, keep_prob, regularizer, layers

def eval_alex_net(faces, params):

    img_input, output, _, dropout, _ = alex_net_actors(params)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    outputs = []
    classes = []
    with sess.as_default():
        sess.run(init)

        for i,face in enumerate(faces):
            out = sess.run(output, feed_dict={img_input: face, dropout: 1.0})
            classes.append(tf.argmax(out,1).eval())
            outputs.append(out)
            sys.stdout.write("\r%.2f%%" %(100*(i+1)/float(len(faces))))

    return outputs, classes

def train_alex_net(sets, params, max_iter=500, lmbda=0.001, keep_prob=0.9):
    x, softmax, logits, dropout, regularizer, _ = alex_net_actors(params)
    costs = []
    accuracies = []
    x_train, t_train, x_validation, t_validation, x_test, t_test = sets
    n_classes = len(t_train[0])
    best_acc = tf.Variable(90)

    y = tf.placeholder(tf.float32, [None, n_classes])

    with tf.name_scope("Training") as scope:
        iter_var = tf.Variable(0)
        cost_batch = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        cost = tf.reduce_mean(cost_batch) + lmbda * regularizer
        optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost, global_step=iter_var)

    with tf.name_scope("Evaluation") as scope:
        correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))

    sess = tf.Session()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess.run(init)
    newbest = False


    with sess.as_default():
        epoch = iter_var.eval() / len(x_train)
        tsize, vsize, tesize = len(x_train), len(x_validation), len(x_test)
        try:
            while epoch < max_iter:
                tcost, vcost, tecost, taccuracy, vaccuracy, teaccuracy =\
                    evaluate_net(sets, x, y, dropout, sess, [cost, accuracy])

                #Train
                for i in np.random.choice(tsize, tsize):
                    sess.run(optimizer, feed_dict={x: x_train[i], y: [t_train[i]], dropout: keep_prob})

                costs.append([tcost, vcost, tecost])
                accuracies.append([taccuracy, vaccuracy, teaccuracy])
                if vaccuracy > best_acc.eval():
                    newbest = True
                    print "Found new best model!"
                    sess.run(best_acc.assign(vaccuracy))
                    saver.save(sess, "trained_alexnet.ckpt")
                    with open('results/part6_result.csv', 'a') as fl:
                        fl.write("%.4f, %.2f, %.4f, %.2f, %.4f, %.2f\n" %(tcost, taccuracy, vcost, vaccuracy, tecost, teaccuracy))

                st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
                print "[%s] Epoch: %3d" % (st, epoch)
                print "\tPenalty:              %8.5f" %(regularizer.eval()*lmbda)
                print "\tTraining Set:   Cost: %8.5f Accuracy: %d%%" %(tcost, taccuracy)
                print "\tValidation Set: Cost: %8.5f Accuracy: %d%%" %(vcost, vaccuracy)
                print "\tTest Set:       Cost: %8.5f Accuracy: %d%%\n\n" %(tecost, teaccuracy)

                epoch = iter_var.eval() / len(x_train)

        except KeyboardInterrupt:
            if raw_input("Save last model (overwrites previous best)? [y/N]").lower() == 'y':
                saver.save(sess, "trained_alexnet.ckpt")
    return np.array(costs), np.array(accuracies), newbest


def gradient_output(face_o, params):
    for i,face in enumerate(face_o):
        face_o[i] = imresize(face, (227,227))
    faces = convert_faces(deepcopy(face_o))
    img_input, output, _, dropout, _, _ = alex_net_actors(params)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    grads = tf.gradients(output, img_input)[0]

    with sess.as_default():
        sess.run(init)
        for i,face in enumerate(faces):
            grads = sess.run(grads, feed_dict={img_input:face, dropout: 1.0})[0]
            if np.max(grads) > 0:
                break

    grads = 255. * grads / np.max(grads)
    grads = np.maximum(grads, 0)
    plt.suptitle('Output Gradient', size=20)
    plt.subplot(121)
    plt.imshow(rgb2gray(grads), cmap=plt.cm.coolwarm)
    plt.title('Gradient')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(face_o[i])
    plt.title('Input image')
    plt.axis('off')

    return grads


def plot_curves(cost, accuracy):
    t = range(len(cost[:,0]))
    plt.plot(t, cost[:,0], label='Train Set')
    plt.plot(t, cost[:,1], label='Validation Set')
    plt.plot(t, cost[:,2], label='Test Set')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('AlexNet Training Curve')
    plt.grid()
    plt.savefig('results/part6_cost_curve.pdf')
    plt.close()
    plt.plot(t, accuracy[:,0], label='Train Set')
    plt.plot(t, accuracy[:,1], label='Validation Set')
    plt.plot(t, accuracy[:,2], label='Test Set')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%%)')
    plt.title('AlexNet Training Curve')
    plt.grid()
    plt.savefig('results/part6_accuracy_curve.pdf')
    plt.close()

def evaluate_net(sets, x, y, dropout, sess, routines):
    #Evaluate
    #Train set
    x_train, t_train, x_validation, t_validation, x_test, t_test = sets
    tsize, vsize, tesize = len(x_train), len(x_validation), len(x_test)
    tcost, taccuracy = 0,0
    for i in range(tsize):
        c, a = sess.run(routines, feed_dict={x: x_train[i], y: [t_train[i]], dropout: 1.0})
        tcost += c
        taccuracy += a
    tcost /= tsize
    taccuracy *= 100.0/tsize

    #Validation set
    vcost, vaccuracy = 0,0
    for i in range(vsize):
        c, a = sess.run(routines, feed_dict={x: x_validation[i], y: [t_validation[i]], dropout: 1.0})
        vcost += c
        vaccuracy += a
    vcost /= vsize
    vaccuracy *= 100.0/vsize

    #Test set
    tecost, teaccuracy = 0,0
    for i in range(tesize):
        c, a = sess.run(routines, feed_dict={x: x_test[i], y: [t_test[i]], dropout: 1.0})
        tecost += c
        teaccuracy += a
    tecost /= tesize
    teaccuracy *= 100.0/tesize

    return tcost, vcost, tecost, taccuracy, vaccuracy, teaccuracy

'''
Custom ReLU gradient which only backpropagates positive gradients
'''
@tf.RegisterGradient("CustomRelu")
def _custom_relu(op, grad):
    zero = tf.Variable(0.)
    relu = op.outputs[0]
    relub0 = tf.cast(tf.greater(relu, zero), tf.float32)
    gradb0 = tf.cast(tf.greater(grad, zero), tf.float32)
    return relub0 * gradb0 * grad

def guided_backprop(face_o, params):

    with tf.Graph().as_default() as g:
        with g.gradient_override_map({"Relu": "CustomRelu"}):
            img = tf.placeholder(tf.float32, [1, 227, 227, 3])

            #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
            conv1W = tf.Variable(net_data["conv1"][0])
            conv1b = tf.Variable(net_data["conv1"][1])
            conv1_in = conv(img, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
            conv1 = tf.nn.relu(conv1_in)

            #lrn1
            #lrn(2, 2e-05, 0.75, name='norm1')
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                              depth_radius=radius,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              bias=bias)

            #maxpool1
            #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


            #conv2
            #conv(5, 5, 256, 1, 1, group=2, name='conv2')
            k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
            conv2W = tf.Variable(net_data["conv2"][0])
            conv2b = tf.Variable(net_data["conv2"][1])
            conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv2 = tf.nn.relu(conv2_in)


            #lrn2
            #lrn(2, 2e-05, 0.75, name='norm2')
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                              depth_radius=radius,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              bias=bias)

            #maxpool2
            #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            #conv3
            #conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
            conv3W = tf.Variable(net_data["conv3"][0])
            conv3b = tf.Variable(net_data["conv3"][1])
            conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv3 = tf.nn.relu(conv3_in)

            #conv4
            #conv(3, 3, 384, 1, 1, group=2, name='conv4')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
            conv4W = tf.Variable(net_data["conv4"][0])
            conv4b = tf.Variable(net_data["conv4"][1])
            conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv4 = tf.nn.relu(conv4_in)


            #conv5
            #conv(3, 3, 256, 1, 1, group=2, name='conv5')
            k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
            conv5W = tf.Variable(net_data["conv5"][0])
            conv5b = tf.Variable(net_data["conv5"][1])
            conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv5 = tf.nn.relu(conv5_in)

            #maxpool5
            #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            layers = [maxpool1, maxpool2, conv3, conv4, maxpool5]
            layers = layers[:params[-1]]
            used = layers[-1]
            used = tf.reshape(used, [1, int(prod(used.get_shape()[1:]))])

            #densely connected
            # input layer
            dcW = tf.Variable(params[0])
            dcB = tf.Variable(params[1])
            dcZ = tf.add(tf.matmul(used, dcW), dcB)
            dcA = tf.nn.relu(dcZ)
            # output layer
            oW = tf.Variable(params[2])
            oB = tf.Variable(params[3])
            oZ = tf.add(tf.matmul(dcA, oW), oB)
            oA = tf.nn.softmax(oZ)


        sess = tf.Session()

        faces = convert_faces(face_o)

        with sess.as_default():
            for i,face in enumerate(faces):
                sess.run(tf.initialize_all_variables())
                pred = sess.run(oA, feed_dict={img: face})
                print pred
                cl = np.argmax(pred)
                oA = tf.split(1, 6, oA)[cl]
                grad = tf.gradients(oA, img)
                sess.run(tf.initialize_all_variables())

                gradient = sess.run(grad[0], feed_dict={img: face})[0]
                if np.max(gradient) > 0:
                    break

        gradient

    return gradient, cl, i

def plot_gradients(face, params, act):
    grad, c, _ = guided_backprop([face], params)
    grad2 = gradient_output([face], params)
    grad = gaussian_filter(grad, 1.5)

    grad /= np.max(grad)
    grad2 /= np.max(grad2)

    plt.subplot(3,2,1)
    plt.imshow(np.maximum(grad,0))
    plt.title('Guided Backpropagation (Pos)')
    plt.axis('off')

    plt.subplot(3,2,2)
    plt.imshow(np.abs(grad))
    plt.title('Guided Backpropagation (Abs)')
    plt.axis('off')

    plt.subplot(3,2,3)
    plt.imshow(np.minimum(grad,0))
    plt.title('Guided Backpropagation (Neg)')
    plt.axis('off')

    plt.subplot(3,2,4)
    plt.imshow(grad)
    plt.title('Guided Backpropagation')
    plt.axis('off')

    plt.subplot(3,2,5)
    plt.imshow(face)
    plt.title('Input Image\n%s' %act[c])
    plt.axis('off')

    plt.subplot(3,2,6)
    plt.imshow(rgb2gray(grad2))
    plt.gray()
    plt.title('Gradient')
    plt.axis('off')

    plt.show()

    return grad, grad2