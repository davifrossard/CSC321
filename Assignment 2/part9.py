from part7 import *

def backpropagate_momentum_epoch2(x, w, b, y, rate, momentum, batch_size):
    '''
    We reimplement the gradient calculation routines in-line for performance optimization
    '''
    new_grads = [[0.0, 0.0], [0.0, 0.0]]
    np.random.seed(0)
    m = len(x)
    ids = np.random.permutation(m)
    x, y = x[ids], y[ids]
    total_batches = m/batch_size

    for i in range(total_batches):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        y_batch = y[i*batch_size:(i+1)*batch_size]
        old_grads = new_grads

        # Gradient Calculation
        w2, b2 = w[1], b[1]
        w1, b1 = w[0], b[0]

        z1 = x_batch.dot(w1) + b1
        a1 = tanh(z1)
        z2 = a1.dot(w2) + b2
        a2 = softmax(z2)

        delta2 = (a2-y_batch)

        grad_weight2 = (1.0/batch_size) * a1.T.dot(delta2)
        grad_bias2 = (1.0/batch_size) * sum(delta2)

        delta1 = np.dot(delta2, w2.T) * dtanh(z1)

        grad_weight1 = (1.0/batch_size) * np.dot(x_batch.T, delta1)
        grad_bias1 = (1.0/batch_size) * np.sum(delta1, axis=0)

        # --------------------

        new_grads = [[grad_weight2, grad_bias2], [grad_weight1, grad_bias1]]
        w[0] += -rate * new_grads[1][0] + momentum * old_grads[1][0]
        b[0] += -rate * new_grads[1][1] + momentum * old_grads[1][1]
        w[1] += -rate * new_grads[0][0] + momentum * old_grads[0][0]
        b[1] += -rate * new_grads[0][1] + momentum * old_grads[0][1]

    return w, b

def evaluate_accuracy2(x, w, b, y):
    output = feed_forward(x, w, b, [tanh, softmax])[0]
    successes = np.equal(np.argmax(output, 1), np.argmax(y, 1))
    matches = np.where(successes == True)[0]
    accuracy = np.sum(successes).astype('float32') / len(y)
    return accuracy, matches, np.argmax(output, 1)

def evaluate_cost2(x, w, b, y):
    cost = cross_entropy(x, w, b, [tanh, softmax], y)
    return np.sum(cost)