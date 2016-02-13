from part3 import *

def backpropagate_momentum_epoch(x, w, b, y, rate, momentum, batch_size):
    '''
    We reimplement the gradient calculation routines in-line for performance optimization
    '''
    new_grads = [0.0, 0.0]
    np.random.seed(0)
    m = len(x)
    ids = np.random.permutation(m)
    x, y = x[ids], y[ids]
    total_batches = m/batch_size

    for i in range(total_batches):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        y_batch = y[i*batch_size:(i+1)*batch_size]

        # Gradient Calculation
        z = x_batch.dot(w) + b
        delta = softmax(z)-y_batch
        grad_weight = (1.0/batch_size) * x_batch.T.dot(delta)
        grad_bias = (1.0/batch_size) * sum(delta)
        # --------------------

        old_grads = new_grads
        new_grads = [grad_weight, grad_bias]
        w += -rate * new_grads[0] + momentum * old_grads[0]
        b += -rate * new_grads[1] + momentum * old_grads[1]

    return w, b

def evaluate_accuracy(x, w, b, y):
    output = feed_forward(x, [w], [b], [softmax])[0]
    successes = np.equal(np.argmax(output, 1), np.argmax(y, 1))
    matches = np.where(successes == True)[0]
    accuracy = np.sum(successes).astype('float32') / len(y)
    return accuracy, matches, np.argmax(output, 1)

def evaluate_cost(x, w, b, y):
    cost = cross_entropy(x, [w], [b], [softmax], y)
    return np.sum(cost)