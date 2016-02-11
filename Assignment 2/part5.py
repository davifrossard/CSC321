from part3 import *

def backpropagate_momentum_step(x, w, b, y, rate, momentum, batch_size):

    new_grads = [dCost_dWeight(x, w, b, y), dCost_dBias(x, w, b, y)]
    total_batches = len(y)/batch_size

    for i in range(total_batches):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        y_batch = y[i*batch_size:(i+1)*batch_size]
        old_grads = new_grads
        new_grads = [dCost_dWeight(x_batch, w, b, y_batch), dCost_dBias(x_batch, w, b, y_batch)]
        w += -rate * new_grads[0].T - momentum * old_grads[0].T
        b += -rate * new_grads[1] - momentum * old_grads[1]

    return w, b

def evaluate_accuracy(x, w, b, y):
    output = feed_forward(x, [w], [b], [softmax])[0]
    accuracy = np.sum(np.equal(np.argmax(output, 1), np.argmax(y, 1)))
    return accuracy.astype('float32')/len(y)

def evaluate_cost(x, w, b, y):
    cost = cross_entropy(x, [w], [b], [softmax], y)
    return np.sum(cost)