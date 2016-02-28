import numpy as np

def feed_forward(inputs, weights, biases, functions):
    if not (len(weights) == len(biases) and len(biases) == len(functions)):
        raise ValueError("Inconsistent Model. \
                          Different number of weights, biases and functions.")
    layers = len(weights)
    outputs = []
    activations = []

    for l in range(layers):
        output = np.dot(inputs, weights[l]) + biases[l]
        outputs.append(output)
        inputs = functions[l](output)
        activations.append(inputs)
    final_activation = inputs
    return final_activation, outputs, activations
