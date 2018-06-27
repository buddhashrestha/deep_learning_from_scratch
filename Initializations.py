import numpy as np


def initialize(activations,layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        if(activations[l-1]=="relu" or activations[l-1]=="leaky_relu"):
            parameters['W' + str(l)] = he_weight_initializer(layer_dims[l],layer_dims[l-1])
        else:
            parameters['W' + str(l)] = xavier_weight_initializer(layer_dims[l],layer_dims[l-1])

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def xavier_weight_initializer(output_dims, input_dims):
    return np.random.randn(output_dims, input_dims) * np.sqrt(1.0 / input_dims)

def he_weight_initializer(output_dims, input_dims):

    return np.random.randn(output_dims, input_dims) * np.sqrt(2.0 / input_dims)

