import numpy as np


def initialize(activations,layer_dims,input_dim,keep_prob = 0.5):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(L):

        if(activations[l]=="relu" or activations[l]=="leaky_relu"):
            parameters['W' + str(l+1)] = he_weight_initializer(input_dim,layer_dims[l])
        else:
            parameters['W' + str(l+1)] = xavier_weight_initializer(input_dim,layer_dims[l])

        parameters['b' + str(l+1)] = np.zeros([layer_dims[l]])
        # parameters['beta' + str(l+1)] = np.zeros([layer_dims[l]])
        # parameters['gamma' + str(l+1)] = np.ones([layer_dims[l]])

        input_dim = layer_dims[l]

    # # Initialise the weights and biases for final FC layer
    # parameters['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, [input_dim, num_classes])
    # parameters['b' + str(self.num_layers)] = np.zeros([num_classes])

    return parameters

def xavier_weight_initializer(output_dims, input_dims):
    return np.random.randn(output_dims, input_dims) * np.sqrt(1.0 / input_dims)

def he_weight_initializer(output_dims, input_dims):

    return np.random.randn(output_dims, input_dims) * np.sqrt(2.0 / input_dims)


def initialize_velocity(parameters):

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
        ### END CODE HERE ###

    return v


def initialize_adam(parameters):

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        ### START CODE HERE ### (approx. 4 lines)
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    ### END CODE HERE ###

    return v, s
