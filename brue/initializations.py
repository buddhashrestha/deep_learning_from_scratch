import numpy as np



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
