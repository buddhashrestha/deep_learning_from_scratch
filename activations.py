import numpy as np



def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    cache = Z

    return A, cache


def softmax(Z):
    """
        Implement the softmax function.

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """
    A = np.exp(Z - np.max(Z, axis=0))
    cache = Z
    return A / A.sum(axis=0), cache


def bipolar_sigmoid_function(Z):
    A = 2.0 / (1.0 + np.exp(-Z)) - 1.0

    cache = Z

    return A, cache


def bipolar_sigmoid_function_derivative(Z):
    A = (1.0 - np.square(Z)) / 2.0

    cache = Z

    return A, cache


def hyperbolic_tangent(Z):
    A = np.tanh(Z)

    cache = Z

    return A, cache


def hyperbolic_tangent_derivative(x):
    A = 1.0 - np.square(x)

    cache = Z

    return A, cache


##THINK ABOUT LEAKY RELU
