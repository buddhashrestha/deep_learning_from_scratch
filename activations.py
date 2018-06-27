import numpy as np

def sigmoid(Z):

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def sigmoid_backward(dA, cache):

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ

def relu(Z):

    A = np.maximum(0, Z)

    cache = Z

    return A, cache


def relu_backward(dA, cache):

    Z = cache

    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    # dZ[Z > 0] = 1
    dZ[Z <= 0] = 0

    return dZ

def softmax(Z):

    A = np.exp(Z - np.max(Z, axis=0))
    cache = Z
    return A / A.sum(axis=0), cache


def bipolar_sigmoid(Z):
    A = (1.0 - np.square(Z)) / 2.0

    cache = Z

    return A, cache

def bipolar_sigmoid_backward(dA, cache):

    Z = cache

    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    # dZ[Z > 0] = 1
    dZ[Z <= 0] = 0

    return dZ

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
