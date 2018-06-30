
import numpy as np



def compute_cost(AL, Y, loss="cross_entropy"):
    if(loss == "cross_entropy"):
        cost = cross_entropy(AL, Y)
        print("COST : ",cost)
        derivative = cross_entropy_backward(AL, Y)
        print("DAL: \n\n\n\n",derivative)
        return cost, derivative
    else:
        cost = mean_squared_error(AL, Y)
        derivative = cross_entropy_backward(AL, Y)
        return cost, derivative

def cross_entropy(AL, Y):
    print("AL: ",AL)
    AL_log = -np.log(AL)

    each_element = np.multiply(AL_log, Y)

    entropy = np.sum(each_element, axis=0)

    return np.mean(entropy)

def cross_entropy_backward(x, y):
    return x - y

def mean_squared_error(AL, Y):
    return np.mean(np.sum(np.square(AL - Y), axis=0))

def mean_squared_error_backward(x,y):
    return x-y

def logistics(AL, Y):

    m = Y.shape[1]
    # Compute loss from aL and y.
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost