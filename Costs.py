
import numpy as np

class Cost(object):

    def cross_entropy(AL, Y):

        AL_log = -np.log(AL)

        each_element = np.multiply(AL_log, Y)

        entropy = np.sum(each_element, axis=0)

        return np.mean(entropy)


    def mean_squared_error(AL, Y):
        return np.mean(np.sum(np.square(AL - Y), axis=0))


    def logistics(AL, Y):

        m = Y.shape[1]
        # Compute loss from aL and y.
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        return cost