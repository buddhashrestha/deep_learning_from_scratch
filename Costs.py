
import numpy as np

class Cost(object):

    def compute(self,AL, Y, loss="cross_entropy"):
        if(loss=="cost_entropy"):
            return self.cross_entropy(AL, Y), self.cross_entropy_backward(AL, Y)

    def cross_entropy(AL, Y):

        AL_log = -np.log(AL)

        each_element = np.multiply(AL_log, Y)

        entropy = np.sum(each_element, axis=0)

        return np.mean(entropy)

    def cross_entropy_backward(x, y):
        return x - y  # lamda rakhne ki narakhne??Z

    def mean_squared_error(AL, Y):
        return np.mean(np.sum(np.square(AL - Y), axis=0))


    def logistics(AL, Y):

        m = Y.shape[1]
        # Compute loss from aL and y.
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        return cost